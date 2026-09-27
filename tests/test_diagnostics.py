"""Не превращать пропуски/будущие точки в причину задержки."""
from datetime import datetime, timedelta, timezone
import asyncio
from fastapi.testclient import TestClient
from backend.app import create_app
from backend.diagnostics import explain
from backend.engine import Engine
from backend.arrivals import CurrentDeviation
from common.contracts import StopTarget, Telemetry

T=datetime(2026,1,6,8,tzinfo=timezone.utc)


def points(offsets, speed=0):
    return [Telemetry(tr_id=101,event_time=T+timedelta(seconds=s),received_at=T+timedelta(seconds=s),
                      lat=55.75,lon=37.60,speed_kmh=speed) for s in offsets]


def test_long_stop_has_measured_evidence_but_not_a_physical_cause_without_delay_context():
    result=explain(points(range(-120,1,15)),T)
    assert result.cause_status=='unknown'
    assert result.observation_status=='observed'
    assert result.observations[0].code=='long_stop'
    assert '120 с' in result.observations[0].evidence
    assert 'пробка' not in result.possible_cause.lower()
    assert 'посадка' not in result.possible_cause.lower()


def test_two_points_separated_by_gap_are_not_a_stop():
    result=explain(points([-180,0]),T)
    assert result.cause_status=='unknown'
    assert result.observations[0].code=='short_history'


def test_future_receive_and_future_events_do_not_provide_evidence():
    history=points(range(-120,1,15))
    history=[p.model_copy(update={'received_at':T+timedelta(seconds=1)}) for p in history[:-1]]+history[-1:]
    history+=points(range(15,121,15))
    assert explain(history,T).cause_status=='unknown'


def test_unknown_speed_breaks_stop_continuity():
    history=points(range(-120,1,15))
    history[5]=history[5].model_copy(update={'speed_kmh':None})
    assert explain(history,T).cause_status=='unknown'


def test_conflicting_duplicates_cannot_fill_history():
    history=points(range(-120,1,15))
    history.append(history[5].model_copy(update={'speed_kmh':25}))
    history.append(history[5])
    assert explain(history,T).cause_status=='unknown'


def test_stale_gps_is_quality_problem_not_traffic_jam():
    result=explain(points(range(-240,-60,15)),T)
    assert result.cause_status=='unknown' and result.observations[0].code=='stale'


def test_speed_drop_and_normal_movement():
    history=points(range(-180,-60,15),30)+points(range(-60,1,15),5)
    result=explain(history,T)
    assert result.cause_status=='unknown' and result.observations[0].code=='speed_drop'
    assert result.observation_status=='observed'
    assert explain(points(range(-180,1,15),30),T).cause_status=='unknown'


def test_incident_explanation_is_frozen_at_issue_time():
    async def exercise():
        e=Engine('http://unavailable')
        await e.tick(0)
        old=e.incidents[0]['explanation']
        for _ in range(10): await e.tick(1)
        assert e.state()['vehicles'][0]['explanation']['observation_status']=='observed'
        assert e.incidents[-1]['explanation']==old
        assert old['cause_status']=='unknown'
    asyncio.run(exercise())


def deviation(id='a', seconds=-30, delay=100, received=None, source='arrival'):
    return CurrentDeviation(delay_s=delay, source=source, planned_stop_id=id,
        observed_at=T+timedelta(seconds=seconds),
        received_at=T+timedelta(seconds=seconds if received is None else received))


def test_short_stop_is_useful_observation_without_inventing_a_cause():
    result=explain(points([-45,-30,-15,0]),T)
    assert result.observation_status=='observed'
    assert result.observations[0].code=='observed_stop'
    assert '45 с' in result.summary
    assert result.cause_status=='unknown'


def test_planned_stop_or_layover_is_not_declared_cause_even_with_delay():
    stop=StopTarget(id='a', name='Конечная', scheduled_at=T, lat=55.75, lon=37.6)
    result=explain(points(range(-120,1,15)),T,current_deviation=deviation(),schedule=[stop])
    assert 'Конечная' in result.summary
    assert result.cause_status=='unknown'
    assert result.observations[0].code=='long_stop'
    approaching=points(range(-180,-60,15),30)+points(range(-60,1,15),5)
    result=explain(approaching,T,current_deviation=deviation(),schedule=[stop])
    assert result.observations[0].code=='speed_drop'
    assert result.cause_status=='unknown'


def test_known_delay_and_long_stop_away_from_planned_stop_allow_only_hypothesis():
    stop=StopTarget(id='a', name='Следующая', scheduled_at=T, lat=55.85, lon=37.6)
    result=explain(points(range(-120,1,15)),T,current_deviation=deviation(),schedule=[stop])
    assert result.cause_status=='hypothesis'
    assert 'может' in result.possible_cause
    assert 'пробк' not in result.possible_cause


def test_low_speed_is_observed_without_requiring_preceding_fast_motion():
    result=explain(points(range(-120,1,15),6),T)
    assert result.observations[0].code=='slow_movement'
    assert result.cause_status=='unknown'
    delayed=explain(points(range(-120,1,15),6),T,current_deviation=deviation())
    assert delayed.cause_status=='hypothesis'


def test_gps_drift_does_not_turn_zero_speed_into_a_long_stationary_observation():
    history=points(range(-120,1,15))
    history=[p.model_copy(update={'lat':55.75+i*.001}) for i,p in enumerate(history)]
    result=explain(history,T)
    assert all(item.code not in ('long_stop','observed_stop') for item in result.observations)


def test_delay_change_compares_distinct_available_visits_not_future_or_revisions():
    current=deviation('current',-30,130)
    previous=deviation('previous',-120,70)
    future_revision=deviation('previous',-120,500,received=1)
    future_visit=deviation('future',10,900)
    same_visit=deviation('current',-40,-999)
    result=explain(points(range(-120,1,15),30),T,current_deviation=current,
        deviation_history=[previous,future_revision,future_visit,same_visit])
    item=next(item for item in result.observations if item.code=='deviation_increased')
    assert '+70 → +130 с' in item.evidence
    assert '60 с' in result.summary
    assert result.cause_status=='unknown'


def test_delay_correction_replaces_old_visit_before_trend_is_computed():
    old=deviation('previous',-120,500,source='gps_estimate')
    corrected=deviation('previous',-125,110,received=-50)
    result=explain(points(range(-120,1,15),30),T,current_deviation=deviation('current',-20,100),
        deviation_history=[old,corrected])
    assert all(item.code not in ('deviation_increased','deviation_decreased') for item in result.observations)


def test_future_or_expired_current_delay_does_not_support_a_cause_or_trend():
    history=points(range(-120,1,15),6)
    for current in [deviation(received=1),deviation(seconds=1),
                    deviation().model_copy(update={'valid_until':T-timedelta(seconds=1)})]:
        result=explain(history,T,current_deviation=current,deviation_history=[deviation('past',-120,0)])
        assert result.cause_status=='unknown'
        assert all(item.kind!='schedule' for item in result.observations)


def test_repeated_timestamp_with_conflicting_position_cannot_create_a_stop():
    history=points(range(-120,1,15))
    history += [history[-1].model_copy(update={'lat':55.8}),history[-1]]
    result=explain(history,T)
    assert result.observation_status=='insufficient_data'


def test_impossible_receipt_time_is_not_diagnostic_evidence():
    history=[p.model_copy(update={'received_at':p.event_time-timedelta(seconds=1)})
             for p in points(range(-120,1,15))]
    result=explain(history,T)
    assert result.observations[0].code=='stale'
    assert result.cause_status=='unknown'


def test_export_has_timestamps_model_section_and_no_actual_event_claim():
    with TestClient(create_app(start_background=False,enable_ndtp=False)) as client:
        client.app.state.engine.reset_demo()
        client.app.state.engine.client = None  # Проверка экспорта не зависит от живого ML.
        asyncio.run(client.app.state.engine.tick(0))
        r=client.get('/api/v1/incidents/export')
        assert r.status_code==200
        assert 'attachment' in r.headers['content-disposition']
        body=r.json()
        assert body['mode']=='demo' and body['horizon_reference']=='scheduled_arrival'
        event=body['incidents'][0]
        assert event['model_version']=='persistence-v1' and '→' in event['section']
        assert 600<event['lead_time_s']<=900
        assert event['explanation']['evaluated_at']==event['created_at']
        client.post('/api/v1/mode',json={'mode':'live'})
        assert client.get('/api/v1/incidents/export').json()['incidents']==[]
