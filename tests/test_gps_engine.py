"""GPS → доступное прибытие → отклонение; без истинных событий генератора/модели."""
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from backend.app import create_app
from backend.arrivals import ArrivalInput
from backend.engine import Engine, LiveContext
from common.contracts import Telemetry

T = datetime(2026, 1, 6, 8, tzinfo=timezone.utc)


def context(*, arrival_mode='gps'):
    return LiveContext.model_validate(dict(
        arrival_mode=arrival_mode,
        vehicles=[dict(tr_id=1, unit_id=2, label='ТС 1', route_id='r')],
        routes=[dict(route_id='r', name='Маршрут', color='#123456',
                     path=[[37.6, 55.75], [37.605, 55.75], [37.615, 55.75]])],
        schedule=[dict(tr_id=1, target=dict(id=name, name=name,
                    scheduled_at=T+timedelta(seconds=seconds), lat=55.75, lon=lon))
                  for name, seconds, lon in [('first', -120, 37.6), ('second', 60, 37.605),
                                            ('target', 720, 37.615)]],
    ))


def point(seconds=0, *, receive=None, lon=37.6, speed=0):
    return Telemetry(tr_id=1, unit_id=2, event_time=T+timedelta(seconds=seconds),
        received_at=T+timedelta(seconds=seconds if receive is None else receive),
        lat=55.75, lon=lon, speed_kmh=speed, event_id=str(seconds))


def engine(monkeypatch, *, arrival_mode='gps'):
    clock = [T]
    monkeypatch.setattr('backend.engine.utcnow', lambda: clock[0])
    e = Engine('http://unused')
    e.set_live(context(arrival_mode=arrival_mode))
    return e, clock


def test_gps_confirmation_has_formula_and_receipt_boundary_and_stable_between_visits(monkeypatch):
    e, clock = engine(monkeypatch)
    e.ingest(point(0))
    assert e.deviation_at(1, T) is None
    clock[0] = T+timedelta(seconds=10)
    e.ingest(point(5, receive=10))
    assert e.deviation_at(1, T+timedelta(seconds=9)) is None
    value = e.deviation_at(1, clock[0])
    assert value.source == 'gps_estimate'
    assert value.delay_s == 120
    assert value.observed_at == T and value.received_at == clock[0]
    assert value.confirmation_span_s == 5
    for seconds, lon in [(20, 37.602), (40, 37.603), (60, 37.604)]:
        clock[0] = T+timedelta(seconds=seconds)
        e.ingest(point(seconds, lon=lon, speed=20))
        assert e.deviation_at(1, clock[0]).delay_s == 120
    for seconds in (80, 85):
        clock[0] = T+timedelta(seconds=seconds)
        e.ingest(point(seconds, lon=37.605))
    second = e.deviation_at(1, clock[0])
    assert second.planned_stop_id == 'second' and second.delay_s == 20
    assert len(e.gps_events) == 2
    assert e.gps_events[0]['available_at'] == clock[0].isoformat()


def test_valid_but_not_yet_available_gps_cannot_confirm_until_both_clocks_arrive(monkeypatch):
    e, clock = engine(monkeypatch)
    e.ingest(point(0))
    e.ingest(point(5, receive=10))
    assert len(e.gps_pending[1]) == 1 and not e.arrivals[1]
    e.process_pending_gps(T+timedelta(seconds=9))
    assert e.deviation_at(1, T+timedelta(seconds=9)) is None
    clock[0] = T+timedelta(seconds=10)
    e.process_pending_gps(clock[0])
    value = e.deviation_at(1, clock[0])
    assert value.delay_s == 120 and value.received_at == clock[0]
    assert e.deviation_at(1, T+timedelta(seconds=9)) is None
    assert not e.gps_pending[1]


def test_impossible_event_after_receipt_never_becomes_usable_when_clock_catches_up(monkeypatch):
    e, clock = engine(monkeypatch)
    e.ingest(point(0))
    e.ingest(point(5, receive=0))
    assert not e.arrivals[1]
    clock[0] = T+timedelta(seconds=10)
    e.process_pending_gps(clock[0])
    assert not e.arrivals[1]
    assert e.gps_detectors[1].status()['confirmed_arrivals'] == 0
    e.ingest(point(10))
    assert e.deviation_at(1, clock[0]).received_at == clock[0]


def test_external_mode_does_not_create_gps_arrivals_or_override_external_fact(monkeypatch):
    e, clock = engine(monkeypatch, arrival_mode='external')
    e.ingest(point())
    clock[0] = T+timedelta(seconds=5)
    e.ingest(point(5))
    assert not e.gps_detectors and e.deviation_at(1, clock[0]) is None
    fact = ArrivalInput(tr_id=1, planned_stop_id='first', arrived_at=T-timedelta(seconds=3))
    e.ingest_arrival(fact, received_at=clock[0])
    assert e.deviation_at(1, clock[0]).delay_s == 117
    assert e.deviation_at(1, clock[0]).source == 'arrival'


def test_operational_fact_can_correct_gps_estimate_and_gps_cannot_rewrite_fact(monkeypatch):
    e, clock = engine(monkeypatch)
    e.ingest(point())
    clock[0] = T+timedelta(seconds=5)
    e.ingest(point(5))
    assert e.deviation_at(1, clock[0]).source == 'gps_estimate'
    clock[0] = T+timedelta(seconds=10)
    correction = ArrivalInput(tr_id=1, planned_stop_id='first', arrived_at=T-timedelta(seconds=3))
    assert e.ingest_arrival(correction, received_at=clock[0])
    corrected = e.deviation_at(1, clock[0])
    assert corrected.source == 'arrival' and corrected.delay_s == 117
    earlier = e.deviation_at(1, T+timedelta(seconds=5))
    assert earlier.source == 'gps_estimate' and earlier.delay_s == 120
    assert e.deviation_at(1, T+timedelta(seconds=4)) is None
    assert not e.ingest_arrival(ArrivalInput(tr_id=1, planned_stop_id='first', arrived_at=T),
                                received_at=clock[0], source='gps_estimate')
    assert e.deviation_at(1, clock[0]).model_dump() == corrected.model_dump()


def test_context_export_contains_only_plan_and_vehicle_mapping_even_when_facts_exist(monkeypatch):
    monkeypatch.setattr('backend.engine.utcnow', lambda: T)
    with TestClient(create_app(start_background=False, enable_ndtp=False)) as client:
        assert client.post('/api/v1/live/context', json=context().model_dump(mode='json')).status_code == 200
        e = client.app.state.engine
        e.ingest_arrival(ArrivalInput(tr_id=1, planned_stop_id='first', arrived_at=T), received_at=T)
        from backend.engine import DelayHint
        e.hints[1].append(DelayHint(tr_id=1, observed_at=T, received_at=T, delay_s=999))
        response = client.get('/api/v1/context')
        assert response.status_code == 200
        exported = response.json()
        assert exported['arrival_mode'] == 'gps' and exported['hints'] == []
        assert len(exported['schedule']) == 3 and exported['vehicles'][0]['unit_id'] == 2
        serialized = response.text
        assert 'arrived_at' not in serialized and 'current_delay' not in serialized
        assert 'observed_at' not in serialized and 'delay_s' not in serialized
        assert LiveContext.model_validate(exported).model_dump() == context().model_dump()
        polluted = {**exported, 'arrivals': [dict(tr_id=1, planned_stop_id='first', arrived_at=T.isoformat())]}
        assert client.post('/api/v1/live/context', json=polluted).status_code == 422


def test_gps_snapshot_expires_but_visit_history_and_operational_fact_remain(monkeypatch):
    e, clock = engine(monkeypatch)
    e.ingest(point())
    clock[0] = T+timedelta(seconds=5)
    e.ingest(point(5))
    assert e.deviation_at(1,T+timedelta(seconds=300)).valid_until == T+timedelta(seconds=300)
    assert e.deviation_at(1,T+timedelta(seconds=301)) is None
    assert e.request_for(1,T+timedelta(seconds=301)) is None  # нет цели в (10,15]
    assert len(e.gps_events) == 1 and len(e.arrivals[1]) == 1
    e.ingest_arrival(ArrivalInput(tr_id=1, planned_stop_id='first', arrived_at=T),
                     received_at=T+timedelta(seconds=310))
    assert e.deviation_at(1,T+timedelta(seconds=900)).source == 'arrival'


def test_door_estimate_waits_for_receipt_expires_and_allows_operational_correction(monkeypatch):
    e, clock = engine(monkeypatch)
    e.ingest(point(0, lon=37.598, speed=25).model_copy(update={'doors_open':False,'door_sensor_key':'fixture:door1'}))
    e.ingest(point(15, receive=20).model_copy(update={'doors_open':True,'door_sensor_key':'fixture:door1'}))
    e.process_pending_gps(T+timedelta(seconds=19))
    assert e.deviation_at(1,T+timedelta(seconds=19)) is None
    clock[0] = T+timedelta(seconds=20)
    e.process_pending_gps(clock[0])
    value = e.deviation_at(1,clock[0])
    assert value.source == 'door_estimate' and value.delay_s == 135
    request = e.request_for(1,clock[0])
    assert request.current_delay_source == 'door_estimate' and request.current_delay_s == 135
    assert e.deviation_at(1,T+timedelta(seconds=315)) is not None
    assert e.deviation_at(1,T+timedelta(seconds=316)) is None
    assert e.target_reached_at(1,'first',T+timedelta(seconds=400))
    e.ingest_arrival(ArrivalInput(tr_id=1,planned_stop_id='first',arrived_at=T+timedelta(seconds=12)),
                     received_at=T+timedelta(seconds=320))
    assert e.deviation_at(1,T+timedelta(seconds=400)).source == 'arrival'
    assert e.deviation_at(1,T+timedelta(seconds=20)).source == 'door_estimate'
