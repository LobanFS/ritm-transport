"""Причина серого маркера должна следовать доступным входам, а не цвету/проценту."""
from datetime import datetime, timedelta, timezone

import pytest

from backend.arrivals import ArrivalInput
from backend.engine import DelayHint, Engine, LiveContext, gate_prediction
from common.contracts import Prediction, Telemetry


T = datetime(2026, 1, 6, 8, tzinfo=timezone.utc)


@pytest.fixture
def engine():
    value = Engine('http://unused')
    value.set_live(LiveContext.model_validate(dict(
        arrival_mode='gps',
        vehicles=[dict(tr_id=1, unit_id=2, label='ТС 1', route_id='r')],
        routes=[dict(route_id='r', name='Маршрут', color='#123456',
                     path=[[37.6, 55.75], [37.61, 55.75]])],
        schedule=[dict(tr_id=1, target=dict(id=name, name=name,
            scheduled_at=T+timedelta(seconds=seconds), lat=55.75, lon=lon))
            for name, seconds, lon in [('past', -600, 37.6), ('target', 720, 37.61)]],
    )))
    value.mode = 'demo'
    value.clock = T
    return value


def arrival(engine, *, age=30, received_offset=0, source='gps_estimate', stop='past'):
    engine.ingest_arrival(ArrivalInput(tr_id=1, planned_stop_id=stop,
        arrived_at=T-timedelta(seconds=age)),
        received_at=T+timedelta(seconds=received_offset), source=source)


def forecast(engine, *, method='learned', risk='green', delay=30):
    return Prediction(request_id='1', tr_id=1, issued_at=T,
        target=engine.schedule[1][1], predicted_delay_s=delay,
        model_version='test', method=method, risk=risk,
        probability_late=None, reasons=[])


def availability(engine, *, at=T, **overrides):
    args = dict(target=engine.schedule[1][1], prediction=forecast(engine), telemetry_age_s=1)
    args.update(overrides)
    return engine.prediction_availability_at(1, at, **args)


@pytest.mark.parametrize('source', ['gps_estimate', 'door_estimate', 'csv_snapshot'])
def test_expiry_exact_boundary_and_late_known_observation(engine, source):
    if source == 'csv_snapshot':
        engine.hints[1].append(DelayHint(tr_id=1, observed_at=T-timedelta(seconds=300),
            received_at=T, delay_s=60, source=source))
    else:
        arrival(engine, age=300, source=source)
    assert availability(engine)['code'] == 'ready'
    expired = availability(engine, at=T+timedelta(seconds=1))
    assert expired['code'] == 'deviation_expired'
    assert '5 мин 1 с' in expired['message']
    assert engine.deviation_at(1, T+timedelta(seconds=1)) is None


@pytest.mark.parametrize('observed_offset,received_offset', [(-600, 10), (10, 10)])
def test_future_arrival_or_delivery_cannot_appear_in_diagnostics(engine, observed_offset, received_offset):
    arrival(engine, age=-observed_offset, received_offset=received_offset)
    assert availability(engine)['code'] == 'deviation_missing'


@pytest.mark.parametrize('observed_offset,received_offset', [(-600, 10), (10, 10)])
def test_future_hint_or_delivery_cannot_appear_in_diagnostics(engine, observed_offset, received_offset):
    engine.hints[1].append(DelayHint(tr_id=1, observed_at=T+timedelta(seconds=observed_offset),
        received_at=T+timedelta(seconds=received_offset), delay_s=60, source='csv_snapshot'))
    assert availability(engine)['code'] == 'deviation_missing'


def test_future_operational_revision_keeps_past_expiry_and_later_fact_is_not_ttl_limited(engine):
    arrival(engine, age=301)
    arrival(engine, age=310, received_offset=10, source='arrival')
    assert availability(engine)['code'] == 'deviation_expired'
    assert availability(engine, at=T+timedelta(seconds=11))['code'] == 'ready'


def test_fresh_hint_can_replace_expired_gps_for_availability(engine):
    arrival(engine, age=600)
    engine.hints[1].append(DelayHint(tr_id=1, observed_at=T, received_at=T,
        delay_s=60, source='csv_snapshot'))
    assert availability(engine)['code'] == 'ready'


def test_single_slow_gps_point_is_candidate_not_current_deviation(engine):
    engine.ingest(Telemetry(tr_id=1, unit_id=2, event_time=T, received_at=T,
        lat=55.75, lon=37.6, speed_kmh=0, event_id='one'))
    assert engine.gps_detectors[1].status()['candidate_points'] == 1
    assert availability(engine)['code'] == 'deviation_missing'


@pytest.mark.parametrize('overrides,code', [
    ({'source_available':False}, 'source_unavailable'),
    ({'target':None}, 'no_target'),
    ({'telemetry_age_s':None}, 'telemetry_missing'),
    ({'telemetry_age_s':61}, 'telemetry_stale'),
    ({'prediction':None}, 'prediction_pending'),
])
def test_primary_input_blockers(engine, overrides, code):
    arrival(engine)
    assert availability(engine, **overrides)['code'] == code


def test_no_target_does_not_blame_unknown_deviation(engine):
    assert availability(engine, target=None)['code'] == 'no_target'


def test_registered_target_is_not_early_prediction(engine):
    arrival(engine, stop='target')
    assert availability(engine)['code'] == 'target_reached'


@pytest.mark.parametrize('method', ['learned', 'fallback', 'persistence'])
def test_numeric_prediction_is_ready_without_probability(engine, method):
    arrival(engine)
    prediction = forecast(engine, method=method)
    assert prediction.probability_late is None
    assert availability(engine, prediction=prediction)['code'] == 'ready'


@pytest.mark.parametrize('risk,delay', [('unknown', 30), ('unknown', None)])
def test_unknown_model_output_is_not_ready(engine, risk, delay):
    arrival(engine)
    assert availability(engine, prediction=forecast(engine, risk=risk, delay=delay))['code'] == 'model_input_unavailable'


def test_gate_does_not_claim_previous_number_when_model_returned_none(engine):
    empty = gate_prediction(forecast(engine, method='unavailable', risk='unknown', delay=None),
        telemetry_age_s=1, has_current_deviation=False, source_available=False)
    assert all('ранее рассчитанный прогноз' not in reason for reason in empty.reasons)
    retained = gate_prediction(forecast(engine), telemetry_age_s=1, has_current_deviation=False)
    assert any('ранее рассчитанный прогноз' in reason for reason in retained.reasons)


def test_state_exposes_reason_without_mutating_prediction(engine):
    arrival(engine, age=600)
    engine.ingest(Telemetry(tr_id=1, unit_id=2, event_time=T, received_at=T,
        lat=55.75, lon=37.605, speed_kmh=25, heading=90, event_id='position'))
    engine.predictions[1] = forecast(engine)
    vehicle = engine.state()['vehicles'][0]
    assert vehicle['prediction_availability']['code'] == 'deviation_expired'
    assert vehicle['prediction']['risk'] == 'unknown'
    assert vehicle['heading'] == 90
    assert engine.predictions[1].risk == 'green'
