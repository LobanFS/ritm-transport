"""What-if: только доступные прогнозы, соседняя линия и явная цена донору."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient
import pytest

from backend.app import create_app
from backend.transfer_advisor import advise_transfer, same_line
from common.contracts import StopTarget

T = datetime(2026, 1, 6, 12, tzinfo=timezone.utc)


def stop(identifier, seconds, lon=37.61, lat=55.75):
    return StopTarget(id=identifier, name=identifier, scheduled_at=T+timedelta(seconds=seconds), lon=lon, lat=lat)


def bus(tr_id, route, delay, lon=37.615, lat=55.75):
    return dict(tr_id=tr_id, label=f'Автобус {tr_id}', route_id=route, lon=lon, lat=lat,
                status='fresh', age_s=0, event_time=T, prediction_availability={'code': 'ready'},
                prediction=dict(issued_at=T, target=stop(f'target-{tr_id}', 720).model_dump(mode='json'),
                                risk='red' if delay > 150 else 'green', predicted_delay_s=delay))


@pytest.fixture
def inputs():
    state = dict(clock_time=T, context={'version': 12, 'plan_complete': True}, vehicles=[
        bus(1, 'plan-1', 240), bus(2, 'plan-2', -10), bus(3, 'plan-3', 20, lon=37.63)], routes=[
        dict(route_id='plan-1', name='Линия А', path=[[37.61, 55.75], [37.62, 55.75], [37.63, 55.75]]),
        dict(route_id='plan-2', name='Линия Б', path=[[37.61, 55.75], [37.61, 55.76], [37.61, 55.77]]),
        dict(route_id='plan-3', name='Линия В', path=[[37.64, 55.75], [37.64, 55.76], [37.64, 55.77]])])
    return state, {2: [stop('Своё ближайшее задание', 100), stop('Следующее', 500)]}


def test_nearest_different_line_and_effect_never_exceeds_existing_delay(inputs):
    state, plan = inputs
    before = deepcopy((state, plan))
    result = advise_transfer(state, plan, 1)
    assert result.status == 'ready' and result.donor.tr_id == 2
    assert result.scenario.service_at == T+timedelta(seconds=720)
    assert result.scenario.donor_arrival_at < result.scenario.service_at
    assert result.scenario.earlier_by_s == 240
    assert result.donor_impact.plan_conflict
    assert result.donor_impact.planned_stops_during_transfer == 2
    assert result.donor_impact.availability == 'requires_dispatcher_confirmation'
    assert 'загрузку и резерв' in result.assumptions[0]
    assert before == (state, plan)


def test_later_relocation_reduces_effect_and_does_not_shift_current_model(inputs):
    state, plan = inputs
    state['vehicles'] = state['vehicles'][:2]
    state['vehicles'][1]['lon'] = 37.651
    result = advise_transfer(state, plan, 1)
    assert result.status == 'ready'
    assert result.scenario.service_at == result.scenario.donor_arrival_at
    assert 60 <= result.scenario.earlier_by_s < 240
    assert state['vehicles'][0]['prediction']['predicted_delay_s'] == 240


@pytest.mark.parametrize('field,value', [
    ('age_s', 61), ('age_s', -1), ('status', 'stale'), ('lat', None),
    ('event_time', T+timedelta(seconds=1)), ('event_time', T-timedelta(seconds=61)),
    ('prediction_availability', {'code': 'deviation_expired'}),
])
def test_invalid_target_inputs_never_get_a_recommendation(inputs, field, value):
    state, plan = inputs
    state['vehicles'][0][field] = value
    assert advise_transfer(state, plan, 1).status == 'unavailable'


@pytest.mark.parametrize('field,value', [
    ('predicted_delay_s', None), ('issued_at', T+timedelta(seconds=1)),
    ('issued_at', T-timedelta(seconds=61)), ('risk', 'unknown'),
])
def test_stale_or_future_prediction_excluded(inputs, field, value):
    state, plan = inputs
    state['vehicles'][0]['prediction'][field] = value
    assert advise_transfer(state, plan, 1).status == 'unavailable'


@pytest.mark.parametrize('delay', [-500, 0, 60, 149.9])
def test_no_intervention_for_minor_delay(inputs, delay):
    state, plan = inputs
    state['vehicles'][0]['prediction']['predicted_delay_s'] = delay
    assert advise_transfer(state, plan, 1).status == 'not_needed'


def test_excludes_late_and_unassessed_donors_instead_of_zero_filling(inputs):
    state, plan = inputs
    state['vehicles'][1]['prediction']['predicted_delay_s'] = 61
    state['vehicles'][2]['prediction'] = None
    assert advise_transfer(state, plan, 1).status == 'unavailable'


def test_stale_nearby_donor_is_not_preferred_to_fresh_farther_one(inputs):
    state, plan = inputs
    state['vehicles'][1]['event_time'] = T-timedelta(seconds=61)
    assert advise_transfer(state, plan, 1).donor.tr_id == 3


@pytest.mark.parametrize('horizon', [600, 901, -1])
def test_out_of_window_prediction_cannot_drive_transfer(inputs, horizon):
    state, plan = inputs
    state['vehicles'][0]['prediction']['target'] = stop('old-target', horizon).model_dump(mode='json')
    assert advise_transfer(state, plan, 1).status == 'unavailable'


def test_identical_route_different_ids_does_not_create_artificial_donor(inputs):
    state, plan = inputs
    state['routes'][1]['path'] = list(reversed(state['routes'][0]['path']))
    result = advise_transfer(state, plan, 1)
    assert result.donor.tr_id == 3
    assert result.candidates_considered == 1


def test_small_coordinate_jitter_same_line_but_crossing_lines_different(inputs):
    state, _ = inputs
    a, b = state['routes'][:2]
    assert not same_line(a, b)
    shifted = {**b, 'path': [[x+.0001, y+.0001] for x, y in a['path']]}
    assert same_line(a, shifted)
    assert not same_line(a, {'route_id': 'missing', 'path': []})


def test_missing_geometry_does_not_turn_unknown_plan_into_a_donor(inputs):
    state, plan = inputs
    state['routes'][1]['path'] = []
    assert advise_transfer(state, plan, 1).donor.tr_id == 3
    state['routes'][0]['path'] = []
    assert advise_transfer(state, plan, 1).status == 'unavailable'


def test_unusable_relocation_and_distant_bus_do_not_get_advice(inputs):
    state, plan = inputs
    state['vehicles'][1]['lon'] = 38.0
    state['vehicles'][2]['lon'] = 37.68
    assert advise_transfer(state, plan, 1).status == 'unavailable'


def test_empty_future_plan_is_not_a_free_bus_claim(inputs):
    state, _ = inputs
    state['context']['plan_complete'] = False
    result = advise_transfer(state, {}, 1)
    assert result.status == 'ready'
    assert result.donor_impact.next_stop_at is None
    assert 'не подтверждает доступность' in result.donor_impact.note
    assert result.donor_impact.availability == 'requires_dispatcher_confirmation'


def test_endpoint_schema_read_only_unknown_vehicle_and_current_context(inputs):
    state, plan = inputs
    with TestClient(create_app(start_background=False, enable_ndtp=False)) as client:
        engine = client.app.state.engine
        engine.state = lambda: state
        engine.schedule = plan
        engine.vehicles = {v['tr_id']: v for v in state['vehicles']}
        before = deepcopy((state, plan))
        response = client.get('/api/v1/vehicles/1/transfer-advice')
        assert response.status_code == 200
        assert response.json()['context_version'] == 12
        assert response.json()['scenario']['earlier_by_s'] == 240
        assert response.headers['cache-control'] == 'no-store'
        assert client.get('/api/v1/vehicles/99/transfer-advice').status_code == 404
        assert client.post('/api/v1/vehicles/1/transfer-advice').status_code == 405
        assert before == (state, plan)
        schema = client.get('/openapi.json').json()
        assert 'TransferAdvice' in schema['components']['schemas']
