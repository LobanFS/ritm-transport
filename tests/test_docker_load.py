"""Нагрузочный отчёт должен проверять правильные ТС и не менять цель по плану."""
from datetime import datetime, timezone
import json

import pytest

from tools.check_docker_load import Docker, build_context, check_bound_endpoint, restored_replay_ready, round_covered


def test_load_plan_has_one_target_and_full_history_without_future_facts():
    current = datetime(2026, 9, 26, tzinfo=timezone.utc)
    context = build_context(current, 40, 400)
    assert len(context['schedule']) == 16000
    assert not context['hints']
    for vehicle in context['vehicles']:
        stops = [row['target'] for row in context['schedule'] if row['tr_id'] == vehicle['tr_id']]
        assert len(stops) == 400
        window = [row for row in stops if 600 < (datetime.fromisoformat(row['scheduled_at']) - current).total_seconds() <= 900]
        assert [row['id'] for row in window] == [f'target-{vehicle["tr_id"]}']
        assert all(set(row) == {'id', 'name', 'scheduled_at', 'lat', 'lon', 'manual_fill'} for row in stops)


def test_progress_requires_each_expected_vehicle_and_correct_prediction_sequence():
    vehicle = lambda identity, sequence, method='learned': dict(tr_id=identity,
        prediction_input_sequence=sequence, prediction=dict(method=method))
    expected = {1001, 1002}
    assert round_covered(dict(vehicles=[vehicle(1001, 3), vehicle(1002, 4)]), expected, 3)
    assert not round_covered(dict(vehicles=[vehicle(1001, 3), vehicle(1002, 2)]), expected, 3)
    assert not round_covered(dict(vehicles=[vehicle(1001, 3), vehicle(1003, 3)]), expected, 3)
    assert not round_covered(dict(vehicles=[vehicle(1001, 3), vehicle(1001, 3)]), expected, 3)
    assert not round_covered(dict(vehicles=[vehicle(1001, 3), vehicle(1002, 3, 'fallback')]), expected, 3)


def test_endpoint_must_be_bound_to_inspected_container():
    inventory = {'ml': {'ports': {'8001/tcp': [{'HostPort': '18001', 'HostIp': '127.0.0.1'}]}}}
    check_bound_endpoint(inventory, 'ml', 'http://127.0.0.1:18001', 8001)
    with pytest.raises(AssertionError, match='published port'):
        check_bound_endpoint(inventory, 'ml', 'http://127.0.0.1:8001', 8001)
    with pytest.raises(AssertionError, match='локальный HTTP'):
        check_bound_endpoint(inventory, 'ml', 'http://example.org:18001', 8001)


def container(service):
    return {
        'Id': f'{service}-id', 'Name': f'/{service}', 'Image': 'image-id',
        'Config': {'Labels': {'com.docker.compose.service': service}, 'Image': 'image', 'Env': []},
        'State': {'StartedAt': '2026-09-27T00:00:00Z', 'Health': {'Status': 'healthy'}},
        'RestartCount': 0, 'NetworkSettings': {'Ports': {}}, 'Mounts': [], 'HostConfig': {},
    }


def mock_inventory(monkeypatch, services):
    docker = Docker(None, 'test-project')
    def run(*command):
        if command[0] == 'ps':
            return '\n'.join(f'{service}-id' for service in services)
        assert command[0] == 'inspect'
        return json.dumps([container(service) for service in services])
    monkeypatch.setattr(docker, 'run', run)
    return docker


@pytest.mark.parametrize('with_generator', [False, True])
def test_inventory_accepts_main_stack_with_optional_generator(monkeypatch, with_generator):
    services = ['backend', 'ml', 'dashboard'] + (['generator'] if with_generator else [])
    docker = mock_inventory(monkeypatch, services)
    assert set(docker.inventory()) == set(services)


@pytest.mark.parametrize('missing', ['backend', 'ml', 'dashboard'])
def test_inventory_requires_each_main_service_even_when_generator_present(monkeypatch, missing):
    services = [name for name in ['backend', 'ml', 'dashboard', 'generator'] if name != missing]
    docker = mock_inventory(monkeypatch, services)
    with pytest.raises(AssertionError, match=f'Не хватает сервисов: {missing}'):
        docker.inventory()


@pytest.mark.parametrize('with_generator', [False, True])
def test_source_hashes_executes_only_present_containers(monkeypatch, with_generator):
    services = ['backend', 'ml', 'dashboard'] + (['generator'] if with_generator else [])
    inventory = {service: {'id': f'{service}-id'} for service in services}
    calls = []
    docker = Docker(None, 'test-project')
    def run(*command):
        assert command[0] == 'exec'
        calls.append(command[1])
        if command[1] == 'dashboard-id':
            return 'abc /usr/share/nginx/html/index.html\n'
        return '{"common/example.py": "def"}'
    monkeypatch.setattr(docker, 'run', run)
    hashes = docker.source_hashes(inventory)
    assert set(hashes) == set(services)
    assert set(calls) == {f'{service}-id' for service in services}
    assert hashes['dashboard'] == {'dashboard/index.html': 'abc'}


def test_dashboard_documentation_hashes_map_to_reference_source_directory(monkeypatch):
    docker = Docker(None, 'test-project')
    inventory = {service: {'id': f'{service}-id'} for service in ['backend', 'ml', 'dashboard']}
    def run(*command):
        if command[1] == 'dashboard-id':
            return ('htmlhash /usr/share/nginx/html/index.html\n'
                    'apihash /usr/share/nginx/html/documentation/backend-openapi.json\n'
                    'nestedhash /usr/share/nginx/html/documentation/pydoc/backend.app.html\n')
        return '{}'
    monkeypatch.setattr(docker, 'run', run)
    assert docker.source_hashes(inventory)['dashboard'] == {
        'dashboard/index.html': 'htmlhash',
        'docs/reference/backend-openapi.json': 'apihash',
        'docs/reference/pydoc/backend.app.html': 'nestedhash',
    }


def restored_state():
    return {
        'mode': 'replay', 'replay': {'running': False}, 'context': {'version': 7},
        'health': {'ml': 'ok'}, 'metrics': {'pipeline_cycles': 11},
        'vehicles': [
            {'prediction': {'method': 'learned'}, 'prediction_availability': {'code': 'ready'}},
            {'prediction': None, 'prediction_availability': {'code': 'no_target'}},
        ],
    }


def test_replay_restoration_accepts_legitimate_no_target_after_completed_cycle():
    state = restored_state()
    assert restored_replay_ready(state, after_cycle=10, context_version=7)
    assert not restored_replay_ready(state, after_cycle=11, context_version=7)
    assert not restored_replay_ready(state, after_cycle=10, context_version=6)


@pytest.mark.parametrize('problem', ['pending', 'ml_starting', 'ml_unavailable', 'running'])
def test_replay_restoration_rejects_pending_or_unhealthy_runtime(problem):
    state = restored_state()
    if problem == 'pending':
        state['vehicles'][1]['prediction_availability']['code'] = 'prediction_pending'
    elif problem == 'running':
        state['replay']['running'] = True
    else:
        state['health']['ml'] = problem.removeprefix('ml_')
    assert not restored_replay_ready(state, after_cycle=10, context_version=7)
