"""Cleanup IRMA-проверки: честный PASS и отсутствие мутаций при unsafe config."""

import copy
import io
import json
import sys
from urllib.error import HTTPError

import pytest

from tools import check_irma


def inspection():
    return {'Id': 'full-local-container-id', 'Image': 'sha256:official-image',
            'Config': {'Image': 'ndtp-telemetry-emulator:1.0'},
            'NetworkSettings': {'Ports': {'18080/tcp': [{'HostIp': '127.0.0.1', 'HostPort': '18080'}]}}}


def method(original, **overrides):
    args = dict(allow_empty_restart=True, emulator_url='http://127.0.0.1:18080',
                inspected=inspection(), docker_host='unix:///local/docker.sock')
    args.update(overrides)
    return check_irma.restoration_method(original, **args)


def test_null_config_requires_explicit_restart_opt_in():
    with pytest.raises(ValueError, match='--restore-empty-by-restart'):
        method(check_irma.EMPTY_CONFIG, allow_empty_restart=False)
    assert method(check_irma.EMPTY_CONFIG) == 'verified_local_container_restart'
    assert method({'targetHost': 'backend', 'targetPort': 9201, 'units': []}) == 'http_post'


@pytest.mark.parametrize('override', [
    {'emulator_url': 'http://example.org:18080'},
    {'emulator_url': 'http://127.0.0.1:9999'},
    {'emulator_url': 'http://127.0.0.1:18080/emulator'},
    {'emulator_url': 'http://user:secret@127.0.0.1:18080'},
    {'docker_host': 'ssh://remote-host'},
    {'docker_host': 'tcp://remote-host:2375'},
    {'inspected': {'Id': 'other', 'NetworkSettings': {'Ports': {}}}},
])
def test_restart_refuses_unverified_service_or_docker_context(override):
    with pytest.raises(ValueError):
        method(check_irma.EMPTY_CONFIG, **override)


def test_guarded_pass_is_set_only_after_both_cleanups():
    report, saved, order = {}, [], []

    def restore():
        assert report['passed'] is False
        report['restored_config'] = True
        order.append('restore')

    def finish():
        assert report['passed'] is False
        order.append('listener')

    check_irma.guarded_check(lambda: order.append('check'), restore, finish, report, saved.append)
    assert order == ['check', 'restore', 'listener']
    assert saved[0]['passed'] is True


@pytest.mark.parametrize('failure', ['http400', 'mismatch', 'listener'])
def test_cleanup_failure_cannot_leave_passed_true(failure):
    report, saved = {}, []

    def restore():
        if failure == 'http400':
            raise HTTPError('http://127.0.0.1:18080/api/config', 400, 'Bad Request', {}, None)
        report['restored_config'] = failure != 'mismatch'

    def finish():
        if failure == 'listener':
            raise RuntimeError('listener cleanup failed')

    with pytest.raises(RuntimeError):
        check_irma.guarded_check(lambda: None, restore, finish, report, saved.append)
    assert saved[0]['passed'] is False
    assert saved[0]['checks_passed'] is True
    assert saved[0]['restored_config'] is (failure == 'listener')
    assert 'listener_cleanup_error' in saved[0] if failure == 'listener' else 'restoration_error' in saved[0]


def test_check_failure_and_cleanup_failure_are_both_preserved():
    report, saved, finished = {}, [], []

    def check():
        raise ValueError('bad packet')

    def restore():
        raise OSError('restore offline')

    with pytest.raises(RuntimeError) as exc:
        check_irma.guarded_check(check, restore, lambda: finished.append(True), report, saved.append)
    assert isinstance(exc.value.__cause__, ValueError)
    assert saved[0]['error'] == 'ValueError: bad packet'
    assert saved[0]['restoration_error'] == 'OSError: restore offline'
    assert saved[0]['passed'] is False and finished == [True]


def test_main_refuses_null_before_post_or_docker(monkeypatch, tmp_path):
    calls = []

    def urlopen(request, **kwargs):
        calls.append((request.get_method(), request.full_url))
        assert request.get_method() == 'GET'
        return io.BytesIO(json.dumps(check_irma.EMPTY_CONFIG).encode())

    def no_docker(*args, **kwargs):
        pytest.fail('Docker должен оставаться нетронутым при отказе preflight')

    monkeypatch.setattr(check_irma, 'urlopen', urlopen)
    monkeypatch.setattr(check_irma.subprocess, 'check_output', no_docker)
    monkeypatch.setattr(check_irma.subprocess, 'Popen', no_docker)
    monkeypatch.setattr(sys, 'argv', ['check_irma.py', '--out', str(tmp_path)])
    with pytest.raises(RuntimeError):
        check_irma.main()
    result = json.loads((tmp_path / 'report.json').read_text())
    assert result['passed'] is False and result['checks_passed'] is False
    assert result['mutation_attempted'] is False and result['restored_config'] is True
    assert result['restoration_method'] == 'unchanged_no_mutation'
    assert '--restore-empty-by-restart' in result['error']
    assert calls == [('GET', 'http://127.0.0.1:18080/api/config')]


def test_main_http_restore_failure_stops_only_test_producer_and_reports_failure(monkeypatch, tmp_path):
    original = {'targetHost': 'original', 'targetPort': 9201, 'units': []}
    current = copy.deepcopy(original)
    posted = []

    def urlopen(request, **kwargs):
        nonlocal current
        if request.full_url.endswith('/api/cells'):
            return io.BytesIO(b'[]')
        if request.get_method() == 'POST':
            body = json.loads(request.data)
            posted.append(body)
            if body == original:
                raise HTTPError(request.full_url, 400, 'Cannot restore', {}, None)
            current = body
            if body['units']:
                # Сервер успел применить конфиг, но ответ закончился ошибкой.
                raise HTTPError(request.full_url, 500, 'After mutation', {}, None)
        return io.BytesIO(json.dumps(current).encode())

    class Listener:
        stdout = io.StringIO('READY\n')
        cleaned = False

        def poll(self):
            return None

        def terminate(self):
            self.cleaned = True

        def communicate(self, **kwargs):
            return '', ''

    listener = Listener()
    monkeypatch.setattr(check_irma, 'urlopen', urlopen)
    monkeypatch.setattr(check_irma.subprocess, 'check_output', lambda *a, **kw: json.dumps([inspection()]).encode())
    monkeypatch.setattr(check_irma.subprocess, 'Popen', lambda *a, **kw: listener)
    monkeypatch.setattr(check_irma.select, 'select', lambda *a: ([listener.stdout], [], []))
    monkeypatch.setattr(sys, 'argv', ['check_irma.py', '--out', str(tmp_path)])
    with pytest.raises(RuntimeError):
        check_irma.main()
    result = json.loads((tmp_path / 'report.json').read_text())
    assert result['passed'] is False and result['restored_config'] is False
    assert result['mutation_attempted'] is True
    assert result['test_producer_stopped_after_restore_failure'] is True
    assert '500' in result['error'] and '400' in result['restoration_error']
    assert current['units'] == [] and listener.cleaned
    assert len(posted) == 3
