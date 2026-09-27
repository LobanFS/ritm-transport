"""Предусловия и cleanup door E2E: null-конфиг не ломает рабочий стенд."""

import io
import json
import sys
from types import SimpleNamespace

import pytest

from tools import check_door_pipeline as tool


def args(**overrides):
    values = dict(restore_empty_by_restart=False, emulator='http://127.0.0.1:18080',
                  context='local-test', emulator_container='official-emulator')
    return SimpleNamespace(**(values | overrides))


def inspection():
    return {'Id': 'verified-full-id',
            'NetworkSettings': {'Ports': {'18080/tcp': [{'HostIp': '127.0.0.1', 'HostPort': '18080'}]}}}


def test_null_preflight_fails_before_any_post_or_docker(monkeypatch, tmp_path):
    calls = []

    def urlopen(request, **kwargs):
        calls.append((request.get_method(), request.full_url))
        assert request.get_method() == 'GET'
        return io.BytesIO(json.dumps(tool.EMPTY_CONFIG).encode())

    def no_docker(*a, **kw):
        pytest.fail('Без явной опции Docker не должен вызываться')

    monkeypatch.setattr(tool, 'urlopen', urlopen)
    monkeypatch.setattr(tool.subprocess, 'check_output', no_docker)
    monkeypatch.setattr(tool.subprocess, 'run', no_docker)
    monkeypatch.setattr(sys, 'argv', ['check_door_pipeline.py', '--out', str(tmp_path)])
    with pytest.raises(ValueError, match='--restore-empty-by-restart'):
        tool.main()
    report = json.loads((tmp_path / 'report.json').read_text())
    assert report['passed'] is False and report['mutation_attempted'] is False
    assert report['restoration']['method'] == 'unchanged_no_mutation'
    assert calls == [('GET', 'http://127.0.0.1:18080/api/config')]


def test_configured_emulator_keeps_http_restore_without_docker(monkeypatch):
    monkeypatch.setattr(tool.subprocess, 'check_output', lambda *a, **kw: pytest.fail('Не нужен Docker'))
    original = {'targetHost': 'backend', 'targetPort': 9201, 'units': []}
    plan = tool.prepare_emulator_restoration(original, args())
    calls = []

    def http(url, body=None):
        calls.append(body)
        return original

    assert tool.restore_emulator(original, plan, args(), http) == original
    assert calls == [original, None]


def test_empty_restart_uses_verified_container_id_and_exact_null_config(monkeypatch):
    calls = []

    def output(command):
        if command[1:3] == ['context', 'inspect']:
            return json.dumps([{'Endpoints': {'docker': {'Host': 'unix:///local.sock'}}}]).encode()
        return json.dumps([inspection()]).encode()

    monkeypatch.setattr(tool.subprocess, 'check_output', output)
    monkeypatch.setattr(tool.subprocess, 'run', lambda command, **kw: calls.append(command))
    options = args(restore_empty_by_restart=True)
    plan = tool.prepare_emulator_restoration(tool.EMPTY_CONFIG, options)
    assert tool.restore_emulator(tool.EMPTY_CONFIG, plan, options, lambda url: tool.EMPTY_CONFIG) == tool.EMPTY_CONFIG
    assert calls == [['docker', '--context', 'local-test', 'restart', 'verified-full-id']]


def test_restart_does_not_accept_foreign_http_service(monkeypatch):
    monkeypatch.setattr(tool.subprocess, 'check_output', lambda command:
        json.dumps([{'Endpoints': {'docker': {'Host': 'unix:///local.sock'}}}]
                   if command[1:3] == ['context', 'inspect'] else [inspection()]).encode())
    with pytest.raises(ValueError, match='127.0.0.1'):
        tool.prepare_emulator_restoration(tool.EMPTY_CONFIG,
            args(restore_empty_by_restart=True, emulator='http://example.org:18080'))


def test_restore_mismatch_is_failure():
    original = {'targetHost': 'backend', 'targetPort': 9201, 'units': []}
    with pytest.raises(RuntimeError, match='не восстановлена'):
        tool.restore_emulator(original, {'method': 'http_post'}, args(),
            lambda url, body=None: tool.EMPTY_CONFIG)
