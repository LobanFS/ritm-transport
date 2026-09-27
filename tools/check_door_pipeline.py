"""Официальный NDTP → двери/GPS → отклонение → HTTP learned → UI proxy.

Заменяет контекст backend синтетическим полным планом без hints/arrivals.
Передаёт реальные бинарные пакеты официальным эмулятором: closed → open.
В finally восстанавливает конфиг эмулятора и загружает replay на паузе
(--restore-mode demo для стека без данных). История прежнего контекста
не восстанавливается. Не запускать параллельно с нагрузочной приёмкой.
При исходном null-конфиге нужна явная опция --restore-empty-by-restart:
только локальный проверенный контейнер эмулятора будет перезапущен в cleanup.
Без опции такой запуск отказывается до любых изменений сервисов.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.check_irma import EMPTY_CONFIG, restoration_method  # noqa: E402

TR_ID = 904001
UNIT_ID = 1904001
LAT, LON = 55.7551234, 37.617321


def prepare_emulator_restoration(original, args):
    """Проверка возможности cleanup до первого POST; guard общий с IRMA wire-check."""
    inspected, docker_host = {}, ''
    if original == EMPTY_CONFIG and args.restore_empty_by_restart:
        inspected = json.loads(subprocess.check_output(
            ['docker', '--context', args.context, 'inspect', args.emulator_container]))[0]
        context = json.loads(subprocess.check_output(['docker', 'context', 'inspect', args.context]))[0]
        docker_host = context['Endpoints']['docker']['Host']
    method = restoration_method(original, allow_empty_restart=args.restore_empty_by_restart,
        emulator_url=args.emulator, inspected=inspected, docker_host=docker_host)
    return {'method': method, 'inspected': inspected, 'docker_host': docker_host}


def restore_emulator(original, plan, args, http):
    """Точный исходный конфиг или исключение; restart только проверенного ID."""
    url = args.emulator.rstrip('/') + '/api/config'
    if plan['method'] == 'verified_local_container_restart':
        docker = ['docker', '--context', args.context]
        container_id = plan['inspected']['Id']
        current = json.loads(subprocess.check_output(docker + ['inspect', container_id]))[0]
        restoration_method(original, allow_empty_restart=True, emulator_url=args.emulator,
                           inspected=current, docker_host=plan['docker_host'])
        subprocess.run(docker + ['restart', container_id], check=True,
                       capture_output=True, text=True, timeout=30)
        deadline = time.monotonic() + 20
        while True:
            try:
                http(url)
                break
            except Exception:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.2)
    else:
        http(url, original)
    restored = http(url)
    if restored != original:
        raise RuntimeError('Конфигурация эмулятора не восстановлена')
    return restored


def build_context(now: datetime) -> dict:
    """Изолированный план: одна текущая зона, следующие остановки вдали от неё."""
    targets = []
    for i in range(20):
        at = now - timedelta(seconds=120) if i == 0 else now + timedelta(minutes=12 + 3 * (i - 1))
        targets.append({'id': f'door-check-{i}', 'name': f'Тест дверей · остановка {i + 1}',
                        'scheduled_at': at.isoformat(), 'lat': LAT + i * 0.002,
                        'lon': LON + i * 0.002, 'manual_fill': False})
    return {'vehicles': [{'tr_id': TR_ID, 'unit_id': UNIT_ID, 'route_id': 'door-check',
                          'label': 'Проверка официального IRMA'}],
            'routes': [{'route_id': 'door-check', 'name': 'Синтетический план IRMA',
                        'color': '#4169e1', 'path': [[t['lon'], t['lat']] for t in targets], 'stops': []}],
            'schedule': [{'tr_id': TR_ID, 'target': target} for target in targets],
            'hints': [], 'arrival_mode': 'gps', 'plan_complete': True,
            'plan_version': f'synthetic-irma-pipeline-{int(now.timestamp())}', 'plan_timezone': 'UTC'}


def build_emulator_config(host: str, port: int, *, closed: bool) -> dict:
    """Форма cells[].fields и IRMA-флаги проверены tools/check_irma.py."""
    return {'targetHost': host, 'targetPort': port, 'units': [{
        'unitId': UNIT_ID, 'intervalMs': 100000 if closed else 1000, 'autoGenerate': False,
        'cells': [
            {'type': 'G6CellNav00', 'fields': {'latitude': round(LAT * 10_000_000),
                'longitude': round(LON * 10_000_000), 'speedAvg': 0, 'speedMax': 0,
                'extraDopBit5': True, 'extraDopBit6': True, 'extraDopBit7': True}},
            {'type': 'G6CellIrma04', 'fields': {'irma_present_door1': True, 'irma_closed_door1': closed}},
        ],
    }]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', default='http://127.0.0.1:8000')
    parser.add_argument('--dashboard', default='http://127.0.0.1:8080')
    parser.add_argument('--ml', default='http://127.0.0.1:8001')
    parser.add_argument('--emulator', default='http://127.0.0.1:18080')
    parser.add_argument('--context', default='colima-mos-transport')
    parser.add_argument('--emulator-container', default='ritm-transport-emulator-1')
    parser.add_argument('--restore-empty-by-restart', action='store_true',
                        help='Вернуть исходный null-конфиг перезапуском проверенного локального контейнера эмулятора')
    parser.add_argument('--target-host', default='backend')
    parser.add_argument('--target-port', type=int, default=9201)
    parser.add_argument('--restore-mode', choices=['replay', 'demo'], default='replay')
    parser.add_argument('--check-only', action='store_true', help='Проверить входной контракт локально, без HTTP и изменения сервисов')
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/audit-fixes/door-pipeline')
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    context = build_context(datetime.now(timezone.utc).replace(microsecond=0))
    closed_config = build_emulator_config(args.target_host, args.target_port, closed=True)
    open_config = build_emulator_config(args.target_host, args.target_port, closed=False)

    def save(name, value):
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

    contract = {'scope': 'Настоящий официальный NDTP и HTTP-инференс, искусственные план/координаты/двери',
        'baseline': 'На одной closed-точке нет текущего отклонения; после closed→open ожидается door_estimate',
        'acceptance': 'learned через HTTP; формула observed_at-plan; вероятность null; актуальные двери через UI proxy; без hints/arrivals',
        'budget_seconds': 90, 'model_quality_claim': False, 'replaces_backend_context': True,
        'restore_mode': args.restore_mode}
    save('contract.json', contract)
    save('synthetic-plan.json', context)
    save('closed-config.json', closed_config)
    save('open-config.json', open_config)
    if args.check_only:
        sys.path.insert(0, str(ROOT))
        from backend.engine import LiveContext
        checked = LiveContext.model_validate(context)
        assert checked.arrival_mode == 'gps' and not checked.hints and checked.plan_complete
        assert len(checked.schedule) == 20
        assert closed_config['units'][0]['cells'][1]['fields']['irma_closed_door1'] is True
        assert open_config['units'][0]['cells'][1]['fields']['irma_closed_door1'] is False
        print('PASS: synthetic plan и ручной IRMA-конфиг проверены; HTTP не вызывался')
        return

    def http(url, body=None):
        request = Request(url, data=None if body is None else json.dumps(body).encode(),
                          headers={'Content-Type': 'application/json'})
        with urlopen(request, timeout=15) as response:
            return json.load(response)

    def state():
        return http(args.dashboard.rstrip('/') + '/api/v1/state')

    def vehicle(snapshot):
        return next((item for item in snapshot.get('vehicles', []) if item['tr_id'] == TR_ID), None)

    def wait(predicate, timeout=20):
        until = time.monotonic() + timeout
        latest = None
        while time.monotonic() < until:
            latest = state()
            if predicate(latest):
                return latest
            time.sleep(0.1)
        save('timeout-state.json', latest)
        raise TimeoutError('Ожидаемое состояние не появилось за отведённое время')

    def dt(value):
        return datetime.fromisoformat(value.replace('Z', '+00:00'))

    report = {'started_at': datetime.now(timezone.utc).isoformat(), 'passed': False,
              'checks_passed': False, 'mutation_attempted': False,
              'contract': contract, 'checks': {}, 'restoration': {}}
    report['sources_sha256'] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in ['backend/ndtp.py', 'backend/gps_arrivals.py', 'backend/engine.py', 'common/contracts.py']}
    try:
        original = http(args.emulator.rstrip('/') + '/api/config')
        save('emulator-before.json', original)
        restore_plan = prepare_emulator_restoration(original, args)
        report['restoration']['method'] = restore_plan['method']
        if restore_plan['inspected']:
            report['emulator_container_id'] = restore_plan['inspected']['Id']
        save('state-before.json', state())
        save('model.json', http(args.ml.rstrip('/') + '/model'))
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        report['restoration']['method'] = 'unchanged_no_mutation'
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        save('report.json', report)
        raise
    try:
        report['mutation_attempted'] = True
        stop_config = closed_config if original == EMPTY_CONFIG else original
        http(args.emulator.rstrip('/') + '/api/config', {**stop_config, 'units': []})
        http(args.backend.rstrip('/') + '/api/v1/live/context', context)
        before = state()['metrics']
        http(args.emulator.rstrip('/') + '/api/config', closed_config)
        closed = wait(lambda s: bool(vehicle(s)) and vehicle(s).get('doors_open') is False
                      and vehicle(s)['telemetry_sequence'] >= 1)
        save('closed-state.json', closed)
        current = vehicle(closed)
        assert current['current_deviation'] is None and current['cur_dev_s'] is None
        assert closed['gps_arrivals'] == []
        assert current['gps_detector']['candidate_points'] == 1
        # Эмулятор использует Unix seconds. Не превращаем два состояния одной
        # секунды в переход времени и не ждём второй closed-точки для GPS-confirm.
        boundary = dt(current['event_time']).timestamp() + 1.1
        while time.time() < boundary:
            time.sleep(0.05)
        http(args.emulator.rstrip('/') + '/api/config', open_config)
        opened = wait(lambda s: bool(vehicle(s)) and vehicle(s).get('doors_open') is True
            and bool(vehicle(s).get('current_deviation'))
            and vehicle(s)['current_deviation']['source'] == 'door_estimate'
            and bool(vehicle(s).get('prediction'))
            and vehicle(s)['prediction']['method'] == 'learned')
        save('open-state.json', opened)
        current = vehicle(opened)
        deviation = current['current_deviation']
        assert deviation['planned_stop_id'] == 'door-check-0'
        plan_time = dt(context['schedule'][0]['target']['scheduled_at'])
        expected_delay = (dt(deviation['observed_at']) - plan_time).total_seconds()
        assert 120 <= expected_delay < 180
        assert deviation['delay_s'] == current['cur_dev_s'] == expected_delay
        assert dt(deviation['received_at']) >= dt(deviation['observed_at'])
        assert len(opened['gps_arrivals']) == 1
        assert opened['gps_arrivals'][0]['source'] == 'door_estimate'
        assert dt(opened['gps_arrivals'][0]['estimated_arrived_at']) == dt(deviation['observed_at'])
        assert opened['health']['ndtp_connections'] == 1 and opened['health']['ml'] == 'ok'
        trace = http(args.dashboard.rstrip('/') + f'/api/v1/vehicles/{TR_ID}/forecast-trace')
        save('forecast-trace.json', trace)
        request, response = trace['request'], trace['response']
        assert trace['execution'] == 'ml_http'
        assert request['current_delay_source'] == 'door_estimate'
        assert request['current_delay_s'] == expected_delay
        assert trace['current_deviation']['source'] == 'door_estimate'
        assert response['method'] == 'learned' and response['model_version'] == 'swiss-transformer-hgbr-prior-v1'
        assert response['probability_late'] is None
        assert current['prediction']['probability_late'] is None
        assert math.isfinite(response['predicted_delay_s'])
        assert response['target']['id'] == 'door-check-1'
        assert 600 < (dt(response['target']['scheduled_at']) - dt(request['issued_at'])).total_seconds() <= 900
        assert not context['hints'] and 'arrivals' not in context
        assert opened['metrics']['ndtp_errors'] == before['ndtp_errors']
        assert opened['metrics']['invalid'] == before['invalid']
        assert opened['metrics']['unknown_units'] == before['unknown_units']
        report['checks'] = {
            'official_ndtp_closed_then_open': True, 'closed_has_no_deviation': True,
            'door_estimate_without_hints_or_arrivals': True, 'delay_formula': True,
            'real_http_learned': True, 'probability_unknown': True, 'dashboard_proxy_doors_open': True,
            'no_ndtp_or_ingest_errors': True, 'current_delay_s': expected_delay,
            'predicted_delay_s': response['predicted_delay_s'], 'trace_sequence': trace['telemetry_sequence'],
            'detector_version': current['gps_detector']['version'],
        }
        report['checks_passed'] = True
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        restoration_error = None
        try:
            restored = restore_emulator(original, restore_plan, args, http)
            save('emulator-after.json', restored)
            report['restoration']['emulator_config_restored'] = restored == original
            if restored != original:
                raise RuntimeError('Конфигурация эмулятора не восстановлена')
        except Exception as error:
            restoration_error = error
            report['restoration']['emulator_error'] = f'{type(error).__name__}: {error}'
        try:
            if args.restore_mode == 'replay':
                http(args.backend.rstrip('/') + '/api/v1/replay/load', {'paused': True})
                restored_state = wait(lambda s: s['mode'] == 'replay' and not s['replay']['running'])
            else:
                http(args.backend.rstrip('/') + '/api/v1/mode', {'mode': 'demo'})
                http(args.backend.rstrip('/') + '/api/v1/demo/control', {'action': 'pause'})
                restored_state = state()
            save('state-after.json', restored_state)
            report['restoration']['backend_mode'] = restored_state['mode']
        except Exception as error:
            restoration_error = restoration_error or error
            report['restoration']['backend_error'] = f'{type(error).__name__}: {error}'
        report['passed'] = report['checks_passed'] and restoration_error is None
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        save('report.json', report)
        if restoration_error:
            raise RuntimeError('Не удалось завершить восстановление сервисов; см. report.json') from restoration_error
    print(f'PASS: официальный IRMA → door_estimate → HTTP learned → UI proxy. {out / "report.json"}')


if __name__ == '__main__':
    main()
