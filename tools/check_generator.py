"""Приёмка ЗАПУЩЕННОГО отдельного генератора через HTTP-прокси дашборда.

Заменяет текущий контекст приложения. Проверяет GPS → оценка прибытия → cur_dev_s,
1–20 маршрутов, pause/resume, потерю связи/восстановление и изоляцию режимов. Оставляет
синтетический normal на паузе. Не проверяет MAE, обученную модель или NDTP.
Нужен запущенный стек с generator:8002; дополнительных Python-пакетов нет.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def timestamp(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise AssertionError('Время не содержит UTC offset')
    return result


def compact_vehicle(vehicle):
    return {key: vehicle.get(key) for key in (
        'tr_id', 'lat', 'lon', 'speed_kmh', 'event_time', 'age_s', 'status',
        'cur_dev_s', 'current_deviation', 'telemetry_sequence', 'prediction_input_sequence')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dashboard', default='http://127.0.0.1:8080', help='HTTP API через тот же proxy, что использует UI')
    parser.add_argument('--producer', default='http://127.0.0.1:8002', help='Только GET /health, без прямого управления')
    parser.add_argument('--timeout', type=float, default=30, help='Таймаут каждого ожидания, секунды')
    parser.add_argument('--check-replay', action='store_true', help='Также проверить переход в CSV replay; нужны примонтированные данные')
    parser.add_argument('--require-learned', action='store_true', help='Требовать обученную модель для известных GPS-отклонений')
    parser.add_argument('--out', type=Path, default=ROOT/'artifacts'/'generator')
    args = parser.parse_args()
    if not 1 <= args.timeout <= 60:
        parser.error('--timeout должен быть от 1 до 60 секунд')
    args.out.mkdir(parents=True, exist_ok=True)
    config = dict(scenario='gps_loss', seed=42, speed=30, paused=True,
                  route_count=4, telemetry_interval_s=15)
    report = dict(started_at=datetime.now(timezone.utc).isoformat(), passed=False,
        scope='Синтетический GPS producer → GPS-детектор backend → cur_dev_s → HTTP ML → API. Истинные прибытия в backend не подаются. Не оценка MAE, NDTP или рендера карты.',
        configuration=config, endpoints=dict(dashboard=args.dashboard, producer=args.producer),
        environment=dict(python=sys.version, platform=platform.platform()), checks={})
    checks = report['checks']

    def save(name, value):
        (args.out/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str)+'\n', encoding='utf-8')

    def http(url, body=None):
        request = Request(url, data=None if body is None else json.dumps(body).encode('utf-8'),
                          headers={'Content-Type':'application/json'})
        try:
            with urlopen(request, timeout=min(10, args.timeout)) as response:
                return json.load(response)
        except HTTPError as exc:
            raise RuntimeError(f'{request.get_method()} {url}: HTTP {exc.code}: {exc.read(1000).decode("utf-8", errors="replace")}') from exc

    def api(path, body=None):
        return http(args.dashboard.rstrip('/')+'/api/v1/'+path, body)

    def state():
        return api('state')

    def wait(label, predicate):
        deadline = time.monotonic()+args.timeout
        last = None
        while time.monotonic() < deadline:
            last = state()
            if predicate(last):
                return last
            if last.get('mode') == 'generator' and last.get('generator', {}).get('error'):
                raise AssertionError(f'{label}: ошибка источника: {last["generator"]["error"]}')
            time.sleep(.15)
        if last is not None:
            save('timeout-state.json', last)
        raise AssertionError(f'{label}: состояние не наступило за {args.timeout:g} с')

    def bus(snapshot, tr_id=104):
        return next(v for v in snapshot['vehicles'] if v['tr_id'] == tr_id)

    def check_causality(snapshot):
        at = timestamp(snapshot['clock_time'])
        assert snapshot['mode'] == 'generator'
        assert snapshot['generator']['connected'] and not snapshot['generator']['gap']
        for vehicle in snapshot['vehicles']:
            if vehicle['event_time']:
                assert timestamp(vehicle['event_time']) <= at
            deviation = vehicle['current_deviation']
            if deviation is None:
                assert vehicle['cur_dev_s'] is None
                continue
            assert deviation['source'] == 'gps_estimate'
            actual, received, planned = (timestamp(deviation[key]) for key in ('observed_at', 'received_at', 'planned_at'))
            assert actual < received <= at
            assert deviation['delay_s'] == vehicle['cur_dev_s'] == (actual-planned).total_seconds()
        assert snapshot['generator']['received_telemetry'] == sum(v['telemetry_sequence'] for v in snapshot['vehicles'])
        assert snapshot['generator']['emitted_telemetry'] >= snapshot['generator']['received_telemetry']

    def check_trace(tr_id, filename, *, known=True):
        trace = api(f'vehicles/{tr_id}/forecast-trace')
        save(filename, trace)
        request, response, deviation = trace['request'], trace['response'], trace['current_deviation']
        assert trace['execution'] == 'ml_http', 'Для этой приёмки ML-контейнер должен отвечать'
        at = timestamp(request['issued_at'])
        assert 600 < (timestamp(request['target']['scheduled_at'])-at).total_seconds() <= 900
        if known:
            assert deviation and deviation['source'] == 'gps_estimate'
            actual, received, planned = (timestamp(deviation[key]) for key in ('observed_at', 'received_at', 'planned_at'))
            assert actual < received <= at
            assert request['current_delay_s'] == deviation['delay_s'] == (actual-planned).total_seconds()
            if args.require_learned:
                assert response['method'] == 'learned' and response['model_version'] == 'swiss-transformer-hgbr-prior-v1'
                assert request['plan_context']['complete'] and response['predicted_delay_s'] is not None
            else:
                assert response['model_version'] == 'persistence-v1'
                assert response['predicted_delay_s'] == request['current_delay_s']
        else:
            assert deviation is None and request['current_delay_s'] is None
            assert response['predicted_delay_s'] is None and response['risk'] == 'unknown'
        return trace

    def leave_normal_paused():
        api('generator/start', {**config, 'scenario':'normal', 'speed':10})
        result = wait('Финальная пауза normal', lambda s: s['mode']=='generator'
            and not s['generator']['running'] and len(s['vehicles'])==8 and all(v['prediction'] for v in s['vehicles']))
        check_causality(result)
        assert all(v['cur_dev_s'] is None for v in result['vehicles'])
        return result

    start_wall = time.monotonic()
    started_generator = False
    try:
        producer = http(args.producer.rstrip('/')+'/health')
        assert producer['status'] == 'ok' and producer['synthetic'] is True
        save('producer-health-before.json', producer)
        api('mode', {'mode':'demo'})
        demo = state()
        assert demo['mode'] == 'demo' and demo['generator'] is None
        checks['demo_mode'] = dict(passed=True)

        api('generator/start', config)
        started_generator = True
        initial = wait('Начальные прогнозы', lambda s: s['mode']=='generator'
            and len(s['vehicles'])==8 and all(v['prediction'] for v in s['vehicles']))
        assert {v['tr_id'] for v in initial['vehicles']} == set(range(101, 109))
        assert not initial['generator']['running'] and initial['generator']['scenario'] == 'gps_loss'
        assert initial['generator']['received_frames'] == 1
        assert initial['generator']['received_telemetry'] == 8
        assert initial['generator']['route_count'] == 4 and initial['generator']['telemetry_interval_s'] == 15
        assert all(v['status']=='fresh' and v['lat'] is not None and v['lon'] is not None for v in initial['vehicles'])
        assert all(v['cur_dev_s'] is None and v['current_deviation'] is None for v in initial['vehicles'])
        check_causality(initial)
        save('initial-state.json', initial)
        trace = check_trace(101, 'initial-trace-101.json', known=False)
        checks['initial_unknown_without_oracle'] = dict(passed=True, clock_time=initial['clock_time'],
            vehicles=[compact_vehicle(v) for v in initial['vehicles']], execution=trace['execution'])

        log_state = api('generator/logs')
        assert log_state['session_id'] == initial['generator']['session_id']
        assert 0 < len(log_state['logs']) <= 100
        assert any(log['kind']=='telemetry' for log in log_state['logs'])
        assert any(log['kind']=='truth_arrival' for log in log_state['logs'])
        assert all(timestamp(log['time']) <= timestamp(initial['clock_time']) for log in log_state['logs'])
        save('initial-generator-logs.json', log_state)
        checks['logs_and_counts'] = dict(passed=True, logs=len(log_state['logs']),
            emitted_telemetry=log_state['emitted_telemetry'], received_telemetry=log_state['received_telemetry'],
            emitted_arrivals=log_state['emitted_arrivals'])

        time.sleep(1.2)
        paused = state()
        assert paused['clock_time'] == initial['clock_time']
        assert [compact_vehicle(v) for v in paused['vehicles']] == [compact_vehicle(v) for v in initial['vehicles']]
        checks['pause_stable'] = dict(passed=True, observed_real_seconds=1.2)

        resumed_wall = time.monotonic()
        api('generator/control', {'action':'resume'})
        cutoff = timestamp(initial['clock_time'])+timedelta(seconds=160)
        stale = wait('Потеря GPS у 104', lambda s: s['mode']=='generator'
            and timestamp(s['clock_time']) >= cutoff and bus(s)['status']=='stale'
            and bus(s)['age_s'] > 60)
        check_causality(stale)
        assert stale['generator']['received_telemetry'] > initial['generator']['received_telemetry']
        assert any((v['lat'], v['lon']) != (bus(initial, v['tr_id'])['lat'], bus(initial, v['tr_id'])['lon'])
                   for v in stale['vehicles'])
        assert all(bus(stale, tr_id)['status']=='fresh' for tr_id in (101, 102, 103))
        # Эталонные прибытия идут лишь в журнал. Без наблюдений у 104 остаётся unknown.
        assert bus(stale)['current_deviation'] is None and bus(stale)['cur_dev_s'] is None
        save('outage-state.json', stale)
        checks['gps_loss_104'] = dict(passed=True, elapsed_real_s=round(time.monotonic()-resumed_wall, 3),
            clock_time=stale['clock_time'], bus=compact_vehicle(bus(stale)))

        api('generator/control', {'action':'pause'})
        held = state()
        time.sleep(1.2)
        still = state()
        assert still['clock_time'] == held['clock_time']
        assert still['generator']['received_telemetry'] == held['generator']['received_telemetry']
        api('generator/control', {'action':'resume'})
        warmed = wait('Подтверждение прибытия по GPS у 101', lambda s: s['mode']=='generator'
            and bus(s, 101)['current_deviation'] is not None
            and bus(s, 101)['prediction_input_sequence'] == bus(s, 101)['telemetry_sequence'])
        check_causality(warmed)
        trace = check_trace(101, 'gps-confirmation-trace-101.json')
        save('gps-confirmation-state.json', warmed)
        checks['gps_confirmation_to_model'] = dict(passed=True, clock_time=warmed['clock_time'],
            bus=compact_vehicle(bus(warmed, 101)), trace_execution=trace['execution'], model=trace['response']['model_version'])
        recovery_at = timestamp(initial['clock_time'])+timedelta(seconds=211)
        recovered = wait('Восстановление GPS у 104', lambda s: s['mode']=='generator'
            and timestamp(s['clock_time']) >= recovery_at and bus(s)['status']=='fresh')
        api('generator/control', {'action':'pause'})
        recovered = wait('Прогноз после восстановления', lambda s: s['mode']=='generator'
            and not s['generator']['running'] and bus(s)['prediction_input_sequence'] == bus(s)['telemetry_sequence'])
        check_causality(recovered)
        save('recovered-state.json', recovered)
        # Восстановление GPS не выдаёт пропущенные истинные прибытия задним числом.
        # cur_dev_s у 104 может остаться unknown до следующей наблюдаемой стоянки.
        trace = check_trace(101, 'recovered-trace-101.json')
        checks['gps_recovery_104'] = dict(passed=True, clock_time=recovered['clock_time'],
            bus=compact_vehicle(bus(recovered)), trace_execution=trace['execution'])

        api('generator/start', {**config, 'scenario':'normal', 'route_count':20})
        large = state()
        assert len(large['vehicles']) == 40 and len(large['routes']) == 20
        assert large['generator']['route_count'] == 20 and large['generator']['telemetry_interval_s'] == 15
        assert large['generator']['received_frames'] == 1 and large['generator']['received_telemetry'] == 40
        assert not large['generator']['running']
        assert all(v['cur_dev_s'] is None and v['current_deviation'] is None for v in large['vehicles'])
        check_causality(large)
        save('twenty-routes-paused-state.json', large)
        checks['twenty_routes_forty_buses'] = dict(passed=True, routes=len(large['routes']),
            vehicles=len(large['vehicles']), telemetry_interval_s=15)

        api('mode', {'mode':'live'})
        live = state()
        assert live['mode']=='live' and live['generator'] is None and live['replay'] is None
        assert live['vehicles'] == [] and live['incidents'] == []
        checks['live_mode_isolation'] = dict(passed=True, vehicles=len(live['vehicles']))
        if args.check_replay:
            api('replay/load', {})
            replay = state()
            assert replay['mode']=='replay' and replay['generator'] is None and replay['replay'] is not None
            assert not replay['replay']['running']
            save('mode-replay-state.json', replay)
            checks['replay_mode_isolation'] = dict(passed=True, vehicles=len(replay['vehicles']))
        final = leave_normal_paused()
        assert final['generator']['session_id'] != initial['generator']['session_id']
        assert final['generator']['received_frames'] == 1 and final['generator']['received_telemetry'] == 8
        save('final-paused-state.json', final)
        checks['new_session_isolation'] = dict(passed=True, clock_time=final['clock_time'], scenario='normal')
        save('producer-health-after.json', http(args.producer.rstrip('/')+'/health'))
        report['passed'] = True
        print('PASS: генератор 4/20 маршрутов, GPS → оценка прибытия → cur_dev_s → HTTP ML, пауза, обрыв и восстановление, изоляция режимов.')
        print('Дашборд оставлен в normal на паузе. Сценарии искусственные; MAE и карта этим скриптом не проверялись.')
    except Exception as exc:
        report['error'] = repr(exc)
        if started_generator:
            try:
                current = state()
                save('failure-state.json', current)
                if current['mode'] == 'generator':
                    api('generator/control', {'action':'pause'})
            except Exception as cleanup_error:
                report['pause_cleanup_error'] = repr(cleanup_error)
        raise
    finally:
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        report['elapsed_real_s'] = round(time.monotonic()-start_wall, 3)
        report['sources_sha256'] = {str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
            for directory in ('generator', 'backend', 'common', 'dashboard')
            for path in sorted((ROOT/directory).glob('*')) if path.is_file()}
        for relative in ('tools/check_generator.py', 'compose.yaml', 'Dockerfile', 'dashboard.Dockerfile', 'requirements.txt'):
            path = ROOT/relative
            if path.exists():
                report['sources_sha256'][relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        report['provenance_note'] = 'SHA-256 локальных исходников на момент проверки; сами хеши не подтверждают состав Docker image. Стек должен быть пересобран из этих файлов перед запуском.'
        save('report.json', report)
        print(args.out/'report.json')


if __name__ == '__main__':
    main()
