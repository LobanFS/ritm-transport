"""Приёмка ЗАПУЩЕННОГО стека и официального эмулятора через HTTP.

Заменяет текущий контекст. Проверяет NDTP/reconnect, затем CSV replay.
Без --require-learned ожидается persistence baseline; при подключённой модели
обязательно передайте --require-learned. Сервис сам этот флаг не переключает.
Оставляет replay на паузе и выключает генерацию эмулятора. Не проверяет MAE.
"""
import argparse
from datetime import datetime, timedelta
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
LEARNED_VERSION = 'swiss-transformer-hgbr-prior-v1'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', default='http://127.0.0.1:8000')
    parser.add_argument('--dashboard', default='http://127.0.0.1:8080')
    parser.add_argument('--emulator', default='http://127.0.0.1:18080')
    parser.add_argument('--target-host', default='backend')
    parser.add_argument('--require-learned', action='store_true', help='Требовать настоящие HTTP-прогнозы frozen обученной модели')
    parser.add_argument('--out', type=Path, default=ROOT/'artifacts'/'stack')
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = dict(started_at=datetime.now().astimezone().isoformat(), checks={},
                  scope='Официальный эмулятор + validate replay; проверка интеграции, не MAE',
                  expected_method='learned' if args.require_learned else 'persistence')

    def http(url, body=None):
        req = Request(url, data=None if body is None else json.dumps(body).encode(),
                      headers={'Content-Type':'application/json'})
        with urlopen(req, timeout=30) as response:
            return json.load(response)

    def state():
        return http(args.dashboard+'/api/v1/state')

    def wait(predicate, timeout=20):
        until = time.monotonic()+timeout
        while time.monotonic() < until:
            value = state()
            if predicate(value):
                return value
            time.sleep(.25)
        raise AssertionError('Истёк таймаут ожидания состояния')

    def save(name, value):
        (out/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str)+'\n')

    def check_prediction(prediction, known_input, *, synthetic=False):
        expected = report['expected_method']
        assert prediction['method'] == expected, (
            f'Ожидался {expected}, получен {prediction["method"]}; '
            'для стека с MODEL_PATH требуется --require-learned'
        )
        assert prediction['model_version'] == (LEARNED_VERSION if args.require_learned else 'persistence-v1')
        assert known_input is not None and math.isfinite(known_input)
        assert prediction['predicted_delay_s'] is not None and math.isfinite(prediction['predicted_delay_s'])
        if not args.require_learned:
            assert prediction['predicted_delay_s'] == known_input
        if synthetic or not args.require_learned:
            assert prediction['probability_late'] is None
        elif prediction['probability_late'] is not None:
            assert 0 <= prediction['probability_late'] <= 1
        issued = datetime.fromisoformat(prediction['issued_at'].replace('Z', '+00:00'))
        target = datetime.fromisoformat(prediction['target']['scheduled_at'].replace('Z', '+00:00'))
        assert 600 < (target-issued).total_seconds() <= 900

    def check_trace(trace, known_input, source, *, synthetic=False):
        assert trace['execution'] == 'ml_http'
        assert trace['request']['current_delay_s'] == known_input
        assert trace['request']['current_delay_source'] == source
        check_prediction(trace['response'], known_input, synthetic=synthetic)
        if args.require_learned:
            context = trace['request']['plan_context']
            assert context['complete'] and context['version'] and context['timezone']
            assert all(isinstance(stop['manual_fill'], bool) for stop in context['stops'])

    config = None
    try:
        subprocess.run([sys.executable, str(ROOT/'tools/configure_emulator.py'),
            '--backend', args.backend, '--emulator', args.emulator, '--target-host', args.target_host],
            check=True, capture_output=True, text=True)
        config = http(args.emulator+'/api/config')
        before = state()['metrics']
        ndtp = wait(lambda s: s['mode']=='live' and s['vehicles'][0]['telemetry_sequence'] >= 3
            and s['vehicles'][0]['prediction_input_sequence'] is not None
            and s['vehicles'][0]['prediction_input_sequence'] >= 3)
        assert ndtp['health']['ndtp_connections'] == 1 and ndtp['health']['ml'] == 'ok'
        assert ndtp['vehicles'][0]['status'] == 'fresh'
        assert ndtp['vehicles'][0]['cur_dev_s'] == 180
        assert ndtp['vehicles'][0]['current_deviation']['source'] == 'synthetic_hint'
        check_prediction(ndtp['vehicles'][0]['prediction'], 180, synthetic=True)
        assert ndtp['metrics']['ndtp_errors'] == before['ndtp_errors']
        trace = http(args.backend+'/api/v1/vehicles/101/forecast-trace')
        check_trace(trace, 180, 'synthetic_hint', synthetic=True)
        save('official-emulator-state.json', ndtp)
        save('official-emulator-trace.json', trace)
        report['checks']['official_ndtp'] = dict(passed=True, received=ndtp['vehicles'][0]['telemetry_sequence'],
            errors_delta=ndtp['metrics']['ndtp_errors']-before['ndtp_errors'], configuration=config,
            prediction=trace['response'], current_delay_source='synthetic_hint',
            limitation='План и начальное отклонение искусственные; NDTP не сообщает факт прибытия')

        http(args.emulator+'/api/config', {**config, 'units':[]})
        stopped = wait(lambda s: s['health']['ndtp_connections'] == 0)
        sequence = stopped['vehicles'][0]['telemetry_sequence']
        started = time.monotonic()
        http(args.emulator+'/api/config', config)
        resumed = wait(lambda s: s['health']['ndtp_connections'] == 1
            and s['vehicles'][0]['telemetry_sequence'] > sequence)
        report['checks']['official_reconnect'] = dict(passed=True, elapsed_s=round(time.monotonic()-started, 3))
        assert resumed['metrics']['ndtp_errors'] == before['ndtp_errors']
        http(args.emulator+'/api/config', {**config, 'units':[]})

        http(args.backend+'/api/v1/replay/load', {})
        first = wait(lambda s: s['mode']=='replay' and all(v['prediction'] for v in s['vehicles']))
        assert not first['replay']['running'] and first['health']['ml'] == 'ok'
        assert {v['tr_id']:v['cur_dev_s'] for v in first['vehicles']} == {133300:211, 122048:-162}
        for v in first['vehicles']:
            assert v['current_deviation']['source'] == 'csv_snapshot'
            check_prediction(v['prediction'], v['cur_dev_s'])
            assert v['status'] == 'fresh' and v['lat'] is not None
        save('replay-start.json', first)
        replay_trace = http(args.backend+'/api/v1/vehicles/133300/forecast-trace')
        check_trace(replay_trace, 211, 'csv_snapshot')
        save('replay-trace.json', replay_trace)
        time.sleep(1.2)
        assert state()['clock_time'] == first['clock_time']
        http(args.backend+'/api/v1/replay/control', {'action':'speed', 'speed':30})
        http(args.backend+'/api/v1/replay/control', {'action':'resume'})
        boundary = datetime.fromisoformat(first['clock_time'].replace('Z', '+00:00')) + timedelta(seconds=305)
        later = wait(lambda s: datetime.fromisoformat(s['clock_time'].replace('Z', '+00:00')) >= boundary, timeout=30)
        http(args.backend+'/api/v1/replay/control', {'action':'pause'})
        later = state()
        assert later['replay']['delivered'] > first['replay']['delivered']
        assert all(v['telemetry_sequence'] > first['vehicles'][i]['telemetry_sequence'] for i, v in enumerate(later['vehicles']))
        assert any((v['lat'],v['lon']) != (first['vehicles'][i]['lat'],first['vehicles'][i]['lon']) for i,v in enumerate(later['vehicles']))
        assert any(v['current_deviation'] and v['current_deviation']['sample_id'] != first['vehicles'][i]['current_deviation']['sample_id'] for i,v in enumerate(later['vehicles']))
        save('replay-later.json', later)
        report['checks']['csv_replay'] = dict(passed=True, start=first['clock_time'], later=later['clock_time'],
            delivered_at_start=first['replay']['delivered'], delivered_later=later['replay']['delivered'],
            sources=first['replay']['sources_sha256'], quality=first['replay']['quality'])
        http(args.backend+'/api/v1/replay/control', {'action':'reset'})
        reset = wait(lambda s: s['clock_time']==first['clock_time'] and all(v['prediction'] for v in s['vehicles']))
        assert reset['replay']['delivered'] == first['replay']['delivered']
        assert reset['vehicles'][0]['telemetry_sequence'] == first['vehicles'][0]['telemetry_sequence']
        report['checks']['replay_reset'] = dict(passed=True)
        report['passed'] = True
        print(f'PASS: официальный NDTP, reconnect, CSV replay, пауза/ход/сброс и HTTP {report["expected_method"]}. Replay оставлен на паузе.')
    except Exception as exc:
        report['passed'] = False
        report['error'] = repr(exc)
        raise
    finally:
        try:
            if state()['mode'] == 'replay':
                http(args.backend+'/api/v1/replay/control', {'action':'pause'})
        except Exception as exc:
            report['pause_cleanup_error'] = repr(exc)
        if config:
            try:
                http(args.emulator+'/api/config', {**config, 'units':[]})
            except Exception as exc:
                report['cleanup_error'] = repr(exc)
        report['sources_sha256'] = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ('backend','common','ml_service','dashboard','tools') for p in (ROOT/folder).rglob('*')
            if p.is_file() and p.suffix in ('.py','.js','.css','.html','.json')}
        save('report.json', report)
        print(out/'report.json')


if __name__ == '__main__':
    main()
