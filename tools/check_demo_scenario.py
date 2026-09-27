"""Проверить демонстрацию диспетчера через работающие сервисы и nginx API.

Четыре главы с явным сбросом: штатное движение, долгая остановка с ростом
отклонения, потеря/возврат GPS и алерт на историческом CSV. Прибытия/hints не подаются, веса не меняются.
Это инженерная проверка синтетики, не оценка качества на реальных автобусах.
В finally загружается replay на паузе (либо --restore-mode demo); прежний
журнал контекста не восстанавливается. Не запускать одновременно с другим
инструментом, меняющим режим backend.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def http(url, body=None):
    request = Request(url, data=None if body is None else json.dumps(body).encode(),
                      headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=15) as response:
        return json.load(response)


def dt(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def vehicle(state, tr_id):
    return next((item for item in state.get('vehicles', []) if item['tr_id'] == tr_id), None)


def predicted(item):
    return (item or {}).get('prediction') or {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dashboard', default='http://127.0.0.1:8080')
    parser.add_argument('--ml', default='http://127.0.0.1:8001')
    parser.add_argument('--restore-mode', choices=['replay', 'demo'], default='replay')
    parser.add_argument('--speed', type=int, choices=[5, 10, 30], default=30)
    parser.add_argument('--out', type=Path, default=ROOT/'artifacts/dispatcher-demo')
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    base = args.dashboard.rstrip('/')
    started = time.monotonic()
    observations = []

    def save(name, value):
        (out/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')

    def state():
        return http(base+'/api/v1/state')

    def checkpoint(name, snapshot, tr_id):
        save(name+'.json', snapshot)
        item = vehicle(snapshot, tr_id)
        row = {'stage': name, 'tr_id': tr_id, 'clock_time': snapshot['clock_time'],
               'status': item['status'], 'cur_dev_s': item['cur_dev_s'],
               'current_deviation': item['current_deviation'], 'prediction': item['prediction'],
               'gps_detector_version': (item.get('gps_detector') or {}).get('version', 'not_enabled'),
               'explanation': item['explanation'], 'metrics': snapshot['metrics']}
        observations.append(row)
        print(json.dumps({'stage': name, 'tr_id': tr_id, 'status': item['status'],
                          'cur_dev_s': item['cur_dev_s'], 'risk': predicted(item).get('risk'),
                          'prediction_s': predicted(item).get('predicted_delay_s')}, ensure_ascii=False), flush=True)
        return row

    def chapter(scenario):
        config = {'scenario': scenario, 'seed': 42, 'speed': args.speed, 'paused': True,
                  'route_count': 4, 'telemetry_interval_s': 15, 'door_sensors': False}
        response = http(base+'/api/v1/generator/start', config)
        save(scenario+'-start.json', {'request': config, 'response': response})
        initial = state()
        assert initial['mode'] == 'generator'
        assert initial['context']['arrival_mode'] == 'gps'
        assert not initial['generator']['door_sensors']
        http(base+'/api/v1/generator/control', {'action': 'resume'})
        return initial

    def wait_until(predicate, *, timeout, mode='generator'):
        deadline = min(time.monotonic()+timeout, started+contract['budget_seconds'])
        latest = None
        while time.monotonic() < deadline:
            latest = state()
            if latest['mode'] != mode:
                raise RuntimeError('Другой оператор изменил режим во время проверки')
            if mode == 'generator' and latest['generator']['error']:
                raise RuntimeError(latest['generator']['error'])
            if predicate(latest):
                return latest
            time.sleep(.2)
        save('timeout-state.json', latest)
        raise TimeoutError('Ожидаемый этап не наступил; последнее состояние сохранено')

    contract = {'scope': 'Separate HTTP generator → backend GPS detector → HTTP learned model → nginx JSON',
        'baseline': 'Normal scenario seed42, four routes, GPS15s, no door sensors or external delay hints',
        'acceptance': ['normal vehicle has fresh green learned forecast from gps_estimate',
            'long-stop observation is followed by a new GPS-estimated arrival, larger current deviation and actual learned prediction',
            'GPS loss makes the affected vehicle stale with unknown risk; delivery resumes with fresh learned forecast',
            'GPS forecast trace uses actual ML HTTP and GPS deviation; separate CSV chapter produces a red alert with target and planned lead in (600,900]',
            'restoration succeeds and playback is paused'],
        'chapters_reset_context': True, 'csv_alert_uses_provided_deviation': True, 'historical_journal_restored': False,
        'quality_or_causal_reason_claim': False, 'door_sensors': False,
        'budget_seconds': 1800/args.speed+40, 'restore_mode': args.restore_mode}
    save('contract.json', contract)
    report = {'started_at': datetime.now(timezone.utc).isoformat(), 'passed': False,
              'checks': {}, 'observations': observations, 'mutated': False,
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'limitations': ['Known synthetic scenarios, not real-world model validation.',
                 'Four chapters restart context. No claim of one continuous operational trip.',
                 'Synthetic long-stop current deviation grows, but learned prediction need not grow or cross red: artificial plans are outside measured model-quality scope.',
                 'Historical alert uses supplied points.csv deviation; it is not a GPS reconstruction quality check.',
                 'Learned model reacts to confirmed current deviation and plan, not directly to GPS speed.',
                 'This check measures API state, not browser paint or transport-wide outage.']}
    failure = None
    try:
        save('before-state.json', state())
        metadata = http(args.ml.rstrip('/')+'/model')
        save('model.json', metadata)
        if not metadata.get('trained') or metadata.get('method') != 'learned':
            raise RuntimeError('Для демонстрации нужна загруженная learned-модель')
        report['mutated'] = True
        chapter('normal')
        normal = wait_until(lambda s: (vehicle(s,104) or {}).get('status') == 'fresh'
            and predicted(vehicle(s,104)).get('method') == 'learned'
            and predicted(vehicle(s,104)).get('risk') == 'green', timeout=360/args.speed+10)
        row = checkpoint('01-normal', normal, 104)
        assert row['current_deviation']['source'] == 'gps_estimate'
        report['checks']['normal'] = True

        chapter('long_stop')
        before = wait_until(lambda s: predicted(vehicle(s,102)).get('method') == 'learned'
            and predicted(vehicle(s,102)).get('predicted_delay_s') is not None
            and predicted(vehicle(s,102))['predicted_delay_s'] <= 120
            and bool((vehicle(s,102) or {}).get('current_deviation')), timeout=240/args.speed+10)
        before_row = checkpoint('02-before-long-stop', before, 102)
        observed = wait_until(lambda s: any(item.get('code') == 'long_stop'
            for item in (vehicle(s,102) or {}).get('explanation', {}).get('observations', [])), timeout=300/args.speed+10)
        checkpoint('03-long-stop-observed', observed, 102)
        alerted = wait_until(lambda s: predicted(vehicle(s,102)).get('method') == 'learned'
            and (vehicle(s,102) or {}).get('cur_dev_s', float('-inf')) is not None
            and vehicle(s,102)['cur_dev_s'] > before_row['cur_dev_s'], timeout=480/args.speed+10)
        alert_row = checkpoint('04-new-arrival-model-output', alerted, 102)
        assert alert_row['current_deviation']['planned_stop_id'] != before_row['current_deviation']['planned_stop_id']
        trace = http(base+'/api/v1/vehicles/102/forecast-trace')
        save('gps-forecast-trace.json', trace)
        assert trace['execution'] == 'ml_http' and trace['response']['method'] == 'learned'
        assert trace['request']['current_delay_source'] == 'gps_estimate'
        report['checks']['observed_stop_then_new_arrival_model_output'] = True
        report['synthetic_prediction_increased'] = alert_row['prediction']['predicted_delay_s'] > before_row['prediction']['predicted_delay_s']
        report['synthetic_forecast_before_s'] = before_row['prediction']['predicted_delay_s']
        report['synthetic_forecast_after_s'] = alert_row['prediction']['predicted_delay_s']

        chapter('gps_loss')
        fresh = wait_until(lambda s: (vehicle(s,104) or {}).get('status') == 'fresh', timeout=180/args.speed+10)
        checkpoint('05-before-gps-loss', fresh, 104)
        stale = wait_until(lambda s: (vehicle(s,104) or {}).get('status') == 'stale', timeout=240/args.speed+10)
        checkpoint('06-gps-stale', stale, 104)
        assert predicted(vehicle(stale,104)).get('risk') in (None, 'unknown')
        resumed = wait_until(lambda s: (vehicle(s,104) or {}).get('status') == 'fresh', timeout=240/args.speed+10)
        checkpoint('07-gps-delivery-resumed', resumed, 104)
        recovered = wait_until(lambda s: (vehicle(s,104) or {}).get('status') == 'fresh'
            and predicted(vehicle(s,104)).get('method') == 'learned'
            and predicted(vehicle(s,104)).get('risk') != 'unknown', timeout=240/args.speed+10)
        checkpoint('08-gps-learned-recovered', recovered, 104)
        report['checks']['gps_loss_and_recovery'] = True
        report['checks']['pipeline_no_invalid_frames'] = recovered['metrics']['invalid'] == 0 and not recovered['generator']['gap']

        # Выбор сценки основан на сохранённом прогнозе validate, без чтения labels.
        # Это отдельная глава с готовым CSV-входом, не продолжение синтетического рейса.
        csv_config = {'start': '2026-01-06T03:50:00Z', 'duration_minutes': 5,
                      'tr_ids': [131672], 'timezone': 'UTC', 'paused': True,
                      'deviation_source': 'csv_snapshot', 'warmup_minutes': 5}
        http(base+'/api/v1/replay/load', csv_config)
        historical = wait_until(lambda s: predicted(vehicle(s,131672)).get('method') == 'learned'
            and predicted(vehicle(s,131672)).get('risk') == 'red'
            and any(item['tr_id'] == 131672 for item in s.get('incidents', [])), timeout=15, mode='replay')
        checkpoint('09-historical-alert', historical, 131672)
        csv_trace = http(base+'/api/v1/vehicles/131672/forecast-trace')
        save('csv-alert-forecast-trace.json', csv_trace)
        assert csv_trace['execution'] == 'ml_http' and csv_trace['request']['current_delay_source'] == 'csv_snapshot'
        incidents = [item for item in historical['incidents'] if item['tr_id'] == 131672]
        assert all(600 < item['lead_time_s'] <= 900 and item['target_name'] for item in incidents)
        exported = http(base+'/api/v1/incidents/export')
        save('csv-alert-journal.json', exported)
        assert any(item['id'] == incidents[0]['id'] for item in exported['incidents'])
        report['checks']['csv_alert_and_export'] = True
    except Exception as error:
        failure = error
        report['error'] = f'{type(error).__name__}: {error}'
    finally:
        if report['mutated']:
            try:
                if state()['mode'] == 'generator':
                    http(base+'/api/v1/generator/control', {'action': 'pause'})
                if args.restore_mode == 'replay':
                    http(base+'/api/v1/replay/load', {'paused': True})
                else:
                    http(base+'/api/v1/mode', {'mode': 'demo'})
                    http(base+'/api/v1/demo/control', {'action': 'pause'})
                restored = state()
                save('restored-state.json', restored)
                assert restored['mode'] == args.restore_mode
                assert not (restored.get('replay') or restored['demo'])['running']
                report['checks']['restored_paused'] = True
            except Exception as error:
                report['restoration_error'] = f'{type(error).__name__}: {error}'
                report['checks']['restored_paused'] = False
        report['passed'] = failure is None and bool(report['checks']) and all(report['checks'].values())
        report['elapsed_s'] = time.monotonic()-started
        report['gps_detector_versions'] = sorted({row['gps_detector_version'] for row in observations})
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        save('report.json', report)
        print(json.dumps({k:report[k] for k in ('passed','checks','elapsed_s')}, ensure_ascii=False), flush=True)
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
