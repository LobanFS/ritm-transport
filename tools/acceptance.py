"""Изолированная приёмка реальных процессов, TCP и HTTP, без данных/обучения.

Запуск: .venv/bin/python tools/acceptance.py
Frozen ML: --model-path artifacts/model/model.joblib --out artifacts/acceptance-learned
Результат: artifacts/acceptance/{report.json,report.html,*.log}.
Стенд получает свободные localhost-порты; работающий пользовательский demo не меняется.
"""
import argparse
import asyncio
from datetime import datetime, timedelta, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import platform
import shutil
import socket
import struct
import subprocess
import sys
import time

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.smoke import frame


def now():
    return datetime.now(timezone.utc)


def percentile(values, q):
    return round(sorted(values)[max(0, math.ceil(len(values)*q)-1)], 2) if values else None


def provenance(root=ROOT):
    paths = [p for folder in ('backend', 'common', 'ml_service', 'dashboard', 'tools')
             for p in (root/folder).rglob('*') if p.suffix in ('.py', '.js', '.css', '.html', '.json')]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def render_report(report):
    esc = lambda value: html.escape(str(value))
    stream = report.get('stream', {})
    cards = [('Результат', report['result']), ('ТС / сообщений', f"{report['config']['vehicles']} / {stream.get('received', '—')}"),
             ('NDTP → API UI, p95', f"{stream.get('e2e_p95_ms', '—')} мс"),
             ('HTTP ML, p95', f"{stream.get('http_ml_p95_ms', '—')} мс"),
             ('Цикл обработки, p95', f"{stream.get('pipeline_p95_ms', '—')} мс"),
             ('Запуск процессов', f"{report.get('startup_s', '—')} с")]
    items = ''.join(f'<article><small>{esc(k)}</small><strong>{esc(v)}</strong></article>' for k, v in cards)
    rows = ''.join(f'<tr><td>{esc(c["name"])}</td><td>{esc(c["result"])}</td><td>{esc(c.get("details", ""))}</td></tr>' for c in report['checks'])
    limitations = ''.join(f'<li>{esc(v)}</li>' for v in report['limitations'])
    return f'''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ритм · проверка системы</title><style>
body{{font:16px/1.5 system-ui,sans-serif;background:#f2f5f8;color:#172d40;margin:0;padding:32px}}
main{{max-width:1100px;margin:auto}}h1{{font-size:32px}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}}
article,section{{background:white;padding:22px;border-radius:14px;margin:12px 0}}small,strong{{display:block}}strong{{font-size:24px;margin-top:8px}}
table{{width:100%;border-collapse:collapse}}td,th{{text-align:left;padding:12px;border-bottom:1px solid #dde4e9}}.table{{overflow:auto}}
code{{overflow-wrap:anywhere}}a{{color:#285c9c}}@media(max-width:600px){{body{{padding:12px}}td{{padding:8px}}}}
</style><main><p>РИТМ / ИНЖЕНЕРНАЯ ПРИЁМКА</p><h1>Поток и восстановление</h1>
<p>{esc(report['checked_at'])} · {esc(report['environment'])}</p>
<p>Синтетический поток через настоящий NDTP TCP, отдельные процессы backend и {esc(report.get('model', {}).get('model_version', 'ML'))}, HTTP-прокси дашборда.</p>
<div class="cards">{items}</div><section class="table"><h2>Проверки</h2><table><thead><tr><th>Сценарий</th><th>Итог</th><th>Наблюдение</th></tr></thead><tbody>{rows}</tbody></table></section>
<section><h2>Границы результата</h2><ul>{limitations}</ul><p>Порог p95 ≤ 2000 мс: {esc(stream.get('p95_budget_met', 'не измерен'))}.</p>
<p>Измеряется время от отправки раунда до наблюдения опубликованного прогноза через HTTP-прокси. Опрос 100 мс; отрисовка браузером сюда не входит.</p>
<p><a href="report.json">Полный JSON: измерения, конфигурация и SHA-256 кода</a></p></section></main></html>'''


class Stand:
    def __init__(self, out, model_path=None):
        self.out = out
        self.processes = {}
        self.logs = []
        self.writers = []
        # Перезапуск ML обязан использовать ту же версию контракта, что backend.
        # Параллельные правки рабочего дерева не должны менять проверяемый стенд.
        self.source_root = out / 'source'
        self.source_root.mkdir(exist_ok=True)
        for folder in ('backend', 'common', 'ml_service', 'dashboard', 'tools'):
            shutil.copytree(ROOT/folder, self.source_root/folder, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.DS_Store'))
        # Одновременно резервируем разные порты. После освобождения остаётся
        # малая гонка bind; её результат — явная ошибка запуска, не чужой процесс.
        reservations = [socket.socket() for _ in range(4)]
        try:
            for item in reservations:
                item.bind(('127.0.0.1', 0))
            self.ports = dict(zip(('ml', 'backend', 'dashboard', 'ndtp'), [s.getsockname()[1] for s in reservations]))
        finally:
            for item in reservations:
                item.close()
        self.api = f'http://127.0.0.1:{self.ports["dashboard"]}'
        self.env = {**os.environ, 'ML_URL': f'http://127.0.0.1:{self.ports["ml"]}',
                    'NDTP_HOST': '127.0.0.1', 'NDTP_PORT': str(self.ports['ndtp'])}
        # Опция определяет проверяемую модель явно, вне зависимости от shell env.
        self.env.pop('MODEL_PATH', None)
        self.env['PYTHONPATH'] = str(self.source_root)
        self.env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', LOKY_MAX_CPU_COUNT='1')
        if model_path is not None:
            self.env['MODEL_PATH'] = str(model_path.resolve())

    def start(self, name):
        if name == 'dashboard':
            command = [sys.executable, 'tools/serve_dashboard.py', '--port', str(self.ports[name]),
                       '--backend', f'http://127.0.0.1:{self.ports["backend"]}']
        else:
            app = 'ml_service.app:app' if name == 'ml' else 'backend.app:app'
            command = [sys.executable, '-m', 'uvicorn', app, '--host', '127.0.0.1', '--port', str(self.ports[name])]
        log = (self.out/f'{name}.log').open('a')
        self.logs.append(log)
        self.processes[name] = subprocess.Popen(command, cwd=self.source_root, env=self.env, stdout=log, stderr=subprocess.STDOUT)

    async def stop(self, name):
        proc = self.processes.get(name)
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        deadline = time.monotonic()+5
        while proc.poll() is None and time.monotonic() < deadline:
            await asyncio.sleep(.05)
        if proc.poll() is None:
            proc.kill()
        proc.wait()

    async def connect(self, count):
        for i in range(count):
            _, writer = await asyncio.open_connection('127.0.0.1', self.ports['ndtp'])
            self.writers.append(writer)
            unit = 1166336+i
            writer.write(frame(unit, 0, 100, struct.pack('<HHHIII', 6, 2, 0, unit, 65535, 0), 1))
            await writer.drain()

    async def disconnect(self):
        for writer in self.writers:
            writer.close()
        for writer in self.writers:
            await writer.wait_closed()
        self.writers.clear()

    async def cleanup(self):
        try:
            await self.disconnect()
        finally:
            for name in ('dashboard', 'backend', 'ml'):
                await self.stop(name)
            for log in self.logs:
                log.close()


async def run(args, out, report):
    stand = Stand(out, args.model_path)
    expected_method = 'learned' if args.model_path else 'persistence'
    report['code_sha256'] = provenance(stand.source_root)
    report['source_snapshot'] = str(stand.source_root)
    report['ports'] = stand.ports
    async with httpx.AsyncClient(timeout=5, trust_env=False) as client:
        async def api(path='/api/v1/state', body=None):
            response = await (client.get(stand.api+path) if body is None else client.post(stand.api+path, json=body))
            response.raise_for_status()
            return response.json()

        async def wait_for(predicate, timeout=10):
            deadline = time.monotonic()+timeout
            while time.monotonic() < deadline:
                state = await api()
                if predicate(state):
                    return state
                await asyncio.sleep(.1)
            (out/'timeout-state.json').write_text(json.dumps(state, ensure_ascii=False, indent=2)+'\n')
            raise AssertionError(f'Условие не выполнено за {timeout} с')

        def passed(name, details=''):
            report['checks'].append(dict(name=name, result='passed', details=details))
            print(f'PASS: {name} {details}', flush=True)

        async def send_round(number):
            payloads = []
            stamp = int(time.time())
            started = time.monotonic()
            for i, writer in enumerate(stand.writers):
                nav = struct.pack('<IIIBBHHHHHBB', stamp, 376000000+i*1000+number,
                                  557500000+i*1000, 0xe0, 100, 22, 25, 45, 0, 150, 10, 1)
                payload = frame(1166336+i, 1, 101, bytes([0, 0])+nav, number+1)
                writer.write(payload)
                payloads.append(payload)
            for writer in stand.writers:
                await writer.drain()
            return started, payloads

        def covered(state, number):
            return len(state['vehicles']) == args.vehicles and all(
                v['prediction'] and (v['prediction_input_sequence'] or 0) >= number for v in state['vehicles'])

        try:
            started = time.monotonic()
            for name in ('ml', 'backend', 'dashboard'):
                stand.start(name)
            deadline = started+15
            while True:
                if any(proc.poll() is not None for proc in stand.processes.values()):
                    raise RuntimeError('Процесс не запустился; подробности в *.log')
                try:
                    state = await api()
                    if state['health']['ml'] == 'ok':
                        break
                except httpx.HTTPError:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError('Сервисы не готовы за 15 с')
                await asyncio.sleep(.1)
            report['startup_s'] = round(time.monotonic()-started, 3)
            model = await client.get(f'http://127.0.0.1:{stand.ports["ml"]}/model')
            model.raise_for_status()
            report['model'] = model.json()
            assert report['model']['trained'] is bool(args.model_path)
            for path in ('/', '/docs', '/openapi.json', '/app.js'):
                response = await client.get(stand.api+path)
                response.raise_for_status()
            passed('Три процесса, UI, Swagger и OpenAPI', f'{report["startup_s"]} с; локально, не Docker')

            current = now()
            context = {'vehicles': [], 'schedule': [], 'hints': [],
                       'arrival_mode': 'external', 'plan_complete': True,
                       'plan_version': 'acceptance-synthetic-v2', 'plan_timezone': 'UTC'}
            for i in range(args.vehicles):
                tr_id = 1001+i
                context['vehicles'].append(dict(tr_id=tr_id, unit_id=1166336+i, label=f'Приёмка {tr_id}', route_id='acceptance'))
                context['schedule'].append(dict(tr_id=tr_id, target=dict(id=f'target-{tr_id}', name='Синтетическая цель',
                    scheduled_at=(current+timedelta(minutes=14)).isoformat(), lat=55.76, lon=37.61, manual_fill=False)))
                context['schedule'].append(dict(tr_id=tr_id, target=dict(id=f'last-{tr_id}', name='Пройденная остановка',
                    scheduled_at=(current-timedelta(seconds=180)).isoformat(), lat=55.75, lon=37.60, manual_fill=False)))
                # Весь план известен до инференса; дополнительные посещения вне
                # тестируемого окна не меняют цель и нагружают реальный контракт.
                for visit in range(args.plan_visits-2):
                    before = visit < (args.plan_visits-2)//2
                    offset = -360-visit*180 if before else 1800+(visit-(args.plan_visits-2)//2)*180
                    context['schedule'].append(dict(tr_id=tr_id, target=dict(
                        id=f'plan-{tr_id}-{visit}', name=f'Плановое посещение {visit}',
                        scheduled_at=(current+timedelta(seconds=offset)).isoformat(),
                        lat=55.75+(visit%10)*.001, lon=37.6+(visit%20)*.001,
                        manual_fill=visit % 5 == 0)))
            report['context_payload_bytes'] = len(json.dumps(context, ensure_ascii=False, separators=(',', ':')).encode())
            await api('/api/v1/live/context', context)
            for i in range(args.vehicles):
                result = await api('/api/v1/arrivals', dict(tr_id=1001+i, planned_stop_id=f'last-{1001+i}', arrived_at=current.isoformat()))
                assert result['current_deviation']['delay_s'] == 180
                assert result['current_deviation']['source'] == 'arrival'
            passed('Подтверждённое прибытие → cur_dev_s', 'Backend вычислил +180 с как факт − план; готовая подсказка не передавалась')
            state = await wait_for(lambda s: len(s['vehicles']) == args.vehicles and
                all(v['prediction'] and v['prediction']['method'] == expected_method for v in s['vehicles']))
            passed('Полный план → прогретый HTTP ML', f'{expected_method}; {args.plan_visits} синтетических посещений на ТС, известный manual_fill')
            initial_received = state['metrics']['received']
            assert state['summary']['unknown'] == args.vehicles
            await stand.connect(args.vehicles)
            pending = {}
            observations = []
            peak_pending = 0
            sending_finished = False
            last_payloads = []

            async def sender():
                nonlocal sending_finished, last_payloads
                start = time.monotonic()
                for number in range(1, args.rounds+1):
                    await asyncio.sleep(max(0, start+(number-1)*args.interval-time.monotonic()))
                    sent_at, last_payloads = await send_round(number)
                    pending[number] = sent_at
                sending_finished = True

            sender_task = asyncio.create_task(sender())
            stream_started = time.monotonic()
            try:
                deadline = stream_started+args.rounds*args.interval+10
                while not sending_finished or pending:
                    if sender_task.done():
                        sender_task.result()
                    if time.monotonic() > deadline:
                        raise AssertionError('Поток не обработан в пределах бюджета')
                    peak_pending = max(peak_pending, len(pending))
                    state = await api()
                    for number, sent_at in list(pending.items()):
                        if covered(state, number):
                            observations.append(dict(round=number, e2e_ms=round((time.monotonic()-sent_at)*1000, 2)))
                            del pending[number]
                    await asyncio.sleep(.1)
            finally:
                sender_task.cancel()
                await asyncio.gather(sender_task, return_exceptions=True)
            received = state['metrics']['received']-initial_received
            assert received == args.vehicles*args.rounds, (received, args.vehicles*args.rounds)
            assert all(v['telemetry_sequence'] == args.rounds and v['prediction_input_sequence'] == args.rounds for v in state['vehicles'])
            latencies = [v['e2e_ms'] for v in observations]
            report['stream'] = dict(received=received, sent=received, nominal_messages_per_s=args.vehicles/args.interval,
                wall_s=round(time.monotonic()-stream_started, 3), e2e_p50_ms=percentile(latencies, .5),
                e2e_p95_ms=percentile(latencies, .95), e2e_max_ms=max(latencies),
                p95_budget_met=percentile(latencies, .95) <= 2000, peak_pending_rounds=peak_pending,
                pending_at_end=len(pending), http_ml_p95_ms=state['metrics']['inference_p95_ms'],
                pipeline_p95_ms=state['metrics']['pipeline_p95_ms'], observations=observations)
            passed('Поток NDTP → прогноз → HTTP дашборда', f'{received}/{received} сообщений; p95 {report["stream"]["e2e_p95_ms"]} мс')
            for vehicle in state['vehicles']:
                prediction = vehicle['prediction']
                lead = (datetime.fromisoformat(prediction['target']['scheduled_at'])-datetime.fromisoformat(prediction['issued_at'])).total_seconds()
                assert 600 < lead <= 900
                published_lead = (datetime.fromisoformat(prediction['target']['scheduled_at'])-datetime.fromisoformat(vehicle['prediction_published_at'])).total_seconds()
                assert 600 < published_lead <= 900
                assert prediction['method'] == expected_method
                assert math.isfinite(prediction['predicted_delay_s'])
                if expected_method == 'persistence':
                    assert prediction['predicted_delay_s'] == 180
                probability = prediction['probability_late']
                if expected_method == 'persistence':
                    assert probability is None
                elif probability is not None:
                    assert math.isfinite(probability) and 0 <= probability <= 1
                    assert prediction['probability_note']
            export = await api('/api/v1/incidents/export')
            trace = await api('/api/v1/vehicles/1001/forecast-trace')
            assert trace['request']['current_delay_s'] == 180
            assert trace['response']['method'] == expected_method
            assert trace['request']['plan_context']['complete'] is True
            assert len(trace['request']['plan_context']['stops']) == args.plan_visits
            assert trace['current_deviation']['planned_stop_id'] == 'last-1001'
            assert trace['execution'] == 'ml_http'
            (out/'forecast-trace.json').write_text(json.dumps(trace, ensure_ascii=False, indent=2)+'\n')
            # Learned регрессия не обязана красить artificial cur_dev=180 в red.
            # Каждый опубликованный red должен быть в журнале; лишних ТС нет.
            red_ids = {v['tr_id'] for v in state['vehicles'] if v['prediction']['risk'] == 'red'}
            incident_ids = {entry['tr_id'] for entry in export['incidents']}
            assert red_ids <= incident_ids <= {1001+i for i in range(args.vehicles)}
            if expected_method == 'persistence':
                assert len(export['incidents']) == args.vehicles
            report['stream']['red_predictions'] = len(red_ids)
            report['stream']['incidents'] = len(export['incidents'])
            assert all(600 < entry['lead_time_s'] <= 900 for entry in export['incidents'])
            (out/'incidents.json').write_text(json.dumps(export, ensure_ascii=False, indent=2)+'\n')
            passed('Горизонт и журнал', '(10,15] минут до плановой цели; не оценка начала нового сбоя')

            duplicates = state['metrics']['duplicates']
            stand.writers[0].write(last_payloads[0])
            await stand.writers[0].drain()
            state = await wait_for(lambda s: s['metrics']['duplicates'] == duplicates+1)
            assert state['metrics']['received']-initial_received == received
            errors = state['metrics']['ndtp_errors']
            _, bad = await asyncio.open_connection('127.0.0.1', stand.ports['ndtp'])
            corrupted = bytearray(last_payloads[0]); corrupted[-1] ^= 1
            try:
                bad.write(corrupted)
                await bad.drain()
                await wait_for(lambda s: s['metrics']['ndtp_errors'] > errors)
            finally:
                bad.close()
                await bad.wait_closed()
            passed('Дубликат и повреждённый CRC', 'Дубль не учтён повторно; повреждённый кадр отклонён, backend жив')

            await stand.disconnect()
            print('Проверка реального обрыва: ждём устаревания GPS >60 секунд…', flush=True)
            state = await wait_for(lambda s: s['health']['ndtp_connections'] == 0 and
                all(v['status'] == 'stale' and v['prediction']['risk'] == 'unknown' for v in s['vehicles']), timeout=67)
            assert (await api('/health'))['status'] == 'ok'
            passed('Обрыв NDTP и stale', 'После >60 реальных секунд риск unknown; API продолжает отвечать')
            recovery = time.monotonic()
            await stand.connect(args.vehicles)
            await send_round(args.rounds+1)
            await wait_for(lambda s: covered(s, args.rounds+1) and all(v['status'] == 'fresh' and
                v['prediction']['risk'] != 'unknown' and v['prediction']['method'] == expected_method for v in s['vehicles']))
            report['ndtp_recovery_s'] = round(time.monotonic()-recovery, 3)
            passed('Новый handshake и восстановление NDTP', f'{report["ndtp_recovery_s"]} с')

            await stand.stop('ml')
            failure = time.monotonic()
            state = await wait_for(lambda s: s['health']['ml'] == 'unavailable' and all(v['prediction']['method'] == 'fallback' for v in s['vehicles']))
            assert all(v['prediction']['predicted_delay_s'] == 180 for v in state['vehicles'])
            report['ml_fallback_s'] = round(time.monotonic()-failure, 3)
            assert (await api('/health'))['status'] == 'ok'
            export = await api('/api/v1/incidents/export')
            assert {entry['tr_id'] for entry in export['incidents']} == {1001+i for i in range(args.vehicles)}
            (out/'fallback-incidents.json').write_text(json.dumps(export, ensure_ascii=False, indent=2)+'\n')
            passed('Остановка процесса ML', f'Fallback за {report["ml_fallback_s"]} с; backend доступен')
            recovery = time.monotonic()
            stand.start('ml')
            await wait_for(lambda s: s['health']['ml'] == 'ok' and all(v['prediction']['method'] == expected_method for v in s['vehicles']))
            report['ml_recovery_s'] = round(time.monotonic()-recovery, 3)
            passed('Перезапуск процесса ML', f'Автоматическое восстановление за {report["ml_recovery_s"]} с')
            report['result'] = 'passed'
        finally:
            await stand.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vehicles', type=int, default=32, choices=range(1, 129), metavar='1..128')
    parser.add_argument('--rounds', type=int, default=24, choices=range(2, 61), metavar='2..60')
    parser.add_argument('--interval', type=float, default=1.)
    parser.add_argument('--out', type=Path, default=ROOT/'artifacts'/'acceptance')
    parser.add_argument('--model-path', type=Path, help='Проверенный frozen model.joblib; без опции проверяется persistence baseline')
    parser.add_argument('--plan-visits', type=int, choices=range(2, 2001), default=2, metavar='2..2000',
                        help='Полный искусственный план на ТС; 400 близко к размеру выданного плана')
    args = parser.parse_args()
    if not .1 <= args.interval <= 1.5:
        parser.error('--interval: от 0.1 до 1.5 секунды (весь сценарий должен оставаться в горизонте и TTL подсказки)')
    if args.model_path is not None and not args.model_path.is_file():
        parser.error('--model-path: файл не найден; сначала tools/prepare_model.py')
    if args.vehicles * args.plan_visits > 20000:
        parser.error('--vehicles × --plan-visits превышает API limit 20000 посещений')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = dict(result='failed', checked_at=now().isoformat(), environment=f'{platform.platform()} / Python {platform.python_version()}',
        config=dict(vehicles=args.vehicles, rounds=args.rounds, interval=args.interval,
                    method='learned' if args.model_path else 'persistence', plan_visits_per_vehicle=args.plan_visits,
                    model_path=str(args.model_path.resolve()) if args.model_path else None,
                    model_sha256=hashlib.sha256(args.model_path.read_bytes()).hexdigest() if args.model_path else None),
        checks=[], code_sha256=provenance(),
        limitations=[(f'Синтетические данные и полный искусственный план по {args.plan_visits} посещений на ТС; MAE не измерялся. '
                      + ('Подключена frozen модель alexchist.' if args.model_path else 'Проверяется persistence baseline.')),
            'Локальные процессы, не Docker; официальный эмулятор и холодный старт контейнеров не проверены.',
            'Горизонт до плановой остановки; упреждение до фактического начала инцидента не оценивалось.',
            'Короткий поток фиксированного размера, не максимальная пропускная способность и не длительный soak test.',
            'Раунды могут объединяться в один прогноз состояния. Pending — раунды, ещё не покрытые прогнозом, не глубина очереди TCP.',
            'p95 HTTP/цикла — окно до 200 вызовов, включая прогрев; end-to-end — только раунды потока.'])
    probability_path = os.environ.get('PROBABILITY_PATH')
    if probability_path:
        report['config']['probability_sha256'] = hashlib.sha256(Path(probability_path).read_bytes()).hexdigest()
    try:
        asyncio.run(run(args, out, report))
    except (Exception, KeyboardInterrupt) as exc:
        report['checks'].append(dict(name='Выполнение приёмки', result='failed', details=f'{type(exc).__name__}: {exc}'))
    finally:
        (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        (out/'report.html').write_text(render_report(report))
        print(out/'report.html', flush=True)
    return 0 if report['result'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
