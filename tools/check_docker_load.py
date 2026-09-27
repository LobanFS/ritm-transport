"""Нагрузка настоящего Docker-стека: NDTP → backend → ML HTTP → nginx API.

Заменяет демонстрационный контекст. Не строит/не удаляет контейнеры.
По завершении (включая ошибку) загружает replay на паузе; --restore demo
возвращает встроенное демо. --check-recovery дополнительно останавливает
и запускает существующий контейнер ML, без пересоздания образа.
Синтетические входы проверяют интеграцию/latency, не точность модели.
"""
import argparse
import asyncio
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import struct
import subprocess
import sys
import time
from urllib.parse import urlparse

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.smoke import frame
from tools.start_solution import replay_readiness


def percentile(values, q):
    return round(sorted(values)[max(0, math.ceil(len(values) * q) - 1)], 2) if values else None


def build_context(current, vehicles, plan_visits):
    """Полный искусственный план и единственная цель в (T+10,T+15]."""
    context = dict(vehicles=[], schedule=[], hints=[], arrival_mode='external',
                   plan_complete=True, plan_timezone='UTC', plan_version='docker-load-synthetic-v1')
    for index in range(vehicles):
        tr_id = 1001 + index
        context['vehicles'].append(dict(tr_id=tr_id, unit_id=1166336 + index,
            label=f'Нагрузка Docker · {tr_id}', route_id='docker-load'))
        entries = [('last', -180), ('target', 840)]
        for visit in range(plan_visits - 2):
            boundary = (plan_visits - 2) // 2
            offset = -360 - visit * 180 if visit < boundary else 1800 + (visit - boundary) * 180
            entries.append((f'plan-{visit}', offset))
        for number, (name, offset) in enumerate(entries):
            context['schedule'].append(dict(tr_id=tr_id, target=dict(
                id=f'{name}-{tr_id}', name=f'Искусственное посещение {name}',
                scheduled_at=(current + timedelta(seconds=offset)).isoformat(),
                lat=55.75 + (number % 10) * .001, lon=37.60 + (number % 20) * .001,
                manual_fill=number > 1 and number % 5 == 0)))
    return context


def round_covered(state, expected_ids, sequence, method='learned'):
    """Покрытие каждого ожидаемого ТС, а не только число ответов."""
    vehicles = state.get('vehicles', [])
    return {v['tr_id'] for v in vehicles} == expected_ids and len(vehicles) == len(expected_ids) and all(
        v.get('prediction') and v['prediction']['method'] == method
        and (v.get('prediction_input_sequence') or 0) >= sequence for v in vehicles)


def restored_replay_ready(state, *, after_cycle, context_version):
    """Ждать цикл загруженного архива; отсутствие цели у части ТС допустимо."""
    return (state.get('mode') == 'replay'
            and (state.get('replay') or {}).get('running') is False
            and (state.get('context') or {}).get('version') == context_version
            and replay_readiness(state, after_cycle=after_cycle)['ready'])


class Docker:
    def __init__(self, context, project):
        self.prefix = ['docker'] + (['--context', context] if context else [])
        self.project = project

    def run(self, *command):
        return subprocess.run(self.prefix + list(command), cwd=ROOT, capture_output=True,
                              text=True, check=True, timeout=30).stdout

    def inventory(self):
        ids = self.run('ps', '-q', '--filter', f'label=com.docker.compose.project={self.project}').split()
        if not ids:
            raise AssertionError(f'Нет запущенных контейнеров Compose-проекта {self.project}')
        containers = json.loads(self.run('inspect', *ids))
        selected = {}
        for item in containers:
            service = item['Config']['Labels'].get('com.docker.compose.service')
            if service not in ('backend', 'ml', 'dashboard', 'generator'):
                continue
            if service in selected:
                raise AssertionError(f'Несколько контейнеров сервиса {service}; стенд неоднозначен')
            allowed_env = {'ML_URL', 'GENERATOR_URL', 'NDTP_HOST', 'NDTP_PORT', 'MODEL_PATH',
                           'PROBABILITY_PATH', 'REPLAY_DATA_DIR', 'OMP_NUM_THREADS',
                           'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'}
            selected[service] = dict(id=item['Id'], name=item['Name'], image_id=item['Image'],
                image_name=item['Config']['Image'], started_at=item['State']['StartedAt'],
                restart_count=item['RestartCount'], health=item['State'].get('Health', {}).get('Status'),
                ports=item['NetworkSettings']['Ports'],
                environment={key: value for entry in item['Config']['Env']
                             for key, _, value in [entry.partition('=')] if key in allowed_env},
                mounts=[dict(destination=mount['Destination'], read_only=not mount['RW'])
                        for mount in item['Mounts']],
                resource_limits={name: item['HostConfig'].get(name) for name in
                    ('NanoCpus', 'CpuQuota', 'CpuPeriod', 'CpusetCpus', 'Memory', 'MemorySwap', 'PidsLimit')})
        missing = {'backend', 'ml', 'dashboard'} - set(selected)
        if missing:
            raise AssertionError(f'Не хватает сервисов: {", ".join(sorted(missing))}')
        return selected

    def source_hashes(self, inventory):
        result = {}
        script = ('import pathlib,hashlib,json; r=pathlib.Path("/app"); '
                  'print(json.dumps({str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() '
                  'for name in ("backend","common","ml_service","generator") '
                  'for p in (r/name).rglob("*.py")}))')
        for service in ('backend', 'ml', 'generator'):
            if service == 'generator' and service not in inventory:
                continue
            result[service] = json.loads(self.run('exec', inventory[service]['id'], 'python', '-c', script))
        files = self.run('exec', inventory['dashboard']['id'], 'sh', '-c',
            'find /usr/share/nginx/html -type f -exec sha256sum {} +')
        result['dashboard'] = {}
        for line in files.splitlines():
            digest, path = line.split(maxsplit=1)
            relative = path.removeprefix('/usr/share/nginx/html/')
            source = ('docs/reference/' + relative.removeprefix('documentation/')
                      if relative.startswith('documentation/') else f'dashboard/{relative}')
            result['dashboard'][source] = digest
        return result


def check_bound_endpoint(inventory, service, url, container_port):
    endpoint = urlparse(url)
    if endpoint.hostname not in ('127.0.0.1', 'localhost') or endpoint.scheme != 'http':
        raise AssertionError('Проверка происхождения endpoint поддерживает только локальный HTTP')
    host_port = endpoint.port or 80
    bindings = inventory[service]['ports'].get(f'{container_port}/tcp') or []
    if not any(int(binding['HostPort']) == host_port for binding in bindings):
        raise AssertionError(f'{url} не соответствует published port контейнера {service}')


async def run(args, out, report):
    docker = Docker(args.context, args.project)
    engine_info = json.loads(await asyncio.to_thread(docker.run, 'info', '--format', '{{json .}}'))
    report['docker_engine'] = {key: engine_info.get(key) for key in
        ('ServerVersion', 'NCPU', 'MemTotal', 'KernelVersion', 'Architecture', 'OSType', 'OperatingSystem')}
    inventory = await asyncio.to_thread(docker.inventory)
    report['containers_before'] = inventory
    check_bound_endpoint(inventory, 'dashboard', args.dashboard, 8080)
    check_bound_endpoint(inventory, 'ml', args.ml_url, 8001)
    check_bound_endpoint(inventory, 'backend', f'http://{args.ndtp_host}:{args.ndtp_port}', 9201)
    hashes = await asyncio.to_thread(docker.source_hashes, inventory)
    report['runtime_code_sha256'] = hashes
    # nginx base image retains its standard error page beside COPY dashboard.
    # Record that asset separately; it is not a stale file from our source tree.
    base_assets = {'dashboard/50x.html'}
    report['base_image_assets_sha256'] = {name: digest for name, digest in hashes['dashboard'].items()
        if name in base_assets and not (ROOT / name).is_file()}
    mismatches = [f'{service}:{name}' for service, files in hashes.items() for name, digest in files.items()
                  if name not in report['base_image_assets_sha256']
                  and (not (ROOT / name).is_file() or hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest)]
    report['runtime_source_mismatches'] = mismatches
    if mismatches:
        raise AssertionError(f'Образы отличаются от исходников: {mismatches[:10]}; сначала пересоберите стек')
    report['tool_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    writers = []
    readers = []
    touched = False
    ml_stopped = False
    expected_ids = set(range(1001, 1001 + args.vehicles))

    def save(name, value):
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

    def passed(name, **details):
        report['checks'].append(dict(name=name, passed=True, **details))
        print(f'PASS {name}: {json.dumps(details, ensure_ascii=False)}', flush=True)

    async def disconnect():
        for writer in writers:
            writer.close()
        await asyncio.gather(*(writer.wait_closed() for writer in writers), return_exceptions=True)
        writers.clear()
        for task in readers:
            task.cancel()
        await asyncio.gather(*readers, return_exceptions=True)
        readers.clear()

    async def drain_ack(reader):
        while await reader.read(4096):
            pass

    async def connect():
        for index in range(args.vehicles):
            reader, writer = await asyncio.open_connection(args.ndtp_host, args.ndtp_port)
            writers.append(writer)
            readers.append(asyncio.create_task(drain_ack(reader)))
            unit = 1166336 + index
            writer.write(frame(unit, 0, 100, struct.pack('<HHHIII', 6, 2, 0, unit, 65535, 0), 1))
            await writer.drain()

    async def send_round(number):
        started = time.monotonic()
        stamp = int(time.time())
        payloads = []
        for index, writer in enumerate(writers):
            nav = struct.pack('<IIIBBHHHHHBB', stamp, 376000000 + index * 1000 + number,
                557500000 + index * 1000, 0xe0, 100, 22, 25, 45, 0, 150, 10, 1)
            payload = frame(1166336 + index, 1, 101, bytes([0, 0]) + nav, number + 1)
            writer.write(payload)
            payloads.append(payload)
        for writer in writers:
            await writer.drain()
        return started, payloads

    async with httpx.AsyncClient(timeout=15, trust_env=False) as client:
        async def api(path='/api/v1/state', body=None):
            response = await (client.get(args.dashboard + path) if body is None
                              else client.post(args.dashboard + path, json=body))
            response.raise_for_status()
            return response.json()

        async def wait(predicate, timeout=15):
            until = time.monotonic() + timeout
            state = None
            while time.monotonic() < until:
                state = await api()
                if predicate(state):
                    return state
                await asyncio.sleep(.1)
            save('timeout-state.json', state)
            raise AssertionError(f'Ожидание условия превысило {timeout} с')

        try:
            before = await api()
            save('state-before.json', before)
            # Не закрываем чужие подключения и не перенастраиваем официальный эмулятор.
            if before['health']['ndtp_connections']:
                raise AssertionError('Сначала остановите другие NDTP producers: нагрузка должна быть изолирована')
            model = await client.get(args.ml_url + '/model')
            model.raise_for_status()
            report['model'] = model.json()
            assert report['model']['trained'] is True
            current = datetime.now(timezone.utc)
            context = build_context(current, args.vehicles, args.plan_visits)
            report['context_payload_bytes'] = len(json.dumps(context, separators=(',', ':')).encode())
            touched = True
            await api('/api/v1/live/context', context)
            for tr_id in sorted(expected_ids):
                result = await api('/api/v1/arrivals', dict(tr_id=tr_id,
                    planned_stop_id=f'last-{tr_id}', arrived_at=current.isoformat()))
                assert result['current_deviation']['delay_s'] == 180
                assert result['current_deviation']['source'] == 'arrival'
            state = await wait(lambda s: round_covered(s, expected_ids, 0))
            assert state['summary']['unknown'] == args.vehicles
            initial = dict(state['metrics'])
            passed('Docker image/source parity и прогретый HTTP learned', vehicles=args.vehicles,
                   visits_per_vehicle=args.plan_visits)
            await connect()
            pending = {}
            observations = []
            pending_history = []
            last_payloads = []
            sending_finished = False
            started = time.monotonic()

            async def sender():
                nonlocal last_payloads, sending_finished
                for number in range(1, args.rounds + 1):
                    await asyncio.sleep(max(0, started + (number - 1) * args.interval - time.monotonic()))
                    sent_at, last_payloads = await send_round(number)
                    pending[number] = sent_at
                sending_finished = True

            sender_task = asyncio.create_task(sender())
            try:
                while not sending_finished or pending:
                    if sender_task.done():
                        sender_task.result()
                    if time.monotonic() - started > args.rounds * args.interval + 15:
                        raise AssertionError('Поток не обработан в бюджете времени')
                    state = await api()
                    observed_at = time.monotonic()
                    for number, sent_at in list(pending.items()):
                        if round_covered(state, expected_ids, number):
                            observations.append(dict(round=number, e2e_ms=round((observed_at - sent_at) * 1000, 2)))
                            del pending[number]
                    pending_history.append(dict(elapsed_s=round(observed_at - started, 3),
                                                pending_rounds=len(pending)))
                    await asyncio.sleep(.1)
            finally:
                sender_task.cancel()
                await asyncio.gather(sender_task, return_exceptions=True)
            latencies = [item['e2e_ms'] for item in observations]
            received = state['metrics']['received'] - initial['received']
            report['stream'] = dict(sent=args.vehicles * args.rounds, received=received,
                nominal_messages_per_s=args.vehicles / args.interval, wall_s=round(time.monotonic() - started, 3),
                e2e_p50_ms=percentile(latencies, .5), e2e_p95_ms=percentile(latencies, .95),
                e2e_max_ms=max(latencies), p95_budget_ms=2000, p95_budget_met=percentile(latencies, .95) <= 2000,
                peak_pending_rounds=max(item['pending_rounds'] for item in pending_history), pending_at_end=len(pending),
                http_ml_p95_ms=state['metrics']['inference_p95_ms'], pipeline_p95_ms=state['metrics']['pipeline_p95_ms'],
                metrics_before=initial, metrics_after=state['metrics'], observations=observations, pending_history=pending_history)
            save('load-state.json', state)
            assert received == args.vehicles * args.rounds
            assert state['metrics']['ndtp_errors'] == initial['ndtp_errors']
            assert len(observations) == args.rounds
            assert all(v['telemetry_sequence'] == args.rounds and v['prediction_input_sequence'] == args.rounds
                       and math.isfinite(v['prediction']['predicted_delay_s']) for v in state['vehicles'])
            trace = await api('/api/v1/vehicles/1001/forecast-trace')
            save('forecast-trace.json', trace)
            assert trace['execution'] == 'ml_http' and trace['response']['method'] == 'learned'
            assert trace['request']['current_delay_s'] == 180
            assert trace['request']['current_delay_source'] == 'arrival'
            assert len(trace['request']['plan_context']['stops']) == args.plan_visits
            passed('NDTP → backend → ML HTTP → nginx API', **{
                key: report['stream'][key] for key in ('sent', 'received', 'e2e_p95_ms', 'pending_at_end')})
            duplicates = state['metrics']['duplicates']
            writers[0].write(last_payloads[0])
            await writers[0].drain()
            state = await wait(lambda s: s['metrics']['duplicates'] == duplicates + 1)
            assert state['metrics']['received'] - initial['received'] == received
            errors = state['metrics']['ndtp_errors']
            _, bad = await asyncio.open_connection(args.ndtp_host, args.ndtp_port)
            try:
                corrupted = bytearray(last_payloads[0]); corrupted[-1] ^= 1
                bad.write(corrupted)
                await bad.drain()
                await wait(lambda s: s['metrics']['ndtp_errors'] > errors)
            finally:
                bad.close()
                await bad.wait_closed()
            passed('Дубликат и повреждённый CRC', duplicate_not_counted=True)
            await disconnect()
            await wait(lambda s: s['health']['ndtp_connections'] == 0)
            restarted = time.monotonic()
            await connect()
            await send_round(args.rounds + 1)
            await wait(lambda s: round_covered(s, expected_ids, args.rounds + 1))
            passed('Повторный NDTP handshake', recovery_s=round(time.monotonic() - restarted, 3))

            if args.check_recovery:
                ml_stopped = True
                await asyncio.to_thread(docker.run, 'stop', '--time', '5', inventory['ml']['id'])
                failed_at = time.monotonic()
                state = await wait(lambda s: s['health']['ml'] == 'unavailable'
                    and round_covered(s, expected_ids, args.rounds + 1, method='fallback'))
                assert all(v['prediction']['predicted_delay_s'] == 180 for v in state['vehicles'])
                passed('Остановлен настоящий ML-контейнер', fallback_s=round(time.monotonic() - failed_at, 3))
                recovered_at = time.monotonic()
                await asyncio.to_thread(docker.run, 'start', inventory['ml']['id'])
                ml_stopped = False
                await wait(lambda s: s['health']['ml'] == 'ok' and round_covered(s, expected_ids, args.rounds + 1))
                passed('Тот же ML-контейнер запущен снова', recovery_s=round(time.monotonic() - recovered_at, 3))
            report['containers_after'] = await asyncio.to_thread(docker.inventory)
            assert all(report['containers_after'][name]['id'] == container['id']
                       for name, container in inventory.items())
            assert report['stream']['p95_budget_met'], 'Измеренный p95 превышает 2000 мс'
            report['passed'] = True
        finally:
            await disconnect()
            if ml_stopped:
                await asyncio.to_thread(docker.run, 'start', inventory['ml']['id'])
            if touched:
                try:
                    if args.restore == 'replay':
                        await api('/api/v1/replay/load', {})
                        loaded = await api()
                        after_cycle = (loaded.get('metrics') or {}).get('pipeline_cycles', 0)
                        context_version = (loaded.get('context') or {}).get('version')
                        restored = await wait(lambda s: restored_replay_ready(s,
                            after_cycle=after_cycle, context_version=context_version))
                    else:
                        await api('/api/v1/mode', dict(mode='demo'))
                        restored = await wait(lambda s: s['mode'] == 'demo' and bool(s['vehicles']))
                    save('restored-state.json', restored)
                    report['restoration'] = dict(passed=True, mode=restored['mode'], clock_time=restored['clock_time'])
                    if args.restore == 'replay':
                        report['restoration']['readiness'] = replay_readiness(restored, after_cycle=after_cycle)
                except Exception as error:
                    report['restoration'] = dict(passed=False, error=repr(error))
                    report['passed'] = False
                    raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context')
    parser.add_argument('--project', default='ritm-transport')
    parser.add_argument('--dashboard', default='http://127.0.0.1:8080')
    parser.add_argument('--ml-url', default='http://127.0.0.1:8001')
    parser.add_argument('--ndtp-host', default='127.0.0.1')
    parser.add_argument('--ndtp-port', type=int, default=9201)
    parser.add_argument('--vehicles', type=int, default=40, choices=range(1, 129))
    parser.add_argument('--rounds', type=int, default=24, choices=range(2, 61))
    parser.add_argument('--interval', type=float, default=1.)
    parser.add_argument('--plan-visits', type=int, default=400, choices=range(2, 2001))
    parser.add_argument('--check-recovery', action='store_true')
    parser.add_argument('--restore', choices=('replay', 'demo'), default='replay')
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/docker-load')
    args = parser.parse_args()
    if not .1 <= args.interval <= 1.5 or args.vehicles * args.plan_visits > 20000:
        parser.error('interval должен быть 0.1..1.5 с; vehicles × plan-visits ≤ 20000')
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    report = dict(passed=False, started_at=datetime.now(timezone.utc).isoformat(),
        environment=f'{platform.platform()} / Python {platform.python_version()}',
        config={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        checks=[], limitations=[
            'Искусственный план и подтверждения arrival; качество ML и вероятность здесь не оцениваются.',
            'Настоящий Linux Docker-стек, хостовый TCP producer, nginx UI proxy; отрисовка браузером не измеряется.',
            'Опрос API каждые 100 мс; end-to-end включает ожидание цикла backend и получение JSON.',
            'Короткий ограниченный поток, не предел пропускной способности и не длительный soak test.',
            'Pending — неподтверждённые раунды, не размер очереди ядра TCP; раунды могут объединяться одним прогнозом.',
            'Готовые запущенные образы; холодная установка, отсутствие интернета и старение GPS >60 с здесь не проверяются.',
            'HTTP/цикл p95 — внутреннее окно backend, включая прогрев; end-to-end — раунды измеряемого потока.'])
    try:
        asyncio.run(run(args, args.out, report))
    except (Exception, KeyboardInterrupt) as error:
        report['passed'] = False
        report['error'] = f'{type(error).__name__}: {error}'
        print(report['error'], file=sys.stderr)
    finally:
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        (args.out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        print(args.out / 'report.json', flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
