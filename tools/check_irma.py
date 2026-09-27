"""Проверяет реальные Nav00+IRMA04+Usi08 кадры официального эмулятора.

На время проверки заменяет конфиг эмулятора, затем восстанавливает его в
finally; его генерация начинает цикл заново. Контекст backend не меняет.
Пустой начальный null-конфиг нельзя вернуть через POST официального API.
Без --restore-empty-by-restart такой случай отклоняется до любых изменений.
С флагом разрешён перезапуск только проверенного локального Docker-контейнера
эмулятора; он возвращает исходный null-конфиг. PASS ставится после cleanup.
Временный TCP-listener запускается внутри backend-контейнера на отдельном
порту. Проверяются ручные synthetic поля, не реальные двери/поездки.
Нужны Docker CLI, запущенные backend и официальный эмулятор; Python stdlib.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import select
import struct
import subprocess
import sys
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.ndtp import parse_frame  # noqa: E402


EMPTY_CONFIG = {'targetHost': None, 'targetPort': None, 'units': []}


def restoration_method(original, *, allow_empty_restart, emulator_url, inspected, docker_host):
    """Выбрать доказуемый способ восстановления ДО изменения /api/config.

    Restart допустим только по явной опции, через локальный Unix Docker socket
    и при совпадении буквального loopback URL с опубликованным портом того же
    контейнера. URL стороннего сервиса не даёт права перезапускать контейнер.
    """
    if original != EMPTY_CONFIG:
        host, port = original.get('targetHost'), original.get('targetPort')
        if (not isinstance(host, str) or not host.strip() or type(port) is not int
                or not 1 <= port <= 65535 or not isinstance(original.get('units'), list)):
            raise ValueError('Исходный конфиг нельзя безопасно восстановить через POST; проверка не начата')
        return 'http_post'
    if not allow_empty_restart:
        raise ValueError('Исходный null-конфиг официальный API не принимает через POST. '
                         'Изменений нет. Для локального Docker явно укажите --restore-empty-by-restart')
    url = urlsplit(emulator_url)
    if (url.scheme != 'http' or url.hostname != '127.0.0.1' or url.username or url.password
            or url.path not in ('', '/') or url.query or url.fragment):
        raise ValueError('Restart требует URL вида http://127.0.0.1:PORT без пути и credentials')
    if not docker_host.startswith('unix://'):
        raise ValueError('Restart разрешён только Docker context с локальным Unix socket')
    bindings = inspected.get('NetworkSettings', {}).get('Ports', {}).get('18080/tcp') or []
    matches = [p for p in bindings if p.get('HostIp') == '127.0.0.1'
               and str(p.get('HostPort')) == str(url.port or 80)]
    if not inspected.get('Id') or not matches:
        raise ValueError('URL эмулятора не совпадает с loopback port binding указанного контейнера')
    return 'verified_local_container_restart'


def guarded_check(check, restore, finish_listener, report, save_report):
    """Единый итог: успех проверки невозможен при ошибке восстановления/cleanup."""
    errors = []
    report.update(passed=False, checks_passed=False, restored_config=False)
    try:
        check()
        report['checks_passed'] = True
    except BaseException as error:
        errors.append(error)
        report['error'] = f'{type(error).__name__}: {error}'
    finally:
        try:
            restore()
            if report.get('restored_config') is not True:
                raise RuntimeError('Восстановление исходного конфига не подтверждено')
        except BaseException as error:
            errors.append(error)
            report['restored_config'] = False
            report['restoration_error'] = f'{type(error).__name__}: {error}'
        try:
            finish_listener()
        except BaseException as error:
            errors.append(error)
            report['listener_cleanup_error'] = f'{type(error).__name__}: {error}'
        report['passed'] = report['checks_passed'] and report['restored_config'] and not errors
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        save_report(report)
    if errors:
        # Не теряем первоначальный сбой проверки, если cleanup тоже отказал:
        # report содержит оба, причина исключения — первый по времени.
        raise RuntimeError('Проверка IRMA или восстановление завершились с ошибкой; см. report.json') from errors[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context', default='colima-mos-transport')
    parser.add_argument('--container', default='ritm-transport-backend-1')
    parser.add_argument('--emulator', default='http://127.0.0.1:18080')
    parser.add_argument('--emulator-container', default='ritm-transport-emulator-1')
    parser.add_argument('--target-host', default='backend')
    parser.add_argument('--listener-port', type=int, default=19204)
    parser.add_argument('--restore-empty-by-restart', action='store_true',
                        help='Разрешить восстановление исходного null-конфига перезапуском проверенного локального контейнера эмулятора')
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/audit-fixes/irma')
    args = parser.parse_args()
    if not 1 <= args.listener_port <= 65535 or args.listener_port == 9201:
        parser.error('Нужен отдельный свободный TCP-порт, не основной 9201')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    def save(name, value):
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

    def http(path, body=None):
        request = Request(args.emulator.rstrip('/') + path,
                          data=None if body is None else json.dumps(body).encode(),
                          headers={'Content-Type': 'application/json'})
        with urlopen(request, timeout=10) as response:
            return json.load(response)

    def line(process):
        if not select.select([process.stdout], [], [], 15)[0]:
            raise TimeoutError('Нет ответа временного TCP-listener за 15 секунд')
        result = process.stdout.readline()
        if not result:
            raise RuntimeError('TCP-listener завершился до получения кадра')
        return result.strip()

    cases = [('no_sensors', 0, None), ('absent_closed_bits', 0xF0, None),
             ('all_closed', 0xFF, False), ('mixed', 0xA5, True)]
    for door in range(4):
        cases.extend([(f'door{door + 1}_open', 1 << door, True),
                      (f'door{door + 1}_closed', (1 << door) | (1 << (door + 4)), False)])
    contract = {
        'scope': 'Реальный wire-format официального эмулятора; synthetic ручные поля, не качество детектора',
        'baseline': 'До поддержки IRMA04 type4 отклонял весь смешанный кадр',
        'acceptance': 'Все 12 кадров имеют Nav00(26)+IRMA04(15)+Usi08(6), верные флаги и nullable doors_open',
        'cases': cases, 'budget_seconds': 120,
        'expected_irma_payload': 'u32 odometer, u16 zone, 8*u8 counters, low4 present/high4 closed',
        'config_warning': 'Значения передаются в cells[].fields; поля рядом с type эмулятор игнорирует',
    }
    save('contract.json', contract)
    docker = ['docker', '--context', args.context]
    report = {
        'started_at': datetime.now(timezone.utc).isoformat(), 'passed': False,
        'parser_sha256': hashlib.sha256((ROOT / 'backend/ndtp.py').read_bytes()).hexdigest(),
        'contract': contract, 'checks': [], 'restored_config': False,
        'mutation_attempted': False,
        'limitations': 'Подтверждена ручная передача. Эмулятор не моделирует физическое открытие дверей автоматически; present — единственный доступный признак наличия датчика.',
    }
    original = process = inspected = method = docker_host = None
    # Listener проверяет framing только для захвата. Собственно CRC, layout,
    # атомарность и нормализацию проверяет локальный production parse_frame.
    code = f'''import socket,json,struct
s=socket.socket();s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
s.bind(('0.0.0.0',{args.listener_port}));s.listen();s.settimeout(15)
print('READY',flush=True)
def read(c,n):
 b=b''
 while len(b)<n:
  z=c.recv(n-len(b))
  if not z:raise EOFError()
  b+=z
 return b
for _ in range({len(cases)}):
 c,_=s.accept();c.settimeout(10)
 with c:
  frames=[]
  for __ in range(2):
   h=read(c,15);n=struct.unpack_from('<H',h,2)[0]
   if n>65520:raise ValueError('oversize')
   frames.append((h+read(c,n)).hex())
  print(json.dumps(frames),flush=True)
'''
    def check():
        nonlocal original, process, inspected, method, docker_host
        original = http('/api/config')
        save('config-before.json', original)
        # Без opt-in null-конфиг отклоняется даже до обращения к Docker.
        if original == EMPTY_CONFIG and not args.restore_empty_by_restart:
            restoration_method(original, allow_empty_restart=False,
                               emulator_url=args.emulator, inspected={}, docker_host='')
        inspected = json.loads(subprocess.check_output(docker + ['inspect', args.emulator_container]))[0]
        docker_host = ''
        if original == EMPTY_CONFIG:
            docker_context = json.loads(subprocess.check_output(['docker', 'context', 'inspect', args.context]))[0]
            docker_host = docker_context['Endpoints']['docker']['Host']
        method = restoration_method(original, allow_empty_restart=args.restore_empty_by_restart,
                                    emulator_url=args.emulator, inspected=inspected, docker_host=docker_host)
        report.update(restoration_method=method, image_id=inspected['Image'],
                      image_name=inspected['Config']['Image'], emulator_container_id=inspected['Id'])
        save('cells.json', http('/api/cells'))
        process = subprocess.Popen(docker + ['exec', '-i', args.container, 'python', '-u', '-c', code],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        assert line(process) == 'READY'
        for name, flags, expected in cases:
            fields = {'odometer': 0x12345678, 'zone': 0x9ABC}
            for door in range(1, 5):
                fields.update({f'irma_door_in{door}': door, f'irma_door_out{door}': door + 4,
                               f'irma_present_door{door}': bool(flags & (1 << (door - 1))),
                               f'irma_closed_door{door}': bool(flags & (1 << (door + 3)))})
            config = {'targetHost': args.target_host, 'targetPort': args.listener_port, 'units': [{
                'unitId': 1166336, 'intervalMs': 100000, 'autoGenerate': False,
                'cells': [
                    {'type': 'G6CellNav00', 'fields': {'latitude': 557551234, 'longitude': 376173210,
                        'extraDopBit5': True, 'extraDopBit6': True, 'extraDopBit7': True}},
                    {'type': 'G6CellIrma04', 'fields': fields},
                    {'type': 'G6CellUsi08', 'fields': {}},
                ],
            }]}
            # HTTP-ошибка может прийти после изменения состояния: cleanup нужен
            # уже после попытки запроса, а не только после успешного ответа.
            report['mutation_attempted'] = True
            http('/api/config', config)
            frames = json.loads(line(process))
            save(f'{name}-capture.json', {'config': config, 'frames_hex': frames})
            handshake, packet = map(bytes.fromhex, frames)
            assert parse_frame(handshake).kind == 'handshake'
            # Строгое ожидание известных полей, не результат собственного encoder.
            assert len(packet) == 78
            assert packet[53:55] == b'\x04\x00'
            assert packet[55:70] == bytes.fromhex('78563412bc9a0102030405060708') + bytes([flags])
            assert packet[70:72] == b'\x08\x00'
            parsed = parse_frame(packet)
            assert parsed.unit_id == 1166336 and parsed.kind == 'realtime'
            nav = parsed.nav
            assert nav['doors_open'] is expected and nav['location_valid']
            assert abs(nav['lat'] - 55.7551234) < 1e-9
            assert abs(nav['lon'] - 37.617321) < 1e-9
            report['checks'].append({'case': name, 'passed': True, 'flags': flags,
                'doors_open': nav['doors_open'], 'bytes': len(packet),
                'frame_sha256': hashlib.sha256(packet).hexdigest()})
    def restore():
        if not report['mutation_attempted']:
            report['restored_config'] = original is not None
            report['restoration_method'] = 'unchanged_no_mutation'
            if original is not None:
                save('config-after.json', original)
            return
        try:
            if method == 'verified_local_container_restart':
                # Используем полный проверенный ID, не произвольное имя из URL.
                # Повторная проверка binding перед restart исключает stale inspect.
                current = json.loads(subprocess.check_output(docker + ['inspect', inspected['Id']]))[0]
                restoration_method(original, allow_empty_restart=True, emulator_url=args.emulator,
                                   inspected=current, docker_host=docker_host)
                subprocess.run(docker + ['restart', inspected['Id']], check=True,
                               capture_output=True, text=True, timeout=30)
                deadline = time.monotonic() + 20
                while True:
                    try:
                        http('/api/config')
                        break
                    except Exception:
                        if time.monotonic() >= deadline:
                            raise
                        time.sleep(0.2)
            else:
                http('/api/config', original)
            restored = http('/api/config')
            save('config-after.json', restored)
            report['restored_config'] = restored == original
            if not report['restored_config']:
                raise RuntimeError('Конфигурация эмулятора не восстановлена')
        except BaseException:
            # При отказе восстановления останавливаем только собственный
            # тестовый producer. Это НЕ восстановление; общий итог остаётся FAIL.
            try:
                current = http('/api/config')
                ours = (current.get('targetHost') == args.target_host
                        and current.get('targetPort') == args.listener_port
                        and [unit.get('unitId') for unit in current.get('units', [])] == [1166336])
                if ours:
                    http('/api/config', {**current, 'units': []})
                    stopped = http('/api/config')
                    save('config-after-failed-restore.json', stopped)
                    report['test_producer_stopped_after_restore_failure'] = stopped.get('units') == []
            except Exception as error:
                report['producer_stop_error'] = f'{type(error).__name__}: {error}'
            raise

    def finish_listener():
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
        try:
            _, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            _, stderr = process.communicate(timeout=5)
        report['listener_stderr'] = stderr

    guarded_check(check, restore, finish_listener, report, lambda value: save('report.json', value))
    print(f'PASS: {len(cases)} официальных IRMA кадров; конфигурация восстановлена. {out / "report.json"}')


if __name__ == '__main__':
    main()
