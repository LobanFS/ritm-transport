"""Один запуск для жюри: Docker + готовый bundle, без Python ML-стека на хосте.

Из корня репозитория: python3 tools/start_solution.py
По умолчанию загружается dataset/train из репозитория, интерфейс — full.
Dispatcher сразу проигрывает архив, full оставляет его на паузе.
--data-dir выбирает внешний архив; --live запускает ожидание живого потока.
Запуск помощника выбирает источник; загрузка config.js сама источник не меняет.
Обучение и отправка данных в облако не нужны.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
UI_MODES = ('dispatcher', 'full')


def json_api(base, path, payload=None):
    request = Request(base+path, data=None if payload is None else json.dumps(payload).encode(),
                      headers={'Content-Type':'application/json'})
    with urlopen(request, timeout=45) as response:
        return json.load(response)


def dashboard_ui_mode():
    """Проверить runtime-конфиг nginx, в том числе при --no-build."""
    with urlopen('http://127.0.0.1:8080/config.js', timeout=5) as response:
        source = response.read().decode('utf-8').strip()
    prefix = 'window.RITM_CONFIG = '
    if not source.startswith(prefix) or not source.endswith(';'):
        raise ValueError('Dashboard не отдаёт runtime config.js: пересоберите образ')
    return json.loads(source[len(prefix):-1]).get('uiMode')


def inspect_bundle(path):
    # Читать константу как AST: не импортируем Pydantic/ML зависимости на хосте.
    module = ast.parse((ROOT/'ml_service/learned.py').read_text())
    expected = next(ast.literal_eval(n.value) for n in module.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'MODEL_SHA256' for t in n.targets))
    expected_encoder = next(ast.literal_eval(n.value) for n in module.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'ENCODER_SHA256' for t in n.targets))
    model = path/'model.joblib'
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError('Неверный SHA-256 весов: нужен подготовленный frozen bundle')
    encoder_digest = hashlib.sha256((path/'encoder.pt').read_bytes()).hexdigest()
    if encoder_digest != expected_encoder:
        raise ValueError('Неверный SHA-256 encoder.pt: нужен подготовленный frozen bundle')
    metadata = {'model_sha256':digest, 'encoder_sha256':encoder_digest}
    probability_path = path/'probability.json'
    if probability_path.exists():
        probability_bytes = probability_path.read_bytes()
        if len(probability_bytes) > 65536:
            raise ValueError('Слишком большой файл коэффициентов вероятности')
        probability = json.loads(probability_bytes)
        if (not isinstance(probability, dict)
            or probability.get('schema_version') not in ('late-probability-logistic-v1', 'late-probability-logistic-v2')):
            raise ValueError('Неизвестная схема калибратора вероятности')
        if probability.get('regression_model_sha256') != digest:
            raise ValueError('Вероятность подготовлена для другой модели')
        if probability.get('encoder_sha256') != encoder_digest:
            raise ValueError('Калибратор не подтверждает текущий Transformer encoder')
        if probability.get('training_protocol', 'vehicle_oof') != 'vehicle_oof':
            raise ValueError('Неизвестный протокол обучения калибратора вероятности')
        history = probability.get('history_protocol')
        expected_history = 'provided_current_delay_strict_past_90m_12steps'
        if (history not in (None, expected_history)
            or (probability['schema_version'] == 'late-probability-logistic-v2' and history != expected_history)):
            raise ValueError('Неподтверждённый контракт причинной истории калибратора')
        metadata['probability_sha256'] = hashlib.sha256(probability_bytes).hexdigest()
    scope_path = ROOT/'ml_service/probability_gps_scope.json'
    if scope_path.exists() and metadata.get('probability_sha256'):
        scope_bytes = scope_path.read_bytes()
        scope = json.loads(scope_bytes)
        if (scope.get('schema_version') != 'gps-probability-scope-v1'
            or scope.get('validation_status') != 'accepted_secondary_diagnostic'
            or scope.get('scope') != 'historical_real_gps_estimate'):
            raise ValueError('Неверный или непринятый GPS-профиль вероятности')
        if scope.get('regression_model_sha256') != digest:
            raise ValueError('GPS-профиль относится к другой регрессионной модели')
        if scope.get('probability_artifact_sha256') != metadata['probability_sha256']:
            raise ValueError('GPS-профиль относится к другому калибратору вероятности')
        detector_path = ROOT/'backend/gps_arrivals.py'
        detector_hash = hashlib.sha256(detector_path.read_bytes()).hexdigest()
        if scope.get('detector_sha256') != detector_hash:
            raise ValueError('GPS-детектор изменён после проверки профиля вероятности')
        detector_module = ast.parse(detector_path.read_text())
        detector_class = next(n for n in detector_module.body if isinstance(n, ast.ClassDef)
                              and n.name == 'GPSArrivalDetector')
        detector_version = next(ast.literal_eval(n.value) for n in detector_class.body if isinstance(n, ast.Assign)
                                and any(isinstance(t, ast.Name) and t.id == 'version' for t in n.targets))
        if scope.get('detector_version') != detector_version:
            raise ValueError('Версия GPS-детектора не совпадает с профилем вероятности')
        metadata.update(gps_scope_sha256=hashlib.sha256(scope_bytes).hexdigest(),
                        gps_detector_sha256=detector_hash,gps_detector_version=detector_version)
    return metadata


def verify_model_card(model, metadata):
    """Проверить реально загруженные артефакты, в том числе при --no-build."""
    if not model.get('trained') or model.get('method') != 'learned':
        raise ValueError('ML контейнер поднялся без обученной модели: проверьте mount и /model')
    if (model.get('provenance') or {}).get('sha256') != metadata['model_sha256']:
        raise ValueError('Контейнер не подтвердил веса из переданного bundle: проверьте /model')
    if (model.get('provenance') or {}).get('encoder_sha256') != metadata['encoder_sha256']:
        raise ValueError('Контейнер не подтвердил encoder из переданного bundle: проверьте /model')
    probability = model.get('probability') or {}
    if metadata.get('probability_sha256') and probability.get('artifact_sha256') != metadata['probability_sha256']:
        raise ValueError('Контейнер не подтвердил калибратор из переданного bundle: проверьте /model')
    if metadata.get('gps_scope_sha256'):
        scope = probability.get('gps_scope') or {}
        if (probability.get('gps_scope_sha256') != metadata['gps_scope_sha256']
            or scope.get('detector_sha256') != metadata['gps_detector_sha256']
            or scope.get('detector_version') != metadata['gps_detector_version']
            or scope.get('regression_model_sha256') != metadata['model_sha256']
            or scope.get('probability_artifact_sha256') != metadata['probability_sha256']):
            raise ValueError('ML контейнер не подтвердил текущий GPS-профиль: пересоберите образ без --no-build и проверьте /model')


def replay_readiness(state, *, after_cycle):
    """Завершённый цикл и объяснённое состояние ТС, а не прогноз для каждого."""
    vehicles = state.get('vehicles') or []
    cycle = (state.get('metrics') or {}).get('pipeline_cycles', 0)
    ml = (state.get('health') or {}).get('ml')
    # Эти состояния данных легитимны после завершения цикла. Само по себе
    # risk=unknown не подтверждает ни публикацию, ни работоспособность ML.
    no_estimate = {'no_target', 'telemetry_missing', 'telemetry_stale',
                   'deviation_missing', 'deviation_expired', 'target_reached'}
    availability = {}
    pending = 0
    published = 0
    for vehicle in vehicles:
        code = (vehicle.get('prediction_availability') or {}).get('code', 'unspecified')
        availability[code] = availability.get(code, 0)+1
        prediction = vehicle.get('prediction')
        has_publication = isinstance(prediction, dict) and bool(prediction.get('method'))
        published += int(has_publication)
        if code in no_estimate:
            continue
        if code in ('ready', 'model_input_unavailable') and has_publication:
            continue
        pending += 1
    return dict(ready=bool(vehicles) and cycle > after_cycle and ml == 'ok' and pending == 0,
                pipeline_cycles=cycle, required_after_cycle=after_cycle, ml=ml,
                vehicles=len(vehicles), published=published, pending=pending,
                availability=availability)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, default=ROOT/'artifacts/model')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--data-dir', type=Path,
                        help='Внешняя раздача для CSV replay; по умолчанию dataset/ из репозитория')
    source.add_argument('--live', action='store_true', help='Ожидать живую телеметрию, не загружая CSV')
    parser.add_argument('--ui-mode', choices=UI_MODES, default=os.getenv('DASHBOARD_UI_MODE', 'full'),
                        help='full — полный интерфейс (по умолчанию), dispatcher — рабочий экран; также DASHBOARD_UI_MODE')
    parser.add_argument('--dataset-split', choices=('validate', 'train'), default='train',
                        help='Набор replay: train по умолчанию, отклонение считается по GPS; validate использует points.csv')
    parser.add_argument('--context', help='Docker context; глобальная настройка не меняется')
    parser.add_argument('--official-emulator', action='store_true', help='Загрузить официальный tar и поднять сервис эмулятора; настройка NDTP отдельной командой')
    parser.add_argument('--no-build', action='store_true', help='Использовать уже собранные образы')
    parser.add_argument('--check-only', action='store_true', help='Проверить входы без запуска/смены контекста')
    args = parser.parse_args()
    if args.ui_mode not in UI_MODES:
        parser.error('DASHBOARD_UI_MODE должен быть dispatcher или full')
    bundle = args.model_dir.resolve()
    metadata = inspect_bundle(bundle)
    data = None if args.live else (args.data_dir or ROOT/'dataset').resolve()
    replay_config = ({'dataset_split':'train', 'deviation_source':'gps', 'warmup_minutes':30}
                     if args.dataset_split == 'train' else {'dataset_split':'validate'})
    if data:
        required = (('train/traffic.csv','train/schedule.csv') if args.dataset_split == 'train' else
                    ('validate/points.csv','validate/traffic.csv','validate/schedule_plan.csv'))
        for name in required:
            if not (data/name).is_file():
                parser.error(f'Нет файла: {data/name}')
    if args.official_emulator and (not args.data_dir or not (data/'ndtp-telemetry-emulator.tar').is_file()):
        parser.error('--official-emulator требует --data-dir с ndtp-telemetry-emulator.tar')
    if args.check_only:
        print(json.dumps(dict(status='inputs_valid',ui_mode=args.ui_mode,
                              source_mode='replay' if data else 'live',
                              data_dir=str(data) if data else None,
                              dataset_split=args.dataset_split if data else None,**metadata),ensure_ascii=False))
        return
    docker = ['docker'] + (['--context',args.context] if args.context else [])
    compose = docker+['compose','-f','compose.yaml','-f','compose.model.yaml']
    env = dict(os.environ, MODEL_DIR=str(bundle), DASHBOARD_UI_MODE=args.ui_mode, INITIAL_MODE='live')
    if data:
        env['REPLAY_DATA_DIR'] = str(data)
        compose += ['-f','compose.replay.yaml']
    if args.official_emulator:
        check = subprocess.run(docker+['image','inspect','ndtp-telemetry-emulator:1.0'],
                               stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        if check.returncode:
            subprocess.run(docker+['load','-i',str(data/'ndtp-telemetry-emulator.tar')],check=True)
        compose += ['-f','compose.emulator.yaml']
    out = ROOT/'artifacts/start-solution'
    out.mkdir(parents=True,exist_ok=True)
    report = dict(started_at=datetime.now(timezone.utc).isoformat(),passed=False,command=compose,
                  ui_mode=args.ui_mode,data_dir=str(data) if data else None,**metadata)
    try:
        start = time.monotonic()
        subprocess.run(compose+['up','-d','--wait','--wait-timeout','90']+([] if args.no_build else ['--build']),
                       cwd=ROOT,env=env,check=True)
        report['up_s_including_requested_build'] = time.monotonic()-start
        model = json_api('http://127.0.0.1:8001','/model')
        verify_model_card(model,metadata)
        report['model'] = model
        if dashboard_ui_mode() != args.ui_mode:
            raise ValueError('Dashboard не подтвердил ui-mode: пересоберите образ и проверьте DASHBOARD_UI_MODE')
        if data:
            json_api('http://127.0.0.1:8000','/api/v1/replay/load',replay_config)
            if args.ui_mode == 'dispatcher':
                json_api('http://127.0.0.1:8000','/api/v1/replay/control',{'action':'resume'})
            expected_mode = 'replay'
        else:
            json_api('http://127.0.0.1:8000','/api/v1/mode',{'mode':'live'})
            expected_mode = 'live'
        state = json_api('http://127.0.0.1:8080','/api/v1/state')
        if state['mode'] != expected_mode:
            raise ValueError('UI proxy не подтвердил выбранный режим')
        if data and state['replay']['dataset_split'] != args.dataset_split:
            raise ValueError('UI proxy не подтвердил выбранный набор replay')
        report['replay_config'] = replay_config if data else None
        # В live отсутствие ТС нормально. У архива ждём новый завершённый цикл:
        # no_target/unknown могут быть корректным результатом, но fallback при
        # недоступном ML или prediction_pending готовностью не считаются.
        if data:
            after_cycle = (state.get('metrics') or {}).get('pipeline_cycles', 0)
            context_version = (state.get('context') or {}).get('version')
            deadline = time.monotonic()+10
            readiness = replay_readiness(state, after_cycle=after_cycle)
            while time.monotonic() < deadline:
                state = json_api('http://127.0.0.1:8080','/api/v1/state')
                if (state.get('mode') != 'replay'
                    or (state.get('context') or {}).get('version') != context_version):
                    raise ValueError('Контекст replay изменился во время проверки готовности')
                readiness = replay_readiness(state, after_cycle=after_cycle)
                if readiness['ready']:
                    break
                time.sleep(.25)
            report['replay_readiness'] = readiness
            if not readiness['ready']:
                raise ValueError('Replay не готов за 10 секунд: '+json.dumps(readiness, ensure_ascii=False))
        report['mode'] = expected_mode
        report['vehicles'] = len(state['vehicles'])
        report['methods'] = sorted({v['prediction']['method'] for v in state['vehicles'] if v['prediction'] is not None})
        report['openapi'] = {}
        for name,port in [('backend',8000),('ml',8001)]:
            schema = json_api(f'http://127.0.0.1:{port}','/openapi.json')
            report['openapi'][name] = {'paths':len(schema['paths']),'version':schema['info']['version']}
        report['passed'] = True
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    message = ('архивный поток запущен' if args.ui_mode == 'dispatcher' else 'архив на паузе, нажмите Продолжить') if data else 'live, ожидание входящих данных'
    print(f'Готово: http://127.0.0.1:8080 — {args.ui_mode}; {message}.')
    print('Backend/ML Swagger: порты 8000/8001, путь /docs.')
    if args.official_emulator:
        print('Эмулятор доступен на 18080. Для NDTP вместо replay: python3 tools/configure_emulator.py --target-host backend')
    print(out/'report.json')


if __name__ == '__main__':
    main()
