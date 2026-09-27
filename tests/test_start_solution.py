"""Запуск не принимает устаревший ML-образ без текущего GPS-профиля."""
from copy import deepcopy
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from tools import start_solution


@pytest.mark.parametrize('ui_mode', ['dispatcher', 'full'])
def test_ui_preflight_uses_explicit_live_without_archive(tmp_path, monkeypatch, capsys, ui_mode):
    bundle, _, _ = setup_bundle(tmp_path, monkeypatch)
    monkeypatch.setenv('DASHBOARD_UI_MODE', ui_mode)
    monkeypatch.setattr(sys, 'argv', ['start_solution.py', '--model-dir', str(bundle), '--live', '--check-only'])
    start_solution.main()
    result = json.loads(capsys.readouterr().out)
    assert result['ui_mode'] == ui_mode
    assert result['source_mode'] == 'live'
    assert result['dataset_split'] is None and result['data_dir'] is None


def test_invalid_ui_environment_is_rejected_before_model_or_runtime_access(monkeypatch):
    monkeypatch.setenv('DASHBOARD_UI_MODE', 'unexpected')
    monkeypatch.setattr(sys, 'argv', ['start_solution.py', '--check-only'])
    monkeypatch.setattr(start_solution, 'inspect_bundle', lambda _: pytest.fail('bundle read before config validation'))
    with pytest.raises(SystemExit) as error:
        start_solution.main()
    assert error.value.code == 2


@pytest.mark.parametrize('ui_mode', ['dispatcher', 'full'])
@pytest.mark.parametrize('source', ['bundled_train', 'bundled_validate', 'external_train', 'external_validate', 'live'])
def test_launch_selects_source_and_passes_ui_mode(tmp_path, monkeypatch, ui_mode, source):
    bundle, _, scope = setup_bundle(tmp_path, monkeypatch)
    metadata = start_solution.inspect_bundle(bundle)
    api_calls, commands = [], []
    with_archive = source != 'live'
    split = 'validate' if source.endswith('validate') else 'train'
    state = {'mode':'replay' if with_archive else 'live', 'vehicles':[],
             'health':{'ml':'ok'}, 'metrics':{'pipeline_cycles':0}, 'context':{'version':1}}
    if with_archive:
        state.update(vehicles=[{'prediction':{'method':'learned'},
                                'prediction_availability':{'code':'ready'}},
                               {'prediction':{'method':'unavailable', 'predicted_delay_s':None},
                                'prediction_availability':{'code':'deviation_missing'}},
                               {'prediction':None, 'prediction_availability':{'code':'no_target'}}],
                     replay={'dataset_split':split})
    def fake_api(base, path, payload=None):
        api_calls.append((path,payload))
        if path == '/model': return runtime_card(metadata, scope)
        if path == '/api/v1/state':
            state['metrics']['pipeline_cycles'] += 1
            return state
        if path == '/openapi.json': return {'paths':{},'info':{'version':'fixture'}}
        return {'ok':True}
    def fake_run(command, **kwargs):
        commands.append((command,kwargs))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(start_solution, 'json_api', fake_api)
    monkeypatch.setattr(start_solution, 'dashboard_ui_mode', lambda: ui_mode)
    monkeypatch.setattr(start_solution.subprocess, 'run', fake_run)
    monkeypatch.setenv('DASHBOARD_UI_MODE', 'full' if ui_mode == 'dispatcher' else 'dispatcher')
    argv = ['start_solution.py','--model-dir',str(bundle),'--ui-mode',ui_mode]
    if with_archive:
        data = tmp_path/'external-dataset' if source.startswith('external') else start_solution.ROOT/'dataset'
        write_dataset(data, split)
        if source.startswith('external'):
            argv += ['--data-dir',str(data)]
        if split == 'validate':
            argv += ['--dataset-split','validate']
    else:
        argv += ['--live']
    monkeypatch.setattr(sys, 'argv', argv)
    start_solution.main()
    assert commands[0][1]['env']['DASHBOARD_UI_MODE'] == ui_mode
    assert commands[0][1]['env']['INITIAL_MODE'] == 'live'
    assert not any('generator' in path for path,_ in api_calls)
    if with_archive:
        expected_config = ({'dataset_split':'train', 'deviation_source':'gps', 'warmup_minutes':30}
                           if split == 'train' else {'dataset_split':'validate'})
        assert ('/api/v1/replay/load',expected_config) in api_calls
        assert commands[0][1]['env']['REPLAY_DATA_DIR'] == str(data)
        assert 'compose.replay.yaml' in commands[0][0]
        assert (('/api/v1/replay/control',{'action':'resume'}) in api_calls) == (ui_mode == 'dispatcher')
        assert not any(path == '/api/v1/mode' for path,_ in api_calls)
    else:
        assert ('/api/v1/mode',{'mode':'live'}) in api_calls
        assert not any(path.startswith('/api/v1/replay/') for path,_ in api_calls)
        assert 'compose.replay.yaml' not in commands[0][0]
    report=json.loads((start_solution.ROOT/'artifacts/start-solution/report.json').read_text())
    assert report['passed'] and report['ui_mode'] == ui_mode
    assert report['mode'] == state['mode']
    assert report['vehicles'] == 3*int(with_archive)
    if with_archive:
        assert report['replay_readiness']['ready']
        assert report['data_dir'] == str(data)


def replay_state(vehicles, *, cycle=2, ml='ok'):
    return {'mode':'replay', 'context':{'version':1}, 'replay':{'dataset_split':'validate'},
            'health':{'ml':ml}, 'metrics':{'pipeline_cycles':cycle}, 'vehicles':vehicles}


def vehicle_state(code, prediction=None):
    return {'prediction_availability':{'code':code}, 'prediction':prediction}


def test_replay_ready_with_mixed_forecasts_no_target_and_unknown_input():
    state = replay_state([
        vehicle_state('ready', {'method':'learned', 'predicted_delay_s':123}),
        vehicle_state('deviation_missing', {'method':'unavailable', 'predicted_delay_s':None}),
        vehicle_state('no_target'),
    ])
    readiness = start_solution.replay_readiness(state, after_cycle=1)
    assert readiness['ready'] and readiness['published'] == 2 and readiness['pending'] == 0
    assert readiness['availability'] == {'ready':1, 'deviation_missing':1, 'no_target':1}


@pytest.mark.parametrize('code', [
    'no_target', 'deviation_missing', 'deviation_expired', 'telemetry_missing',
    'telemetry_stale', 'target_reached',
])
def test_replay_without_numeric_forecast_is_ready_after_healthy_completed_cycle(code):
    state = replay_state([vehicle_state(code)])
    assert start_solution.replay_readiness(state, after_cycle=1)['ready']
    assert not start_solution.replay_readiness(state, after_cycle=2)['ready']


@pytest.mark.parametrize('code,prediction', [
    ('prediction_pending', None), ('ready', None), ('model_input_unavailable', None),
    ('unexpected_unknown', None), ('source_unavailable', {'method':'learned'}),
])
def test_replay_does_not_accept_pending_or_unexplained_missing_publication(code, prediction):
    state = replay_state([vehicle_state('no_target'), vehicle_state(code, prediction)])
    readiness = start_solution.replay_readiness(state, after_cycle=1)
    assert not readiness['ready'] and readiness['pending'] == 1


@pytest.mark.parametrize('ml', ['starting', 'unavailable', None])
def test_replay_does_not_accept_backend_fallback_as_healthy_ml(ml):
    state = replay_state([vehicle_state('ready', {'method':'fallback', 'predicted_delay_s':123})], ml=ml)
    assert not start_solution.replay_readiness(state, after_cycle=1)['ready']


def test_replay_model_input_unavailable_response_is_legitimate_publication():
    state = replay_state([vehicle_state('model_input_unavailable',
                                       {'method':'unavailable', 'predicted_delay_s':None})])
    assert start_solution.replay_readiness(state, after_cycle=1)['ready']


def test_replay_empty_context_is_not_ready():
    assert not start_solution.replay_readiness(replay_state([]), after_cycle=1)['ready']


def test_train_preflight_accepts_two_train_files_without_validate_points_or_labels(tmp_path, monkeypatch, capsys):
    bundle, _, _ = setup_bundle(tmp_path, monkeypatch)
    data = tmp_path/'dataset'
    (data/'train').mkdir(parents=True)
    for name in ('traffic.csv', 'schedule.csv'):
        (data/'train'/name).write_text('fixture header\n')
    monkeypatch.setattr(sys, 'argv', ['start_solution.py', '--model-dir', str(bundle),
        '--data-dir', str(data), '--check-only'])
    start_solution.main()
    result = json.loads(capsys.readouterr().out)
    assert result['status'] == 'inputs_valid' and result['dataset_split'] == 'train'
    # Отсутствующие validate входы не заменяются данными train неявно.
    monkeypatch.setattr(sys, 'argv', ['start_solution.py', '--model-dir', str(bundle),
        '--data-dir', str(data), '--dataset-split', 'validate', '--check-only'])
    with pytest.raises(SystemExit) as error:
        start_solution.main()
    assert error.value.code == 2


@pytest.mark.parametrize('split', ['train', 'validate'])
def test_missing_bundled_csv_is_reported_without_runtime_start(tmp_path, monkeypatch, capsys, split):
    bundle, _, _ = setup_bundle(tmp_path, monkeypatch)
    monkeypatch.setattr(start_solution.subprocess, 'run', lambda *_args, **_kwargs: pytest.fail('runtime called'))
    monkeypatch.setattr(sys, 'argv', ['start_solution.py', '--model-dir', str(bundle),
                                    '--dataset-split', split, '--check-only'])
    with pytest.raises(SystemExit) as error:
        start_solution.main()
    assert error.value.code == 2
    assert str(start_solution.ROOT/'dataset'/split) in capsys.readouterr().err


def write_dataset(data, split):
    (data/split).mkdir(parents=True)
    names = ('traffic.csv', 'schedule.csv') if split == 'train' else ('traffic.csv', 'schedule_plan.csv', 'points.csv')
    for name in names:
        (data/split/name).write_text('fixture header\n')


@pytest.mark.parametrize('split', ['train', 'validate'])
def test_preflight_uses_bundled_dataset_without_path_argument(tmp_path, monkeypatch, capsys, split):
    bundle, _, _ = setup_bundle(tmp_path, monkeypatch)
    write_dataset(start_solution.ROOT/'dataset', split)
    argv = ['start_solution.py', '--model-dir', str(bundle), '--check-only']
    if split == 'validate':
        argv += ['--dataset-split', 'validate']
    monkeypatch.setattr(sys, 'argv', argv)
    start_solution.main()
    result = json.loads(capsys.readouterr().out)
    assert result['source_mode'] == 'replay' and result['dataset_split'] == split
    assert result['data_dir'] == str(start_solution.ROOT/'dataset')


def test_live_and_external_dataset_are_mutually_exclusive(monkeypatch):
    monkeypatch.setattr(start_solution, 'inspect_bundle', lambda _: pytest.fail('model read before argument validation'))
    monkeypatch.setattr(sys, 'argv', ['start_solution.py', '--live', '--data-dir', '/unused'])
    with pytest.raises(SystemExit) as error:
        start_solution.main()
    assert error.value.code == 2


def test_official_emulator_requires_explicit_external_dataset_with_tar(tmp_path, monkeypatch, capsys):
    bundle, _, _ = setup_bundle(tmp_path, monkeypatch)
    write_dataset(start_solution.ROOT/'dataset', 'train')
    argv = ['start_solution.py', '--model-dir', str(bundle), '--official-emulator', '--check-only']
    monkeypatch.setattr(sys, 'argv', argv)
    with pytest.raises(SystemExit) as error:
        start_solution.main()
    assert error.value.code == 2
    assert '--data-dir' in capsys.readouterr().err
    data = tmp_path/'external'
    write_dataset(data, 'train')
    monkeypatch.setattr(sys, 'argv', argv + ['--data-dir', str(data)])
    with pytest.raises(SystemExit) as error:
        start_solution.main()
    assert error.value.code == 2
    assert 'ndtp-telemetry-emulator.tar' in capsys.readouterr().err
    (data/'ndtp-telemetry-emulator.tar').write_bytes(b'fixture archive')
    start_solution.main()
    assert json.loads(capsys.readouterr().out)['data_dir'] == str(data)


def setup_bundle(tmp_path,monkeypatch,*,with_scope=True):
    root = tmp_path/'source'
    model_dir = tmp_path/'model'
    model_dir.mkdir()
    (root/'ml_service').mkdir(parents=True)
    (root/'backend').mkdir()
    model_bytes = b'trusted test model'
    model_hash = hashlib.sha256(model_bytes).hexdigest()
    encoder_bytes = b'trusted test encoder'
    encoder_hash = hashlib.sha256(encoder_bytes).hexdigest()
    (model_dir/'model.joblib').write_bytes(model_bytes)
    (model_dir/'encoder.pt').write_bytes(encoder_bytes)
    (root/'ml_service/learned.py').write_text(
        f"MODEL_SHA256 = {model_hash!r}\nENCODER_SHA256 = {encoder_hash!r}\n")
    probability_bytes = json.dumps({
        'schema_version':'late-probability-logistic-v2',
        'regression_model_sha256':model_hash, 'encoder_sha256':encoder_hash,
        'training_protocol':'vehicle_oof',
        'history_protocol':'provided_current_delay_strict_past_90m_12steps',
    }).encode()
    probability_hash = hashlib.sha256(probability_bytes).hexdigest()
    (model_dir/'probability.json').write_bytes(probability_bytes)
    detector_bytes = b"class GPSArrivalDetector:\n    version = 'fixture-v1'\n"
    (root/'backend/gps_arrivals.py').write_bytes(detector_bytes)
    scope = dict(schema_version='gps-probability-scope-v1',validation_status='accepted_secondary_diagnostic',
                 scope='historical_real_gps_estimate',regression_model_sha256=model_hash,
                 probability_artifact_sha256=probability_hash,detector_version='fixture-v1',
                 detector_sha256=hashlib.sha256(detector_bytes).hexdigest())
    scope_path = root/'ml_service/probability_gps_scope.json'
    if with_scope:
        scope_path.write_text(json.dumps(scope))
    monkeypatch.setattr(start_solution,'ROOT',root)
    return model_dir,scope_path,scope


def runtime_card(metadata,scope=None):
    result = dict(trained=True,method='learned',provenance={
                      'sha256':metadata['model_sha256'],'encoder_sha256':metadata['encoder_sha256']},
                  probability={'artifact_sha256':metadata['probability_sha256']})
    if scope:
        result['probability'].update(gps_scope=scope,gps_scope_sha256=metadata['gps_scope_sha256'])
    return result


def test_optional_gps_scope_absence_keeps_previous_csv_only_bundle_working(tmp_path,monkeypatch):
    bundle,_,_ = setup_bundle(tmp_path,monkeypatch,with_scope=False)
    metadata = start_solution.inspect_bundle(bundle)
    assert set(metadata) == {'model_sha256','encoder_sha256','probability_sha256'}
    start_solution.verify_model_card(runtime_card(metadata),metadata)


def test_current_profile_matches_local_bundle_and_loaded_runtime(tmp_path,monkeypatch):
    bundle,path,scope = setup_bundle(tmp_path,monkeypatch)
    metadata = start_solution.inspect_bundle(bundle)
    assert metadata['gps_scope_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    start_solution.verify_model_card(runtime_card(metadata,scope),metadata)


@pytest.mark.parametrize('field,value,message', [
    ('regression_model_sha256', '0'*64, 'другой модели'),
    ('encoder_sha256', '0'*64, 'encoder'),
    ('encoder_sha256', None, 'encoder'),
    ('schema_version', 'late-probability-logistic-v3', 'схема'),
    ('schema_version', None, 'схема'),
    ('history_protocol', 'future_rows_allowed', 'истории'),
    ('history_protocol', None, 'истории'),
    ('training_protocol', 'row_random_split', 'обучения'),
])
def test_probability_preflight_rejects_incompatible_hybrid_metadata(tmp_path, monkeypatch, field, value, message):
    bundle, _, _ = setup_bundle(tmp_path, monkeypatch, with_scope=False)
    path = bundle/'probability.json'
    payload = json.loads(path.read_bytes())
    if value is None:
        payload.pop(field)
    else:
        payload[field] = value
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=message):
        start_solution.inspect_bundle(bundle)


def test_probability_v1_with_matching_encoder_does_not_require_history(tmp_path, monkeypatch):
    bundle, _, _ = setup_bundle(tmp_path, monkeypatch, with_scope=False)
    path = bundle/'probability.json'
    payload = json.loads(path.read_bytes())
    payload['schema_version'] = 'late-probability-logistic-v1'
    payload.pop('history_protocol')
    payload.pop('training_protocol')  # Runtime default remains vehicle_oof.
    path.write_text(json.dumps(payload))
    metadata = start_solution.inspect_bundle(bundle)
    assert metadata['probability_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize('payload', [[], None, 'invalid', {'unexpected':'schema'}])
def test_probability_preflight_rejects_non_artifact_json(tmp_path, monkeypatch, payload):
    bundle, _, _ = setup_bundle(tmp_path, monkeypatch, with_scope=False)
    (bundle/'probability.json').write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='схема'):
        start_solution.inspect_bundle(bundle)


@pytest.mark.parametrize('field,value',[
    ('regression_model_sha256','0'*64),('probability_artifact_sha256','0'*64),
    ('detector_sha256','0'*64),('detector_version','old-v1'),
    ('validation_status','rejected_transfer'),
])
def test_local_profile_with_wrong_model_calibrator_or_detector_is_rejected(tmp_path,monkeypatch,field,value):
    bundle,path,scope = setup_bundle(tmp_path,monkeypatch)
    scope[field] = value
    path.write_text(json.dumps(scope))
    with pytest.raises(ValueError):
        start_solution.inspect_bundle(bundle)


@pytest.mark.parametrize('problem',['absent','old_scope_hash','old_detector','old_version','old_model','old_probability'])
def test_no_build_cannot_silently_use_old_or_mismatched_runtime_scope(tmp_path,monkeypatch,problem):
    bundle,_,scope = setup_bundle(tmp_path,monkeypatch)
    metadata = start_solution.inspect_bundle(bundle)
    card = runtime_card(metadata,deepcopy(scope))
    if problem == 'absent':
        card['probability'].pop('gps_scope')
        card['probability'].pop('gps_scope_sha256')
    elif problem == 'old_scope_hash':
        card['probability']['gps_scope_sha256'] = '0'*64
    else:
        field = {'old_detector':'detector_sha256','old_version':'detector_version',
                 'old_model':'regression_model_sha256','old_probability':'probability_artifact_sha256'}[problem]
        card['probability']['gps_scope'][field] = 'old'
    with pytest.raises(ValueError,match='без --no-build'):
        start_solution.verify_model_card(card,metadata)


def test_runtime_weights_must_match_even_when_probability_metadata_claims_match(tmp_path,monkeypatch):
    bundle,_,scope = setup_bundle(tmp_path,monkeypatch)
    metadata = start_solution.inspect_bundle(bundle)
    card = runtime_card(metadata,scope)
    card['provenance']['sha256'] = '0'*64
    with pytest.raises(ValueError,match='веса'):
        start_solution.verify_model_card(card,metadata)
