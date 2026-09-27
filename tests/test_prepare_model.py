"""Миграция runtime-весов сохраняет старый bundle и не переносит его калибратор."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

from tools import prepare_model


def setup(tmp_path, monkeypatch):
    source, out = tmp_path/'source', tmp_path/'model'
    source.mkdir(); out.mkdir()
    model, encoder = b'new verified HGBR', b'new verified Transformer'
    (source/'hybrid_model.joblib').write_bytes(model)
    (source/'swiss_encoder.pt').write_bytes(encoder)
    monkeypatch.setattr(prepare_model, 'MODEL_SHA256', hashlib.sha256(model).hexdigest())
    monkeypatch.setattr(prepare_model, 'ENCODER_SHA256', hashlib.sha256(encoder).hexdigest())
    return source, out, model, encoder


def snapshot(path):
    return {str(p.relative_to(path)): p.read_bytes() for p in path.rglob('*') if p.is_file()}


def probability_payload(model, encoder, **changes):
    """Синтетический контракт теста, без утверждений о качестве реальной модели."""
    payload = {
        'schema_version': 'late-probability-logistic-v2',
        'regression_model_sha256': hashlib.sha256(model).hexdigest(),
        'encoder_sha256': hashlib.sha256(encoder).hexdigest(),
        'training_protocol': 'vehicle_oof',
        'history_protocol': 'provided_current_delay_strict_past_90m_12steps',
        'contract_sha256': 'a'*64, 'sources_sha256': {'test_fixture': 'b'*64},
        'event': 'target_delay_s > 120', 'feature_offset_s': 120.0,
        'feature_scale_s': 120.0, 'coefficient': 1.23, 'intercept': -0.5,
        'train_rows': 4434, 'test_rows': 353, 'train_constant': 0.5,
        'test_brier': 0.2, 'test_constant_brier': 0.25,
        'validation_status': 'accepted_secondary_test',
        'scope': 'real_data_with_provided_current_deviation',
    }
    payload.update(changes)
    return json.dumps(payload).encode()


def test_migration_preserves_entire_old_bundle_and_archives_old_probability(tmp_path, monkeypatch):
    source, out, model, encoder = setup(tmp_path, monkeypatch)
    old = {'model.joblib': b'old HGBR', 'encoder.pt': b'old Transformer',
           'manifest.json': b'{"previous": true}',
           'probability.json': b'{"regression_model_sha256": "old hash"}'}
    for name, content in old.items():
        (out/name).write_bytes(content)
    result = prepare_model.prepare_bundle(source, out)
    assert result['stale_probability_archived'] is True
    assert snapshot(Path(result['previous_bundle_backup'])) == old
    assert (out/'model.joblib').read_bytes() == model
    assert (out/'encoder.pt').read_bytes() == encoder
    assert not (out/'probability.json').exists()
    manifest = json.loads((out/'manifest.json').read_text())
    assert manifest['sha256'] == hashlib.sha256(model).hexdigest()
    assert manifest['encoder_sha256'] == hashlib.sha256(encoder).hexdigest()
    before = snapshot(out)
    repeated = prepare_model.prepare_bundle(source, out)
    assert repeated['previous_bundle_backup'] is None
    assert snapshot(out) == before


@pytest.mark.parametrize('bad_source', ['hybrid_model.joblib', 'swiss_encoder.pt'])
def test_bad_source_leaves_existing_bundle_untouched(tmp_path, monkeypatch, bad_source):
    source, out, _, _ = setup(tmp_path, monkeypatch)
    (out/'model.joblib').write_bytes(b'valuable existing weights')
    (out/'probability.json').write_bytes(b'valuable existing calibration')
    (source/bad_source).write_bytes(b'corrupted source')
    before = snapshot(out)
    with pytest.raises(SystemExit, match='неподтверждённый hash'):
        prepare_model.prepare_bundle(source, out)
    assert snapshot(out) == before
    assert not (out/'backups').exists()


def test_compatible_current_probability_is_retained(tmp_path, monkeypatch):
    source, out, model, encoder = setup(tmp_path, monkeypatch)
    (out/'model.joblib').write_bytes(model)
    (out/'encoder.pt').write_bytes(encoder)
    probability = probability_payload(model, encoder)
    (out/'probability.json').write_bytes(probability)
    # Соседний файл источника не означает запрос установить другие коэффициенты.
    (source/'probability.json').write_bytes(probability_payload(model, encoder, coefficient=9.0))
    result = prepare_model.prepare_bundle(source, out)
    assert (out/'probability.json').read_bytes() == probability
    assert result['stale_probability_archived'] is False
    assert result['previous_bundle_backup'] is None
    assert result['environment']['PROBABILITY_PATH'] == str((out/'probability.json').resolve())


def test_malformed_probability_is_preserved_outside_active_bundle(tmp_path, monkeypatch):
    source, out, model, encoder = setup(tmp_path, monkeypatch)
    (out/'model.joblib').write_bytes(model)
    (out/'encoder.pt').write_bytes(encoder)
    (out/'probability.json').write_bytes(b'not JSON')
    result = prepare_model.prepare_bundle(source, out)
    assert not (out/'probability.json').exists()
    assert (Path(result['previous_bundle_backup'])/'probability.json').read_bytes() == b'not JSON'


@pytest.mark.parametrize('encoder_hash', ['c'*64, None])
def test_current_probability_requires_exact_encoder(tmp_path, monkeypatch, encoder_hash):
    source, out, model, encoder = setup(tmp_path, monkeypatch)
    (out/'model.joblib').write_bytes(model)
    (out/'encoder.pt').write_bytes(encoder)
    previous = probability_payload(model, encoder, encoder_sha256=encoder_hash)
    (out/'probability.json').write_bytes(previous)
    result = prepare_model.prepare_bundle(source, out)
    assert result['stale_probability_archived'] is True
    assert not (out/'probability.json').exists()
    assert (Path(result['previous_bundle_backup'])/'probability.json').read_bytes() == previous
    assert 'PROBABILITY_PATH' not in result['environment']


def test_explicit_probability_copies_verified_bytes_and_reports_environment(tmp_path, monkeypatch):
    source, out, model, encoder = setup(tmp_path, monkeypatch)
    path = tmp_path/'accepted.json'
    content = probability_payload(model, encoder)
    path.write_bytes(content)
    result = prepare_model.prepare_bundle(source, out, path)
    assert (out/'probability.json').read_bytes() == content == path.read_bytes()
    assert result['probability_sha256'] == hashlib.sha256(content).hexdigest()
    assert result['probability'] == str((out/'probability.json').resolve())
    assert result['environment']['PROBABILITY_PATH'] == result['probability']
    assert json.loads((out/'manifest.json').read_text())['probability_sha256'] == result['probability_sha256']
    before = snapshot(out)
    repeated = prepare_model.prepare_bundle(source, out, probability_path=path)
    assert repeated['previous_bundle_backup'] is None
    assert snapshot(out) == before


def test_probability_replacement_preserves_entire_bundle(tmp_path, monkeypatch):
    source, out, model, encoder = setup(tmp_path, monkeypatch)
    path = tmp_path/'accepted.json'
    path.write_bytes(probability_payload(model, encoder))
    prepare_model.prepare_bundle(source, out, path)
    old = snapshot(out)
    path.write_bytes(probability_payload(model, encoder, coefficient=1.5))
    result = prepare_model.prepare_bundle(source, out, probability_path=path)
    assert snapshot(Path(result['previous_bundle_backup'])) == old
    assert (out/'probability.json').read_bytes() == path.read_bytes()
    assert result['stale_probability_archived'] is False


@pytest.mark.parametrize('changes', [
    {'regression_model_sha256': 'c'*64}, {'encoder_sha256': 'c'*64},
    {'schema_version': 'unknown'}, {'test_brier': 0.3},
])
def test_invalid_explicit_probability_does_not_mutate_bundle(tmp_path, monkeypatch, changes):
    source, out, model, encoder = setup(tmp_path, monkeypatch)
    (out/'model.joblib').write_bytes(b'valuable previous model')
    (out/'probability.json').write_bytes(b'valuable previous calibration')
    path = tmp_path/'rejected.json'
    path.write_bytes(probability_payload(model, encoder, **changes))
    before = snapshot(out)
    with pytest.raises(SystemExit, match='калибратор не прошёл проверку'):
        prepare_model.prepare_bundle(source, out, path)
    assert snapshot(out) == before
    assert not (out/'backups').exists()


@pytest.mark.parametrize('content', [b'not JSON', None])
def test_unreadable_explicit_probability_does_not_create_output(tmp_path, monkeypatch, content):
    source, out, _, _ = setup(tmp_path, monkeypatch)
    out.rmdir()
    path = tmp_path/'rejected.json'
    if content is not None:
        path.write_bytes(content)
    with pytest.raises(SystemExit, match='калибратор не прошёл проверку'):
        prepare_model.prepare_bundle(source, out, path)
    assert not out.exists()


def test_probability_changed_during_validation_does_not_mutate_bundle(tmp_path, monkeypatch):
    source, out, model, encoder = setup(tmp_path, monkeypatch)
    path = tmp_path/'accepted.json'
    path.write_bytes(probability_payload(model, encoder))
    loader = prepare_model.ProbabilityModel.load

    def changed_loader(*args, **kwargs):
        path.write_bytes(probability_payload(model, encoder, coefficient=1.5))
        return loader(*args, **kwargs)

    monkeypatch.setattr(prepare_model.ProbabilityModel, 'load', changed_loader)
    before = snapshot(out)
    with pytest.raises(SystemExit, match='изменился во время проверки'):
        prepare_model.prepare_bundle(source, out, path)
    assert snapshot(out) == before


def test_default_does_not_discover_probability_in_source(tmp_path, monkeypatch):
    source, out, model, encoder = setup(tmp_path, monkeypatch)
    (source/'probability.json').write_bytes(probability_payload(model, encoder))
    result = prepare_model.prepare_bundle(source, out)
    assert not (out/'probability.json').exists()
    assert result['probability'] is None
    assert 'PROBABILITY_PATH' not in result['environment']


def test_cli_forwards_explicit_probability(tmp_path, monkeypatch, capsys):
    source, out, model, encoder = setup(tmp_path, monkeypatch)
    path = tmp_path/'accepted.json'
    path.write_bytes(probability_payload(model, encoder))
    monkeypatch.setattr(sys, 'argv', ['prepare_model.py', '--source', str(source),
                                    '--out', str(out), '--probability', str(path)])
    prepare_model.main()
    result = json.loads(capsys.readouterr().out)
    assert result['environment']['PROBABILITY_PATH'] == str((out/'probability.json').resolve())
    assert (out/'probability.json').read_bytes() == path.read_bytes()
