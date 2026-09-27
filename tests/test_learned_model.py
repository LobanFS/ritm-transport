"""Границы frozen ML интеграции; полный паритет 504 строк — tools/check_model.py."""
from datetime import datetime, timedelta, timezone
import importlib
import math
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from common.contracts import PredictionRequest
from ml_service.learned import ENCODER_SHA256, LearnedModel, MODEL_SHA256, MODEL_VERSION


def payload():
    now = datetime(2026, 1, 6, 9, tzinfo=timezone.utc)
    stops = [{
        "id": str(index), "name": f"Остановка {index}",
        "scheduled_at": (now + timedelta(seconds=offset)).isoformat(),
        "lat": 55.75, "lon": 37.61 + index * .001, "manual_fill": index == 2,
    } for index, offset in enumerate((-600, 750, 1800))]
    return {"request_id": "sample-1", "tr_id": 123, "issued_at": now.isoformat(),
            "target": dict(stops[1]), "current_delay_s": 130,
            "features": {"telemetry_age_s": 10},
            "plan_context": {"version": "plan-1", "timezone": "UTC", "complete": True, "stops": stops}}


@pytest.fixture(scope="module")
def learned():
    path = Path(__file__).resolve().parents[1] / "artifacts/model/model.joblib"
    if not path.exists():
        pytest.skip("Для проверки frozen weights выполните python tools/prepare_model.py")
    return LearnedModel(path)


@pytest.fixture
def client(learned, monkeypatch):
    module = importlib.import_module("ml_service.app")
    monkeypatch.setattr(module, "runtime", lambda: (learned, None))
    with TestClient(module.app) as result:
        yield result


def test_hash_is_checked_before_deserializing(tmp_path):
    artifact = tmp_path / "untrusted.joblib"
    artifact.write_bytes(b"not a trusted pickle")
    with pytest.raises(ValueError, match="SHA-256"):
        LearnedModel(artifact)


def test_health_and_provenance_describe_actual_loaded_model(client, learned):
    assert client.get('/health').json() == {"status": "ok", "model_version": MODEL_VERSION, "trained": True}
    card = client.get('/model').json()
    assert card['provenance']['sha256'] == MODEL_SHA256
    assert len(card['features']) == 14
    assert card['features'][-1] == 'neural_prior'
    assert card['provenance']['encoder_sha256'] == learned.encoder_sha256
    assert card['output']['probability_late'] is None


def test_full_plan_frozen_features_use_past_state_and_future_plan(learned):
    features, reason = learned.features(PredictionRequest.model_validate(payload()))
    assert reason is None
    assert features.iloc[0].route_len == 3
    assert features.iloc[0].state_rel_pos == 0
    assert features.iloc[0].plan_manual_target == 0
    assert features.iloc[0].plan_manual_rate == pytest.approx(1/3)
    assert features.iloc[0].plan_interval_s == 1350


def test_plan_order_and_timezone_offset_do_not_change_features(learned):
    original = PredictionRequest.model_validate(payload())
    changed = payload()
    changed['plan_context']['stops'].reverse()
    changed['issued_at'] = '2026-01-06T12:00:00+03:00'
    a, _ = learned.features(original)
    b, _ = learned.features(PredictionRequest.model_validate(changed))
    assert a.equals(b)


@pytest.mark.parametrize('problem', ['missing_plan', 'partial_plan', 'missing_manual'])
def test_incomplete_inputs_use_explicit_fallback(client, problem):
    request = payload()
    if problem == 'missing_plan':
        request.pop('plan_context')
    elif problem == 'partial_plan':
        request['plan_context']['complete'] = False
    else:
        request['plan_context']['stops'][0]['manual_fill'] = None
    response = client.post('/predict', json=request)
    assert response.status_code == 200
    result = response.json()
    assert result['method'] == 'fallback'
    assert result['model_version'] == 'persistence-v1'
    assert result['predicted_delay_s'] == 130
    assert result['fallback_reason']


def test_unknown_deviation_remains_unknown_with_complete_plan(client):
    request = payload()
    request['current_delay_s'] = None
    result = client.post('/predict', json=request).json()
    assert result['method'] == 'unavailable'
    assert result['predicted_delay_s'] is None
    assert result['risk'] == 'unknown'


def test_stale_gps_keeps_numeric_model_prediction_but_unknown_risk(client):
    request = payload()
    fresh = client.post('/predict', json=request).json()
    request['features']['telemetry_age_s'] = 61
    stale = client.post('/predict', json=request).json()
    assert fresh['method'] == stale['method'] == 'learned'
    assert fresh['predicted_delay_s'] == stale['predicted_delay_s']
    assert stale['risk'] == 'unknown'
    assert stale['probability_late'] is None


def test_learned_prediction_has_additive_noncausal_explanation(client):
    result = client.post('/predict', json=payload()).json()
    explanation = result['forecast_explanation']
    assert explanation['method'] == 'exact_grouped_shapley_reference_v1'
    assert explanation['prediction_s'] == pytest.approx(result['predicted_delay_s'])
    assert explanation['model_adjustment_s'] == pytest.approx(
        result['predicted_delay_s'] - payload()['current_delay_s']
    )
    assert abs(explanation['reconstruction_error_s']) < 1e-8
    assert {factor['code'] for factor in explanation['factors']} == {
        'current_state', 'forecast_horizon', 'route_position',
        'remaining_path', 'plan_context', 'sequence_history',
    }
    assert explanation['history_points'] == 1
    assert explanation['history_effect_on_neural_prior_s'] == pytest.approx(0)
    assert explanation['transformer_analysis']['influential_history'] == []
    assert explanation['transformer_analysis']['summary'] == (
        'Недостаточно истории: прогноз рассчитан по текущему отклонению и плану.'
    )
    assert 'не являются доказанными причинами' in explanation['interpretation']


def test_past_delay_history_changes_transformer_prior_and_is_reported(client):
    request = payload()
    issued = datetime.fromisoformat(request['issued_at'])
    request['delay_history'] = [
        {'observed_at': (issued-timedelta(minutes=10)).isoformat(), 'delay_s': 20},
        {'observed_at': (issued-timedelta(minutes=5)).isoformat(), 'delay_s': 70},
    ]
    result = client.post('/predict', json=request).json()
    explanation = result['forecast_explanation']
    assert result['method'] == 'learned'
    assert explanation['history_points'] == 3
    assert explanation['history_span_s'] == 600
    assert abs(explanation['history_effect_on_neural_prior_s']) > 1e-6
    analysis = explanation['transformer_analysis']
    assert analysis['method'] == 'leave_one_history_observation_out_v1'
    assert analysis['delay_trend_s'] == 110
    assert analysis['recent_change_s'] == 60
    assert analysis['step_volatility_s'] == pytest.approx(55)
    assert len(analysis['influential_history']) == 2
    assert 'Прошлая динамика усиливает прогноз задержки' in analysis['summary']
    assert 'neural prior' not in analysis['summary']
    assert all(item['age_s'] > 0 for item in analysis['influential_history'])
    strongest = analysis['influential_history'][0]
    without = payload()
    strongest_at = datetime.fromisoformat(strongest['observed_at'].replace('Z', '+00:00'))
    without['delay_history'] = [item for item in request['delay_history']
                                if datetime.fromisoformat(item['observed_at']) != strongest_at]
    without_result = client.post('/predict', json=without).json()
    assert strongest['prior_effect_s'] == pytest.approx(
        explanation['neural_prior_s']-
        without_result['forecast_explanation']['neural_prior_s'], abs=1e-5)


@pytest.mark.parametrize('offset', [0, 1])
def test_delay_history_rejects_current_or_future_observations(client, offset):
    request = payload()
    issued = datetime.fromisoformat(request['issued_at'])
    request['delay_history'] = [{'observed_at': (issued+timedelta(seconds=offset)).isoformat(), 'delay_s': 20}]
    assert client.post('/predict', json=request).status_code == 422


def test_fallback_does_not_invent_model_explanation(client):
    request = payload()
    request.pop('plan_context')
    result = client.post('/predict', json=request).json()
    assert result['method'] == 'fallback'
    assert 'forecast_explanation' not in result


@pytest.mark.parametrize('location', ['request', 'target', 'plan'])
def test_future_facts_cannot_enter_feature_builder(client, location):
    request = payload()
    obj = request if location == 'request' else request['target'] if location == 'target' else request['plan_context']['stops'][0]
    obj['time_fact_begin'] = '2026-01-06T09:20:00+00:00'
    assert client.post('/predict', json=request).status_code == 422


def test_target_and_versioned_plan_must_agree(client):
    request = payload()
    request['plan_context']['stops'][1]['scheduled_at'] = '2026-01-06T09:14:00+00:00'
    assert client.post('/predict', json=request).status_code == 422


def test_batch_mixed_fallback_order_and_prediction_parity(client):
    first, second, third = payload(), payload(), payload()
    second['request_id'] = 'sample-2'
    second.pop('plan_context')
    third['request_id'] = 'sample-3'
    third['current_delay_s'] = None
    response = client.post('/predict/batch', json=[first, second, third])
    assert response.status_code == 200
    rows = response.json()
    assert [row['request_id'] for row in rows] == ['sample-1', 'sample-2', 'sample-3']
    assert [row['method'] for row in rows] == ['learned', 'fallback', 'unavailable']
    assert rows[0]['predicted_delay_s'] == client.post('/predict', json=first).json()['predicted_delay_s']


def test_failed_model_load_does_not_claim_learned_prediction(monkeypatch):
    module = importlib.import_module("ml_service.app")
    monkeypatch.setattr(module, 'runtime', lambda: (None, 'SHA-256 mismatch'))
    with TestClient(module.app) as client:
        assert client.get('/health').json()['status'] == 'degraded'
        result = client.post('/predict', json=payload()).json()
        assert result['method'] == 'fallback'
        assert result['predicted_delay_s'] == 130
        assert result['fallback_reason']


def probability_fixture(tmp_path, *, model_hash=MODEL_SHA256):
    """Искусственные коэффициенты из test_probability, не измеренное качество."""
    from ml_service.probability import ProbabilityArtifact
    artifact = ProbabilityArtifact(
        schema_version='late-probability-logistic-v2', regression_model_sha256=model_hash,
        encoder_sha256=ENCODER_SHA256,history_protocol='provided_current_delay_strict_past_90m_12steps',
        contract_sha256='b'*64, sources_sha256={'train': 'c'*64}, event='target_delay_s > 120',
        feature_offset_s=120., feature_scale_s=120., coefficient=1., intercept=0.,
        train_rows=4434, test_rows=353, train_constant=.225, test_brier=.1, test_constant_brier=.2,
        validation_status='accepted_secondary_test', scope='real_data_with_provided_current_deviation',
    )
    path = tmp_path/'fixture-probability.json'
    path.write_text(artifact.model_dump_json())
    return path


@pytest.fixture
def calibrated_client(learned, tmp_path, monkeypatch):
    """Проверить настоящее подключение через env и LearnedModel.__init__."""
    monkeypatch.setenv('PROBABILITY_PATH', str(probability_fixture(tmp_path)))
    path = Path(__file__).resolve().parents[1]/'artifacts/model/model.joblib'
    calibrated = LearnedModel(path)
    assert calibrated.probability is not None and calibrated.probability_error is None
    module = importlib.import_module('ml_service.app')
    monkeypatch.setattr(module, 'runtime', lambda: (calibrated, None))
    with TestClient(module.app) as result:
        yield result


def test_calibrated_probability_accepts_only_provided_csv_without_changing_regression(calibrated_client, learned):
    request = payload()
    request['current_delay_source'] = 'csv_snapshot'
    expected = learned.predict(PredictionRequest.model_validate(request)).predicted_delay_s
    response = calibrated_client.post('/predict', json=request)
    assert response.status_code == 200
    result = response.json()
    assert result['method'] == 'learned'
    assert result['predicted_delay_s'] == expected
    assert result['probability_late'] == pytest.approx(1/(1+math.exp(-(expected-120)/120)))
    assert result['probability_status'] == 'validated'
    assert result['probability_note']


@pytest.mark.parametrize('source', ['arrival', 'external_hint', 'gps_estimate', 'door_estimate', 'demo_arrival',
                                  'generator_arrival', 'synthetic_hint', None])
def test_calibration_never_transfers_outside_provided_csv_source(calibrated_client, source):
    confirmed, unverified = payload(), payload()
    confirmed['current_delay_source'] = 'csv_snapshot'
    unverified['request_id'] = 'unverified'
    unverified['current_delay_source'] = source
    response = calibrated_client.post('/predict/batch', json=[confirmed, unverified])
    assert response.status_code == 200
    first, second = response.json()
    assert first['probability_late'] is not None
    assert second['probability_late'] is None
    if source == 'gps_estimate':
        assert 'ошибки распознавания' in second['probability_note']
    elif source == 'door_estimate':
        assert 'нет размеченной реальной истории' in second['probability_note']
    else:
        assert second['probability_note'] == 'Вероятность проверена для выданных CSV-подсказок; для этого источника требуется отдельная проверка'
    assert second['predicted_delay_s'] == first['predicted_delay_s']
    assert second['risk'] == first['risk']
    assert first['method'] == second['method'] == 'learned'


@pytest.mark.parametrize('age,available', [(60, True), (60.001, False), (None, False)])
def test_calibrated_probability_respects_freshness_without_changing_seconds(calibrated_client, age, available):
    request = payload()
    request['current_delay_source'] = 'csv_snapshot'
    fresh = calibrated_client.post('/predict', json=request).json()
    request['features']['telemetry_age_s'] = age
    result = calibrated_client.post('/predict', json=request).json()
    assert result['predicted_delay_s'] == fresh['predicted_delay_s']
    assert (result['probability_late'] is not None) is available
    if not available:
        assert result['risk'] == 'unknown' and result['probability_note']


@pytest.mark.parametrize('problem', ['missing_delay', 'partial_plan'])
def test_unavailable_and_fallback_do_not_receive_calibrated_probability(calibrated_client, problem):
    request = payload()
    request['current_delay_source'] = 'csv_snapshot'
    if problem == 'missing_delay':
        request['current_delay_s'] = None
    else:
        request['plan_context']['complete'] = False
    result = calibrated_client.post('/predict', json=request).json()
    assert result['method'] == ('unavailable' if problem == 'missing_delay' else 'fallback')
    assert result['predicted_delay_s'] == (None if problem == 'missing_delay' else 130)
    assert result['probability_late'] is None


def test_bad_calibration_preserves_loaded_regression_and_marks_probability_unavailable(learned, tmp_path, monkeypatch):
    monkeypatch.setenv('PROBABILITY_PATH', str(probability_fixture(tmp_path, model_hash='d'*64)))
    path = Path(__file__).resolve().parents[1]/'artifacts/model/model.joblib'
    rejected = LearnedModel(path)
    assert rejected.probability is None and rejected.probability_error
    request = payload()
    request['current_delay_source'] = 'csv_snapshot'
    validated = PredictionRequest.model_validate(request)
    result = rejected.predict(validated)
    assert result.method == 'learned'
    assert result.predicted_delay_s == learned.predict(validated).predicted_delay_s
    assert result.probability_late is None and 'проверку' in result.probability_note
    assert result.probability_status == 'unavailable'


def test_model_card_limits_probability_to_provided_csv_distribution(calibrated_client):
    card = calibrated_client.get('/model').json()
    assert card['probability']['allowed_current_delay_sources'] == ['csv_snapshot']
    assert card['probability']['source_validation']['gps_estimate'] == 'transfer_not_confirmed'
    assert card['probability']['source_validation']['door_estimate'] == 'no_labeled_real_history'
    assert 'current_delay_source=csv_snapshot' in card['output']['probability_late']
    assert any('операционное arrival' in line and 'external_hint' in line for line in card['limitations'])


def test_gps_scope_http_response_requires_complete_provenance_and_preserves_seconds(learned,tmp_path,monkeypatch):
    from ml_service.probability import ProbabilityModel
    from tests.test_probability import gps_scope_fixture
    path = probability_fixture(tmp_path)
    probability = ProbabilityModel.load(path,MODEL_SHA256)
    scope_path = gps_scope_fixture(tmp_path,probability)
    probability = ProbabilityModel.load(path,MODEL_SHA256,gps_scope_path=scope_path)
    monkeypatch.setattr(learned,'probability',probability)
    module = importlib.import_module('ml_service.app')
    monkeypatch.setattr(module,'runtime',lambda:(learned,None))
    with TestClient(module.app) as client:
        request = dict(payload(),current_delay_source='gps_estimate',telemetry_domain='historical_real',
                       current_delay_detector_version='fixture-v1',current_delay_detector_sha256='d'*64)
        accepted = client.post('/predict',json=request).json()
        assert accepted['probability_late'] is not None
        assert accepted['probability_status'] == 'validated'
        assert 'Вторичная проверка' in accepted['probability_note']
        assert client.get('/model').json()['probability']['gps_scope']
        mixed_request = {**request,'telemetry_domain':'historical_mixed'}
        transferred = client.post('/predict',json=mixed_request).json()
        assert transferred['probability_status'] == 'transferred'
        assert transferred['probability_late'] == accepted['probability_late']
        assert transferred['predicted_delay_s'] == accepted['predicted_delay_s']
        assert 'Приближённая оценка' in transferred['probability_note']
        assert 'качество здесь не проверено' in transferred['probability_note']
        for changes in ({'telemetry_domain':'synthetic'}, {'telemetry_domain':'live_unverified'},
                        {'current_delay_detector_sha256':'0'*64}, {'current_delay_detector_version':None},
                        {'current_delay_source':'door_estimate'}):
            rejected = client.post('/predict',json={**request,**changes}).json()
            assert rejected['probability_late'] is None
            assert rejected['probability_status'] == 'unavailable'
            assert rejected['predicted_delay_s'] == accepted['predicted_delay_s']
        stale = client.post('/predict',json={**request,'features':{'telemetry_age_s':61}}).json()
        assert stale['probability_late'] is None and stale['predicted_delay_s'] == accepted['predicted_delay_s']
        for changes in ({'features':{'telemetry_age_s':61}}, {'current_delay_s':None},
                        {'current_delay_detector_sha256':'0'*64}, {'current_delay_detector_version':None},
                        {'current_delay_source':'door_estimate'}):
            rejected = client.post('/predict',json={**mixed_request,**changes}).json()
            assert rejected['probability_late'] is None
            assert rejected['probability_status'] == 'unavailable'
