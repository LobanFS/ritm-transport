"""Проверки контракта ML-сервиса: честный baseline, время, неизвестность и ошибки."""

from datetime import datetime, timedelta, timezone
import json

from fastapi.testclient import TestClient
import pytest

from ml_service.app import app


@pytest.fixture
def client():
    """Предоставить изолированный HTTP-клиент приложения без отдельного сервера."""
    with TestClient(app) as test_client:
        yield test_client


def prediction_request(*, horizon_s=750, delay_s=130.0, age_s=10.0):
    """Собрать запрос с явно заданным временем, горизонтом и состоянием подсказки."""
    issued_at = datetime(2026, 1, 6, 9, 0, tzinfo=timezone.utc)
    return {
        "request_id": "sample-1",
        "tr_id": 123,
        "issued_at": issued_at.isoformat(),
        "target": {
            "id": "arrival-1",
            "name": "Тестовая остановка",
            "scheduled_at": (issued_at + timedelta(seconds=horizon_s)).isoformat(),
            "lat": 55.75,
            "lon": 37.61,
        },
        "current_delay_s": delay_s,
        "features": {
            "telemetry_age_s": age_s,
            "valid_points_5m": 0 if age_s is None else 3,
        },
    }


def test_health_is_explicit_about_untrained_model(client):
    """Проверить полную карточку работоспособности, включая факт отсутствия обучения."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok", "model_version": "persistence-v1", "trained": False
    }


def test_model_card_describes_contract_and_limitations(client):
    """Проверить, что карточка не обещает обученную модель или вероятности."""
    response = client.get("/model")
    assert response.status_code == 200
    card = response.json()
    assert card["model_version"] == client.get("/health").json()["model_version"]
    assert card["trained"] is False
    assert card["units"] == "seconds"
    assert card["prediction_horizon_seconds"] == {
        "lower_exclusive": 600, "upper_inclusive": 900
    }
    assert card["output"]["probability_late"] is None
    assert card["limitations"]


def test_negative_delay_preserves_early_arrival(client):
    """Отрицательная задержка не обрезается до нуля и не выдаётся за вероятность."""
    response = client.post("/predict", json=prediction_request(delay_s=-90.5))
    assert response.status_code == 200
    result = response.json()
    assert result["predicted_delay_s"] == -90.5
    assert result["method"] == "persistence"
    assert result["model_version"] == "persistence-v1"
    assert result["probability_late"] is None
    assert result["risk"] == "green"


def test_missing_hint_is_unavailable_not_zero(client):
    """Отсутствующая подсказка сохраняет неизвестность результата."""
    response = client.post("/predict", json=prediction_request(delay_s=None))
    assert response.status_code == 200
    result = response.json()
    assert result["predicted_delay_s"] is None
    assert result["method"] == "unavailable"
    assert result["risk"] == "unknown"
    assert result["probability_late"] is None


@pytest.mark.parametrize("horizon_s,status", [(600, 422), (600.001, 200), (900, 200), (900.001, 422)])
def test_prediction_horizon_has_correct_open_and_closed_boundaries(client, horizon_s, status):
    """Окно цели строго открыто слева и закрыто справа: (600, 900] секунд."""
    response = client.post("/predict", json=prediction_request(horizon_s=horizon_s))
    assert response.status_code == status


@pytest.mark.parametrize("field", ["issued_at", "scheduled_at"])
def test_naive_timestamp_is_rejected(client, field):
    """Ни момент решения, ни плановое прибытие не могут иметь неизвестный часовой пояс."""
    payload = prediction_request()
    target = payload["target"] if field == "scheduled_at" else payload
    target[field] = target[field].replace("+00:00", "")
    assert client.post("/predict", json=payload).status_code == 422


def test_horizon_is_computed_across_timezone_offsets(client):
    """Разные UTC offsets описывают одни часы и не меняют смысл горизонта."""
    payload = prediction_request(horizon_s=900)
    payload["issued_at"] = "2026-01-06T12:00:00+03:00"
    assert client.post("/predict", json=payload).status_code == 200


@pytest.mark.parametrize("scope", ["request", "target", "features"])
def test_extra_fields_are_rejected_at_every_input_level(client, scope):
    """Неизвестные поля отклоняются, в том числе внутри признаков и остановки."""
    payload = prediction_request()
    target = payload if scope == "request" else payload[scope]
    target["future_actual_arrival"] = "2026-01-06T09:15:00Z"
    assert client.post("/predict", json=payload).status_code == 422


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_delay_is_a_serializable_validation_error(client, value):
    """Неконечное число даёт 422, а не 500 при сериализации текста ошибки."""
    payload = prediction_request(delay_s=value)
    response = client.post(
        "/predict", content=json.dumps(payload), headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    assert response.json()["detail"]
    assert "current_delay_s" in str(response.json()["detail"])


@pytest.mark.parametrize("field", ["telemetry_age_s", "distance_to_target_m"])
def test_unbounded_feature_still_requires_a_finite_number(client, field):
    """Неотрицательность признака не должна допускать положительную бесконечность."""
    payload = prediction_request()
    payload["features"][field] = float("inf")
    response = client.post(
        "/predict", content=json.dumps(payload), headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    assert field in str(response.json()["detail"])


@pytest.mark.parametrize("age_s,risk", [(60, "red"), (60.001, "unknown"), (None, "unknown")])
def test_missing_or_stale_gps_changes_risk_without_fabricating_delay(client, age_s, risk):
    """Свежесть GPS ограничивает цвет риска; численный fallback остаётся явной подсказкой."""
    response = client.post("/predict", json=prediction_request(delay_s=130, age_s=age_s))
    assert response.status_code == 200
    result = response.json()
    assert result["risk"] == risk
    assert result["predicted_delay_s"] == 130
    assert result["probability_late"] is None


def test_batch_preserves_order_and_missing_values(client):
    """Пакет сохраняет соответствие ID и результата, включая неизвестную подсказку."""
    first = prediction_request(delay_s=-20)
    second = prediction_request(delay_s=None)
    second["request_id"] = "sample-2"
    response = client.post("/predict/batch", json=[first, second])
    assert response.status_code == 200
    assert [(row["request_id"], row["predicted_delay_s"]) for row in response.json()] == [
        ("sample-1", -20), ("sample-2", None)
    ]


@pytest.mark.parametrize("size,status", [(0, 200), (128, 200), (129, 422)])
def test_batch_size_limit(client, size, status):
    """Проверить ограничение входного пакета до 128 включительно."""
    response = client.post("/predict/batch", json=[prediction_request()] * size)
    assert response.status_code == status
    if status == 200:
        assert len(response.json()) == size


def test_batch_rejects_invalid_horizon_without_partial_results(client):
    """Невалидная цель в одной строке отклоняет весь пакет."""
    response = client.post(
        "/predict/batch", json=[prediction_request(), prediction_request(horizon_s=600)]
    )
    assert response.status_code == 422


def test_openapi_exposes_separate_prediction_endpoints(client):
    """Проверить доступность схемы API для отдельного и пакетного вызова."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "post" in paths["/predict"]
    assert "post" in paths["/predict/batch"]
    assert paths["/predict/batch"]["post"]["requestBody"]["content"]["application/json"]["schema"]["maxItems"] == 128
