"""Единые границы цветов от ответа ML до счётчика, маршрута и журнала."""
import asyncio
from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace

import httpx
import pytest

from backend.engine import Engine, LiveContext
from common.contracts import PredictionRequest, StopTarget, Telemetry, baseline
from common.risk import RISK_POLICY, risk_for_delay
from ml_service.learned import LearnedModel


T = datetime(2026, 1, 6, 12, tzinfo=timezone.utc)
TARGET = StopTarget(id="next", name="Следующая", scheduled_at=T+timedelta(minutes=12),
                    lat=55.75, lon=37.61)
BOUNDARIES = [
    (-90, "green"), (0, "green"), (59.999, "green"),
    (60, "amber"), (120, "amber"), (130, "amber"), (150, "amber"),
    (150.001, "red"), (240, "red"),
]


def request(delay_s, age_s=0):
    return PredictionRequest(request_id="boundary", tr_id=101, issued_at=T,
        target=TARGET, current_delay_s=delay_s, features={"telemetry_age_s": age_s})


@pytest.mark.parametrize("delay_s,expected", BOUNDARIES)
def test_policy_baseline_and_learned_response_have_ui_boundaries(delay_s, expected):
    """Проверяем реальный сборщик ответа learned без загрузки весов регрессии."""
    model = SimpleNamespace(probability=None, probability_error=None, model_version="test")
    prediction = LearnedModel.response(model, request(delay_s), delay_s, None)
    assert risk_for_delay(delay_s) == expected
    assert baseline(request(delay_s)).risk == expected
    assert baseline(request(delay_s), fallback=True).risk == expected
    assert prediction.risk == expected
    assert prediction.predicted_delay_s == delay_s


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), float("-inf")])
def test_policy_does_not_color_unknown_delay_as_zero(value):
    assert risk_for_delay(value) == "unknown"


@pytest.mark.parametrize("age_s", [None, 60.001])
def test_both_ml_paths_preserve_stale_number_without_current_risk(age_s):
    model = SimpleNamespace(probability=None, probability_error=None, model_version="test")
    payload = request(130, age_s)
    for prediction in (baseline(payload), LearnedModel.response(model, payload, 130, None)):
        assert prediction.risk == "unknown"
        assert prediction.predicted_delay_s == 130
        assert prediction.probability_late is None


@pytest.mark.parametrize("delay_s,expected", BOUNDARIES)
def test_engine_tick_agrees_with_ml_summary_and_incident_boundary(monkeypatch, delay_s, expected):
    asyncio.run(_engine_case(monkeypatch, delay_s, expected))


@pytest.mark.parametrize("delay_s,ml_risk,expected", [
    (60, "green", "amber"), (130, "red", "amber"),
    (150, "red", "amber"), (150.001, "green", "red"),
    (130, "unknown", "unknown"), (None, "unknown", "unknown"),
])
def test_engine_normalizes_legacy_ml_colors_but_respects_unknown(monkeypatch, delay_s, ml_risk, expected):
    asyncio.run(_engine_case(monkeypatch, delay_s, expected, ml_risk=ml_risk))


async def _engine_case(monkeypatch, delay_s, expected, ml_risk=None):
    monkeypatch.setattr("backend.engine.utcnow", lambda: T)

    def handler(http_request):
        assert http_request.url.path == "/predict/batch"
        results = []
        for row in json.loads(http_request.content):
            result = baseline(PredictionRequest.model_validate(row)).model_dump(mode="json")
            if ml_risk is not None:
                result["risk"] = ml_risk
            results.append(result)
        return httpx.Response(200, json=results)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        engine = Engine("http://ml", client)
        engine.set_live(LiveContext(
            vehicles=[dict(tr_id=101, unit_id=101, label="Автобус", route_id="line")],
            schedule=[
                dict(tr_id=101, target=TARGET.model_copy(update={"id": "previous",
                     "scheduled_at": T-timedelta(minutes=1)})),
                dict(tr_id=101, target=TARGET),
            ],
            hints=[] if delay_s is None else [dict(tr_id=101, observed_at=T, delay_s=delay_s)],
        ))
        engine.ingest(Telemetry(tr_id=101, event_time=T, received_at=T,
                               lat=55.75, lon=37.60, speed_kmh=20))
        await engine.tick(0)
        state = engine.state()
        assert state["health"]["ml"] == "ok"
        assert state["risk_policy"] == RISK_POLICY.model_dump(mode="json")
        prediction = state["vehicles"][0]["prediction"]
        assert prediction["risk"] == expected
        assert prediction["predicted_delay_s"] == delay_s
        for summary in [state["summary"], state["route_risks"][0], state["section_risks"][0]]:
            assert summary["red"] == int(expected == "red")
            assert summary["amber"] == int(expected == "amber")
            assert summary["unknown"] == int(expected == "unknown")
        assert state["route_risks"][0]["risk"] == expected
        assert state["section_risks"][0]["risk"] == expected
        assert len(state["incidents"]) == int(expected == "red")
        if expected == "red":
            assert state["incidents"][0]["risk"] == "red"
            assert state["incidents"][0]["predicted_delay_s"] == delay_s
        # Второй расчёт того же посещения не дублирует предупреждение.
        await engine.tick(0)
        assert len(engine.incidents) == int(expected == "red")
