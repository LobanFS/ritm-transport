"""Production collection is causal, idempotent and isolated from dispatcher state."""
from datetime import datetime, timedelta, timezone
import asyncio
import json
from pathlib import Path

import httpx

from backend.arrivals import CurrentDeviation
from backend.engine import Engine, LiveContext
from backend.learning_store import LearningStore, sampling_horizon_s
from common.contracts import PredictionRequest, StopTarget, baseline
from tools.export_learning_data import export
from training.production_data import load_examples, one_snapshot_per_target


T = datetime(2026, 1, 6, 8, tzinfo=timezone.utc)


def request(request_id="r1", issued_at=T, target_at=None):
    target_at = target_at or T+timedelta(minutes=12)
    target = StopTarget(id="target", name="Target",
        scheduled_at=target_at, lat=55.7, lon=37.6)
    return PredictionRequest(request_id=request_id, tr_id=1, issued_at=issued_at,
        target=target, current_delay_s=30,
        features={"telemetry_age_s": 1})


def learned(req):
    return baseline(req).model_copy(update={"method": "learned", "model_version": "fixture-v1"})


def label(*, received_at=T+timedelta(minutes=13)):
    return CurrentDeviation(delay_s=60, source="arrival",
        observed_at=T+timedelta(minutes=13), received_at=received_at,
        planned_stop_id="target", stop_name="Target",
        planned_at=T+timedelta(minutes=12))


def test_store_is_idempotent_and_exports_only_pre_label_requests(tmp_path):
    store = LearningStore(tmp_path/"learning.sqlite3")
    target_at = T+timedelta(minutes=12)
    first = request(issued_at=target_at-timedelta(seconds=601), target_at=target_at)
    assert store.record_prediction(first, learned(first), published_at=first.issued_at+timedelta(seconds=1))
    assert not store.record_prediction(first, learned(first), published_at=first.issued_at+timedelta(seconds=1))
    assert store.record_arrival(1, label())
    assert not store.record_arrival(1, label())
    assert store.stats() == {"predictions": 1, "labels": 1, "examples": 1}

    result = export(store.path, tmp_path/"examples.jsonl")
    examples = load_examples(Path(result["output"]))
    assert len(examples) == 1
    assert examples[0].request.request_id == "r1"
    assert examples[0].label.target_delay_s == 60


def test_store_rejects_fallback_and_snapshot_selection_removes_frequency_bias(tmp_path):
    store = LearningStore(tmp_path/"learning.sqlite3")
    fallback_request = request("fallback")
    assert not store.record_prediction(fallback_request, baseline(fallback_request), published_at=T)
    threshold = sampling_horizon_s(1, "target")
    if threshold < 900:
        lead = threshold+1
        issued = request().target.scheduled_at-timedelta(seconds=lead)
        item = request(f"lead-{lead}", issued, request().target.scheduled_at)
        assert not store.record_prediction(item, learned(item), published_at=issued+timedelta(seconds=1))
    lead = max(601, threshold-1)
    issued = request().target.scheduled_at-timedelta(seconds=lead)
    item = request(f"lead-{lead}", issued, request().target.scheduled_at)
    assert store.record_prediction(item, learned(item), published_at=issued+timedelta(seconds=1))
    later = request("later", request().target.scheduled_at-timedelta(seconds=601),
                    request().target.scheduled_at)
    assert not store.record_prediction(later, learned(later), published_at=later.issued_at+timedelta(seconds=1))
    store.record_arrival(1, label())
    export(store.path, tmp_path/"examples.jsonl")
    selected = one_snapshot_per_target(load_examples(tmp_path/"examples.jsonl"))
    assert [item.request.request_id for item in selected] == [f"lead-{lead}"]


def test_engine_collects_only_live_learned_predictions_and_operational_arrivals(monkeypatch):
    import backend.engine as module
    monkeypatch.setattr(module, "utcnow", lambda: T)

    class Spy:
        def __init__(self): self.predictions=[]; self.arrivals=[]
        def record_prediction(self, req, prediction, *, published_at):
            self.predictions.append((req, prediction, published_at))
        def record_arrival(self, tr_id, deviation): self.arrivals.append((tr_id, deviation))

    async def exercise():
        async def handler(http_request):
            inputs = [PredictionRequest.model_validate(row) for row in json.loads(http_request.content)]
            return httpx.Response(200, json=[learned(item).model_dump(mode="json") for item in inputs])
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            spy = Spy()
            engine = Engine("http://ml", client, learning_store=spy)
            engine.set_live(LiveContext(
                vehicles=[dict(tr_id=1, unit_id=2, label="Bus", route_id="r")],
                schedule=[dict(tr_id=1, target=request().target), dict(tr_id=1, target={
                    "id":"other", "name":"Other", "scheduled_at":T+timedelta(minutes=24),
                    "lat":55.7, "lon":37.6})],
                hints=[dict(tr_id=1, observed_at=T, delay_s=30)],
            ))
            await engine.tick(0)
            assert len(spy.predictions) == 1
            event = {"tr_id": 1, "planned_stop_id": "target",
                     "arrived_at": T+timedelta(minutes=13)}
            from backend.arrivals import ArrivalInput
            engine.ingest_arrival(ArrivalInput(**event), received_at=T+timedelta(minutes=13))
            assert len(spy.arrivals) == 1
            engine.mode = "replay"
            engine.ingest_arrival(ArrivalInput(tr_id=1, planned_stop_id="other",
                arrived_at=T+timedelta(minutes=25)), received_at=T+timedelta(minutes=25))
            assert len(spy.arrivals) == 1
    asyncio.run(exercise())
