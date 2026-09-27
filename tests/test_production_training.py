from datetime import datetime, timedelta, timezone
import hashlib
import json

import pytest

from ml_service.artifacts import SCHEMA_VERSION, load_manifest
from ml_service.frozen_plan.plan import PLAN_FEATURES
from training.production_data import ProductionExample, temporal_split


T = datetime(2026, 1, 6, 8, tzinfo=timezone.utc)
FEATURES = tuple(PLAN_FEATURES) + ("neural_prior",)


def payload(index=0, *, issued_offset=0, available_offset=800):
    issued = T+timedelta(seconds=issued_offset)
    scheduled = issued+timedelta(seconds=750)
    arrived = scheduled+timedelta(seconds=40)
    request = {
        "request_id": f"r{index}", "tr_id": 1, "issued_at": issued.isoformat(),
        "target": {"id": f"s{index}", "name": "S", "scheduled_at": scheduled.isoformat(),
                   "lat": 55.7, "lon": 37.6},
        "current_delay_s": 10, "features": {"telemetry_age_s": 1},
    }
    prediction = {"request_id": f"r{index}", "tr_id": 1,
        "issued_at": issued.isoformat(), "target": request["target"],
        "predicted_delay_s": 20, "model_version": "v", "method": "learned",
        "risk": "green", "reasons": []}
    return {"schema_version": "production-delay-example-v1", "request": request,
        "prediction": prediction, "label": {"target_delay_s": 40,
            "arrived_at": arrived.isoformat(),
            "available_at": (issued+timedelta(seconds=available_offset)).isoformat(),
            "source": "arrival"}}


def test_production_example_rejects_label_available_at_issue_time():
    with pytest.raises(ValueError, match="already available"):
        ProductionExample.model_validate(payload(available_offset=0))


def test_temporal_split_keeps_later_labels_for_validation():
    examples = [ProductionExample.model_validate(payload(i, issued_offset=i*1000)) for i in range(5)]
    fit, validation = temporal_split(examples, .2)
    assert len(fit) == 4 and len(validation) == 1
    assert max(x.label.available_at for x in fit) < validation[0].request.issued_at


def test_temporal_split_purges_future_and_boundary_labels():
    # The last train label becomes known exactly when validation starts;
    # it is not a past input to this fold. Another arrives even later.
    examples = [ProductionExample.model_validate(payload(i,
        issued_offset=i*1000, available_offset=offset))
        for i, offset in enumerate([800, 800, 2500, 1000, 800])]
    fit, validation = temporal_split(examples, .2)
    assert [item.request.request_id for item in fit] == ["r0", "r1"]
    assert [item.request.request_id for item in validation] == ["r4"]
    cutoff = min(item.request.issued_at for item in validation)
    assert all(item.label.available_at < cutoff for item in fit)


def test_temporal_split_refuses_dense_requests_without_mature_labels():
    # Ordering by label times alone previously accepted all four train labels,
    # although none was available at the first validation prediction.
    examples = [ProductionExample.model_validate(payload(i, issued_offset=i*100))
                for i in range(5)]
    with pytest.raises(ValueError, match="no mature training labels"):
        temporal_split(examples, .2)


def test_temporal_split_keeps_simultaneous_requests_in_one_fold():
    examples = [ProductionExample.model_validate(payload(i, issued_offset=offset))
                for i, offset in enumerate([0, 1000, 2000, 3000, 3000])]
    fit, validation = temporal_split(examples, .2)
    assert len(fit) == 3 and len(validation) == 2
    assert {item.request.issued_at for item in fit}.isdisjoint(
        {item.request.issued_at for item in validation})


def test_versioned_manifest_binds_model_encoder_and_reference(tmp_path):
    model, encoder, reference = tmp_path/"model.joblib", tmp_path/"encoder.pt", tmp_path/"explanation_reference.json"
    model.write_bytes(b"model"); encoder.write_bytes(b"encoder"); reference.write_bytes(b"reference")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": SCHEMA_VERSION, "model_version": "candidate-v2",
        "model_sha256": digest(model), "encoder_sha256": digest(encoder),
        "explanation_reference_sha256": digest(reference),
        "estimator": "HistGradientBoostingRegressor", "feature_names": list(FEATURES),
        "frozen_encoder": True,
        "training": {"update_strategy": "full_hgbr_refit", "official_rows": 10,
                     "new_targets": 2},
    }
    path = tmp_path/"manifest.json"
    path.write_text(json.dumps(manifest))
    assert load_manifest(path, model_path=model, encoder_path=encoder,
                         feature_names=FEATURES)["model_version"] == "candidate-v2"
    encoder.write_bytes(b"changed")
    with pytest.raises(ValueError, match="encoder.pt"):
        load_manifest(path, model_path=model, encoder_path=encoder, feature_names=FEATURES)


def test_current_bundle_can_be_loaded_through_versioned_manifest(tmp_path):
    from pathlib import Path
    import shutil
    from ml_service.learned import LearnedModel

    root = Path(__file__).resolve().parents[1]
    source = root/"artifacts/model"
    if not (source/"model.joblib").is_file():
        pytest.skip("runtime bundle is not present")
    model, encoder = tmp_path/"model.joblib", tmp_path/"encoder.pt"
    shutil.copyfile(source/"model.joblib", model)
    shutil.copyfile(source/"encoder.pt", encoder)
    reference_payload = json.loads((root/"artifacts/model/explanation_reference.json").read_text())
    reference = tmp_path/"explanation_reference.json"
    reference.write_text(json.dumps(reference_payload))
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": SCHEMA_VERSION, "model_version": "versioned-fixture",
        "model_sha256": digest(model), "encoder_sha256": digest(encoder),
        "explanation_reference_sha256": digest(reference),
        "estimator": "HistGradientBoostingRegressor", "feature_names": list(FEATURES),
        "frozen_encoder": True,
        "training": {"update_strategy": "full_hgbr_refit", "official_rows": 4434,
                     "new_targets": 0},
    }
    path = tmp_path/"manifest.json"; path.write_text(json.dumps(manifest))
    loaded = LearnedModel(model, encoder, path)
    assert loaded.model_version == "versioned-fixture"
    assert loaded.encoder_sha256 == reference_payload["encoder_sha256"]


def test_refit_rejects_candidate_that_worsens_later_production_targets(tmp_path, monkeypatch):
    """The weekly gate must leave a worse model as a report-only candidate."""
    from types import SimpleNamespace
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingRegressor
    import training.refit_hgbr as module

    class CurrentEstimator(HistGradientBoostingRegressor):
        def predict(self, matrix):
            return np.zeros(len(matrix))

    class WorseCandidate:
        def predict(self, matrix):
            return np.full(len(matrix), 10.0)

    current = SimpleNamespace(estimator=CurrentEstimator(), transformer=object())
    examples = [ProductionExample.model_validate(payload(i, issued_offset=i*1000))
                for i in range(2)]
    matrix = pd.DataFrame([{name: 0.0 for name in FEATURES}])
    monkeypatch.setattr(module, "LearnedModel", lambda *args, **kwargs: current)
    monkeypatch.setattr(module, "load_examples", lambda path: examples)
    monkeypatch.setattr(module, "one_snapshot_per_target", lambda rows: rows)
    monkeypatch.setattr(module, "temporal_split", lambda rows, fraction: ([rows[0]], [rows[1]]))
    monkeypatch.setattr(module, "official_matrix", lambda *args: (matrix, np.array([0.0])))
    monkeypatch.setattr(module, "production_matrix", lambda *args: (matrix, np.array([0.0])))
    monkeypatch.setattr(module, "_fit", lambda *args: WorseCandidate())
    encoder = tmp_path/"encoder.pt"
    encoder.write_bytes(b"frozen")
    output = tmp_path/"candidate"

    result = module.refit(model_path=tmp_path/"model.joblib", encoder_path=encoder,
        production_path=tmp_path/"examples.jsonl", dataset=tmp_path/"dataset",
        sequences=tmp_path/"sequences.npz", output=output, model_version="worse",
        minimum_new_targets=2, maximum_mae_regression_s=0)

    assert result["accepted"] is False
    assert result["old_validation_mae_s"] == 0
    assert result["candidate_validation_mae_s"] == 10
    assert (output/"refit_report.json").is_file()
    assert not (output/"model.joblib").exists()
