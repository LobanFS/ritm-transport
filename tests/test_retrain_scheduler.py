from datetime import datetime, timezone
import fcntl
from pathlib import Path
import pytest

from backend.learning_store import LearningStore
from training.scheduler import SchedulerConfig, locked_run_once, run_once


T = datetime(2026, 10, 1, tzinfo=timezone.utc)


def config(tmp_path, **changes):
    values = dict(store=tmp_path/"learning.sqlite3", model=tmp_path/"model.joblib",
        encoder=tmp_path/"encoder.pt", manifest=None, dataset=tmp_path/"dataset",
        sequences=tmp_path/"sequences.npz", output=tmp_path/"output",
        interval_s=604800, minimum_new_targets=200, validation_fraction=.2,
        maximum_mae_regression_s=0, run_on_start=True)
    values.update(changes)
    LearningStore(values["store"])
    return SchedulerConfig(**values)


def test_weekly_run_skips_insufficient_and_then_unchanged_corpus(tmp_path):
    cfg = config(tmp_path)
    first = run_once(cfg, now=T)
    assert first["status"] == "skipped_insufficient_data"
    assert first["unique_targets"] == 0
    second = run_once(cfg, now=T.replace(day=8))
    assert second["status"] == "skipped_unchanged_corpus"
    state = (cfg.output/"scheduler_state.json").read_text()
    assert "2026-10-15" in state


def test_weekly_run_creates_checked_candidate_without_promotion(tmp_path, monkeypatch):
    import training.scheduler as module
    cfg = config(tmp_path, minimum_new_targets=2)

    def fake_export(store, output):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("fixture\n")
        return {"exported": 2}

    monkeypatch.setattr(module, "export_examples", fake_export)
    monkeypatch.setattr(module, "load_examples", lambda _: [object(), object()])
    monkeypatch.setattr(module, "one_snapshot_per_target", lambda values: values)
    calls = []
    def fake_refit(**kwargs):
        calls.append(kwargs)
        kwargs["output"].mkdir(parents=True)
        (kwargs["output"]/"refit_report.json").write_text("{}")
        return {"accepted": True}
    monkeypatch.setattr(module, "refit", fake_refit)

    result = run_once(cfg, now=T)
    assert result["status"] == "accepted_candidate"
    assert len(calls) == 1 and calls[0]["minimum_new_targets"] == 2
    assert Path(result["candidate_path"]).is_dir()
    assert not hasattr(cfg, "production_path")  # scheduler has no deployment target


def test_scheduler_environment_defaults_to_weekly(monkeypatch, tmp_path):
    values = {
        "LEARNING_STORE_PATH": tmp_path/"learning.sqlite3",
        "CURRENT_MODEL_PATH": tmp_path/"model.joblib",
        "CURRENT_ENCODER_PATH": tmp_path/"encoder.pt",
        "RETRAIN_DATASET_PATH": tmp_path/"dataset",
        "RETRAIN_SEQUENCES_PATH": tmp_path/"sequences.npz",
        "RETRAIN_OUTPUT_PATH": tmp_path/"output",
    }
    for name, value in values.items(): monkeypatch.setenv(name, str(value))
    cfg = SchedulerConfig.from_env()
    assert cfg.interval_s == 7*24*60*60 and cfg.run_on_start is True


def test_manual_run_cannot_overlap_daemon(tmp_path):
    cfg = config(tmp_path)
    cfg.output.mkdir(parents=True)
    with (cfg.output/"scheduler.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="already running"):
            locked_run_once(cfg)
