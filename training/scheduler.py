"""Weekly export, full HGBR refit and temporal quality gate."""
from __future__ import annotations

from dataclasses import dataclass
import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import threading
import traceback

from training.export import export_examples
from training.production_data import load_examples, one_snapshot_per_target
from training.refit_hgbr import refit


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _boolean(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise ValueError(f"{name} must be true or false")


@dataclass(frozen=True)
class SchedulerConfig:
    store: Path
    model: Path
    encoder: Path
    manifest: Path | None
    dataset: Path
    sequences: Path
    output: Path
    interval_s: int = 7 * 24 * 60 * 60
    minimum_new_targets: int = 200
    validation_fraction: float = .2
    maximum_mae_regression_s: float = 0.0
    run_on_start: bool = True

    @classmethod
    def from_env(cls):
        manifest = os.getenv("CURRENT_MANIFEST_PATH")
        config = cls(
            store=Path(os.environ["LEARNING_STORE_PATH"]),
            model=Path(os.environ["CURRENT_MODEL_PATH"]),
            encoder=Path(os.environ["CURRENT_ENCODER_PATH"]),
            manifest=Path(manifest) if manifest else None,
            dataset=Path(os.environ["RETRAIN_DATASET_PATH"]),
            sequences=Path(os.environ["RETRAIN_SEQUENCES_PATH"]),
            output=Path(os.environ["RETRAIN_OUTPUT_PATH"]),
            interval_s=int(os.getenv("RETRAIN_INTERVAL_SECONDS", str(7*24*60*60))),
            minimum_new_targets=int(os.getenv("RETRAIN_MINIMUM_NEW_TARGETS", "200")),
            validation_fraction=float(os.getenv("RETRAIN_VALIDATION_FRACTION", ".2")),
            maximum_mae_regression_s=float(os.getenv("RETRAIN_MAXIMUM_MAE_REGRESSION_S", "0")),
            run_on_start=_boolean("RETRAIN_RUN_ON_START", True),
        )
        if config.interval_s < 3600:
            raise ValueError("RETRAIN_INTERVAL_SECONDS must be at least 3600")
        if config.minimum_new_targets < 2:
            raise ValueError("RETRAIN_MINIMUM_NEW_TARGETS must be at least 2")
        if not 0 < config.validation_fraction < .5:
            raise ValueError("RETRAIN_VALIDATION_FRACTION must be between 0 and 0.5")
        return config


def _read_state(path: Path) -> dict:
    if not path.exists():
        return {}
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("scheduler state must be an object")
    return value


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n")
    temporary.replace(path)


def run_once(config: SchedulerConfig, *, now: datetime | None = None) -> dict:
    now = now or utcnow()
    config.output.mkdir(parents=True, exist_ok=True)
    state_path = config.output/"scheduler_state.json"
    state = _read_state(state_path)
    run_id = now.strftime("hgbr-weekly-%Y%m%dT%H%M%SZ")
    inputs = config.output/"inputs"
    export_path = inputs/f"{run_id}.jsonl"
    export_result = export_examples(config.store, export_path)
    corpus_sha = hashlib.sha256(export_path.read_bytes()).hexdigest()
    examples = load_examples(export_path) if export_result["exported"] else []
    unique_targets = len(one_snapshot_per_target(examples)) if examples else 0
    report = {
        "schema_version": "weekly-hgbr-run-v1", "run_id": run_id,
        "started_at": now.isoformat(), "corpus_sha256": corpus_sha,
        "exported_predictions": export_result["exported"],
        "unique_targets": unique_targets,
        "minimum_new_targets": config.minimum_new_targets,
        "next_run_at": (now+timedelta(seconds=config.interval_s)).isoformat(),
    }
    if corpus_sha == state.get("last_corpus_sha256"):
        report["status"] = "skipped_unchanged_corpus"
    elif unique_targets < config.minimum_new_targets:
        report["status"] = "skipped_insufficient_data"
    else:
        candidate = config.output/"candidates"/run_id
        result = refit(model_path=config.model, encoder_path=config.encoder,
            manifest_path=config.manifest, production_path=export_path,
            dataset=config.dataset, sequences=config.sequences, output=candidate,
            model_version=run_id, minimum_new_targets=config.minimum_new_targets,
            validation_fraction=config.validation_fraction,
            maximum_mae_regression_s=config.maximum_mae_regression_s)
        report.update({"status": "accepted_candidate" if result["accepted"] else "rejected_candidate",
                       "accepted": result["accepted"], "candidate_path": str(candidate),
                       "quality_report": str(candidate/"refit_report.json")})
    state.update({"last_attempt_at": now.isoformat(), "next_run_at": report["next_run_at"],
                  "last_status": report["status"], "last_corpus_sha256": corpus_sha,
                  "last_run_id": run_id})
    _write_json(config.output/"runs"/f"{run_id}.json", report)
    _write_json(state_path, state)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


def serve(config: SchedulerConfig) -> None:
    config.output.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: stop.set())
    lock_path = config.output/"scheduler.lock"
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another weekly retraining scheduler is already running") from exc
        state_path = config.output/"scheduler_state.json"
        while not stop.is_set():
            state = _read_state(state_path)
            next_text = state.get("next_run_at")
            if not next_text and not config.run_on_start:
                next_text = (utcnow()+timedelta(seconds=config.interval_s)).isoformat()
                state["next_run_at"] = next_text
                _write_json(state_path, state)
            due = utcnow() if config.run_on_start and not next_text else (
                datetime.fromisoformat(next_text) if next_text else utcnow()+timedelta(seconds=config.interval_s))
            delay = (due-utcnow()).total_seconds()
            if delay > 0:
                stop.wait(min(delay, 300))
                continue
            try:
                run_once(config)
            except Exception as exc:
                failed_at = utcnow()
                failure = {"schema_version": "weekly-hgbr-run-v1", "status": "failed",
                    "failed_at": failed_at.isoformat(), "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=20),
                    "next_run_at": (failed_at+timedelta(seconds=config.interval_s)).isoformat()}
                _write_json(config.output/"runs"/f"failed-{failed_at.strftime('%Y%m%dT%H%M%SZ')}.json", failure)
                state.update({"last_attempt_at": failed_at.isoformat(), "last_status": "failed",
                              "next_run_at": failure["next_run_at"]})
                _write_json(state_path, state)
                print(json.dumps(failure, ensure_ascii=False), flush=True)


def locked_run_once(config: SchedulerConfig) -> dict:
    """One-shot operator run obeying the same single-scheduler lock."""
    config.output.mkdir(parents=True, exist_ok=True)
    with (config.output/"scheduler.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another weekly retraining scheduler is already running") from exc
        return run_once(config)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="run one export/refit check and exit")
    args = parser.parse_args()
    config = SchedulerConfig.from_env()
    locked_run_once(config) if args.once else serve(config)


if __name__ == "__main__":
    main()
