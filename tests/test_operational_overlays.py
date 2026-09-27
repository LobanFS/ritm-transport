from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_learning_collection_is_opt_in_and_persistent():
    base = yaml.safe_load((ROOT/"compose.yaml").read_text())
    assert "LEARNING_STORE_PATH" not in base["services"]["backend"]["environment"]
    overlay = yaml.safe_load((ROOT/"compose.learning.yaml").read_text())
    backend = overlay["services"]["backend"]
    assert backend["environment"]["LEARNING_STORE_PATH"].endswith("learning.sqlite3")
    assert backend["volumes"] and "learning-data" in overlay["volumes"]


def test_scale_overlay_routes_backend_through_gateway_without_host_port_collision():
    text = (ROOT/"compose.scale.yaml").read_text()
    assert "ports: !reset []" in text
    assert "ML_URL: http://ml-gateway:8001" in text
    assert "http://127.0.0.1:8001/health" in text
    assert "http://localhost:8001/health" not in text
    assert "server ml:8001 resolve" in (ROOT/"config/ml-gateway.conf").read_text()


def test_training_image_contains_only_current_hgbr_stack():
    text = (ROOT/"Dockerfile.train").read_text()
    assert "training.refit_hgbr" in text
    assert "requirements.txt" in text


def test_weekly_scheduler_uses_shared_journal_and_never_mounts_candidates_into_ml():
    text = (ROOT/"compose.retrain.yaml").read_text()
    assert "training.scheduler" in text
    assert "RETRAIN_INTERVAL_SECONDS:-604800" in text
    assert "learning-data:/learning" in text
    assert "target: /outputs" in text
    assert "services:\n  ml:" not in text


def test_trainer_keeps_sqlite_owner_uid_with_configurable_output_group():
    # Backend creates the volume/database under UID 10001. A host UID override
    # cannot read WAL safely: SQLite sometimes must recreate its WAL/SHM files.
    overlay = yaml.safe_load((ROOT/"compose.retrain.yaml").read_text())
    trainer = overlay["services"]["model-trainer"]
    assert trainer["user"] == "10001:${RETRAIN_GID:-10001}"
    assert "--uid 10001 app" in (ROOT/"Dockerfile").read_text()
