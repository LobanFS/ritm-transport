"""Strict manifest for replaceable HGBR bundles with a frozen encoder."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


SCHEMA_VERSION = "hybrid-runtime-bundle-v2"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(path: Path, *, model_path: Path, encoder_path: Path,
                  feature_names: tuple[str, ...]) -> dict:
    payload = json.loads(path.read_text())
    required = {
        "schema_version", "model_version", "model_sha256", "encoder_sha256",
        "explanation_reference_sha256", "estimator", "feature_names",
        "frozen_encoder", "training",
    }
    if set(payload) != required:
        raise ValueError("manifest содержит неожиданную схему")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError("неподдерживаемая версия manifest")
    if payload["estimator"] != "HistGradientBoostingRegressor":
        raise ValueError("manifest разрешает только текущий HGBR")
    if payload["frozen_encoder"] is not True:
        raise ValueError("encoder должен оставаться замороженным")
    if tuple(payload["feature_names"]) != feature_names:
        raise ValueError("manifest не совпадает со схемой признаков")
    if (not isinstance(payload["model_version"], str) or not payload["model_version"]
            or len(payload["model_version"]) > 120):
        raise ValueError("manifest не содержит версию модели")
    if sha256(model_path) != payload["model_sha256"]:
        raise ValueError("SHA-256 model.joblib не совпадает с manifest")
    if sha256(encoder_path) != payload["encoder_sha256"]:
        raise ValueError("SHA-256 encoder.pt не совпадает с manifest")
    reference = path.with_name("explanation_reference.json")
    if sha256(reference) != payload["explanation_reference_sha256"]:
        raise ValueError("SHA-256 explanation reference не совпадает с manifest")
    training = payload["training"]
    if not isinstance(training, dict) or training.get("update_strategy") != "full_hgbr_refit":
        raise ValueError("manifest содержит неподдерживаемую стратегию обновления")
    if (not isinstance(training.get("official_rows"), int) or training["official_rows"] <= 0
            or not isinstance(training.get("new_targets"), int) or training["new_targets"] < 0):
        raise ValueError("manifest содержит неверные размеры обучающей выборки")
    return payload
