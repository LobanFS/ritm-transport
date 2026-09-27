#!/usr/bin/env python3
"""Build a reviewable HGBR candidate while keeping Transformer weights frozen."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml_service.artifacts import SCHEMA_VERSION, sha256
from ml_service.hybrid import transformer_prior
from ml_service.learned import LearnedModel
from ml_service.frozen_plan.plan import PLAN_FEATURES, build_plan_features, load_plan
from training.production_data import load_examples, one_snapshot_per_target, temporal_split


FEATURE_NAMES = tuple(PLAN_FEATURES) + ("neural_prior",)


def official_matrix(dataset: Path, sequences: Path, transformer):
    data = np.load(sequences)
    ids = data["mt_id"].astype(str)
    if len(ids) == 0 or len(set(ids)) != len(ids):
        raise ValueError("Moscow sequence ids must be non-empty and unique")
    labels = pd.read_csv(dataset/"labels/labels_train.csv", dtype={"sample_id": str, "tr_id": str})
    target_by_id = labels.set_index("sample_id")["target_delay_s"]
    points = labels.drop(columns=["target_delay_s", "target_class"])
    plan = build_plan_features(points, load_plan(dataset, "train")).set_index("sample_id").loc[ids]
    current = data["mt_cur"].astype(float)
    if not np.allclose(current, plan["cur_dev_s"].to_numpy(float), rtol=0, atol=1e-6):
        raise ValueError("sequence current delays do not match official plan features")
    if "mt_target" in data and not np.allclose(
            data["mt_target"].astype(float), target_by_id.loc[ids].to_numpy(float), rtol=0, atol=1e-6):
        raise ValueError("sequence targets do not match official labels")
    prior = transformer_prior(transformer, data["mt_seq"], data["mt_context"], current)
    matrix = plan.loc[:, list(PLAN_FEATURES)].copy()
    matrix["neural_prior"] = prior
    return matrix.reset_index(drop=True), target_by_id.loc[ids].to_numpy(float)


def production_matrix(model: LearnedModel, examples):
    frames = []
    for example in examples:
        frame, reason = model.features(example.request)
        if reason:
            raise ValueError(f"request {example.request.request_id} cannot use learned model: {reason}")
        frames.append(frame)
    plan = pd.concat(frames, ignore_index=True).loc[:, list(PLAN_FEATURES)]
    requests = [item.request for item in examples]
    sequence, context, *_ = model.sequence_inputs(requests, plan)
    current = np.asarray([item.request.current_delay_s for item in examples], dtype=float)
    plan["neural_prior"] = transformer_prior(model.transformer, sequence, context, current)
    target = np.asarray([item.label.target_delay_s for item in examples], dtype=float)
    return plan, target


def _fit(template, matrix, target):
    params = template.get_params(deep=False)
    candidate = HistGradientBoostingRegressor(**params)
    with threadpool_limits(limits=1):
        candidate.fit(matrix, target)
    return candidate


def _mae(target, prediction):
    return float(np.mean(np.abs(np.asarray(target)-np.asarray(prediction))))


def _reference(matrix: pd.DataFrame, ids: list[str]):
    values = matrix.to_numpy(float)
    median = np.median(values, axis=0)
    q25, q75 = np.percentile(values, [25, 75], axis=0)
    scale = np.where(q75 > q25, q75-q25, 1.0)
    index = int(np.argmin(np.sum(np.abs((values-median)/scale), axis=1)))
    return index, {name: float(matrix.iloc[index][name]) for name in FEATURE_NAMES}, ids[index]


def refit(*, model_path: Path, encoder_path: Path, production_path: Path,
          dataset: Path, sequences: Path, output: Path, model_version: str,
          manifest_path: Path | None = None,
          minimum_new_targets: int = 200, validation_fraction: float = .2,
          maximum_mae_regression_s: float = 0.0) -> dict:
    work = output.with_name(output.name+".tmp")
    if output.exists() or work.exists():
        raise FileExistsError(f"candidate output already exists: {output}")
    current = LearnedModel(model_path, encoder_path, manifest_path)
    if not isinstance(current.estimator, HistGradientBoostingRegressor):
        raise TypeError("current estimator is not HistGradientBoostingRegressor")
    raw = load_examples(production_path)
    unique = one_snapshot_per_target(raw)
    if len(unique) < minimum_new_targets:
        raise ValueError(f"need at least {minimum_new_targets} unique new targets, got {len(unique)}")
    fit_examples, validation_examples = temporal_split(raw, validation_fraction)
    base_x, base_y = official_matrix(dataset, sequences, current.transformer)
    new_fit_x, new_fit_y = production_matrix(current, fit_examples)
    validation_x, validation_y = production_matrix(current, validation_examples)
    trial = _fit(current.estimator, pd.concat([base_x, new_fit_x], ignore_index=True),
                 np.concatenate([base_y, new_fit_y]))
    old_mae = _mae(validation_y, current.estimator.predict(validation_x))
    candidate_mae = _mae(validation_y, trial.predict(validation_x))
    accepted = candidate_mae <= old_mae + maximum_mae_regression_s
    report = {
        "schema_version": "hgbr-refit-report-v1",
        "model_version": model_version,
        "frozen_encoder_sha256": sha256(encoder_path),
        "update_strategy": "full_hgbr_refit",
        "raw_predictions": len(raw), "unique_new_targets": len(unique),
        "fit_new_targets": len(fit_examples), "validation_new_targets": len(validation_examples),
        "purged_new_targets": len(unique)-len(fit_examples)-len(validation_examples),
        "validation_start": min(item.request.issued_at for item in validation_examples).isoformat(),
        "latest_fit_label_available_at": max(item.label.available_at for item in fit_examples).isoformat(),
        "official_rows": len(base_x), "old_validation_mae_s": old_mae,
        "candidate_validation_mae_s": candidate_mae,
        "maximum_mae_regression_s": maximum_mae_regression_s,
        "accepted": accepted,
    }
    work.mkdir(parents=True)
    (work/"refit_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    if not accepted:
        work.replace(output)
        return report

    all_new_x, all_new_y = production_matrix(current, unique)
    final_x = pd.concat([base_x, all_new_x], ignore_index=True)
    final_y = np.concatenate([base_y, all_new_y])
    final = _fit(current.estimator, final_x, final_y)
    payload = {"model": final, "projection": None, "variant": "prior",
               "accepted": True, "features": PLAN_FEATURES}
    joblib.dump(payload, work/"model.joblib")
    shutil.copyfile(encoder_path, work/"encoder.pt")
    _, reference, reference_id = _reference(
        final_x, [f"official:{i}" for i in range(len(base_x))]
        + [f"production:{item.request.request_id}" for item in unique])
    reference_payload = {
        "schema_version": "hybrid-explanation-reference-v1",
        "model_sha256": sha256(work/"model.joblib"),
        "encoder_sha256": sha256(work/"encoder.pt"),
        "description": f"Обучающий пример, ближайший к robust-медиане признаков; id={reference_id}",
        "source_rows": len(final_x), "sample_id": reference_id,
        "feature_names": list(FEATURE_NAMES), "values": reference,
    }
    (work/"explanation_reference.json").write_text(
        json.dumps(reference_payload, ensure_ascii=False, indent=2)+"\n")
    manifest = {
        "schema_version": SCHEMA_VERSION, "model_version": model_version,
        "model_sha256": sha256(work/"model.joblib"),
        "encoder_sha256": sha256(work/"encoder.pt"),
        "explanation_reference_sha256": sha256(work/"explanation_reference.json"),
        "estimator": "HistGradientBoostingRegressor", "feature_names": list(FEATURE_NAMES),
        "frozen_encoder": True,
        "training": {"update_strategy": "full_hgbr_refit", "official_rows": len(base_x),
                     "new_targets": len(unique), "created_at": datetime.now(timezone.utc).isoformat()},
    }
    (work/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    # Full runtime load checks hashes, payload schema, frozen encoder and explainer
    # before the directory becomes a selectable candidate.
    LearnedModel(work/"model.joblib", work/"encoder.pt", work/"manifest.json")
    work.replace(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--production-examples", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--sequences", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--minimum-new-targets", type=int, default=200)
    parser.add_argument("--validation-fraction", type=float, default=.2)
    parser.add_argument("--maximum-mae-regression-s", type=float, default=0.0)
    args = parser.parse_args()
    result = refit(model_path=args.model, encoder_path=args.encoder,
        production_path=args.production_examples, dataset=args.dataset,
        sequences=args.sequences, output=args.out, model_version=args.model_version,
        manifest_path=args.manifest,
        minimum_new_targets=args.minimum_new_targets,
        validation_fraction=args.validation_fraction,
        maximum_mae_regression_s=args.maximum_mae_regression_s)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
