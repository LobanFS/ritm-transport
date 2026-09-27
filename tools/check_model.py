"""Без обучения: паритет HTTP/runtime и frozen builder на 151 validate + 353 test.

Пример: .venv/bin/python tools/check_model.py --dataset <dataset> --url http://127.0.0.1:8001
Без --url проверяется локальный адаптер. Все будущие labels читаются только после
инференса для отдельно обозначенной диагностической MAE, не для выбора модели.
--require-probability дополнительно сверяет 353 сохранённые вероятности и не
пересчитывает MAE по меткам: только POINT_COLUMNS и готовые ID/вероятность.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def probability_reference(path):
    """Проекция JSONL только на ID/вероятность; сохранённый late_gt120 не используется."""
    selected = {}
    with path.open() as stream:
        for line in stream:
            # Аналог CSV usecols: остальные поля не участвуют ни в сравнении,
            # ни в запросах, ни в каком-либо обучении или пересчёте качества.
            row = json.loads(line, object_hook=lambda item: {
                key: item[key] for key in ('sample_id', 'probability_late') if key in item})
            sample_id, value = row['sample_id'], row['probability_late']
            if not isinstance(sample_id, str) or not sample_id or sample_id in selected:
                raise ValueError('Непустые уникальные ID обязательны в probability reference')
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('Некорректная сохранённая вероятность')
            selected[sample_id] = value
    if len(selected) != 353:
        raise ValueError('Probability reference должен содержать ровно 353 test ID')
    return selected


def build_requests(points, plan, timezone_name, plan_version):
    """CSV allowlist → тот же внешний контракт, который отправляет backend."""
    from common.contracts import PredictionRequest
    tz = ZoneInfo(timezone_name)
    def aware(value):
        timestamp = datetime.fromisoformat(str(value))
        return timestamp.replace(tzinfo=tz) if timestamp.tzinfo is None else timestamp
    by_vehicle = {}
    for row in plan.itertuples(index=False):
        coords = re.fullmatch(r"POINT\s*\(\s*([-+\d.eE]+)\s+([-+\d.eE]+)\s*\)", str(row.geom))
        if not coords:
            raise ValueError("Invalid plan WKT")
        # CSV reader должен распознать boolean, строка 'False' не truthy.
        manual_text = str(row.manual_fill).lower()
        if manual_text not in ("true", "false"):
            raise ValueError("Unknown manual_fill")
        by_vehicle.setdefault(int(row.tr_id), []).append({
            "id": str(row.tt_action_item_id), "name": str(row.building_address)[:120],
            "scheduled_at": aware(row.time_begin), "manual_fill": manual_text == "true",
            "lat": float(coords[2]), "lon": float(coords[1]),
        })
    histories = {}
    ordered = points.copy()
    ordered['aware_T'] = [aware(value) for value in ordered['T']]
    for _, group in ordered.sort_values(['tr_id', 'aware_T'], kind='mergesort').groupby('tr_id', sort=False):
        previous = []
        for row in group.itertuples(index=False):
            now = row.aware_T
            histories[str(row.sample_id)] = [
                {'observed_at': stamp, 'delay_s': delay}
                for stamp, delay in previous if 0 <= (now-stamp).total_seconds() <= 90*60
            ][-11:]
            previous.append((now, float(row.cur_dev_s)))
    result = []
    for row in points.itertuples(index=False):
        stops = by_vehicle[int(row.tr_id)]
        target = dict(next(stop for stop in stops if stop["id"] == str(row.target_stop_id)))
        target["scheduled_at"] = aware(row.target_time_begin)
        result.append(PredictionRequest.model_validate({
            "request_id": str(row.sample_id), "tr_id": int(row.tr_id), "issued_at": aware(row.T),
            "target": target, "current_delay_s": float(row.cur_dev_s),
            "current_delay_source": "csv_snapshot",
            "delay_history": histories[str(row.sample_id)],
            "features": {"telemetry_age_s": 0},
            "plan_context": {"version": plan_version, "timezone": timezone_name,
                             "complete": True, "stops": stops},
        }))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=ROOT.parent / "alexchist/reports/experiments/swiss_sequence_hybrid")
    parser.add_argument("--model", type=Path, default=ROOT / "artifacts/model/model.joblib")
    parser.add_argument("--timezone", default="UTC")
    parser.add_argument("--url", help="ML service URL; без него local runtime")
    parser.add_argument("--require-probability", action="store_true",
                        help="Требовать калибратор и паритет 353 saved probabilities; не читать target labels")
    parser.add_argument("--probability-reference", type=Path,
                        default=ROOT / "artifacts/probability/test-probabilities.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/model")
    args = parser.parse_args()
    start = time.perf_counter()
    write(args.out / "parity-contract.json", {
        "started_at": datetime.now(timezone.utc).isoformat(), "budget_seconds": 120,
        "comparison": "Неизменённый frozen builder + saved predictions, без обучения/подбора",
        "populations": {"validate": 151, "test": 353},
        "acceptance": "Все ID/порядок, exact feature parity, |prediction difference| <= 1e-9, 0 fallback",
        "probability_required": args.require_probability,
        "probability_acceptance": "504 finite probabilities; 353 stored test probabilities |difference| <= 1e-12; same calibrator SHA-256" if args.require_probability else None,
        "input_source": "csv_snapshot; telemetry_age_s=0 is a declared parity-check input, not measured live freshness",
        "future_facts_policy": "Test читается только с usecols=POINT_COLUMNS; target labels не загружаются." if args.require_probability else "До всех прогнозов test читается только с usecols=POINT_COLUMNS. Labels отдельно после.",
        "url": args.url,
    })
    import numpy as np
    import pandas as pd
    import httpx
    from ml_service.frozen_plan.contracts import POINT_COLUMNS
    from ml_service.frozen_plan.plan import PLAN_COLUMNS, PLAN_FEATURES, build_plan_features
    from ml_service.learned import LearnedModel, MODEL_SHA256
    runtime = LearnedModel(args.model)
    saved_probabilities = None
    expected_probability_metadata = None
    if args.require_probability:
        from ml_service.probability import ProbabilityModel
        saved_probabilities = probability_reference(args.probability_reference)
        expected_probability_metadata = ProbabilityModel.load(
            args.model.with_name('probability.json'), MODEL_SHA256).metadata()
    plan_path = args.dataset / "validate/schedule_plan.csv"
    plan = pd.read_csv(plan_path, usecols=list(PLAN_COLUMNS))
    results = {}
    predictions_by_split = {}
    with httpx.Client(timeout=90, trust_env=False) as client:
        if args.url:
            card = client.get(args.url.rstrip('/') + "/model").raise_for_status().json()
            if not card.get("trained") or card.get("provenance", {}).get("sha256") != MODEL_SHA256:
                raise ValueError("HTTP service does not expose expected frozen model")
        else:
            card = {"runtime_versions": runtime.versions, "sha256": runtime.sha256,
                    "probability": runtime.probability.metadata() if runtime.probability else None,
                    "probability_error": runtime.probability_error}
        if args.require_probability:
            probability_card = card.get('probability')
            if (not probability_card or card.get('probability_error')
                or probability_card.get('artifact_sha256') != expected_probability_metadata['artifact_sha256']
                or probability_card.get('regression_model_sha256') != MODEL_SHA256
                or probability_card.get('event') != 'target_delay_s > 120'):
                raise ValueError('Сервис не загрузил ожидаемый калибратор; проверьте PROBABILITY_PATH и /model')
        for split, filename, expected_count in (("validate", "validate/points.csv", 151), ("test", "labels/labels_test.csv", 353)):
            path = args.dataset / filename
            # Explicit projection: target_delay_s и target_class не загружаются.
            points = pd.read_csv(path, usecols=list(POINT_COLUMNS))
            assert len(points) == expected_count
            requests = build_requests(points, plan, args.timezone, sha(plan_path))
            reference_features = build_plan_features(points, plan)
            observed_features = pd.concat([runtime.features(request)[0] for request in requests], ignore_index=True)
            np.testing.assert_array_equal(reference_features[list(PLAN_FEATURES)].to_numpy(), observed_features[list(PLAN_FEATURES)].to_numpy())
            responses = []
            timings = []
            for offset in range(0, len(requests), 128):
                batch = requests[offset:offset + 128]
                before = time.perf_counter()
                if args.url:
                    rows = client.post(args.url.rstrip('/') + "/predict/batch", json=[r.model_dump(mode="json") for r in batch]).raise_for_status().json()
                else:
                    rows = [row.model_dump(mode="json") for row in runtime.predict_batch(batch)]
                timings.append({"rows": len(batch), "seconds": time.perf_counter() - before})
                responses.extend(rows)
            assert [row["request_id"] for row in responses] == points.sample_id.tolist()
            assert all(row["method"] == "learned" for row in responses)
            probability_values = [row['probability_late'] for row in responses]
            assert all(value is None or (not isinstance(value, bool) and isinstance(value, (int, float))
                       and math.isfinite(value) and 0 <= value <= 1) for value in probability_values)
            probability_result = {"available_rows": sum(value is not None for value in probability_values),
                                  "required": args.require_probability, "reference_rows": 0}
            if args.require_probability:
                assert all(value is not None for value in probability_values), 'Вероятность отсутствует при корректных CSV-входах'
                if split == 'test':
                    assert set(saved_probabilities) == set(points.sample_id), 'Probability reference IDs не совпали'
                    reference_values = np.array([saved_probabilities[sample_id] for sample_id in points.sample_id])
                    observed_values = np.array(probability_values)
                    np.testing.assert_allclose(observed_values, reference_values, rtol=0, atol=1e-12)
                    probability_result.update(reference_rows=len(reference_values),
                        max_absolute_difference=float(np.max(np.abs(observed_values-reference_values))),
                        reference_sha256=sha(args.probability_reference))
                    write(args.out / 'test-runtime-probabilities.json', [
                        {"sample_id": sample_id, "probability_late": value}
                        for sample_id, value in zip(points.sample_id, probability_values, strict=True)])
            predictions = np.array([row["predicted_delay_s"] for row in responses])
            saved_path = args.source / ("submission_predictions.csv" if split == "validate" else "holdout_predictions.csv")
            saved = pd.read_csv(saved_path)
            expected = saved.set_index("sample_id").loc[points.sample_id].candidate_prediction.to_numpy()
            assert set(saved.sample_id) == set(points.sample_id)
            np.testing.assert_allclose(predictions, expected, rtol=0, atol=1e-9)
            prediction_frame = pd.DataFrame({"sample_id": points.sample_id, "prediction": predictions})
            prediction_frame.to_csv(args.out / f"{split}-runtime-predictions.csv", index=False)
            predictions_by_split[split] = prediction_frame
            results[split] = {"rows": len(points), "feature_parity_exact": True,
                              "max_prediction_difference_s": float(np.max(np.abs(predictions-expected))),
                              "fallback_rows": 0, "batches": timings,
                              "probability": probability_result,
                              "point_file_sha256": sha(path), "saved_predictions_sha256": sha(saved_path)}
    diagnostic = None
    if not args.require_probability:
        # Legacy workflow: future target впервые читается после всего инференса.
        labels = pd.read_csv(args.dataset / "labels/labels_test.csv", usecols=["sample_id", "cur_dev_s", "target_delay_s"])
        compared = labels.merge(predictions_by_split["test"], on="sample_id", validate="one_to_one")
        diagnostic = {"rows": len(compared),
                      "mae_s": float((compared.prediction-compared.target_delay_s).abs().mean()),
                      "baseline_mae_s": float((compared.cur_dev_s-compared.target_delay_s).abs().mean()),
                      "model_selected_or_tuned_here": False}
    report = {
        "status": "PASS", "runtime_seconds": time.perf_counter()-start,
        "model_sha256": runtime.sha256, "runtime_versions": runtime.versions,
        "service_card": card, "plan_sha256": sha(plan_path), "results": results,
        "test_diagnostic_only": diagnostic,
        "target_labels_loaded": not args.require_probability,
        "probability_reference_metadata": expected_probability_metadata,
        "limitations": ["Это проверка интеграции прежних прогнозов, не новый независимый ML holdout.",
                        "Известный организаторами cur_dev_s не доказывает качество GPS-оценки.",
                        "SciPy 1.11.4 отличается от training 1.11.1; точность этих 504 прогнозов проверена."],
        "source_sha256": {str(path.relative_to(ROOT)): sha(path) for path in (ROOT/'ml_service/learned.py', ROOT/'ml_service/app.py', ROOT/'common/contracts.py', Path(__file__).resolve())},
    }
    write(args.out / ("http-parity.json" if args.url else "local-parity.json"), report)
    print(json.dumps({"status": "PASS", "rows": 504, "runtime_seconds": report["runtime_seconds"],
                      "test_mae_s": diagnostic['mae_s'] if diagnostic else None,
                      "probability_reference_rows": results['test']['probability']['reference_rows']}))


if __name__ == "__main__":
    main()
