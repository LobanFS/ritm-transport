"""Проверка frozen вероятности на GPS HTTP-предиктах; без fit и runtime-правок.

Сначала сохраняется contract и вычисляются вероятности без меток. Только затем
читается target_delay_s. Срез CSV строго совпадает по sample_id. Bootstrap
выполняется по автобусам, а не по коррелированным строкам одного автобуса.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml_service.learned import MODEL_SHA256
from ml_service.probability import GPSProbabilityScope, ProbabilityModel
from tools.prepare_probability import probability_metrics


RECIPE = dict(
    name='gps-probability-frozen-transfer-v1', event='target_delay_s > 120',
    baseline='Existing training base-rate, fixed in probability.json; no GPS fitting',
    split='Provided test, secondary diagnostic; no new holdout or independence claim',
    probability='Unchanged train-OOF logistic mapping of frozen regression seconds',
    selection='All numeric learned GPS-estimate rows with valid current deviation and GPS age <=60s',
    comparison='CSV predictions restricted to the same sample IDs',
    reliability_bins=[0, .2, .4, .6, .8, 1],
    bootstrap=dict(unit='tr_id', draws=2000, random_seed=20260926, interval=[.05, .95]),
    acceptance=dict(min_rows=50, min_positives=10, min_negatives=10, min_vehicles=5,
                    brier_below_train_constant=True, log_loss_below_train_constant=True,
                    brier_cluster_bootstrap_p05_above_zero=True,
                    maximum_expected_calibration_error=.15),
    budget_s=60, no_fit=True, no_threshold_search=True, no_regression_change=True,
    door_transfer='Not evaluated: historical CSV has no door sensors',
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def read_rows(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or len({row['sample_id'] for row in rows}) != len(rows):
        raise ValueError('Нужны непустые уникальные sample_id')
    return rows


def freeze_predictions(gps_rows, csv_rows, model):
    """Чистая функция без labels/файлов; неизвестные входы остаются неизвестными."""
    reference = {row['sample_id']: row for row in csv_rows}
    if not {row['sample_id'] for row in gps_rows}.issubset(reference):
        raise ValueError('У GPS sample_id нет CSV-reference')
    rows = []
    for gps in gps_rows:
        if gps['method'] != 'learned' or gps['input_source'] != 'gps_estimate':
            continue
        probability = model.predict(gps['prediction'], current_delay_s=gps['input_cur_dev_s'],
                                    telemetry_age_s=gps['telemetry_age_s'])
        if probability is None:
            continue
        csv = reference[gps['sample_id']]
        if (gps['tr_id'], gps['T']) != (csv['tr_id'], csv['T']):
            raise ValueError('GPS и CSV sample_id относятся к разным ТС/моментам')
        if any(gps.get(key) != csv.get(key) for key in ('target_stop_id', 'target_time_begin')):
            raise ValueError('GPS и CSV sample_id относятся к разным целям')
        if csv['method'] != 'learned' or csv['input_source'] != 'csv_snapshot':
            raise ValueError('У сравниваемой GPS-точки нет frozen CSV-reference')
        csv_probability = model.predict(csv['prediction'], current_delay_s=csv['input_cur_dev_s'],
                                        telemetry_age_s=csv['telemetry_age_s'])
        if csv_probability is None:
            raise ValueError('У сравниваемой CSV-точки нет свежего входа')
        rows.append(dict(sample_id=gps['sample_id'], tr_id=gps['tr_id'], T=gps['T'],
                         gps_probability=probability, csv_probability=csv_probability,
                         gps_prediction_s=gps['prediction'], csv_prediction_s=csv['prediction']))
    return rows


def expected_calibration_error(metrics):
    return sum(row['rows'] * abs(row['mean_probability']-row['observed_frequency'])
               for row in metrics['reliability'] if row['rows']) / metrics['rows']


def cluster_improvement(rows, constant):
    """Положительное значение означает выигрыш GPS p над train-константой."""
    by_vehicle = []
    for _, group in rows.groupby('tr_id', sort=True):
        y = group['label'].to_numpy()
        difference = (y-constant)**2-(y-group['gps_probability'].to_numpy())**2
        by_vehicle.append((float(difference.sum()), len(difference)))
    rng = np.random.default_rng(RECIPE['bootstrap']['random_seed'])
    groups = np.asarray(by_vehicle)
    sample = rng.integers(0, len(groups), (RECIPE['bootstrap']['draws'], len(groups)))
    totals = groups[sample].sum(axis=1)
    improvements = totals[:, 0]/totals[:, 1]
    return dict(vehicles=len(groups), mean=float(improvements.mean()),
                p05=float(np.quantile(improvements, .05)),
                p95=float(np.quantile(improvements, .95)),
                method='Percentile cluster bootstrap by vehicle; repeated-use test remains diagnostic')


def evaluate(gps, reference, labels, artifact, out):
    started = time.monotonic()
    # Hashing label bytes does not parse labels and is allowed before prediction freeze.
    inputs = dict(gps_predictions=sha(gps/'predictions.jsonl'),
                  gps_contract=sha(gps/'contract.json'), gps_report=sha(gps/'report.json'),
                  csv_predictions=sha(reference/'predictions.jsonl'), labels=sha(labels),
                  probability=sha(artifact), code=sha(__file__),
                  probability_runtime_code=sha(ROOT/'ml_service/probability.py'))
    contract = dict(RECIPE, inputs_sha256=inputs, created_at=datetime.now(timezone.utc).isoformat())
    save(out/'contract.json', contract)
    model = ProbabilityModel.load(artifact, MODEL_SHA256)
    gps_rows, csv_rows = read_rows(gps/'predictions.jsonl'), read_rows(reference/'predictions.jsonl')
    frozen = freeze_predictions(gps_rows, csv_rows, model)
    save(out/'frozen-probabilities.json', frozen)
    predictions_sha = sha(out/'frozen-probabilities.json')
    targets = pd.read_csv(labels, usecols=['sample_id', 'target_delay_s'], dtype={'sample_id': str})
    if targets.sample_id.duplicated().any():
        raise ValueError('Дубли меток')
    rows = pd.DataFrame(frozen).merge(targets, on='sample_id', how='left', validate='one_to_one')
    if rows.empty or not np.isfinite(rows.target_delay_s).all():
        raise ValueError('Нет полного конечного набора меток для frozen предиктов')
    rows['label'] = (rows.target_delay_s > 120).astype(int)
    y = rows.label.to_numpy()
    constant = model.artifact.train_constant
    metrics = dict(gps=probability_metrics(y, rows.gps_probability),
                   csv_same_ids=probability_metrics(y, rows.csv_probability),
                   train_constant=probability_metrics(y, np.full(len(rows), constant)))
    for value in metrics.values():
        value['expected_calibration_error'] = expected_calibration_error(value)
        value['base_rate'] = float(y.mean())
    boot = cluster_improvement(rows, constant)
    gps_metric, base = metrics['gps'], metrics['train_constant']
    checks = dict(
        enough_rows=len(rows) >= RECIPE['acceptance']['min_rows'],
        enough_positives=int(y.sum()) >= RECIPE['acceptance']['min_positives'],
        enough_negatives=int((1-y).sum()) >= RECIPE['acceptance']['min_negatives'],
        enough_vehicles=boot['vehicles'] >= RECIPE['acceptance']['min_vehicles'],
        brier_better=gps_metric['brier'] < base['brier'],
        logloss_better=gps_metric['log_loss'] < base['log_loss'],
        bootstrap_improvement=boot['p05'] > 0,
        calibration_error=gps_metric['expected_calibration_error'] <= RECIPE['acceptance']['maximum_expected_calibration_error'],
    )
    if time.monotonic()-started > RECIPE['budget_s']:
        raise TimeoutError('Бюджет 60 с исчерпан')
    report = dict(status='accepted_secondary_diagnostic' if all(checks.values()) else 'rejected_transfer',
                  coverage=dict(all_rows=len(csv_rows), recorded_gps_predictions=len(gps_rows),
                                missing_gps_prediction_ids=sorted({r['sample_id'] for r in csv_rows}-{r['sample_id'] for r in gps_rows}),
                                evaluated=len(rows), fraction=len(rows)/len(csv_rows)),
                  metrics=metrics, bootstrap_brier_improvement=boot, checks=checks,
                  runtime_probability_changed=False, regression_unchanged=True,
                  independent_holdout=False, door_probability_validated=False,
                  frozen_predictions_sha256=predictions_sha,
                  contract_sha256=sha(out/'contract.json'), elapsed_s=time.monotonic()-started)
    save(out/'scored-probabilities.json', rows.to_dict(orient='records'))
    save(out/'report.json', report)
    return report


def export_scope(gps: Path, artifact: Path, out: Path, destination: Path):
    """Явная публикация допуска только после положительной HTTP-диагностики."""
    report = json.loads((out/'report.json').read_text())
    contract = json.loads((out/'contract.json').read_text())
    stream_contract = json.loads((gps/'contract.json').read_text())
    stream_report = json.loads((gps/'report.json').read_text())
    if report['status'] != 'accepted_secondary_diagnostic' or not all(report['checks'].values()):
        raise ValueError('GPS-перенос не прошёл все предварительно заданные gates')
    if (report['contract_sha256'] != sha(out/'contract.json')
        or contract['inputs_sha256']['gps_report'] != sha(gps/'report.json')
        or contract['inputs_sha256']['gps_contract'] != sha(gps/'contract.json')
        or contract['inputs_sha256']['gps_predictions'] != sha(gps/'predictions.jsonl')
        or contract['inputs_sha256']['probability'] != sha(artifact)):
        raise ValueError('Исходники диагностики изменились после проверки')
    if not stream_report.get('pipeline_checks_passed') or stream_report['deviation_source'] != 'gps':
        raise ValueError('Нужна успешная причинная GPS stream-проверка')
    if stream_report['model_card']['provenance']['sha256'] != MODEL_SHA256:
        raise ValueError('HTTP-поток использовал другую регрессионную модель')
    rows = read_rows(gps/'predictions.jsonl')
    if any(row.get('execution') != 'ml_http' for row in rows if row.get('prediction') is not None):
        raise ValueError('Допуск требует подтверждённой HTTP-цепочки')
    versions = {row['detector']['version'] for row in read_rows(gps/'outcomes.jsonl') if row.get('detector')}
    if len(versions) != 1:
        raise ValueError('Неоднозначная версия GPS-детектора')
    detector_hash = stream_contract['code_sha256']['backend/gps_arrivals.py']
    if detector_hash != sha(ROOT/'backend/gps_arrivals.py'):
        raise ValueError('GPS-детектор изменился после измерения')
    calibrated = ProbabilityModel.load(artifact,MODEL_SHA256)
    if stream_report['model_card']['probability']['artifact_sha256'] != calibrated.artifact_sha256:
        raise ValueError('HTTP-поток использовал другие коэффициенты вероятности')
    metric, baseline = report['metrics']['gps'],report['metrics']['train_constant']
    artifact_report_path = str((out/'report.json').relative_to(ROOT)) if out.is_relative_to(ROOT) else str(out/'report.json')
    scope = GPSProbabilityScope(schema_version='gps-probability-scope-v1',
        validation_status='accepted_secondary_diagnostic', scope='historical_real_gps_estimate',
        regression_model_sha256=MODEL_SHA256,probability_artifact_sha256=calibrated.artifact_sha256,
        detector_version=next(iter(versions)),detector_sha256=detector_hash,
        contract_sha256=sha(out/'contract.json'), report_sha256=sha(out/'report.json'),report_path=artifact_report_path,
        evaluated_rows=metric['rows'],total_rows=report['coverage']['all_rows'],positives=metric['positives'],
        vehicles=report['bootstrap_brier_improvement']['vehicles'],brier=metric['brier'],constant_brier=baseline['brier'],
        log_loss=metric['log_loss'],constant_log_loss=baseline['log_loss'],
        cluster_improvement_p05=report['bootstrap_brier_improvement']['p05'],
        expected_calibration_error=metric['expected_calibration_error'])
    save(destination,scope.model_dump())
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gps', type=Path, default=ROOT/'artifacts/gps-stream-http/gps')
    parser.add_argument('--reference', type=Path, default=ROOT/'artifacts/gps-stream-http/csv-reference')
    parser.add_argument('--labels', type=Path, default=ROOT.parent.parent/'data/raw/dataset/labels/labels_test.csv')
    parser.add_argument('--artifact', type=Path, default=ROOT/'artifacts/model/probability.json')
    parser.add_argument('--out', type=Path, default=ROOT/'artifacts/gps-probability/baseline')
    parser.add_argument('--export-scope', type=Path, help='После полного PASS явно записать GPS source-profile JSON')
    args = parser.parse_args()
    print(json.dumps(evaluate(args.gps, args.reference, args.labels, args.artifact, args.out), ensure_ascii=False, indent=2))
    if args.export_scope:
        print(export_scope(args.gps,args.artifact,args.out,args.export_scope))


if __name__ == '__main__':
    main()
