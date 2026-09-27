"""Один диагностический GPS-профиль: 50% frozen p + 50% ранней train-частоты.

Регрессия не обучается; её train-прогнозы in-sample. Ранний prior использует
только доступность GPS и метки до cutoff, поэтому OOF регрессии для prior не
требуется. Независимое качество frozen регрессии этим не доказывается.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.engine import Engine
from common.contracts import Telemetry
from ml_service.learned import LearnedModel
from ml_service.probability import ProbabilityModel
from tools.evaluate_stream_real import GPS_COLUMNS, load_inputs, parse_time
from tools.gps_probability_evaluate import cluster_improvement, expected_calibration_error, save, sha
from tools.prepare_probability import probability_metrics


RECIPE = dict(
    name='gps-probability-fixed-shrinkage-v1',
    candidate='p = 0.5 * existing_frozen_logistic_p + 0.5 * Laplace_early_GPS_train_frequency',
    cutoff='2026-01-06T12:00:00+00:00', earliest_point='2026-01-06T00:00:00+00:00',
    train='Real vehicles with provided test forecast points; train T < cutoff, actual target < cutoff; purge whole T if any label immature',
    test='Provided test with T >= cutoff, secondary known-day diagnostic',
    late_train='Separate diagnostic only: frozen regressor trained on these labels',
    available_at='plan+target lower bound; label receipt/finalization absent; NOT confirmed causal online training',
    acceptance=dict(min_early_rows=20, min_early_positives=5, min_early_negatives=5,
                    min_late_test_rows=50, min_late_test_positives=10,
                    min_late_test_negatives=10, min_vehicles=5,
                    brier_and_logloss_below_original_train_constant=True,
                    brier_and_logloss_not_worse_than_frozen_mapping=True,
                    cluster_brier_improvement_p05_above_zero=True, maximum_ece=.15),
    budget_s=180, no_threshold_search=True, variants=1, no_regression_fit=True,
    door_transfer=False, independent_holdout=False,
)


def sources():
    paths = [*sorted((ROOT/'backend').glob('*.py')), *sorted((ROOT/'common').glob('*.py')),
             Path(__file__), ROOT/'ml_service/learned.py', ROOT/'ml_service/probability.py']
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def replay_frozen_predictions(data: Path, model_path: Path, probability_path: Path, budget_started):
    # Input projection contains no cur_dev_s or future labels.
    base = load_inputs(data, deviation_source='gps')
    real_ids = {vehicle.tr_id for vehicle in base.context.vehicles}
    columns = ['sample_id', 'tr_id', 'T', 'target_stop_id', 'target_time_begin']
    frames = []
    for split in ('train', 'test'):
        frame = pd.read_csv(data/f'labels/labels_{split}.csv', usecols=columns, dtype={'sample_id':str, 'target_stop_id':str})
        frame = frame[frame.tr_id.isin(real_ids)].copy()
        frame['split'] = split
        frames.append(frame)
    points = pd.concat(frames, ignore_index=True)
    if points.sample_id.duplicated().any():
        raise ValueError('train/test IDs пересекаются')
    points['T'] = pd.to_datetime(points['T'], utc=True)
    groups = {at.to_pydatetime(): list(group.to_dict(orient='records'))
              for at, group in points.groupby('T', sort=True)}
    # Test traffic is explicitly real; train traffic adds synthetic rows. Here
    # all selected real vehicles are already present in test/traffic.csv.
    raw = pd.read_csv(data/'test/traffic.csv', usecols=list(GPS_COLUMNS))
    raw = raw[raw.tr_id.isin(real_ids)]
    telemetry = []
    for row in raw.itertuples(index=False):
        def nullable(value):
            return None if pd.isna(value) else float(value)
        try:
            telemetry.append(Telemetry(tr_id=int(row.tr_id), unit_id=int(row.unit_id),
                event_time=parse_time(row.event_time), received_at=parse_time(row.receive_time),
                lat=nullable(row.lat), lon=nullable(row.lon), speed_kmh=nullable(row.speed),
                heading=nullable(row.heading), location_valid=str(row.location_valid).lower()=='true',
                source='replay', event_id=str(row.packet_id)))
        except ValueError:
            continue
    telemetry.sort(key=lambda point:(max(point.event_time,point.received_at),point.event_time,point.event_id or ''))
    engine = Engine('unused')
    engine.set_live(base.context)
    engine.mode = 'replay'
    engine.running = False
    cursor = 0
    requests, rows = [], []
    for at, samples in groups.items():
        if time.monotonic()-budget_started > RECIPE['budget_s']:
            raise TimeoutError('GPS replay превысил бюджет')
        while cursor < len(telemetry) and max(telemetry[cursor].event_time,telemetry[cursor].received_at) <= at:
            point = telemetry[cursor]
            engine.clock = max(point.event_time,point.received_at)
            engine.ingest(point)
            cursor += 1
        engine.clock = at
        for sample in samples:
            request = engine.request_for(int(sample['tr_id']), at)
            reason = None
            if request is None:
                reason = 'no_target'
            elif request.target.id != str(sample['target_stop_id']) or request.target.scheduled_at != parse_time(sample['target_time_begin']):
                reason = 'target_mismatch'
            elif request.current_delay_s is None or request.current_delay_source != 'gps_estimate':
                reason = 'no_gps_deviation'
            elif request.features.telemetry_age_s is None or request.features.telemetry_age_s > 60:
                reason = 'stale_gps'
            row = {**sample, 'T': at.isoformat(), 'status': reason or 'available'}
            rows.append(row)
            if reason is None:
                current = engine.deviation_at(sample['tr_id'], at)
                assert current and current.received_at <= at and current.observed_at <= at
                row.update(current_delay_s=request.current_delay_s, source=request.current_delay_source,
                           telemetry_age_s=request.features.telemetry_age_s)
                requests.append((request, row))
    learned = LearnedModel(model_path)
    probability = ProbabilityModel.load(probability_path, learned.sha256)
    for offset in range(0, len(requests), 128):
        chunk = requests[offset:offset+128]
        values = learned.predict_batch([request for request, _ in chunk])
        for (request, row), value in zip(chunk, values, strict=True):
            if value.method != 'learned':
                row['status'] = 'model_fallback'
                continue
            row.update(prediction_s=value.predicted_delay_s, frozen_probability=probability.predict(
                value.predicted_delay_s, current_delay_s=request.current_delay_s,
                telemetry_age_s=request.features.telemetry_age_s))
    return rows, probability, dict(vehicles=len(real_ids), raw_messages=len(telemetry), delivered=cursor,
                                  total_points=len(points), detector_versions=sorted({v.status()['version'] for v in engine.gps_detectors.values()}))


def mature_prior(train_targets, frozen_rows, cutoff):
    """Не читает поздние/test файлы; поздние и незрелые train строки не входят в fit."""
    labels = train_targets.copy()
    labels['T'] = pd.to_datetime(labels['T'], utc=True)
    labels = labels[labels['T'] < cutoff].copy()
    labels['earliest_label_at'] = pd.to_datetime(labels.target_time_begin, utc=True)+pd.to_timedelta(labels.target_delay_s, unit='s')
    immature = labels.earliest_label_at.isna() | (labels.earliest_label_at >= cutoff)
    excluded_times = set(labels.loc[immature, 'T'])
    purged = labels[labels['T'].isin(excluded_times)]
    mature = labels[~labels['T'].isin(excluded_times)]
    available = frozen_rows[(frozen_rows.split=='train') & (frozen_rows.status=='available')]
    selected = mature.merge(available[['sample_id']], on='sample_id', validate='one_to_one')
    if not np.isfinite(selected.target_delay_s).all():
        raise ValueError('Неконечные метки train')
    positives = int((selected.target_delay_s > 120).sum())
    prior = (positives+1)/(len(selected)+2)
    report = dict(prior=prior, rows=len(selected), positives=positives, negatives=len(selected)-positives,
                  label_availability_mode='arrival_lower_bound_only',
                  early_candidates=len(labels), immature_rows=int(immature.sum()),
                  purged_rows=len(purged), purged_times=len(excluded_times),
                  latest_earliest_label_at=selected.earliest_label_at.max().isoformat() if len(selected) else None,
                  selected_sample_ids=selected.sample_id.tolist(), purged_sample_ids=purged.sample_id.tolist())
    return prior, report


def score(frozen_rows, labels, prior, old_constant, cutoff, split):
    available = frozen_rows[(frozen_rows.split==split)&(frozen_rows.status=='available')].copy()
    available = available[pd.to_datetime(available['T'], utc=True) >= cutoff]
    rows = available.merge(labels[['sample_id','target_delay_s']], on='sample_id', validate='one_to_one')
    if rows.empty:
        return dict(rows=0, metrics=None, bootstrap=None), rows
    rows['label'] = (rows.target_delay_s > 120).astype(int)
    rows['gps_probability'] = .5*rows.frozen_probability+.5*prior
    y = rows.label.to_numpy()
    values = dict(shrinkage=rows.gps_probability, frozen=rows.frozen_probability,
                  old_train_constant=np.full(len(rows),old_constant), early_gps_prior=np.full(len(rows),prior))
    metrics = {name:probability_metrics(y,value) for name,value in values.items()}
    for metric in metrics.values():
        metric['expected_calibration_error'] = expected_calibration_error(metric)
    return dict(rows=len(rows), metrics=metrics, bootstrap=cluster_improvement(rows,old_constant)), rows


def run(data, out, model_path, probability_path):
    started = time.monotonic()
    source_before = sources()
    contract = dict(RECIPE, code_sha256=source_before, inputs_sha256={
        name:sha(data/name) for name in ('test/traffic.csv','validate/schedule_plan.csv','labels/labels_train.csv','labels/labels_test.csv')},
        model_sha256=sha(model_path), probability_sha256=sha(probability_path))
    save(out/'contract.json',contract)
    frozen, model, population = replay_frozen_predictions(data,model_path,probability_path,started)
    save(out/'frozen-predictions.json',frozen)
    frozen_rows = pd.DataFrame(frozen)
    # Regression seconds and raw probabilities already immutable before targets.
    train = pd.read_csv(data/'labels/labels_train.csv',dtype={'sample_id':str})
    train = train[train.tr_id.isin({row['tr_id'] for row in frozen})]
    cutoff = pd.Timestamp(RECIPE['cutoff'])
    prior, training = mature_prior(train,frozen_rows,cutoff)
    save(out/'frozen-prior.json',dict(training, frozen_predictions_sha256=sha(out/'frozen-predictions.json')))
    test = pd.read_csv(data/'labels/labels_test.csv',dtype={'sample_id':str})
    late_test, scored_test = score(frozen_rows,test,prior,model.artifact.train_constant,cutoff,'test')
    late_train, scored_train = score(frozen_rows,train,prior,model.artifact.train_constant,cutoff,'train')
    metrics, boot = late_test['metrics'], late_test['bootstrap']
    checks = dict(early_rows=training['rows'] >= 20, early_positives=training['positives'] >= 5,
                  early_negatives=training['negatives'] >= 5, late_rows=late_test['rows'] >= 50)
    if metrics:
        new, original, base = metrics['shrinkage'],metrics['frozen'],metrics['old_train_constant']
        checks.update(late_positives=new['positives'] >= 10, late_negatives=new['rows']-new['positives'] >= 10,
            vehicles=boot['vehicles'] >= 5, brier_better=new['brier'] < base['brier'],
            logloss_better=new['log_loss'] < base['log_loss'], brier_no_worse=new['brier'] <= original['brier'],
            logloss_no_worse=new['log_loss'] <= original['log_loss'], cluster_improvement=boot['p05'] > 0,
            calibration_error=new['expected_calibration_error'] <= .15)
    unchanged_sources = source_before == sources()
    if not unchanged_sources:
        raise RuntimeError('Исходники изменились во время эксперимента; повторить после фиксации')
    if time.monotonic()-started > RECIPE['budget_s']:
        raise TimeoutError('Полный эксперимент превысил бюджет')
    report = dict(status='accepted_secondary_diagnostic' if all(checks.values()) else 'rejected_transfer',
                  population=population, training=training, late_test=late_test, late_train_in_sample=late_train,
                  checks=checks, regression_fit=False, runtime_changed=False, independent_holdout=False,
                  sources_unchanged=unchanged_sources, elapsed_s=time.monotonic()-started,
                  contract_sha256=sha(out/'contract.json'), frozen_prior_sha256=sha(out/'frozen-prior.json'))
    save(out/'scored-test.json',scored_test.to_dict(orient='records'))
    save(out/'scored-train.json',scored_train.to_dict(orient='records'))
    save(out/'report.json',report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT.parent.parent/'data/raw/dataset')
    parser.add_argument('--out',type=Path,default=ROOT/'artifacts/gps-probability/shrinkage')
    parser.add_argument('--model',type=Path,default=ROOT/'artifacts/model/model.joblib')
    parser.add_argument('--probability',type=Path,default=ROOT/'artifacts/model/probability.json')
    args = parser.parse_args()
    report = run(args.data,args.out,args.model,args.probability)
    print(json.dumps({key:report[key] for key in ('status','population','checks','elapsed_s')},ensure_ascii=False,indent=2))
    print(args.out/'report.json')


if __name__=='__main__':
    main()
