"""Диагностика фиксированного GPS на размеченных посещениях, без подбора.

Детектор получает только пять плановых полей и доступную GPS-телеметрию.
Перед детекцией из labels читаются только tr_id для выбора 13 автобусов.
После неё отдельный evaluator читает метки. Другие предсказанные посещения
не считаются false positive: разметка посещений в labels частичная.

Из filipp/: .venv/bin/python tools/evaluate_gps_real.py \
    --data ../../data/raw/dataset --out artifacts/gps-real --budget-seconds 60
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import csv
import hashlib
import json
from pathlib import Path
import platform
import re
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.gps_arrivals import GPSArrivalDetector, GPSDetectorConfig  # noqa: E402
from common.contracts import StopTarget, Telemetry  # noqa: E402

PLAN_FIELDS = ('tt_action_item_id', 'tr_id', 'time_begin', 'geom', 'building_address')
NAV_FIELDS = ('tr_id', 'unit_id', 'event_time', 'receive_time', 'lat', 'lon', 'speed',
              'heading', 'location_valid', 'packet_id')
NUMBER = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
POINT = re.compile(rf'POINT\s*\(\s*({NUMBER})\s+({NUMBER})\s*\)', re.IGNORECASE)


def projected_rows(path: Path, fields):
    """Проекция при чтении: поля фактов не извлекаются из строк расписания."""
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream)
        header = next(reader)
        if len(header) != len(set(header)):
            raise ValueError(f'Повтор поля: {path}')
        indexes = [header.index(name) for name in fields]
        for row in reader:
            yield {field: row[index] for field, index in zip(fields, indexes)}


def dt(value):
    result = datetime.fromisoformat(value)
    # Допущение ранее принятого replay; организаторы пояс naive CSV не подтвердили.
    return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result


def describe(values):
    if not values:
        return {'n': 0, 'mean_s': None, 'median_s': None, 'p95_s': None, 'min_s': None, 'max_s': None}
    ordered = sorted(values)
    return {'n': len(values), 'mean_s': statistics.fmean(values), 'median_s': statistics.median(values),
            'p95_s': ordered[min(len(values)-1, int(.95*len(values)))],
            'min_s': ordered[0], 'max_s': ordered[-1]}


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def evaluate(data: Path, out: Path, budget_seconds=60.0, split='test', cohort='all'):
    started = time.perf_counter()
    deadline = started+budget_seconds
    root = Path(__file__).resolve().parents[1]
    files = {'traffic': data/split/'traffic.csv', 'plan': data/split/'schedule.csv',
             'labels': data/f'labels/labels_{split}.csv', 'script': Path(__file__),
             'detector': root/'backend/gps_arrivals.py', 'contracts': root/'common/contracts.py'}
    provenance = {key: {'path': str(path.resolve()), 'sha256': sha(path), 'bytes': path.stat().st_size}
                  for key, path in files.items()}
    def check_budget():
        if time.perf_counter() > deadline:
            raise TimeoutError(f'Диагностика превысила бюджет {budget_seconds} с; результат не сохраняется')
    ids = {int(row['tr_id']) for row in projected_rows(files['labels'], ('tr_id',))
           if int(row['tr_id']) < 9_000_000}
    plans = defaultdict(list)
    for row in projected_rows(files['plan'], PLAN_FIELDS):
        tr_id = int(row['tr_id'])
        if tr_id not in ids:
            continue
        match = POINT.fullmatch(row['geom'].strip())
        if not match:
            raise ValueError('План содержит неподдерживаемый WKT')
        lon, lat = map(float, match.groups())
        plans[tr_id].append(StopTarget(id=row['tt_action_item_id'], name=row['building_address'] or row['tt_action_item_id'],
                                     scheduled_at=dt(row['time_begin']), lat=lat, lon=lon))
    config = GPSDetectorConfig()
    detectors = {tr_id: GPSArrivalDetector(plans[tr_id], config) for tr_id in sorted(ids)}
    plan_by_key = {(tr_id, stop.id): stop for tr_id, stops in plans.items() for stop in stops}
    telemetry, counters = [], Counter()
    for row in projected_rows(files['traffic'], NAV_FIELDS):
        counters['traffic_rows_total'] += 1
        tr_id = int(row['tr_id'])
        if tr_id not in ids:
            continue
        counters['traffic_rows_selected'] += 1
        event, received = dt(row['event_time']), dt(row['receive_time'])
        if received < event:
            # Детектор отклоняет такие сообщения. Не исправляем времена молча.
            counters['receive_before_event_skipped'] += 1
            continue
        try:
            point = Telemetry(tr_id=tr_id, unit_id=int(row['unit_id']), event_time=event, received_at=received,
                lat=float(row['lat']) if row['lat'] else None, lon=float(row['lon']) if row['lon'] else None,
                speed_kmh=float(row['speed']) if row['speed'] else None,
                heading=float(row['heading']) if row['heading'] else None,
                location_valid=row['location_valid'].lower() == 'true', source='replay', event_id=row['packet_id'])
        except ValueError:
            counters['invalid_contract_skipped'] += 1
            continue
        telemetry.append((max(event, received), counters['traffic_rows_total'], point))
    # Stable source-row order for equal availability; no reordering by future events.
    telemetry.sort(key=lambda item: (item[0], item[1]))
    detected, reasons, per_vehicle = {}, Counter(), defaultdict(list)
    for index, (available, _, point) in enumerate(telemetry):
        if index % 1000 == 0:
            check_budget()
        assert point.event_time <= available and point.received_at <= available
        detector = detectors[point.tr_id]
        result = detector.observe(point)
        reasons[detector.status()['reason']] += 1
        if result:
            assert result.received_at <= available and result.arrived_at <= result.received_at
            key = (result.tr_id, result.planned_stop_id)
            if key in detected:
                raise AssertionError('Детектор повторно выдал посещение')
            detected[key] = result
            per_vehicle[result.tr_id].append(result)
    prediction_completed_after_s = time.perf_counter()-started

    # Только сейчас читаются времена целей/метки/cur_dev_s для независимой оценки.
    label_fields = ('sample_id', 'tr_id', 'T', 'target_stop_id', 'target_time_begin', 'target_delay_s', 'cur_dev_s')
    label_rows = [row for row in projected_rows(files['labels'], label_fields)
                  if int(row['tr_id']) in ids
                  and (cohort != 'calibration' or dt(row['T']) < dt('2026-01-06T11:45:00Z'))
                  and (cohort != 'holdout' or dt(row['T']) >= dt('2026-01-06T12:15:00Z'))]
    label_keys, comparisons, deviations = set(), [], []
    arrival_errors, confirmation_lags, own_confirmation = [], [], []
    known_cur = []
    for row in label_rows:
        tr_id, cutoff = int(row['tr_id']), dt(row['T'])
        key = (tr_id, row['target_stop_id'])
        if key in label_keys:
            raise ValueError('Повтор labeled посещения: требуется отдельно агрегировать метрики')
        label_keys.add(key)
        scheduled = dt(row['target_time_begin'])
        if key not in plan_by_key or plan_by_key[key].scheduled_at != scheduled:
            raise ValueError('Label не совпал с планом по ID/времени')
        actual = scheduled+timedelta(seconds=float(row['target_delay_s']))
        match = detected.get(key)
        record = dict(sample_id=row['sample_id'], tr_id=tr_id, planned_stop_id=key[1],
                      actual_arrival_at=actual.isoformat(), detected_same_visit=bool(match))
        if match:
            error = (match.arrived_at-actual).total_seconds()
            lag = (match.received_at-actual).total_seconds()
            arrival_errors.append(abs(error))
            confirmation_lags.append(lag)
            own_confirmation.append((match.received_at-match.arrived_at).total_seconds())
            record.update(estimated_arrival_at=match.arrived_at.isoformat(), available_at=match.received_at.isoformat(),
                          signed_error_s=error, confirmation_lag_from_truth_s=lag)
        comparisons.append(record)
        available_predictions = [event for event in per_vehicle[tr_id]
                                 if event.received_at <= cutoff and event.arrived_at <= cutoff]
        latest = max(available_predictions, key=lambda event: event.arrived_at, default=None)
        deviation_record = dict(sample_id=row['sample_id'], tr_id=tr_id, cutoff=cutoff.isoformat(), known=latest is not None)
        if latest:
            value = (latest.arrived_at-plan_by_key[(tr_id, latest.planned_stop_id)].scheduled_at).total_seconds()
            reference = float(row['cur_dev_s'])
            error = abs(value-reference)
            known_cur.append(error)
            deviation_record.update(estimated_delay_s=value, supplied_cur_dev_s=reference,
                                    absolute_error_s=error, based_on_visit=latest.planned_stop_id,
                                    known_since=latest.received_at.isoformat())
        deviations.append(deviation_record)
    check_budget()
    for key, path in files.items():
        if sha(path) != provenance[key]['sha256']:
            raise RuntimeError(f'Исходник изменился во время проверки: {key}; повторите запуск')
    report = {
        'purpose': f'Диагностика фиксированного GPS на частично размеченных {split}-посещениях ({cohort}), не новый независимый день',
        'created_at': datetime.now(timezone.utc).isoformat(), 'runtime_python': platform.python_version(),
        'elapsed_s': time.perf_counter()-started, 'prediction_completed_after_s': prediction_completed_after_s,
        'experiment': {'change': f'Применить фиксированный {GPSArrivalDetector.version} к реальной GPS',
            'baseline': 'Нет отдельного GPS baseline; cur_dev_s сравнивается только как оценочный reference после детекции',
            'split': f'Выданные {split}/traffic.csv + schedule.csv, реальные ТС из labels_{split}, cohort={cohort}; не новый день',
            'acceptance': 'Измерение покрытия/ошибок без подбора порогов; это диагностика, не условие production-ready',
            'budget_seconds': budget_seconds, 'fit_or_tuning': False},
        'sources': provenance, 'detector_version': GPSArrivalDetector.version, 'thresholds': asdict(config),
        'causality': {'plan_fields': list(PLAN_FIELDS), 'never_used_as_input': ['time_fact_begin', 'target_delay_s', 'cur_dev_s'],
            'ordering': 'max(event_time, receive_time), затем исходный номер строки',
            'receive_before_event': 'отклоняются, считаются отдельно', 'naive_timezone_assumption': 'UTC',
            'ground_truth': f'только labels_{split}: target_time_begin + target_delay_s после вычисления предсказаний'},
        'vehicles': sorted(ids), 'counts': dict(counters),
        'plan_visits': len(plan_by_key), 'processed_messages': len(telemetry),
        'first_available_at': telemetry[0][0].isoformat() if telemetry else None,
        'last_available_at': telemetry[-1][0].isoformat() if telemetry else None,
        'labeled_visits': len(label_keys), 'detected_visits_total': len(detected),
        'detected_labeled_same_id': len(arrival_errors),
        'labeled_same_id_fraction': len(arrival_errors)/len(label_keys) if label_keys else None,
        'labeled_matches_within_s': {str(tolerance): sum(error <= tolerance for error in arrival_errors)
                                    for tolerance in (30, 60, 120)},
        'labeled_errors_over_900s': sum(error > 900 for error in arrival_errors),
        'arrival_absolute_error': describe(arrival_errors),
        'confirmation_lag_from_actual_signed': describe(confirmation_lags),
        'confirmed_before_ground_truth_count': sum(lag < 0 for lag in confirmation_lags),
        'confirmation_after_own_estimate': describe(own_confirmation),
        'other_detected_visits_unassessable': len(set(detected)-label_keys),
        'overall_precision': None,
        'current_deviation_at_T': {'known': len(known_cur), 'unknown': len(label_rows)-len(known_cur),
            'known_fraction': len(known_cur)/len(label_rows) if label_rows else None,
            'absolute_error_on_known_only': describe(known_cur),
            'note': 'Последнее доступное GPS-прибытие; cur_dev_s из labels используется только при сравнении'},
        'detector_observation_reasons': dict(reasons),
        'limitations': [
            'Labels содержат только выбранные будущие посещения, не полную разметку всех остановок. Общий precision неизвестен.',
            'Совпадение ID при большой ошибке времени может означать неверную привязку; смотрите окна 30/60/120 секунд.',
            'UTC для naive времени — допущение. Точные остановки и надежность фактической разметки независимо не подтверждены.',
            'Один день/те же ТС; это не статистически независимая проверка и не доказательство раннего предупреждения.',
            'Подбор параметров по этому отчету сделает test калибровочным; требуется новый независимый набор.',
            'Known current deviation означает наличие доступного estimate, а не его правильность или свежесть GPS.',
        ],
    }
    out.mkdir(parents=True, exist_ok=True)
    (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (out/'labeled-visits.json').write_text(json.dumps(comparisons, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (out/'current-deviation.json').write_text(json.dumps(deviations, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (out/'predictions.json').write_text(json.dumps([asdict(event) for event in detected.values()],
        default=lambda value: value.isoformat(), ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('../../data/raw/dataset'))
    parser.add_argument('--out', type=Path, default=Path('artifacts/gps-real'))
    parser.add_argument('--budget-seconds', type=float, default=60.0)
    parser.add_argument('--split', choices=('train', 'test'), default='test')
    parser.add_argument('--cohort', choices=('all', 'calibration', 'holdout'), default='all')
    args = parser.parse_args()
    report = evaluate(args.data, args.out, args.budget_seconds, args.split, args.cohort)
    print(json.dumps({key: report[key] for key in ('elapsed_s', 'labeled_visits', 'detected_labeled_same_id',
        'labeled_matches_within_s', 'arrival_absolute_error', 'other_detected_visits_unassessable',
        'current_deviation_at_T')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
