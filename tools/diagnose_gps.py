"""Аудит согласованности GPS/плана/выданных фактов ПОСЛЕ детекции.

Не обучает детектор и не меняет его настройки. Читает уже сохранённые прогнозы,
затем факты и метки только как оценочные reference. Полезен и при отрицательном
результате: отличает отсутствие наблюдений от неверного сопоставления визита.

Из filipp/: .venv/bin/python tools/diagnose_gps.py --data ../../data/raw/dataset
"""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import csv
from datetime import timedelta
import json
import math
from pathlib import Path
import statistics
import time

from evaluate_gps_real import dt, sha, POINT


def describe(values):
    values = sorted(values)
    return {'n': len(values), 'median': statistics.median(values) if values else None,
            'mean': statistics.fmean(values) if values else None,
            'p95': values[min(len(values)-1, int(.95*len(values)))] if values else None,
            'max': values[-1] if values else None}


def distance(point, lat, lon):
    return math.hypot((point['lat']-lat)*111195,
                      (point['lon']-lon)*111195*math.cos(math.radians(lat)))


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        yield from csv.DictReader(stream)


def run(data, predictions, output, budget):
    started = time.monotonic()
    # Файл предсказаний фиксируется до загрузки evaluation-only truth.
    predicted = json.loads(predictions.read_text())
    paths = {'predictions': predictions, 'traffic': data/'test/traffic.csv',
             'plan_and_evaluation_facts': data/'test/schedule.csv',
             'train_plan_for_overlap_only': data/'train/schedule.csv',
             'train_traffic_for_overlap_only': data/'train/traffic.csv',
             'labels_evaluation_only': data/'labels/labels_test.csv',
             'script': Path(__file__)}
    provenance = {key: {'path': str(path.resolve()), 'sha256': sha(path)} for key, path in paths.items()}
    plans = {row['tt_action_item_id']: row for row in rows(paths['plan_and_evaluation_facts'])}
    labels = list(rows(paths['labels_evaluation_only']))
    ids = {row['tr_id'] for row in labels}
    gps, all_messages = defaultdict(list), defaultdict(list)
    invalid, receive_before_event = Counter(), []
    for row in rows(paths['traffic']):
        if row['tr_id'] not in ids:
            continue
        point = {'tr_id': row['tr_id'], 'event': dt(row['event_time']),
                 'receipt': dt(row['receive_time']), 'lat': float(row['lat']) if row['lat'] else None,
                 'lon': float(row['lon']) if row['lon'] else None,
                 'speed': float(row['speed']) if row['speed'] else None,
                 'valid': row['location_valid'].lower() == 'true'}
        all_messages[row['tr_id']].append(point)
        if point['receipt'] < point['event']:
            receive_before_event.append((point['event']-point['receipt']).total_seconds())
        if point['valid'] and point['lat'] is not None and point['lon'] is not None:
            gps[row['tr_id']].append(point)
        else:
            invalid['invalid_or_missing_coordinates'] += 1
    for values in gps.values():
        values.sort(key=lambda point: point['event'])
    times = {tr: [point['event'] for point in values] for tr, values in gps.items()}
    by_visit = {event['planned_stop_id']: event for event in predicted}
    def near(tr, timestamp, window=60):
        values, stamps = gps[tr], times[tr]
        lo, hi = bisect_left(stamps, timestamp-timedelta(seconds=window)), bisect_right(stamps, timestamp+timedelta(seconds=window))
        selected = values[lo:hi]
        index = bisect_left(stamps, timestamp)
        candidates = values[max(0, index-1):index+1]
        closest = min(candidates, key=lambda p: abs((p['event']-timestamp).total_seconds())) if candidates else None
        return selected, closest

    comparisons = []
    for label in labels:
        plan = plans[label['target_stop_id']]
        lon, lat = map(float, POINT.fullmatch(plan['geom']).groups())
        actual = dt(label['target_time_begin'])+timedelta(seconds=float(label['target_delay_s']))
        selected, closest = near(label['tr_id'], actual)
        slow = [p for p in selected if p['speed'] is not None and p['speed'] <= 5 and distance(p,lat,lon) <= 35]
        # Возможность наблюдения, не oracle-признак: известна только оценщику.
        distinct_slow = {point['event'] for point in slow}
        matched = by_visit.get(label['target_stop_id'])
        record = dict(sample_id=label['sample_id'], tr_id=int(label['tr_id']),
            planned_stop_id=label['target_stop_id'], address=plan['building_address'],
            manual_fill=plan['manual_fill'], actual_at=actual.isoformat(),
            label_equals_schedule_fact=bool(plan['time_fact_begin']) and actual == dt(plan['time_fact_begin']),
            gps_within_60s=len(selected), distinct_stopped_gps_in_35m=len(distinct_slow),
            nearest_gps_gap_s=abs((closest['event']-actual).total_seconds()) if closest else None,
            distance_nearest_time_gps_m=distance(closest,lat,lon) if closest else None,
            min_distance_within_60s_m=min((distance(p,lat,lon) for p in selected),default=None),
            detector_same_visit=matched is not None,
            estimated_error_s=(dt(matched['arrived_at'])-actual).total_seconds() if matched else None)
        comparisons.append(record)

    full_fact_errors, errors_by_manual = [], defaultdict(list)
    unassessable, predicted_raw_delay, delay_age, confirmation_age = [], [], [], []
    full_comparisons = []
    for event in predicted:
        plan = plans.get(event['planned_stop_id'])
        if plan is None or not plan['time_fact_begin']:
            unassessable.append(event['planned_stop_id'])
            continue
        raw_delay = (dt(event['arrived_at'])-dt(plan['time_begin'])).total_seconds()
        error = (dt(event['arrived_at'])-dt(plan['time_fact_begin'])).total_seconds()
        predicted_raw_delay.append(abs(raw_delay))
        full_fact_errors.append(abs(error))
        errors_by_manual[plan['manual_fill']].append(abs(error))
        full_comparisons.append(dict(tr_id=event['tr_id'],planned_stop_id=event['planned_stop_id'],
            planned_at=plan['time_begin'], supplied_fact_at=plan['time_fact_begin'],
            estimated_arrival_at=event['arrived_at'], signed_error_s=error,
            predicted_delay_s=raw_delay, manual_fill=plan['manual_fill']))

    deviations = []
    for label in labels:
        timestamp = dt(label['T'])
        available = [event for event in predicted if event['tr_id'] == int(label['tr_id'])
                     and dt(event['received_at']) <= timestamp and dt(event['arrived_at']) <= timestamp]
        latest = max(available,key=lambda event:dt(event['arrived_at']), default=None)
        if latest:
            # Та же граница, что у Engine.deviation_at: TTL от оценённого
            # прибытия observed_at, а не от более позднего подтверждения.
            age = (timestamp-dt(latest['arrived_at'])).total_seconds()
            estimate = (dt(latest['arrived_at'])-dt(plans[latest['planned_stop_id']]['time_begin'])).total_seconds()
            delay_age.append(age)
            confirmation_age.append((timestamp-dt(latest['received_at'])).total_seconds())
            deviations.append(dict(age_s=age,error_s=abs(estimate-float(label['cur_dev_s']))))
    # Не выбираем TTL по этим результатам: показываем, как отличаются coverage/MAE.
    ttl_sensitivity = [{'ttl_s': ttl, 'known': sum(row['age_s'] <= ttl for row in deviations),
        'total_labels':len(labels),'mae_on_known_s':statistics.fmean([row['error_s'] for row in deviations if row['age_s']<=ttl])
        if any(row['age_s']<=ttl for row in deviations) else None} for ttl in (60,120,300,600)]

    # Относительное совпадение часов, а не доказательство абсолютной timezone.
    offset_check = {}
    for seconds in (-10800,0,10800):
        distances=[]
        for row in labels:
            plan=plans[row['target_stop_id']];lon,lat=map(float,POINT.fullmatch(plan['geom']).groups())
            timestamp=dt(row['target_time_begin'])+timedelta(seconds=float(row['target_delay_s'])+seconds)
            selected,_=near(row['tr_id'],timestamp)
            if selected:
                distances.append(min(distance(p,lat,lon) for p in selected))
        offset_check[str(seconds)]={'with_gps_within60s':len(distances),'min_distance_m':describe(distances)}

    duplicate_times=0
    event_intervals=[]
    for tr, points in gps.items():
        stamps=sorted(set(p['event'] for p in points));duplicate_times+=len(points)-len(stamps)
        event_intervals.extend((b-a).total_seconds() for a,b in zip(stamps,stamps[1:]))
    repeated_coordinates=Counter()
    for plan in plans.values():
        repeated_coordinates[(plan['tr_id'],plan['geom'])]+=1

    plan_sets=[]
    traffic_sets=[]
    for split in ('train','test'):
        plan_sets.append({(r['tr_id'],r['tt_action_item_id'],dt(r['time_begin']).isoformat(),r['geom'])
                          for r in rows(data/split/'schedule.csv') if int(r['tr_id'])<9_000_000})
        traffic_sets.append({(r['tr_id'],r['event_time'],r['lat'],r['lon'],r['speed'])
                             for r in rows(data/split/'traffic.csv') if int(r['tr_id'])<9_000_000})
    overlap={'real_plan_unique_train':len(plan_sets[0]),'real_plan_unique_test':len(plan_sets[1]),
             'real_plan_intersection':len(plan_sets[0]&plan_sets[1]),
             'gps_projection_unique_train':len(traffic_sets[0]),'gps_projection_unique_test':len(traffic_sets[1]),
             'gps_projection_exact_intersection':len(traffic_sets[0]&traffic_sets[1]),
             'gps_projection':['tr_id','event_time','lat','lon','speed'],
             'note':'Строковое совпадение GPS может недооценивать overlap из-за форматирования наносекунд; это не новые дни.'}

    support=Counter()
    for record in comparisons:
        if not record['gps_within_60s']:
            support['no_valid_gps_within60s']+=1
        elif record['min_distance_within_60s_m']>70:
            support['all_gps_further70m_within60s']+=1
        elif record['distinct_stopped_gps_in_35m']<2:
            support['fewer2_distinct_slow_points35m']+=1
        else:
            support['at_least2_distinct_slow_points35m']+=1

    report={'purpose':'Post-prediction forensic diagnostics, no fitting or threshold selection',
        'elapsed_s':time.monotonic()-started,'budget_seconds':budget,'sources':provenance,
        'facts_only_in_evaluation':True,'labels':len(labels),'label_equals_schedule_fact':sum(r['label_equals_schedule_fact'] for r in comparisons),
        'label_observability_partition':dict(support),
        'nearest_gps_time_gap_s':describe([r['nearest_gps_gap_s'] for r in comparisons if r['nearest_gps_gap_s'] is not None]),
        'nearest_time_gps_distance_m':describe([r['distance_nearest_time_gps_m'] for r in comparisons if r['distance_nearest_time_gps_m'] is not None]),
        'relative_clock_offset_sensitivity':offset_check,
        'gps_event_interval_s':describe(event_intervals),'repeated_valid_event_times':duplicate_times,
        'invalid_messages':dict(invalid),'receive_before_event_s':describe(receive_before_event),
        'physical_coordinate_groups':len(repeated_coordinates),'groups_repeated':sum(n>1 for n in repeated_coordinates.values()),
        'train_test_overlap':overlap,
        'predictions_with_schedule_fact':len(full_fact_errors),'predictions_without_fact':len(unassessable),
        'full_schedule_fact_mae_s':describe(full_fact_errors),
        'full_schedule_errors_by_manual_fill':{k:describe(v) for k,v in errors_by_manual.items()},
        'max_absolute_predicted_deviation_s':max(predicted_raw_delay,default=None),
        'known_deviation_confirmation_age_s':describe(confirmation_age),
        'known_deviation_observed_age_s':describe(delay_age),
        'ttl_basis':'T minus estimated arrival (observed_at), inclusive <= TTL; same as engine GPS snapshot policy',
        'ttl_diagnostic_not_selection':ttl_sensitivity,
        'worst_labeled_geometry_cases':sorted(comparisons,key=lambda r:r['distance_nearest_time_gps_m'] or 0,reverse=True)[:12],
        'worst_detected_time_cases':sorted(full_comparisons,key=lambda r:abs(r['signed_error_s']),reverse=True)[:12],
        'limitations':['Schedule fact is supplied reference, not independently verified passenger-stop arrival.',
            'No GPS near a supplied fact may mean missing coverage, incorrect mapping or incorrect fact; diagnostic does not assign a cause without evidence.',
            'Nearby stationary GPS does not distinguish terminal layover, next departure, boarding or traffic light.',
            'Offset comparison supports relative time consistency; UTC vs Moscow absolute timezone still requires organizer confirmation.',
            'The test day is already inspected. These reports are diagnostic, not untouched generalisation evidence.']}
    if report['elapsed_s']>budget:
        raise TimeoutError('Diagnostic budget exceeded')
    for key,path in paths.items():
        if sha(path)!=provenance[key]['sha256']:
            raise RuntimeError(f'Changed source during diagnostic: {key}')
    output.mkdir(parents=True,exist_ok=True)
    (output/'data-diagnostic.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (output/'visit-support.json').write_text(json.dumps(comparisons,ensure_ascii=False,indent=2)+'\n')
    (output/'full-schedule-comparison.json').write_text(json.dumps(full_comparisons,ensure_ascii=False,indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('../../data/raw/dataset'))
    parser.add_argument('--predictions',type=Path,default=Path('artifacts/gps-diagnosis/v2-test/predictions.json'))
    parser.add_argument('--out',type=Path,default=Path('artifacts/gps-diagnosis'))
    parser.add_argument('--budget-seconds',type=float,default=60)
    args=parser.parse_args()
    report=run(args.data,args.predictions,args.out,args.budget_seconds)
    print(json.dumps({key:report[key] for key in ('elapsed_s','label_observability_partition',
        'full_schedule_fact_mae_s','known_deviation_confirmation_age_s','ttl_diagnostic_not_selection')},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
