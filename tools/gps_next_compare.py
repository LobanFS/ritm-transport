"""Постфактум-диагностика GPS-кандидата: общие, новые и грубо ошибочные ID.

Читает уже сохранённые predictions.json. Факты/метки используются только
для оценки, не передаются в детектор или модель; validate не открывается.
Повторно просмотренный test не является новым holdout.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import time


def dt(text):
    value = datetime.fromisoformat(text)
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def describe(values):
    values = sorted(values)
    return dict(n=len(values), mae_s=statistics.fmean(values) if values else None,
                p95_s=values[min(len(values)-1, int(.95*len(values)))] if values else None,
                correct_60s=sum(value <= 60 for value in values),
                errors_gt300s=sum(value > 300 for value in values),
                errors_gt900s=sum(value > 900 for value in values))


def compare(data, baseline, candidate, out):
    began = time.monotonic()
    # Freeze predicted events before reading any evaluation fact.
    runs = {name: dict(path=path, predictions=json.loads((path/'predictions.json').read_text()),
                       report=json.loads((path/'report.json').read_text()),
                       deviations=json.loads((path/'current-deviation.json').read_text()))
            for name, path in [('baseline', baseline), ('candidate', candidate)]}
    for field in ('traffic', 'plan', 'labels'):
        if len({run['report']['sources'][field]['sha256'] for run in runs.values()}) != 1:
            raise ValueError(f'Несопоставимые исходники {field}')
    schedule_path = data/'test/schedule.csv'
    if sha(schedule_path) != runs['baseline']['report']['sources']['plan']['sha256']:
        raise ValueError('Расписание изменилось после детекции')
    with schedule_path.open(encoding='utf-8-sig', newline='') as stream:
        schedule = {row['tt_action_item_id']: row for row in csv.DictReader(stream)}
    per_bus = defaultdict(list)
    for stop in schedule.values():
        per_bus[int(stop['tr_id'])].append(stop)
    for stops in per_bus.values():
        stops.sort(key=lambda stop: stop['time_begin'])

    errors, metrics, by_run_id = {}, {}, {}
    for name, run in runs.items():
        by_id = {row['planned_stop_id']: row for row in run['predictions']}
        by_run_id[name] = by_id
        errors[name] = {key: abs((dt(event['arrived_at'])-dt(schedule[key]['time_fact_begin'])).total_seconds())
                       for key, event in by_id.items() if schedule[key]['time_fact_begin']}
        values = errors[name]
        fresh = [row for row in run['deviations'] if row['known']
                 and 0 <= (dt(row['cutoff'])-dt(by_id[row['based_on_visit']]['arrived_at'])).total_seconds() <= 300]
        report = run['report']
        metrics[name] = dict(full_schedule=describe(values.values()), fresh_at_T=len(fresh),
            forecast_moments=len(run['deviations']), labeled_same_id=report['detected_labeled_same_id'],
            labeled_within_s=report['labeled_matches_within_s'],
            labeled_errors_gt900=report['labeled_errors_over_900s'])
    old, new = errors['baseline'], errors['candidate']
    common = old.keys() & new.keys()
    added, lost = new.keys()-old.keys(), old.keys()-new.keys()
    old_correct = {key for key, value in old.items() if value <= 60}
    retained = sum(new.get(key, float('inf')) <= 60 for key in old_correct)
    common_stats = {name: describe(values[key] for key in common) for name, values in errors.items()}
    partitions = dict(common=common_stats, new=describe(new[key] for key in added),
                      lost=describe(old[key] for key in lost), retained_correct_60s=retained,
                      baseline_correct_60s=len(old_correct))
    gross_keys = [key for key in new if new[key] > 300
                  and (key not in old or new[key] > old[key])]
    detailed = []
    for key in gross_keys:
        stop, event = schedule[key], by_run_id['candidate'][key]
        tr_id = int(stop['tr_id'])
        stops = per_bus[tr_id]
        index = next(index for index, item in enumerate(stops) if item['tt_action_item_id'] == key)
        lon, lat = map(float, stop['geom'].replace('POINT (', '').rstrip(')').split())
        detailed.append(dict(visit_id=key, tr_id=tr_id, plan=stop['time_begin'], fact=stop['time_fact_begin'],
            estimate=event['arrived_at'], available=event['received_at'],
            candidate_error_s=new[key], baseline_error_s=old.get(key),
            signed_error_s=(dt(event['arrived_at'])-dt(stop['time_fact_begin'])).total_seconds(),
            manual_fill=stop.get('manual_fill'), address=stop.get('building_address'),
            context=stops[max(0,index-2):index+3],
            geometry=dict(lat=lat, lon=lon), gps_near_estimate=[], gps_near_fact=[]))
    # Only diagnostic excerpts, not model features. Include invalid and delayed
    # packets with original event/receive clocks instead of silently fixing them.
    traffic_path = data/'test/traffic.csv'
    with traffic_path.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            matches = [case for case in detailed if case['tr_id'] == int(row['tr_id'])]
            if not matches:
                continue
            event = dt(row['event_time'])
            for case in matches:
                for field, ref in [('gps_near_estimate','estimate'), ('gps_near_fact','fact')]:
                    if abs((event-dt(case[ref])).total_seconds()) > 60:
                        continue
                    point = {key: row[key] for key in ('event_time','receive_time','lat','lon','speed','location_valid')}
                    if row['lat'] and row['lon']:
                        lat, lon = float(row['lat']), float(row['lon'])
                        ref = case['geometry']
                        point['distance_m'] = round(math.hypot((lat-ref['lat'])*111195,
                            (lon-ref['lon'])*111195*math.cos(math.radians(lat))), 2)
                    case[field].append(point)
    for case in detailed:
        for field in ('gps_near_estimate','gps_near_fact'):
            case[field].sort(key=lambda point: point['event_time'])
    base, fresh = metrics['baseline'], metrics['candidate']
    gates = dict(coverage_increased=fresh['fresh_at_T'] > base['fresh_at_T'],
        correct_visits_increased=fresh['full_schedule']['correct_60s'] > base['full_schedule']['correct_60s'],
        common_mae_within_5s=common_stats['candidate']['mae_s'] <= common_stats['baseline']['mae_s']+5,
        gross_fraction_not_materially_increased=fresh['full_schedule']['errors_gt300s']/fresh['full_schedule']['n']
            <= base['full_schedule']['errors_gt300s']/base['full_schedule']['n']+.005,
        catastrophic_count_bounded=fresh['full_schedule']['errors_gt900s'] <= base['full_schedule']['errors_gt900s']
            or (fresh['full_schedule']['errors_gt900s'] <= base['full_schedule']['errors_gt900s']+1
                and fresh['full_schedule']['correct_60s'] >= 1.2*base['full_schedule']['correct_60s']))
    source_files = [Path(__file__), schedule_path, traffic_path]
    source_files += [run['path']/filename for run in runs.values()
                     for filename in ('predictions.json','report.json','current-deviation.json')]
    report = dict(purpose='Post-prediction diagnostic, previously inspected same-day test, not holdout',
        metrics=metrics, partitions=partitions, data_gates=gates, data_gates_passed=all(gates.values()),
        new_or_worsened_gross_cases=detailed, elapsed_s=time.monotonic()-began,
        sources_sha256={str(path.resolve()):sha(path) for path in source_files},
        limitations=['Официальный факт может смешивать прибытие/отстой/отправление; независимого операционного эталона нет.',
            'Нет вывода о качестве ML по MAE на разных подмножествах посещений.',
            'Тесты причинности, синтетики и ручной разбор новых грубых случаев обязательны отдельно.'])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(metrics=metrics, partitions=partitions, gates=gates,
                          changed_gross_ids=[case['visit_id'] for case in detailed]), ensure_ascii=False, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('../../data/raw/dataset'))
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    compare(args.data,args.baseline,args.candidate,args.out)
