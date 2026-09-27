"""Разобрать пропуски GPS по причинному журналу, затем сверить с фактами.

Сначала detector получает только проекцию плана/GPS evaluate_gps_real.
Затем evaluator читает фактические прибытия для объяснения пропусков.
Никакие поля факта не передаются детектору. Не новый holdout.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import importlib.util
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import evaluate_gps_real as runner
from backend.gps_arrivals import GPSArrivalDetector, distance_m
from common.contracts import StopTarget


def run(data, out, snapshot=None):
    started = time.monotonic()
    base = GPSArrivalDetector
    if snapshot is not None:
        spec = importlib.util.spec_from_file_location('gps_diagnostic_snapshot', snapshot)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        base = module.GPSArrivalDetector
    traces = defaultdict(list)
    order_diagnostics = Counter()

    class Traced(base):
        def observe(self, point):
            if (point.event_time == self._last_event and self._last_valid is None
                    and self._last_received is not None and point.received_at >= self._last_received
                    and point.received_at >= point.event_time and point.location_valid
                    and point.lat is not None and point.lon is not None
                    and point.speed_kmh is not None and point.speed_kmh <= 130):
                order_diagnostics['valid_replacement_of_same_time_invalid'] += 1
            result = super().observe(point)
            if (point.location_valid and point.lat is not None and point.lon is not None
                    and point.speed_kmh is not None and point.speed_kmh <= self.config.max_stop_speed_kmh):
                state = self.status()
                traces[point.tr_id].append(dict(time=point.event_time, received=point.received_at,
                    lat=point.lat, lon=point.lon, reason=state['reason'],
                    candidate=state['candidate_stop_id'], next_visit=state['next_stop_id'],
                    confirmed_visit=result.planned_stop_id if result else None,
                    previous_delay_s=self._delay_s, needs_anchor=self._needs_anchor))
            return result

    runner.GPSArrivalDetector = Traced
    report = runner.evaluate(data, out, budget_seconds=120)
    actual_detector = snapshot or Path(__file__).resolve().parents[1]/'backend/gps_arrivals.py'
    report['sources']['detector'] = dict(path=str(actual_detector.resolve()),
        sha256=runner.sha(actual_detector), bytes=actual_detector.stat().st_size)
    report['sources']['trace_wrapper'] = dict(path=str(Path(__file__).resolve()),
        sha256=runner.sha(Path(__file__)), bytes=Path(__file__).stat().st_size)
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    predictions = json.loads((out/'predictions.json').read_text())
    predicted = {row['planned_stop_id']:row for row in predictions}
    # These facts are first read after all predictions have been persisted.
    with (data/'test/schedule.csv').open(encoding='utf-8-sig',newline='') as stream:
        stops = {row['tt_action_item_id']:row for row in csv.DictReader(stream)}
    with (data/'labels/labels_test.csv').open(encoding='utf-8-sig',newline='') as stream:
        labels = list(csv.DictReader(stream))
    rows, categories = [], Counter()
    for row in labels:
        stop = stops[row['target_stop_id']]
        actual = runner.dt(stop['time_fact_begin'])
        lon,lat = map(float, runner.POINT.fullmatch(stop['geom']).groups())
        target = StopTarget(id=row['target_stop_id'],name='evaluation only',
                           scheduled_at=runner.dt(stop['time_begin']),lat=lat,lon=lon)
        near = [point for point in traces[int(row['tr_id'])]
            if abs((point['time']-actual).total_seconds()) <= 60
            and distance_m(point['lat'],point['lon'],target) <= 35]
        pred = predicted.get(target.id)
        nearest = min(near,key=lambda point:abs((point['time']-actual).total_seconds()),default=None)
        useful = [point for point in near if point['reason'] not in
                  ('late_or_duplicate','receipt_out_of_order','event_after_receipt')]
        representative = min(useful,key=lambda point:abs((point['time']-actual).total_seconds()),default=nearest)
        if pred:
            error = abs((runner.dt(pred['arrived_at'])-actual).total_seconds())
            category = 'detected_correct_60s' if error <= 60 else 'detected_wrong_time'
        elif not near:
            category = 'no_available_slow_gps_within_35m_60s'
        elif len({point['time'] for point in near}) == 1:
            category = 'only_one_distinct_slow_point'
        else:
            category = 'missed_with_slow_points:'+representative['reason']
        categories[category] += 1
        rows.append(dict(sample_id=row['sample_id'],visit_id=target.id,category=category,
            actual=actual.isoformat(),plan=stop['time_begin'],slow_points_near_fact=near))
    diagnostic = dict(detector_version=base.version, split='Previously inspected test day',
        prediction_count=report['detected_visits_total'],classification=dict(categories),
        order_diagnostics=dict(order_diagnostics),rows=rows,
        inputs='Same projected causal GPS/plan as evaluate_gps_real; labels and facts are evaluation only',
        caveats=['Points within +/-60s of fact are a post-hoc observability diagnostic, not model features.',
                 'Two points do not imply consecutive confirmation: invalid/gaps/moving samples may occur between them.',
                 'Official arrival facts have no receive timestamp and may represent terminal departure.'],
        snapshot_sha256=runner.sha(snapshot) if snapshot else None,
        script_sha256=runner.sha(Path(__file__)), elapsed_s=time.monotonic()-started)
    (out/'missed-reasons.json').write_text(json.dumps(diagnostic,ensure_ascii=False,indent=2,default=str)+'\n')
    print(json.dumps(dict(classification=dict(categories),order_diagnostics=dict(order_diagnostics),
                          elapsed_s=diagnostic['elapsed_s']),ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('../../data/raw/dataset'))
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--detector-snapshot',type=Path)
    args=parser.parse_args()
    run(args.data,args.out,args.detector_snapshot)
