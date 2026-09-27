"""Оценка GPS-детектора на отдельной истине синтетического источника, без ML.

Один эксперимент: подтверждение посещения по нескольким GPS вместо одиночной
близкой точки. Пороги v1 фиксированы; seed7 — разработка, seed42 — проверка без
подбора параметров. Это не измерение на реальных автобусах или DS submission.
"""
import dataclasses
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.engine import Engine
from backend.generator_bridge import ProducerState
from backend.gps_arrivals import GPSDetectorConfig, distance_m
from common.contracts import Telemetry
from generator.scenarios import GeneratorSession, ResetRequest, START


def dt(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def percentile(values, fraction):
    return sorted(values)[max(0, math.ceil(len(values)*fraction)-1)] if values else None


def run_case(seed, scenario, *, route_count=4, cadence=15, stress=False, isolated_outlier=False, door_sensors=False):
    producer = GeneratorSession(ResetRequest(seed=seed, scenario=scenario,
        route_count=route_count, telemetry_interval_s=cadence, speed=1, paused=False, door_sensors=door_sensors))
    engine = Engine('unused')
    engine.set_generator(producer.context, ProducerState.model_validate(producer.status()))
    baseline, pending, snapshots = {}, [], []
    rng = random.Random(seed+891)
    packet = 0
    for tick in range(1801):
        if tick:
            producer.advance(1)
        engine.clock = producer.clock_time
        frame = producer.frames[-1]
        assert not frame['arrivals'], 'Oracle must not enter the input stream'
        for raw in frame['telemetry']:
            point = Telemetry.model_validate(raw)
            packet += 1
            if isolated_outlier and point.tr_id == 101 and tick == 60:
                future_stop = next(s for s in engine.schedule[101] if s.scheduled_at >= engine.clock+timedelta(seconds=150))
                point = point.model_copy(update={'lat':future_stop.lat,'lon':future_stop.lon,'speed_kmh':0})
            if stress:
                point = point.model_copy(update={
                    'lat':point.lat+rng.uniform(-10,10)/111320,
                    'lon':point.lon+rng.uniform(-10,10)/(111320*math.cos(math.radians(point.lat))),
                    'received_at':point.received_at+timedelta(seconds=45 if packet%7 == 0 else 0),
                    'location_valid':packet%13 != 0})
            pending.append(point)
        ready = sorted((p for p in pending if max(p.event_time,p.received_at) <= engine.clock),
                       key=lambda p:(p.received_at,p.event_time))
        pending = [p for p in pending if max(p.event_time,p.received_at) > engine.clock]
        for point in ready:
            engine.ingest(point.model_copy(update={'source':'generator'}))
            # Намеренно слабый причинный baseline: первый медленный пакет в зоне,
            # среди повторных посещений — ближайшее плановое время.
            if point.location_valid and point.lat is not None and point.speed_kmh is not None and point.speed_kmh <= 5:
                candidates = [s for s in engine.schedule[point.tr_id]
                    if abs((point.event_time-s.scheduled_at).total_seconds()) <= 900
                    and distance_m(point.lat,point.lon,s) <= 35]
                if candidates:
                    stop = min(candidates,key=lambda s:abs((point.event_time-s.scheduled_at).total_seconds()))
                    baseline.setdefault((point.tr_id,stop.id),dict(arrived_at=point.event_time,received_at=point.received_at))
        if tick%15 == 0:
            for tr in engine.vehicles:
                value = engine.deviation_at(tr,engine.clock)
                snapshots.append((tr,engine.clock,value.delay_s if value else None))
    # Эталон впервые читается ПОСЛЕ работы детектора и сохранения его результатов.
    truth = producer.truth()['observed_truth_arrivals']
    facts = {(a['tr_id'],a['planned_stop_id']):dt(a['arrived_at']) for a in truth}
    end = producer.clock_time-timedelta(seconds=30)
    eligible = {key:actual for key,actual in facts.items() if START <= actual <= end}
    detected = {(tr,k):dict(arrived_at=a.observed_at,received_at=a.received_at)
                for tr,records in engine.arrivals.items() for k,a in records.items()}
    def score(predictions):
        scored = {key:p for key,p in predictions.items() if key in eligible}
        matched = {key:p for key,p in scored.items() if abs((p['arrived_at']-eligible[key]).total_seconds()) <= 30}
        errors = [abs((p['arrived_at']-eligible[key]).total_seconds()) for key,p in matched.items()]
        lags = [(p['received_at']-eligible[key]).total_seconds() for key,p in matched.items()]
        return dict(ground_truth_visits=len(eligible),predicted_visits=len(scored),matched=len(matched),
            excluded_boundary_or_unmatured=len(predictions)-len(scored),
            unassessable_ids=sum(key not in facts for key in predictions),
            excluded_preexisting=sum(key in facts and facts[key] < START for key in predictions),
            excluded_right_boundary=sum(key in facts and facts[key] > end for key in predictions),
            precision=len(matched)/len(scored) if scored else None,
            recall=len(matched)/len(eligible) if eligible else None,
            arrival_mae_s=statistics.mean(errors) if errors else None,
            arrival_mae_all_scored_ids_s=statistics.mean(abs((p['arrived_at']-eligible[key]).total_seconds()) for key,p in scored.items()) if scored else None,
            confirmation_lag_mean_s=statistics.mean(lags) if lags else None,
            confirmation_lag_p95_s=percentile(lags,.95))
    plans = {(tr,s.id):s.scheduled_at for tr,stops in engine.schedule.items() for s in stops}
    current_errors = []
    for tr,at,value in snapshots:
        actual = max(((key,t) for key,t in facts.items() if key[0] == tr and t <= at),key=lambda pair:pair[1],default=None)
        if actual and value is not None:
            key,t = actual
            current_errors.append(abs(value-(t-plans[key]).total_seconds()))
    metrics = score(detected)
    metrics['sources'] = {source:sum(a.source == source for records in engine.arrivals.values() for a in records.values())
                          for source in ('gps_estimate', 'door_estimate')}
    metrics.update(unknown_fraction=sum(value is None for _,_,value in snapshots)/len(snapshots),
                   current_delay_mae_s=statistics.mean(current_errors) if current_errors else None,
                   current_delay_scored_samples=len(current_errors),sample_count=len(snapshots))
    return dict(seed=seed,scenario=scenario,route_count=route_count,cadence_s=cadence,stress=stress,isolated_outlier=isolated_outlier,door_sensors=door_sensors,
        metrics=metrics,baseline=score(baseline),
        final_diagnostics={str(tr):d.status() for tr,d in engine.gps_detectors.items()},
        detected=[dict(tr_id=key[0],planned_stop_id=key[1],**value) for key,value in detected.items()])


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'artifacts'/'gps')
    output = parser.parse_args().out
    output.mkdir(parents=True,exist_ok=True)
    contract = dict(change='Arrival-detection policy: confirmation+ordered visits vs single-nearby-packet baseline; not a single-feature ablation',
        development_seeds=[7],verification_seeds=[42],parameters=dataclasses.asdict(GPSDetectorConfig()),
        event_match='same planned visit ID and arrival error <=30s; boundary visits excluded',
        metric_scope='Precision on mature labeled visits; arrival_mae_s only matched visits. All-ID MAE and unknown/boundary counts reported separately.',
        acceptance='clean 15s cadence: precision>=0.95, recall>=0.70, matched confirmation p95<=45s',
        budget_seconds=60,stress='10m coordinate jitter; every7th packet delayed45s; every13th invalid',
        caveat='Synthetic validation only; no parameter search; no real visit ground truth used')
    (output/'contract.json').write_text(json.dumps(contract,ensure_ascii=False,indent=2)+'\n')
    started = time.monotonic()
    cases = [run_case(7,'normal')]
    for scenario in ('normal','slow_segment','long_stop','gps_loss'):
        cases.append(run_case(42,scenario))
    cases.extend([run_case(42,'normal',stress=True),run_case(42,'normal',isolated_outlier=True),run_case(42,'normal',cadence=30),run_case(42,'normal',route_count=20)])
    clean = cases[1]['metrics']
    passed = ((clean['precision'] or 0)>=.95 and (clean['recall'] or 0)>=.70
              and clean['unassessable_ids'] == 0
              and clean['confirmation_lag_p95_s'] is not None and clean['confirmation_lag_p95_s']<=45)
    files = [*Path(ROOT/'backend').glob('*.py'),*Path(ROOT/'generator').glob('*.py'),Path(__file__)]
    report = dict(created_at=datetime.now(timezone.utc).isoformat(),acceptance_passed=passed,
        elapsed_s=round(time.monotonic()-started,3),contract=contract,cases=cases,
        sources_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str)+'\n')
    for case in cases:
        print(json.dumps({k:v for k,v in case.items() if k not in ('detected','final_diagnostics')},ensure_ascii=False))
    print('Acceptance:',passed, '|',output/'report.json')


if __name__ == '__main__':
    main()
