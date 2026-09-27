"""Парное сравнение наличия дверных датчиков; только синтетический эксперимент.

Меняется ровно доступность doors_open. GPS, план, seed и скрытые прибытия
одинаковы. Это оценка распознавания, не MAE модели на реальных автобусах.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.evaluate_gps import run_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'artifacts/audit-fixes/doors')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    contract = dict(change='Only optional door sensor; matching and GPS-only policy unchanged',
        seed=42, scenarios=['normal', 'slow_segment', 'long_stop', 'gps_loss'],
        cadences_s=[15,30], routes=4, duration_s=1800, budget_s=60,
        baseline='Same stream with doors_open=null',
        acceptance='Each case: matched visits >= GPS-only; wrong mature visit matches do not increase; '
                   'mean confirmation lag does not increase. Some improvement required overall.',
        limitation='Known synthetic scenario family, engineering regression, not an independent real-data holdout')
    (args.out/'contract.json').write_text(json.dumps(contract,ensure_ascii=False,indent=2)+'\n')
    began = time.monotonic()
    cases = []
    for cadence in contract['cadences_s']:
        for scenario in contract['scenarios']:
            pair = [run_case(42,scenario,cadence=cadence,door_sensors=present) for present in (False,True)]
            old, new = [result['metrics'] for result in pair]
            passed = (new['matched'] >= old['matched']
                and new['predicted_visits']-new['matched'] <= old['predicted_visits']-old['matched']
                and (old['confirmation_lag_mean_s'] is None or new['confirmation_lag_mean_s'] <= old['confirmation_lag_mean_s']))
            cases.append(dict(scenario=scenario,cadence_s=cadence,gps_only=old,gps_doors=new,passed=passed))
            (args.out/f'{scenario}-{cadence}-observations.json').write_text(json.dumps(pair,ensure_ascii=False,indent=2,default=str)+'\n')
            if time.monotonic()-began > contract['budget_s']:
                raise RuntimeError('Budget exceeded')
    improved = any(c['gps_doors']['matched'] > c['gps_only']['matched']
                   or c['gps_doors']['confirmation_lag_mean_s'] < c['gps_only']['confirmation_lag_mean_s'] for c in cases)
    report = dict(checked_at=datetime.now(timezone.utc).isoformat(),cases=cases,
        passed=all(c['passed'] for c in cases) and improved,elapsed_s=time.monotonic()-began,
        real_data_improvement_claimed=False, model_changed=False,
        sources_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in (
            'tools/evaluate_doors.py','tools/evaluate_gps.py','backend/engine.py',
            'backend/gps_arrivals.py','common/contracts.py','generator/scenarios.py')})
    (args.out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
