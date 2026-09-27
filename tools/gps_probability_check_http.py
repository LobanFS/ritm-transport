"""HTTP-проверка опубликованного GPS probability scope без смены backend-режима.

Восстанавливает точные исторические запросы из compact traces + plan-input,
проверяет SHA каждого запроса и frozen вероятности. Затем меняет только
происхождение (live/synthetic/door/wrong detector SHA): p должна исчезнуть,
а численный прогноз регрессии — сохраниться. Никаких меток или fit.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.engine import Engine, LiveContext
from common.contracts import Prediction, PredictionRequest


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def reconstruct_requests(stream, frozen_path):
    """Ни один запрос не обогащается фактами, которых не было в сохранённой trace."""
    expected = json.loads(frozen_path.read_text())
    expected_by_id = {row['sample_id']:row for row in expected}
    if len(expected_by_id) != len(expected) or not expected:
        raise ValueError('Frozen IDs должны быть непустыми и уникальными')
    context = LiveContext.model_validate_json((stream/'plan-input.json').read_text())
    engine = Engine('unused')
    engine.set_live(context)
    traces = [json.loads(line) for line in (stream/'traces.jsonl').read_text().splitlines() if line.strip()]
    selected = [row for row in traces if row['sample_id'] in expected_by_id]
    if len(selected) != len(expected) or {row['sample_id'] for row in selected} != set(expected_by_id):
        raise ValueError('Traces не покрывают frozen GPS IDs ровно один раз')
    result = []
    for trace in selected:
        request = deepcopy(trace['request'])
        compact = request['plan_context']
        plan = engine.model_plans[request['tr_id']].model_dump(mode='json')
        if (len(plan['stops']) != compact['stop_count']
            or any(plan[name] != compact[name] for name in ('version','timezone','complete'))):
            raise ValueError('Полный план отличается от compact trace')
        request['plan_context'] = plan
        PredictionRequest.model_validate(request)
        digest = hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()
        if digest != trace['request_sha256']:
            raise ValueError(f"Не совпал SHA восстановленного запроса {trace['sample_id']}")
        if request['current_delay_source'] != 'gps_estimate' or request['telemetry_domain'] != 'historical_real':
            raise ValueError('Frozen GPS-point не относится к реальной истории')
        result.append((trace['sample_id'],request,expected_by_id[trace['sample_id']]))
    return result


def check(stream, frozen_path, scope_path, url, out, *, ready_timeout_s=45, budget_s=90):
    started = time.monotonic()
    expected_scope_hash = sha(scope_path)
    contract = dict(check='GPS scope HTTP parity and negative provenance controls',
        endpoint=url, ready_timeout_s=ready_timeout_s, budget_s=budget_s,
        probability_atol=1e-12, seconds_atol=1e-9, negative_cases=['live_unverified','synthetic','door_estimate','wrong_detector_sha'],
        backend_mode_changed=False, labels_read=False, trained=False,
        input_sha256={str(path):sha(path) for path in
            (stream/'traces.jsonl',stream/'plan-input.json',frozen_path,scope_path,Path(__file__))})
    save(out/'contract.json',contract)
    save(out/'report.json',dict(status='RUNNING',contract_sha256=sha(out/'contract.json')))
    observed = []
    try:
        rows = reconstruct_requests(stream,frozen_path)
        with httpx.Client(base_url=url,timeout=15) as client:
            deadline = time.monotonic()+ready_timeout_s
            last_error = None
            while True:
                try:
                    response = client.get('/model')
                    response.raise_for_status()
                    card = response.json()
                    if (card.get('trained') and card.get('probability')
                        and card['probability'].get('gps_scope_sha256') == expected_scope_hash):
                        break
                    last_error = 'HTTP ML ещё не загрузил ожидаемый GPS scope'
                except (httpx.HTTPError, ValueError) as error:
                    last_error = str(error)
                if time.monotonic() >= deadline:
                    raise TimeoutError(last_error)
                time.sleep(.5)
            save(out/'model-card.json',card)
            jobs = []
            for sample_id, request, expected in rows:
                jobs.append((sample_id,'historical_real',request,expected))
                for name, mutation in (
                    ('live_unverified',{'telemetry_domain':'live_unverified'}),
                    ('synthetic',{'telemetry_domain':'synthetic'}),
                    ('door_estimate',{'current_delay_source':'door_estimate'}),
                    ('wrong_detector_sha',{'current_delay_detector_sha256':'0'*64}),
                ):
                    changed = {**request,**mutation,'request_id':request['request_id']+':'+name}
                    jobs.append((sample_id,name,changed,expected))
            for start in range(0,len(jobs),16):
                if time.monotonic()-started > budget_s:
                    raise TimeoutError('Превышен общий бюджет HTTP-проверки')
                batch = jobs[start:start+16]
                response = client.post('/predict/batch',json=[job[2] for job in batch])
                response.raise_for_status()
                values = response.json()
                if len(values) != len(batch):
                    raise ValueError('Ответ ML не покрывает весь batch')
                for (sample_id,case,request,expected),raw in zip(batch,values,strict=True):
                    result = Prediction.model_validate(raw)
                    if result.request_id != request['request_id'] or result.method != 'learned':
                        raise ValueError('Нарушена идентичность запроса либо learned-инференс')
                    delta = abs(result.predicted_delay_s-expected['gps_prediction_s'])
                    probability = result.probability_late
                    p_delta = abs(probability-expected['gps_probability']) if probability is not None else None
                    passed = delta <= 1e-9 and (p_delta is not None and p_delta <= 1e-12
                              if case=='historical_real' else probability is None)
                    observed.append(dict(sample_id=sample_id,case=case,passed=passed,seconds_delta=delta,
                                         probability_late=probability,probability_delta=p_delta,
                                         probability_note=result.probability_note))
                    if not passed:
                        raise AssertionError(f'HTTP scope/parity mismatch: {sample_id}, {case}')
            final_card = client.get('/model')
            final_card.raise_for_status()
            if final_card.json()['probability']['gps_scope_sha256'] != expected_scope_hash:
                raise AssertionError('ML scope изменился во время проверки')
        report = dict(status='PASS',historical_rows=len(rows),negative_control_rows=len(observed)-len(rows),
            total_http_predictions=len(observed),request_sha256_verified=len(rows),
            probability_max_abs_error=max(row['probability_delta'] for row in observed if row['case']=='historical_real'),
            regression_max_abs_error=max(row['seconds_delta'] for row in observed),
            cases={case:sum(row['case']==case for row in observed) for case in ['historical_real',*contract['negative_cases']]},
            gps_scope_sha256=expected_scope_hash,contract_sha256=sha(out/'contract.json'),
            backend_mode_changed=False,elapsed_s=time.monotonic()-started)
        save(out/'responses.json',observed)
        save(out/'report.json',report)
        return report
    except Exception as error:
        save(out/'responses.json',observed)
        save(out/'report.json',dict(status='FAIL',error=f'{type(error).__name__}: {error}',
                                   completed_predictions=len(observed),elapsed_s=time.monotonic()-started,
                                   contract_sha256=sha(out/'contract.json')))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stream',type=Path,default=ROOT/'artifacts/gps-next/http-gps')
    parser.add_argument('--frozen',type=Path,default=ROOT/'artifacts/gps-probability/final/frozen-probabilities.json')
    parser.add_argument('--scope',type=Path,default=ROOT/'ml_service/probability_gps_scope.json')
    parser.add_argument('--url',default='http://127.0.0.1:8001')
    parser.add_argument('--out',type=Path,default=ROOT/'artifacts/gps-probability/http-scope')
    args = parser.parse_args()
    print(json.dumps(check(args.stream,args.frozen,args.scope,args.url,args.out),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
