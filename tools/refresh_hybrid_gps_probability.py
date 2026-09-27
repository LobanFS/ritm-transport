"""Причинный GPS → history → frozen hybrid: допуск нового p на реальной истории.

ASGI выполняет настоящий ML API в этом процессе, без изменения Docker/стенда.
Это проверка качества цепочки, не измерение сетевого latency. Коэффициенты уже
заморожены train-OOF калибратором. Labels читает лишь итоговый оценщик.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import sys
import time

import httpx
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ml_service.learned import LearnedModel,MODEL_SHA256,ENCODER_SHA256
from ml_service.probability import ProbabilityModel,GPSProbabilityScope
from tools.evaluate_stream_real import load_inputs,replay,sources
from tools.gps_probability_evaluate import evaluate,save,sha,RECIPE


async def run(data,model_path,probability_path,out,publish_scope=False):
    started=time.monotonic();torch.set_num_threads(1)
    stream=out/'stream';reference=out/'csv-reference';measurement=out/'evaluation'
    source_hashes=sources()
    contract=dict(change='Only regression+calibrator identity: current frozen hybrid with existing GPS detector',
        model_sha256=MODEL_SHA256,encoder_sha256=ENCODER_SHA256,probability_sha256=sha(probability_path),
        calibration_fit=False,scope='historical_real, gps_estimate only',transport='ASGI in process',
        official_sources={name:sha(data/name) for name in ('test/traffic.csv','validate/schedule_plan.csv','labels/labels_test.csv')},
        acceptance=RECIPE['acceptance'],budget_s=600,code_sha256=source_hashes,
        no_container_changes=True,no_backend_mode_changes=True)
    save(stream/'contract.json',contract)
    save(out/'report.json',dict(status='running'))
    try:
        model=LearnedModel(model_path)
        model.probability=ProbabilityModel.load(probability_path,MODEL_SHA256,expected_encoder_sha256=ENCODER_SHA256)
        model.probability_error=None
        import ml_service.app as app_module
        original_runtime=app_module.runtime
        app_module.runtime=lambda:(model,None)
        try:
            inputs=load_inputs(data,deviation_source='gps')
            save(stream/'plan-input.json',inputs.context.model_dump(mode='json'))
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app_module.app),base_url='http://local-ml',timeout=90) as client:
                result=await replay(inputs,client,'http://local-ml',budget_s=550,out=stream,deviation_source='gps')
                card=(await client.get('/model')).json()
        finally:
            app_module.runtime=original_runtime
        if source_hashes!=sources():raise RuntimeError('Backend/common изменились во время проверки')
        pipeline_ok=(result['outside_planned_horizon']==0
                     and all(row['reason']=='target_mismatch' for row in result['failures'])
                     and all(row['method'] in ('learned','unavailable') for row in result['predictions']))
        save(stream/'report.json',dict(status='DIAGNOSTIC',pipeline_checks_passed=pipeline_ok,
             deviation_source='gps',transport='ASGI_in_process_not_network',model_card=card,
             population=inputs.metadata,all_forecasts=result['all_forecasts'],
             outside_planned_horizon=result['outside_planned_horizon'],target_mismatches=len(result['failures'])))
        # Reference probabilities are already frozen before this script; input
        # cur_dev_s is projected without reading target labels.
        frozen=json.loads((probability_path.parent/'frozen-test-predictions.json').read_text())
        points=pd.read_csv(data/'labels/labels_test.csv',usecols=['sample_id','tr_id','T','target_stop_id','target_time_begin','cur_dev_s'],
                           dtype={'sample_id':str,'target_stop_id':str}).set_index('sample_id')
        reference.mkdir(parents=True,exist_ok=True)
        with (reference/'predictions.jsonl').open('w') as destination:
            for row in frozen:
                point=points.loc[row['sample_id']]
                value=dict(sample_id=row['sample_id'],tr_id=int(point.tr_id),T=pd.Timestamp(point['T'],tz='UTC').isoformat(),
                    target_stop_id=str(point.target_stop_id),target_time_begin=pd.Timestamp(point.target_time_begin,tz='UTC').isoformat(),
                    input_cur_dev_s=float(point.cur_dev_s),input_source='csv_snapshot',prediction=row['prediction'],
                    method='learned',telemetry_age_s=0,probability_late=row['probability_late'])
                destination.write(json.dumps(value)+'\n')
        report=evaluate(stream,reference,data/'labels/labels_test.csv',probability_path,measurement)
        if not pipeline_ok:raise ValueError('Причинная GPS/ML цепочка не прошла проверку')
        scope_path=None
        if report['status']=='accepted_secondary_diagnostic':
            versions={row['detector']['version'] for row in result['outcomes'] if row['detector']}
            if len(versions)!=1:raise ValueError('Неоднозначная версия детектора')
            metric,baseline=report['metrics']['gps'],report['metrics']['train_constant']
            scope=GPSProbabilityScope(schema_version='gps-probability-scope-v1',validation_status='accepted_secondary_diagnostic',
                scope='historical_real_gps_estimate',regression_model_sha256=MODEL_SHA256,
                probability_artifact_sha256=sha(probability_path),detector_version=next(iter(versions)),
                detector_sha256=source_hashes['backend/gps_arrivals.py'],contract_sha256=sha(measurement/'contract.json'),
                report_sha256=sha(measurement/'report.json'),report_path=str((measurement/'report.json').relative_to(ROOT)),
                evaluated_rows=metric['rows'],total_rows=report['coverage']['all_rows'],positives=metric['positives'],
                vehicles=report['bootstrap_brier_improvement']['vehicles'],brier=metric['brier'],constant_brier=baseline['brier'],
                log_loss=metric['log_loss'],constant_log_loss=baseline['log_loss'],
                cluster_improvement_p05=report['bootstrap_brier_improvement']['p05'],
                expected_calibration_error=metric['expected_calibration_error'])
            scope_path=out/'probability_gps_scope.json';save(scope_path,scope.model_dump())
            if publish_scope:
                active=ROOT/'ml_service/probability_gps_scope.json'
                if active.exists() and not (out/'previous-gps-scope.json').exists():
                    (out/'previous-gps-scope.json').write_bytes(active.read_bytes())
                active.write_bytes(scope_path.read_bytes())
        summary=dict(status=report['status'],probability_sha256=sha(probability_path),
            scope_sha256=sha(scope_path) if scope_path else None,published=bool(scope_path and publish_scope),
            pipeline_checks_passed=pipeline_ok,coverage=report['coverage'],metrics=report['metrics'],
            bootstrap=report['bootstrap_brier_improvement'],transport='ASGI in process, not measured network',
            regression_unchanged=True,independent_holdout=False,elapsed_s=time.monotonic()-started)
        save(out/'report.json',summary)
        return summary
    except Exception as error:
        save(out/'report.json',dict(status='failed',error=f'{type(error).__name__}: {error}',elapsed_s=time.monotonic()-started))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT.parent.parent/'data/raw/dataset')
    parser.add_argument('--model',type=Path,default=ROOT/'artifacts/model/model.joblib')
    parser.add_argument('--probability',type=Path,default=ROOT/'artifacts/probability-refresh/probability.json')
    parser.add_argument('--out',type=Path,default=ROOT/'artifacts/probability-refresh/gps')
    parser.add_argument('--publish-scope',action='store_true')
    args=parser.parse_args()
    print(json.dumps(asyncio.run(run(args.data,args.model,args.probability,args.out,args.publish_scope)),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
