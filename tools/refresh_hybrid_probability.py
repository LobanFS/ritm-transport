"""Один train-OOF калибратор для точного frozen Swiss Transformer + HGBR.

Проверяет причинную проекцию последовательностей и воспроизводит три vehicle
fold-модели как аудит сохранённого OOF. Runtime-регрессия не переобучается.
Test читается по allowlist входов; его метки — только после freeze коэффициентов
и всех прогнозов. Это вторичная проверка известного дня, не новый holdout.
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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ml_service.learned import LearnedModel, MODEL_SHA256, ENCODER_SHA256
from ml_service.hybrid import context_row, sequence_row, transformer_prior
from ml_service.frozen_plan.plan import PLAN_COLUMNS, PLAN_FEATURES, build_plan_features
from ml_service.probability import ProbabilityArtifact, ProbabilityModel
from tools.prepare_probability import probability_metrics
from tools.gps_probability_evaluate import cluster_improvement, expected_calibration_error

POINTS=['sample_id','tr_id','T','target_stop_id','target_time_begin','cur_dev_s']
RECIPE=dict(name='hybrid-probability-logistic-v2',event='target_delay_s > 120',
    feature='(hybrid_prediction-120)/120',folds=['vehicle_01','vehicle_02','vehicle_03'],
    estimator=dict(C=1.0,solver='lbfgs',max_iter=1000,class_weight=None,random_state=0),
    baseline='Empirical positive frequency of unique training labels',
    acceptance='Test Brier and logloss below train constant, ECE<=0.15, vehicle-bootstrap p05 improvement>0',
    oof_audit='Retrain only three shadow fold regressors with frozen encoder; maximum OOF difference<=1e-8',
    sequence_audit='Exact equality to stored causal arrays; last12 current/strictly-past inputs<=90min, never targets',
    split='Train vehicle-OOF for fit; provided test353 for secondary diagnostics',
    limitations=['Vehicle-OOF is not chronological rolling validation.',
                 'Prior uses only input cur_dev_s, declared available by dataset; no factual arrivals.',
                 'Known day/model family selected in prior research; no independent test claim.',
                 'Synthetic train relatives may share dynamics; no claim of new-family independence.',
                 'New GPS, doors, live and synthetic sources require separate diagnostics.'],
    budget_s=600,no_hyperparameter_search=True,runtime_regression_changed=False)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def causal_arrays(points,plan_features):
    """Чистая проекция входов; принимает ровно POINTS без target/fact полей."""
    if set(points.columns)!=set(POINTS):
        raise ValueError('Последовательности требуют только разрешённые входы POINTS')
    ordered=points.copy()
    ordered['T']=pd.to_datetime(ordered['T'],utc=True)
    if ordered.duplicated(['tr_id','T']).any():
        raise ValueError('Неоднозначные одновременные состояния одного ТС')
    sequence=np.zeros((len(points),12,5),dtype=np.float32)
    context=np.zeros((len(points),6),dtype=np.float32)
    max_age=0.
    for _,group in ordered.sort_values(['tr_id','T']).groupby('tr_id',sort=False):
        previous=[]
        for index,row in group.iterrows():
            at=row['T'].to_pydatetime()
            history=[item for item in previous if (at-item[0]).total_seconds()<=5400][-11:]
            assert all(stamp<at for stamp,_ in history)
            times=[stamp for stamp,_ in history]+[at]
            delays=[delay for _,delay in history]+[float(row.cur_dev_s)]
            sequence[index],count,span=sequence_row(times,delays,at)
            context[index]=context_row(plan_features.iloc[index],count)
            max_age=max(max_age,span)
            previous.append((at,float(row.cur_dev_s)))
    return sequence,context,dict(rows=len(points),future_observations=0,maximum_history_age_s=max_age)


def projected_split(data,split,stored,prefix,model):
    points=pd.read_csv(data/f'labels/labels_{split}.csv',usecols=POINTS,dtype={'sample_id':str,'target_stop_id':str})
    plan_file=data/('train/schedule.csv' if split=='train' else 'validate/schedule_plan.csv')
    plan=pd.read_csv(plan_file,usecols=list(PLAN_COLUMNS))
    features=build_plan_features(points,plan).loc[:,list(PLAN_FEATURES)]
    sequence,context,audit=causal_arrays(points,features)
    ids=stored[prefix+'_id'].astype(str)
    if not np.array_equal(points.sample_id.to_numpy(),ids):
        raise ValueError('Порядок входов отличается от архивных sequence IDs')
    if not np.array_equal(sequence,stored[prefix+'_seq']) or not np.array_equal(context,stored[prefix+'_context']):
        raise ValueError('Runtime causal sequence/context отличаются от обучающей проекции')
    prior=np.concatenate([transformer_prior(model.transformer,sequence[i:i+512],context[i:i+512],
                          points.cur_dev_s.to_numpy()[i:i+512]) for i in range(0,len(points),512)])
    hybrid=features.copy();hybrid['neural_prior']=prior
    return points,plan,hybrid,audit,sequence,context


def verified_oof(team,data,features,points,model,started):
    saved=pd.read_csv(team/'reports/experiments/swiss_sequence_hybrid/fold_predictions.csv',
                      dtype={'sample_id':str})
    selected=saved[(saved.variant=='prior') & saved.fold.isin(RECIPE['folds'])].copy()
    labels=pd.read_csv(data/'labels/labels_train.csv',usecols=['sample_id','tr_id','target_delay_s'],dtype={'sample_id':str})
    if labels.sample_id.duplicated().any() or len(labels)!=4434:
        raise ValueError('Неверный train IDs/size')
    if selected.sample_id.duplicated().any() or set(selected.sample_id)!=set(labels.sample_id):
        raise ValueError('Hybrid vehicle-OOF должен покрывать train ровно один раз')
    fold_defs=json.loads((team/'artifacts/final_model/frozen_folds.json').read_text())
    indexed=labels.set_index('sample_id');idx={identity:i for i,identity in enumerate(points.sample_id)}
    checks=[]
    for name in RECIPE['folds']:
        if time.monotonic()-started>RECIPE['budget_s']:raise TimeoutError('Бюджет исчерпан')
        fold=next(item for item in fold_defs if item['name']==name)
        train_ids,valid_ids=fold['train_sample_ids'],fold['validation_sample_ids']
        if set(train_ids)&set(valid_ids) or set(train_ids)|set(valid_ids)!=set(labels.sample_id):
            raise ValueError('Нарушена train/validation граница OOF')
        if set(indexed.loc[train_ids,'tr_id'])&set(indexed.loc[valid_ids,'tr_id']):
            raise ValueError('Одно ТС одновременно в fit и validation OOF')
        part=selected[selected.fold==name].set_index('sample_id')
        if set(part.index)!=set(valid_ids):raise ValueError('OOF IDs не совпали с frozen fold')
        train_indices=[idx[sid] for sid in train_ids];valid_indices=[idx[sid] for sid in valid_ids]
        shadow=HistGradientBoostingRegressor(**model.estimator.get_params())
        with threadpool_limits(limits=1):
            shadow.fit(features.iloc[train_indices].to_numpy(),indexed.loc[train_ids,'target_delay_s'].to_numpy())
            prediction=shadow.predict(features.iloc[valid_indices].to_numpy())
        delta=float(np.max(np.abs(prediction-part.loc[valid_ids,'prediction'].to_numpy())))
        if delta>1e-8:raise ValueError(f'Не воспроизводится OOF {name}: {delta}')
        checks.append(dict(fold=name,train_rows=len(train_ids),validation_rows=len(valid_ids),max_abs_delta=delta,
                           disjoint_vehicles=True,disjoint_sample_ids=True))
    joined=selected.merge(labels,on='sample_id',validate='one_to_one')
    if not np.array_equal(joined.target.to_numpy(),joined.target_delay_s.to_numpy()):
        raise ValueError('OOF labels отличаются от раздачи')
    return joined,checks


def run(data,team,out,model_path):
    started=time.monotonic();torch.set_num_threads(1)
    source_paths={
        'hybrid/model':model_path,'hybrid/encoder':model_path.with_name('encoder.pt'),
        'hybrid/fold_predictions':team/'reports/experiments/swiss_sequence_hybrid/fold_predictions.csv',
        'hybrid/holdout_predictions':team/'reports/experiments/swiss_sequence_hybrid/holdout_predictions.csv',
        'frozen_folds':team/'artifacts/final_model/frozen_folds.json',
        'sequence_archive':team/'artifacts/experiments/swiss_sequence_transfer/sequences.npz',
        'labels_train':data/'labels/labels_train.csv','labels_test':data/'labels/labels_test.csv',
        'plan_train':data/'train/schedule.csv','plan_test':data/'validate/schedule_plan.csv',
        'experiment_code':team/'experiments/swiss_sequence_hybrid/experiment.py',
        'sequence_builder':team/'experiments/swiss_sequence_transfer/build_sequences.py',
        'runtime_hybrid':ROOT/'ml_service/hybrid.py','calibration_code':Path(__file__),
    }
    fingerprints={key:sha(path) for key,path in source_paths.items()}
    save(out/'contract.json',dict(RECIPE,sources_sha256=fingerprints))
    save(out/'report.json',dict(status='running'))
    try:
        model=LearnedModel(model_path)
        stored=np.load(source_paths['sequence_archive'],allow_pickle=False)
        train,_,x_train,train_audit,_,_=projected_split(data,'train',stored,'mt',model)
        oof,folds=verified_oof(team,data,x_train,train,model,started)
        save(out/'oof-audit.json',dict(folds=folds,sequence=train_audit,encoder_sha256=ENCODER_SHA256))
        y=(oof.target_delay_s.to_numpy()>120).astype(int)
        constant=float(y.mean())
        classifier=LogisticRegression(**RECIPE['estimator']).fit(((oof.prediction.to_numpy()-120)/120).reshape(-1,1),y)
        coefficient,intercept=float(classifier.coef_[0,0]),float(classifier.intercept_[0])
        if coefficient<=0 or int(classifier.n_iter_.max())>=1000:raise ValueError('Калибратор не сошёлся/немонотонен')
        save(out/'frozen-fit.json',dict(coefficient=coefficient,intercept=intercept,train_rows=len(oof),
              train_constant=constant,regression_model_sha256=MODEL_SHA256,encoder_sha256=ENCODER_SHA256,
              contract_sha256=sha(out/'contract.json')))
        points,plan,x_test,test_audit,sequence,context=projected_split(data,'test',stored,'me',model)
        with threadpool_limits(limits=1):prediction=model.estimator.predict(x_test.to_numpy())
        saved=pd.read_csv(source_paths['hybrid/holdout_predictions'],usecols=['sample_id','candidate_prediction'],dtype={'sample_id':str})
        reference=saved.set_index('sample_id').loc[points.sample_id,'candidate_prediction'].to_numpy()
        parity=float(np.max(np.abs(prediction-reference)))
        if parity>1e-8:raise ValueError(f'Точная frozen регрессия не совпала с holdout artifact: {parity}')
        # HTTP/runtime получает тот же sequence/context, включая strictly-past history.
        from tools.check_model import build_requests
        requests=build_requests(points,plan,'UTC',sha(data/'validate/schedule_plan.csv'))
        runtime_sequence,runtime_context,*_=model.sequence_inputs(requests,x_test[list(PLAN_FEATURES)])
        if not np.array_equal(runtime_sequence,sequence) or not np.array_equal(runtime_context,context):
            raise ValueError('Runtime sequence projection не совпала с проверенной историей')
        probability=classifier.predict_proba(((prediction-120)/120).reshape(-1,1))[:,1]
        frozen=[dict(sample_id=sid,prediction=float(value),probability_late=float(p))
                for sid,value,p in zip(points.sample_id,prediction,probability,strict=True)]
        save(out/'frozen-test-predictions.json',frozen)
        # Только сейчас метки test становятся доступны оценщику.
        labels=pd.read_csv(data/'labels/labels_test.csv',usecols=['sample_id','target_delay_s'],dtype={'sample_id':str})
        scored=pd.DataFrame(frozen).merge(labels,on='sample_id',validate='one_to_one').merge(
            points[['sample_id','tr_id']],on='sample_id',validate='one_to_one')
        scored['label']=(scored.target_delay_s>120).astype(int)
        measured=probability_metrics(scored.label,scored.probability_late)
        measured['expected_calibration_error']=expected_calibration_error(measured)
        baseline=probability_metrics(scored.label,np.full(len(scored),constant))
        boot=cluster_improvement(scored.rename(columns={'probability_late':'gps_probability'}),constant)
        checks=dict(brier=measured['brier']<baseline['brier'],logloss=measured['log_loss']<baseline['log_loss'],
                    ece=measured['expected_calibration_error']<=.15,cluster_gain=boot['p05']>0)
        if time.monotonic()-started>RECIPE['budget_s']:raise TimeoutError('Бюджет 600с исчерпан')
        report=dict(status='accepted' if all(checks.values()) else 'rejected',checks=checks,test=measured,
                    constant_baseline=baseline,bootstrap=boot,train_rows=len(oof),test_rows=len(scored),
                    fold_audit=folds,sequence_audit=dict(train=train_audit,test=test_audit),
                    holdout_regression_parity_max_abs=parity,runtime_sequence_exact=True,
                    regression_weights_unchanged=True,independent_holdout=False,elapsed_s=time.monotonic()-started)
        save(out/'scored-test.json',scored.to_dict(orient='records'))
        if all(checks.values()):
            artifact=ProbabilityArtifact(schema_version='late-probability-logistic-v2',
                regression_model_sha256=MODEL_SHA256,encoder_sha256=ENCODER_SHA256,
                history_protocol='provided_current_delay_strict_past_90m_12steps',
                contract_sha256=sha(out/'contract.json'),sources_sha256=fingerprints,
                event='target_delay_s > 120',feature_offset_s=120.,feature_scale_s=120.,
                coefficient=coefficient,intercept=intercept,train_rows=len(oof),test_rows=len(scored),
                train_constant=constant,test_brier=measured['brier'],test_constant_brier=baseline['brier'],
                validation_status='accepted_secondary_test',scope='real_data_with_provided_current_deviation')
            save(out/'probability.json',artifact.model_dump())
            ProbabilityModel.load(out/'probability.json',MODEL_SHA256,expected_encoder_sha256=ENCODER_SHA256)
            report['probability_sha256']=sha(out/'probability.json')
        save(out/'report.json',report)
        return report
    except Exception as error:
        save(out/'report.json',dict(status='failed',error=f'{type(error).__name__}: {error}',elapsed_s=time.monotonic()-started))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT.parent.parent/'data/raw/dataset')
    parser.add_argument('--team',type=Path,default=ROOT.parent/'alexchist')
    parser.add_argument('--model',type=Path,default=ROOT/'artifacts/model/model.joblib')
    parser.add_argument('--out',type=Path,default=ROOT/'artifacts/probability-refresh')
    args=parser.parse_args()
    report=run(args.data,args.team,args.out,args.model)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
