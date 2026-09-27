"""Один frozen train-only logistic-рецепт; test используется только после fit.

Предварительно фиксируется artifacts/probability/contract.json. Ни estimator,
ни split, ни порог события не подбираются. На test проверяется единственный
критерий Brier < константы частоты train; при отказе runtime JSON не сохраняется.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ml_service.learned import MODEL_SHA256
from ml_service.probability import ProbabilityArtifact


RECIPE=dict(name='late-probability-logistic-v1',folds=['vehicle_01','vehicle_02','vehicle_03'],
    feature='(prediction-120)/120',event='target_delay_s > 120',
    estimator=dict(C=1.0,solver='lbfgs',max_iter=1000,class_weight=None,random_state=0),
    baseline='998/4434: empirical late frequency over all unique training labels',
    acceptance='secondary test Brier strictly below train-constant Brier',
    reliability_bins=[0,.2,.4,.6,.8,1],budget_seconds=60,
    no_search=True,no_test_fit=True,no_regression_retraining=True,
    expected_train_rows=4434,expected_train_positives=998,expected_test_rows=353,
    scope='real_data_with_provided_current_deviation',
    limitations=['Vehicle-OOF covers unseen vehicles, not a new independent day.',
                 'Regressor/model family had already been selected in prior research.',
                 'All-train regressor differs from OOF fold estimators.',
                 'Official test was used previously in the research; this is secondary confirmation.',
                 'No claim of calibration for GPS estimates or synthetic generator.'])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def probability_metrics(y,p):
    y,p=np.asarray(y,dtype=float),np.asarray(p,dtype=float)
    if y.ndim!=1 or y.shape!=p.shape or not len(y) or not np.isfinite(p).all() or not np.isin(y,[0,1]).all() or ((p<0)|(p>1)).any():
        raise ValueError('Неверные вероятности/метки')
    clipped=np.clip(p,1e-15,1-1e-15)
    bins=[]
    for index,(lo,hi) in enumerate(zip(RECIPE['reliability_bins'],RECIPE['reliability_bins'][1:])):
        mask=(p>=lo)&((p<hi) if index<4 else (p<=hi))
        bins.append(dict(lower=lo,upper=hi,upper_inclusive=index==4,rows=int(mask.sum()),
                         mean_probability=float(p[mask].mean()) if mask.any() else None,
                         observed_frequency=float(y[mask].mean()) if mask.any() else None))
    return dict(rows=len(y),positives=int(y.sum()),brier=float(np.square(y-p).mean()),
                log_loss=float(-(y*np.log(clipped)+(1-y)*np.log1p(-clipped)).mean()),
                reliability=bins)


def training_rows(team: Path,data: Path):
    predictions=pd.read_csv(team/'artifacts/final_model/fold_predictions.csv',dtype={'sample_id':str,'fold':str})
    folds=json.loads((team/'artifacts/final_model/frozen_folds.json').read_text())
    labels=pd.read_csv(data/'labels/labels_train.csv',dtype={'sample_id':str})
    if labels.sample_id.duplicated().any() or len(labels)!=RECIPE['expected_train_rows']:
        raise ValueError('Нарушен контракт train IDs/числа строк')
    selected=predictions.loc[predictions.fold.isin(RECIPE['folds'])].copy()
    if selected.sample_id.duplicated().any() or set(selected.sample_id)!=set(labels.sample_id):
        raise ValueError('vehicle OOF должен покрывать train ровно один раз')
    for name in RECIPE['folds']:
        matching=[f for f in folds if f['name']==name]
        if len(matching)!=1:
            raise ValueError('Неоднозначные frozen fold IDs')
        fold=matching[0]
        train_ids,valid_ids=set(fold['train_sample_ids']),set(fold['validation_sample_ids'])
        if (train_ids & valid_ids or train_ids|valid_ids!=set(labels.sample_id)
            or len(valid_ids)!=len(fold['validation_sample_ids'])
            or set(selected.loc[selected.fold==name,'sample_id'])!=valid_ids):
            raise ValueError('OOF/frozen membership не совпал')
    joined=selected.merge(labels[['sample_id','target_delay_s','cur_dev_s']],on='sample_id',validate='one_to_one')
    if not np.isfinite(joined[['prediction','target_delay_s','cur_dev_s']].to_numpy()).all():
        raise ValueError('Нечисловые train значения')
    if not np.array_equal(joined.baseline.to_numpy(),joined.cur_dev_s.to_numpy()):
        raise ValueError('Baseline OOF не совпал с cur_dev_s выданных labels')
    return joined


def fit_train_only(rows):
    """Функция не получает путь test/его данные: только уникальные train OOF."""
    x=((rows.prediction.to_numpy()-120)/120).reshape(-1,1)
    y=(rows.target_delay_s.to_numpy()>120).astype(int)
    if len(np.unique(y))!=2:
        raise ValueError('Для калибровки нужны оба класса')
    model=LogisticRegression(**RECIPE['estimator']).fit(x,y)
    if int(model.n_iter_.max())>=RECIPE['estimator']['max_iter']:
        raise ValueError('Калибратор не сошёлся в frozen budget')
    return model,float(y.mean())


def test_rows(team: Path,data: Path):
    # Вызывать только ПОСЛЕ fit_train_only и freeze coefficients.
    predictions=pd.read_csv(team/'artifacts/final_model/test_predictions.csv',dtype={'sample_id':str})
    labels=pd.read_csv(data/'labels/labels_test.csv',dtype={'sample_id':str})
    if len(labels)!=RECIPE['expected_test_rows'] or labels.sample_id.duplicated().any() or predictions.sample_id.duplicated().any():
        raise ValueError('Неверное число test строк/дубли')
    if set(labels.sample_id)!=set(predictions.sample_id):
        raise ValueError('Неполное one-to-one test сопоставление')
    rows=labels[['sample_id','target_delay_s']].merge(predictions,on='sample_id',validate='one_to_one')
    if not np.isfinite(rows[['prediction','target_delay_s']].to_numpy()).all():
        raise ValueError('Нечисловые test значения')
    return rows


def prepare(team: Path,data: Path,out: Path,runtime_path: Path):
    source=team/'artifacts/final_model'
    started=time.monotonic()
    train_inputs={name:sha(path) for name,path in {
        'team/fold_predictions.csv':source/'fold_predictions.csv',
        'team/frozen_folds.json':source/'frozen_folds.json',
        'team/model.joblib':source/'model.joblib',
        'team/acceptance.json':source/'acceptance.json',
        'dataset/labels_train.csv':data/'labels/labels_train.csv'}.items()}
    contract=dict(RECIPE,regression_model_sha256=MODEL_SHA256,train_sources_sha256=train_inputs,
                  code_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'ml_service/probability.py')},
                  versions={name:version(name) for name in ('scikit-learn','numpy','pandas')})
    save(out/'contract.json',contract)  # До fit и до первого чтения test.
    save(out/'report.json',dict(status='running',contract=contract))
    try:
        if train_inputs['team/model.joblib']!=MODEL_SHA256:
            raise ValueError('Frozen regression artifact hash не совпал')
        acceptance=json.loads((source/'acceptance.json').read_text())
        if (acceptance['status']!='PASS'
            or acceptance['fingerprint']['labels/labels_train.csv']!=train_inputs['dataset/labels_train.csv']):
            raise ValueError('Train labels не соответствуют frozen regression provenance')
        train=training_rows(team,data)
        if int((train.target_delay_s>120).sum())!=RECIPE['expected_train_positives']:
            raise ValueError('Изменилось число положительных train меток')
        model,constant=fit_train_only(train)
        coefficients=dict(coefficient=float(model.coef_[0,0]),intercept=float(model.intercept_[0]))
        frozen=dict(**coefficients,train_constant=constant,train_rows=len(train),
                    regression_model_sha256=MODEL_SHA256,contract_sha256=sha(out/'contract.json'))
        save(out/'frozen-fit.json',frozen)  # Веса/рецепт зафиксированы до test.
        test=test_rows(team,data)
        p=model.predict_proba(((test.prediction.to_numpy()-120)/120).reshape(-1,1))[:,1]
        y=(test.target_delay_s.to_numpy()>120).astype(int)
        calibrated=probability_metrics(y,p)
        baseline=probability_metrics(y,np.full(len(y),constant))
        accepted=calibrated['brier']<baseline['brier']
        sources={**train_inputs,'team/test_predictions.csv':sha(source/'test_predictions.csv'),
                 'dataset/labels_test.csv':sha(data/'labels/labels_test.csv')}
        elapsed=time.monotonic()-started
        if elapsed>RECIPE['budget_seconds']:
            raise TimeoutError('Превышен предварительно заданный бюджет 60с')
        report=dict(status='accepted' if accepted else 'rejected',contract=contract,
            created_at=datetime.now(timezone.utc).isoformat(),elapsed_s=round(elapsed,3),
            test=calibrated,constant_baseline=baseline,brier_improvement=baseline['brier']-calibrated['brier'],
            frozen_fit=frozen,source_sha256=sources,train_test_ids_disjoint=not bool(set(train.sample_id)&set(test.sample_id)),
            regression_predictions_unchanged=True,threshold_selected_on_test=False)
        if not report['train_test_ids_disjoint']:
            raise ValueError('Train/test sample IDs пересекаются')
        with (out/'test-probabilities.jsonl').open('w') as stream:
            for sample_id,probability,label in zip(test.sample_id,p,y,strict=True):
                stream.write(json.dumps(dict(sample_id=sample_id,probability_late=float(probability),late_gt120=bool(label)))+'\n')
        if accepted:
            artifact=ProbabilityArtifact(schema_version=RECIPE['name'],regression_model_sha256=MODEL_SHA256,
                contract_sha256=frozen['contract_sha256'],sources_sha256=sources,
                event=RECIPE['event'],feature_offset_s=120.,feature_scale_s=120.,**coefficients,
                train_rows=len(train),test_rows=len(test),train_constant=constant,
                test_brier=calibrated['brier'],test_constant_brier=baseline['brier'],
                validation_status='accepted_secondary_test',scope=RECIPE['scope'])
            save(runtime_path,artifact.model_dump())
            report.update(runtime_artifact=str(runtime_path),runtime_sha256=sha(runtime_path))
        else:
            runtime_path.unlink(missing_ok=True)
            report.update(runtime_artifact=None,runtime_sha256=None)
        save(out/'report.json',report)
        return report
    except Exception as error:
        save(out/'report.json',dict(status='failed',contract=contract,error=f'{type(error).__name__}: {error}',
                                   elapsed_s=round(time.monotonic()-started,3)))
        runtime_path.unlink(missing_ok=True)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--team-dir',type=Path,default=ROOT.parent/'alexchist')
    parser.add_argument('--data-dir',type=Path,default=ROOT.parent.parent/'data/raw/dataset')
    parser.add_argument('--out',type=Path,default=ROOT/'artifacts/probability')
    parser.add_argument('--runtime-out',type=Path,default=ROOT/'artifacts/model/probability.json')
    args=parser.parse_args()
    report=prepare(args.team_dir,args.data_dir,args.out,args.runtime_out)
    print(json.dumps({k:report[k] for k in ('status','elapsed_s','brier_improvement','runtime_artifact')},ensure_ascii=False))
    print('Brier:',report['test']['brier'],'constant:',report['constant_baseline']['brier'])
    print(args.out/'report.json')


if __name__=='__main__':
    main()
