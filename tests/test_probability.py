"""Вероятность не подменяет отсутствующие/устаревшие входы и не меняет секунды."""
import json
import math

import numpy as np
import pandas as pd
import pytest

from ml_service.probability import ProbabilityArtifact, ProbabilityModel, load_probability
from tools.prepare_probability import fit_train_only, probability_metrics, training_rows

SHA='a'*64


def artifact(**overrides):
    return dict(schema_version='late-probability-logistic-v1',regression_model_sha256=SHA,
        contract_sha256='b'*64,sources_sha256={'train':'c'*64},event='target_delay_s > 120',
        feature_offset_s=120.,feature_scale_s=120.,coefficient=1.,intercept=0.,
        train_rows=4434,test_rows=353,train_constant=.225,test_brier=.1,test_constant_brier=.2,
        validation_status='accepted_secondary_test',scope='real_data_with_provided_current_deviation',**overrides)


def model(tmp_path):
    path=tmp_path/'probability.json'
    path.write_text(json.dumps(artifact()))
    return ProbabilityModel.load(path,SHA),path


def test_logistic_mapping_is_numerically_correct_and_finite_for_extremes(tmp_path):
    calibrated,_=model(tmp_path)
    pred=lambda value:calibrated.predict(value,current_delay_s=0,telemetry_age_s=0)
    assert pred(120)==.5
    assert pred(240)==pytest.approx(1/(1+math.exp(-1)))
    assert pred(0)==pytest.approx(1/(1+math.exp(1)))
    assert pred(-1e308)==0 and pred(1e308)==1
    assert [pred(v) for v in (-120,0,120,240)]==sorted(pred(v) for v in (-120,0,120,240))


@pytest.mark.parametrize('field,value',[
    ('predicted_delay_s',None),('predicted_delay_s',float('nan')),('predicted_delay_s',float('inf')),
    ('predicted_delay_s',True),('current_delay_s',None),('current_delay_s',float('nan')),
    ('telemetry_age_s',None),('telemetry_age_s',-1),('telemetry_age_s',60.001),
    ('telemetry_age_s',float('inf')),('telemetry_age_s','bad'),
])
def test_unknown_invalid_or_stale_returns_null(tmp_path,field,value):
    calibrated,_=model(tmp_path)
    values=dict(predicted_delay_s=180,current_delay_s=160,telemetry_age_s=15)
    values[field]=value
    assert calibrated.predict(**values) is None


def test_freshness_boundary_sixty_seconds_is_inclusive(tmp_path):
    calibrated,_=model(tmp_path)
    assert calibrated.predict(120,current_delay_s=0,telemetry_age_s=60)==.5


def test_model_hash_mismatch_is_unavailable_not_a_wrong_probability(tmp_path):
    calibrated,path=model(tmp_path)
    with pytest.raises(ValueError,match='другому'):
        ProbabilityModel.load(path,'d'*64)
    loaded,error=load_probability('d'*64,path)
    assert loaded is None and 'другому' in error
    assert calibrated.metadata()['event']=='target_delay_s > 120'
    assert len(calibrated.metadata()['artifact_sha256'])==64


def test_no_implicit_activation_and_explicit_env_path(monkeypatch,tmp_path):
    _,path=model(tmp_path)
    monkeypatch.delenv('PROBABILITY_PATH',raising=False)
    assert load_probability(SHA)==(None,None)
    monkeypatch.setenv('PROBABILITY_PATH',str(path))
    loaded,error=load_probability(SHA)
    assert loaded is not None and error is None


@pytest.mark.parametrize('changes',[
    {'test_brier':.2}, {'test_brier':.3}, {'coefficient':float('nan')},
    {'sources_sha256':{}}, {'sources_sha256':{'train':'bad'}},
    {'feature_scale_s':1.}, {'unknown_parameter':42},
])
def test_invalid_unaccepted_or_untracked_artifact_cannot_load(tmp_path,changes):
    payload=artifact();payload.update(changes)
    path=tmp_path/'bad.json';path.write_text(json.dumps(payload))
    loaded,error=load_probability(SHA,path)
    assert loaded is None and error


def test_corrupt_or_missing_json_degrades_to_none(tmp_path):
    path=tmp_path/'bad.json';path.write_text('{')
    assert load_probability(SHA,path)[0] is None
    assert load_probability(SHA,tmp_path/'missing.json')[0] is None


def test_metrics_are_independently_checkable_and_bins_include_endpoints():
    result=probability_metrics([0,1],[.25,.75])
    assert result['brier']==.0625
    assert result['log_loss']==pytest.approx(-math.log(.75))
    assert sum(row['rows'] for row in result['reliability'])==2
    edges=probability_metrics([0,1],[0,1])
    assert edges['brier']==0 and math.isfinite(edges['log_loss'])
    assert edges['reliability'][0]['rows']==edges['reliability'][-1]['rows']==1
    assert edges['reliability'][1]['mean_probability'] is None
    with pytest.raises(ValueError):
        probability_metrics([0,float('nan')],[.1,.5])
    with pytest.raises(ValueError):
        probability_metrics([0,1],[.1,1.1])


def small_training(tmp_path,monkeypatch):
    from tools import prepare_probability as module
    monkeypatch.setitem(module.RECIPE,'expected_train_rows',6)
    team,data=tmp_path/'team',tmp_path/'data'
    final=team/'artifacts/final_model';final.mkdir(parents=True)
    (data/'labels').mkdir(parents=True)
    labels=pd.DataFrame(dict(sample_id=list('abcdef'),cur_dev_s=[0,50,100,150,200,250],
                             target_delay_s=[0,70,60,180,300,240]))
    labels.to_csv(data/'labels/labels_train.csv',index=False)
    rows=[];folds=[]
    for n,name in enumerate(module.RECIPE['folds']):
        valid=list('abcdef')[n*2:n*2+2]
        folds.append(dict(name=name,validation_sample_ids=valid,train_sample_ids=[v for v in 'abcdef' if v not in valid]))
        for sample in valid:
            value=labels.set_index('sample_id').loc[sample,'cur_dev_s']
            rows.append(dict(fold=name,sample_id=sample,baseline=value,prediction=value))
    # Time-fold duplicate is deliberately present and must not enter calibration.
    rows.append(dict(fold='time_01',sample_id='a',baseline=0,prediction=999))
    pd.DataFrame(rows).to_csv(final/'fold_predictions.csv',index=False)
    (final/'frozen_folds.json').write_text(json.dumps(folds))
    return team,data,final


def test_only_unique_vehicle_oof_is_used_and_fit_never_reads_test(tmp_path,monkeypatch):
    team,data,_=small_training(tmp_path,monkeypatch)
    rows=training_rows(team,data)
    assert len(rows)==6 and rows.sample_id.nunique()==6
    assert rows.loc[rows.sample_id=='a','prediction'].item()==0
    def forbidden(*args,**kwargs):
        raise AssertionError('fit не должен читать файлы, особенно test')
    monkeypatch.setattr(pd,'read_csv',forbidden)
    fitted,constant=fit_train_only(rows)
    assert constant==.5 and np.isfinite(fitted.coef_).all()


def test_duplicate_oof_and_train_validation_overlap_fail(tmp_path,monkeypatch):
    team,data,final=small_training(tmp_path,monkeypatch)
    original=pd.read_csv(final/'fold_predictions.csv')
    pd.concat([original,original.iloc[:1]]).to_csv(final/'fold_predictions.csv',index=False)
    with pytest.raises(ValueError,match='ровно один'):
        training_rows(team,data)
    original.to_csv(final/'fold_predictions.csv',index=False)
    folds=json.loads((final/'frozen_folds.json').read_text())
    folds[0]['train_sample_ids'].append('a')
    (final/'frozen_folds.json').write_text(json.dumps(folds))
    with pytest.raises(ValueError,match='membership'):
        training_rows(team,data)


def test_gps_probability_diagnostic_freezes_without_labels_or_file_access(tmp_path, monkeypatch):
    from tools.gps_probability_evaluate import freeze_predictions
    calibrated, _ = model(tmp_path)
    gps = dict(sample_id='gps-1', tr_id=1, T='2026-01-06T09:00:00Z',
               method='learned', input_source='gps_estimate', input_cur_dev_s=10,
               prediction=120, telemetry_age_s=10)
    csv = dict(gps, input_source='csv_snapshot', prediction=240)
    def forbidden(*args, **kwargs):
        raise AssertionError('Freeze не должен читать метки или файлы')
    monkeypatch.setattr(pd, 'read_csv', forbidden)
    rows = freeze_predictions([gps], [csv], calibrated)
    assert rows[0]['gps_probability'] == .5
    assert rows[0]['csv_probability'] == pytest.approx(1/(1+math.exp(-1)))
    assert 'label' not in rows[0] and 'target_delay_s' not in rows[0]


@pytest.mark.parametrize('changes', [
    {'input_source': 'door_estimate'}, {'method': 'unavailable'},
    {'input_cur_dev_s': None}, {'prediction': None}, {'telemetry_age_s': 61},
])
def test_gps_probability_diagnostic_never_counts_unknown_as_zero_or_transfers_doors(tmp_path, changes):
    from tools.gps_probability_evaluate import freeze_predictions
    calibrated, _ = model(tmp_path)
    gps = dict(sample_id='gps-1', tr_id=1, T='2026-01-06T09:00:00Z',
               method='learned', input_source='gps_estimate', input_cur_dev_s=10,
               prediction=120, telemetry_age_s=10)
    csv = dict(gps, input_source='csv_snapshot')
    gps.update(changes)
    assert freeze_predictions([gps], [csv], calibrated) == []


def test_gps_probability_diagnostic_requires_matching_reference(tmp_path):
    from tools.gps_probability_evaluate import freeze_predictions
    calibrated, _ = model(tmp_path)
    with pytest.raises(ValueError, match='CSV-reference'):
        freeze_predictions([{'sample_id': 'missing'}], [], calibrated)


def test_gps_probability_diagnostic_rejects_reused_sample_id_for_a_different_target(tmp_path):
    from tools.gps_probability_evaluate import freeze_predictions
    calibrated, _ = model(tmp_path)
    gps = dict(sample_id='gps-1', tr_id=1, T='2026-01-06T09:00:00Z',
               target_stop_id='a', method='learned', input_source='gps_estimate',
               input_cur_dev_s=10, prediction=120, telemetry_age_s=10)
    csv = dict(gps, input_source='csv_snapshot', target_stop_id='b')
    with pytest.raises(ValueError, match='разным целям'):
        freeze_predictions([gps], [csv], calibrated)


def test_gps_shrinkage_prior_purges_whole_timestamp_and_never_fits_late_or_unavailable_rows():
    from tools.gps_probability_shrinkage import mature_prior
    cutoff = pd.Timestamp('2026-01-06T12:00:00Z')
    labels = pd.DataFrame([
        dict(sample_id='early', T='2026-01-06T10:00:00Z', target_time_begin='2026-01-06T10:15:00Z', target_delay_s=200),
        dict(sample_id='available', T='2026-01-06T11:00:00Z', target_time_begin='2026-01-06T11:15:00Z', target_delay_s=0),
        dict(sample_id='immature', T='2026-01-06T11:45:00Z', target_time_begin='2026-01-06T11:59:00Z', target_delay_s=60),
        dict(sample_id='same-T', T='2026-01-06T11:45:00Z', target_time_begin='2026-01-06T11:56:00Z', target_delay_s=0),
        dict(sample_id='late', T='2026-01-06T12:00:00Z', target_time_begin='2026-01-06T12:15:00Z', target_delay_s=1000),
        dict(sample_id='no-gps', T='2026-01-06T10:30:00Z', target_time_begin='2026-01-06T10:45:00Z', target_delay_s=1000),
    ])
    frozen = pd.DataFrame([dict(sample_id=identity,split='train',status='available' if identity!='no-gps' else 'no_gps_deviation')
                           for identity in labels.sample_id])
    prior, report = mature_prior(labels, frozen, cutoff)
    assert prior == .5  # Laplace (1+1)/(2+2)
    assert report['selected_sample_ids'] == ['early','available']
    assert report['purged_sample_ids'] == ['immature','same-T']
    assert report['immature_rows'] == 1 and report['purged_rows'] == 2
    assert report['label_availability_mode'] == 'arrival_lower_bound_only'


def gps_scope_fixture(tmp_path, calibrated, **changes):
    """Искусственная положительная проверка протокола, не измеренные метрики."""
    scope = dict(schema_version='gps-probability-scope-v1',validation_status='accepted_secondary_diagnostic',
        scope='historical_real_gps_estimate',regression_model_sha256=calibrated.artifact.regression_model_sha256,
        probability_artifact_sha256=calibrated.artifact_sha256,detector_version='fixture-v1',detector_sha256='d'*64,
        contract_sha256='e'*64,report_sha256='f'*64,report_path='fixture.json',evaluated_rows=60,total_rows=353,
        positives=15,vehicles=10,brier=.1,constant_brier=.2,log_loss=.3,constant_log_loss=.5,
        cluster_improvement_p05=.01,expected_calibration_error=.05)
    scope.update(changes)
    path = tmp_path/'gps-scope.json'
    path.write_text(json.dumps(scope))
    return path


@pytest.mark.parametrize('source,domain,version,digest,expected',[
    ('gps_estimate','historical_real','fixture-v1','d'*64,True),
    ('gps_estimate','historical_mixed','fixture-v1','d'*64,False),
    ('gps_estimate','live_unverified','fixture-v1','d'*64,False),
    ('gps_estimate','synthetic','fixture-v1','d'*64,False),
    ('gps_estimate','unknown','fixture-v1','d'*64,False),
    ('door_estimate','historical_real','fixture-v1','d'*64,False),
    ('gps_estimate','historical_real','old-v1','d'*64,False),
    ('gps_estimate','historical_real','fixture-v1','a'*64,False),
    ('gps_estimate','historical_real',None,None,False),
    ('csv_snapshot','unknown',None,None,True),
])
def test_accepted_gps_scope_is_bound_to_domain_detector_and_coefficients(tmp_path,source,domain,version,digest,expected):
    calibrated, path = model(tmp_path)
    scope_path = gps_scope_fixture(tmp_path,calibrated)
    loaded = ProbabilityModel.load(path,SHA,gps_scope_path=scope_path)
    assert loaded.gps_scope is not None and loaded.gps_scope_error is None
    assert loaded.source_allowed(source,telemetry_domain=domain,detector_version=version,detector_sha256=digest) is expected
    assert len(loaded.metadata()['gps_scope_sha256']) == 64


@pytest.mark.parametrize('domain,source,version,digest,expected',[
    ('historical_real','gps_estimate','fixture-v1','d'*64,'validated'),
    ('historical_mixed','gps_estimate','fixture-v1','d'*64,'transferred'),
    ('live_unverified','gps_estimate','fixture-v1','d'*64,'unavailable'),
    ('synthetic','gps_estimate','fixture-v1','d'*64,'unavailable'),
    ('unknown','gps_estimate','fixture-v1','d'*64,'unavailable'),
    ('historical_mixed','door_estimate','fixture-v1','d'*64,'unavailable'),
    ('historical_mixed','external_hint','fixture-v1','d'*64,'unavailable'),
    ('historical_mixed','gps_estimate','old-v1','d'*64,'unavailable'),
    ('historical_mixed','gps_estimate','fixture-v1','a'*64,'unavailable'),
    ('historical_mixed','gps_estimate',None,None,'unavailable'),
])
def test_train_probability_transfer_is_explicit_and_keeps_detector_gate(tmp_path,domain,source,version,digest,expected):
    calibrated, path = model(tmp_path)
    scope_path = gps_scope_fixture(tmp_path,calibrated)
    loaded = ProbabilityModel.load(path,SHA,gps_scope_path=scope_path)
    assert loaded.source_status(source,telemetry_domain=domain,detector_version=version,
                                detector_sha256=digest) == expected
    if expected == 'transferred':
        assert not loaded.source_allowed(source,telemetry_domain=domain,detector_version=version,
                                         detector_sha256=digest)


@pytest.mark.parametrize('changes',[
    {'probability_artifact_sha256':'0'*64}, {'regression_model_sha256':'0'*64},
    {'brier':.3}, {'log_loss':.6}, {'cluster_improvement_p05':0}, {'evaluated_rows':49},
    {'expected_calibration_error':.16}, {'positives':59}, {'total_rows':50},
])
def test_bad_gps_scope_keeps_csv_probability_but_cannot_admit_gps(tmp_path,changes):
    calibrated, path = model(tmp_path)
    scope_path = gps_scope_fixture(tmp_path,calibrated,**changes)
    loaded = ProbabilityModel.load(path,SHA,gps_scope_path=scope_path)
    assert loaded.gps_scope is None and loaded.gps_scope_error
    assert loaded.source_allowed('csv_snapshot')
    assert loaded.predict(120,current_delay_s=0,telemetry_age_s=0) == .5
    assert not loaded.source_allowed('gps_estimate',telemetry_domain='historical_real',
                                     detector_version='fixture-v1',detector_sha256='d'*64)
    assert loaded.source_status('gps_estimate',telemetry_domain='historical_mixed',
                                detector_version='fixture-v1',detector_sha256='d'*64) == 'unavailable'


def test_rejected_probability_transfer_cannot_export_runtime_profile(tmp_path):
    from tools.gps_probability_evaluate import export_scope
    result, gps = tmp_path/'result', tmp_path/'gps'
    result.mkdir(); gps.mkdir()
    (result/'report.json').write_text(json.dumps(dict(status='rejected_transfer',checks={'bootstrap':False})))
    (result/'contract.json').write_text('{}')
    (gps/'contract.json').write_text('{}')
    (gps/'report.json').write_text('{}')
    destination = tmp_path/'scope.json'
    with pytest.raises(ValueError, match='gates'):
        export_scope(gps,tmp_path/'unused.json',result,destination)
    assert not destination.exists()


def test_hybrid_calibration_pins_both_hgbr_and_encoder(tmp_path):
    payload=artifact()
    payload.update(schema_version='late-probability-logistic-v2',encoder_sha256='d'*64,
                   history_protocol='provided_current_delay_strict_past_90m_12steps')
    path=tmp_path/'hybrid.json';path.write_text(json.dumps(payload))
    valid,error=load_probability(SHA,path,expected_encoder_sha256='d'*64)
    assert valid is not None and error is None
    rejected,error=load_probability(SHA,path,expected_encoder_sha256='e'*64)
    assert rejected is None and 'encoder' in error
    assert valid.metadata()['encoder_sha256']=='d'*64


def test_old_single_model_calibrator_cannot_be_attached_to_hybrid_encoder(tmp_path):
    _,path=model(tmp_path)
    rejected,error=load_probability(SHA,path,expected_encoder_sha256='d'*64)
    assert rejected is None and 'encoder' in error


def test_hybrid_schema_requires_explicit_encoder_and_history_contract():
    value=artifact();value['schema_version']='late-probability-logistic-v2'
    with pytest.raises(ValueError,match='encoder'):
        ProbabilityArtifact.model_validate(value)


def test_hybrid_probability_sequence_projection_has_no_future_dependency():
    from tools.refresh_hybrid_probability import causal_arrays,POINTS
    points=pd.DataFrame([
        dict(sample_id='a',tr_id=1,T='2026-01-06T08:00:00Z',target_stop_id='s1',target_time_begin='2026-01-06T08:12:00Z',cur_dev_s=10),
        dict(sample_id='b',tr_id=1,T='2026-01-06T08:05:00Z',target_stop_id='s2',target_time_begin='2026-01-06T08:17:00Z',cur_dev_s=20),
    ],columns=POINTS)
    plan=pd.DataFrame([dict(cur_dev_s=10,horizon_min=12,stops_ahead=2,state_rel_pos=.2,route_len=10),
                       dict(cur_dev_s=20,horizon_min=12,stops_ahead=2,state_rel_pos=.3,route_len=10)])
    original,_,audit=causal_arrays(points,plan)
    changed=points.copy();changed.loc[1,'cur_dev_s']=999
    mutated,_,_=causal_arrays(changed,plan)
    assert np.array_equal(original[0],mutated[0])
    assert original[0,:,-1].sum()==1 and original[1,:,-1].sum()==2
    assert audit['future_observations']==0
    with pytest.raises(ValueError,match='разрешённые'):
        causal_arrays(points.assign(target_delay_s=0),plan)
