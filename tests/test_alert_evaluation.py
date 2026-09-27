"""Границы окна, цензурирование, новый сбой и отделение oracle от входов."""
import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from tools.evaluate_alerts import IncidentTruth, evaluate_records, run_case

T = datetime(2026,1,6,8,tzinfo=timezone.utc)


def at(seconds):
    return T+timedelta(seconds=seconds)


def warning(*, tr_id=1, visit='target', issued=600, target=1400, cutoff=None):
    return dict(id=f'{tr_id}:{visit}',tr_id=tr_id,created_at=at(issued),
                data_cutoff=at(issued if cutoff is None else cutoff),
                target_time=at(target),predicted_delay_s=180,risk='red')


def event(*, name='incident', tr_id=1, onset=1500, visits=('target',), already_late=False):
    return IncidentTruth(event_id=name,tr_id=tr_id,onset_at=at(onset),
                         affected_visit_ids=visits,already_late=already_late)


def evaluate(alerts=None, *, plans=None, arrivals=None, events=None, start=0, end=3000):
    return evaluate_records([warning()] if alerts is None else alerts,
        {(1,'target'):at(1400)} if plans is None else plans,
        {(1,'target'):at(1600)} if arrivals is None else arrivals,
        [event()] if events is None else events,started_at=at(start),ended_at=at(end))


@pytest.mark.parametrize('lead,valid',[(600,False),(600.001,True),(900,True),(900.001,False)])
def test_plan_publication_window_is_lower_exclusive(lead,valid):
    result = evaluate([warning(issued=1400-lead)],events=[])
    assert result['planned_horizon']['valid']==int(valid)
    assert result['planned_horizon']['invalid']==int(not valid)


@pytest.mark.parametrize('lead,matched',[(599,False),(600,True),(900,True),(901,False)])
def test_actual_onset_window_is_independent_from_plan_window(lead,matched):
    issued = 1500-lead
    result = evaluate([warning(issued=issued,target=issued+800)],
                      plans={(1,'target'):at(issued+800)})
    assert result['planned_horizon']['valid']==1
    assert result['new_event_onsets']['eligible']==1
    assert result['new_event_onsets']['matched']==int(matched)


def test_future_cutoff_and_slow_publication_do_not_create_early_success():
    future = evaluate([warning(cutoff=601)])
    assert future['planned_horizon']['violations'][0]['reason']=='future_input_cutoff'
    assert future['new_event_onsets']['recall']==0
    slow = evaluate([warning(issued=810,cutoff=600)])
    assert slow['planned_horizon']['violations'][0]['reason']=='publication_outside_plan_window'
    assert slow['new_event_onsets']['recall']==0


def test_actual_arrival_horizon_does_not_claim_new_incident_onset():
    result = evaluate(arrivals={(1,'target'):at(1600)},events=[])
    assert result['planned_horizon']['fraction']==1
    assert result['target_outcomes']['late_precision']==1
    assert result['target_outcomes']['actual_arrival_10_15m']==0  # 1000 с до прибытия
    assert result['new_event_onsets']['recall'] is None
    assert result['new_event_onsets']['status'].startswith('not_assessable')


@pytest.mark.parametrize('arrivals',[{}, {(1,'target'):at(4000)}])
def test_unknown_and_unmatured_truth_is_not_a_negative(arrivals):
    result = evaluate(arrivals=arrivals,events=[])
    assert result['target_outcomes']['assessable']==0
    assert result['target_outcomes']['unmatured_or_unlabeled']==1
    assert result['target_outcomes']['late_precision'] is None


@pytest.mark.parametrize('delay,is_late',[(120,False),(120.001,True),(-300,False)])
def test_late_outcome_threshold_is_independent_of_display_risk(delay,is_late):
    result = evaluate(arrivals={(1,'target'):at(1400+delay)},events=[])
    assert result['target_outcomes']['late']==int(is_late)


@pytest.mark.parametrize('truth,reason',[
    (event(onset=-1),'preexisting_event'),
    (event(onset=3500),'unobserved_event'),
    (event(already_late=None),'prior_delay_unknown'),
    (event(already_late=True),'already_late_before_disruption'),
    (event(onset=899),'left_censored_warning_window'),
])
def test_incomplete_or_existing_event_windows_are_excluded_not_success(truth,reason):
    result = evaluate(events=[truth])
    assert result['new_event_onsets']['eligible']==0
    assert result['new_event_onsets']['recall'] is None
    assert result['new_event_onsets']['excluded']=={reason:1}


def test_first_publication_is_retained_and_one_alert_cannot_match_two_events():
    result = evaluate([warning(),warning(issued=650)],
                      events=[event(),event(name='second',onset=1501)])
    assert result['duplicate_alerts']==1 and result['unique_alerts']==1
    assert result['new_event_onsets']['matched']==1
    assert result['new_event_onsets']['recall']==.5
    assert result['new_event_onsets']['rows'][0]['lead_s']==900


@pytest.mark.parametrize('truth',[event(tr_id=2),event(visits=('other',))])
def test_event_match_requires_vehicle_and_affected_visit(truth):
    assert evaluate(events=[truth])['new_event_onsets']['matched']==0


def test_late_visit_is_only_a_proxy_and_missing_previous_visit_is_not_filled():
    plans={(1,'a'):at(100),(1,'b'):at(500),(1,'c'):at(900),(1,'d'):at(1200)}
    facts={(1,'a'):at(130),(1,'b'):at(680),(1,'d'):at(1500)}
    result = evaluate([],plans=plans,arrivals=facts,events=[])
    proxy=result['first_late_visit_proxies']
    assert len(proxy)==1 and proxy[0]['planned_stop_id']=='b'
    assert proxy[0]['previous_observed_at']==at(130)
    assert proxy[0]['first_late_visit_at']==at(680)
    assert proxy[0]['exact_onset_known'] is False
    assert result['new_event_onsets']['recall'] is None


def test_invalid_and_naive_inputs_fail_instead_of_claiming_success():
    alert=warning()
    alert['created_at']=datetime(2026,1,6,8)
    with pytest.raises(ValueError,match='часовой пояс'):
        evaluate([alert])
    with pytest.raises(ValueError,match='интервал'):
        evaluate(end=-1)


def test_synthetic_run_freezes_predictions_before_oracle_and_keeps_unmatured_unknown(monkeypatch):
    from tools import evaluate_alerts as module
    original_truth=module.GeneratorSession.truth
    calls=[]
    def truth(self):
        assert self.clock_time==module.START+timedelta(seconds=module.DURATION_S)
        calls.append(self.clock_time)
        return original_truth(self)
    monkeypatch.setattr(module.GeneratorSession,'truth',truth)
    result=asyncio.run(run_case('normal',route_count=1,forecast_interval=30))
    assert len(calls)==1
    assert result['oracle_read_after_prediction_freeze']
    assert result['predictions']==122 and result['ml_calls']==61
    assert result['prediction_plan_window_violations']==0
    assert result['planned_horizon']['invalid']==0
    assert result['target_outcomes']['unmatured_or_unlabeled']>0
    assert result['new_event_onsets']['recall'] is None
    assert result['execution']=={'ml_http':122}
