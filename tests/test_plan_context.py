"""Необрезанный план и семантика manual_fill при подключении frozen-модели."""
from datetime import datetime, timedelta, timezone

import pytest

from backend.engine import Engine, LiveContext, ScheduledStop, VehicleConfig
from common.contracts import ModelPlanContext, PredictionRequest, StopTarget, parse_manual_fill


T = datetime(2026, 1, 6, 11, 30, tzinfo=timezone.utc)


def stop(key, offset, manual=False):
    return StopTarget(id=key, name=key, scheduled_at=T+timedelta(seconds=offset), lat=55, lon=37,
                      manual_fill=manual)


def test_engine_keeps_full_plan_in_request_but_no_actual_times():
    stops = [stop('old', -3600), stop('target', 750, True), stop('later', 7200)]
    engine = Engine('http://ml')
    engine.set_live(LiveContext(vehicles=[VehicleConfig(tr_id=1, unit_id=1, label='1', route_id='r')],
        schedule=[ScheduledStop(tr_id=1, target=s) for s in stops], plan_complete=True,
        plan_version='audited-plan', plan_timezone='UTC'))
    request = engine.request_for(1, T)
    assert request.plan_context.stops == stops
    assert request.plan_context.complete and request.target.manual_fill
    assert request.current_delay_s is None
    assert 'time_fact_begin' not in request.model_dump_json()


@pytest.mark.parametrize('raw,expected', [('False',False), ('True',True), ('0',False), ('1',True), ('',None), (None,None)])
def test_manual_unknown_is_not_false(raw, expected):
    assert parse_manual_fill(raw) is expected


def test_incompatible_target_and_plan_rejected():
    target = stop('target', 750)
    plan = ModelPlanContext(version='v1', timezone='UTC', complete=True,
                            stops=[stop('target', 850)])
    with pytest.raises(ValueError, match='Цель должна совпадать'):
        PredictionRequest(request_id='1', tr_id=1, issued_at=T, target=target, plan_context=plan)


def test_claim_of_full_plan_needs_provenance():
    with pytest.raises(ValueError, match='версия и часовой пояс'):
        LiveContext(vehicles=[VehicleConfig(tr_id=1, unit_id=1, label='1', route_id='r')], plan_complete=True)


def test_explicit_point_target_resolves_plan_time_tie_only_when_available(monkeypatch):
    monkeypatch.setattr("backend.engine.utcnow", lambda: T+timedelta(seconds=1))
    from backend.engine import DelayHint
    a, b, later = stop('a',750), stop('b',750), stop('later',850)
    hint = DelayHint(tr_id=1, observed_at=T, received_at=T+timedelta(seconds=1), delay_s=90,
                     source='csv_snapshot', target_stop_id='b', target_time_begin=b.scheduled_at)
    context = LiveContext(vehicles=[VehicleConfig(tr_id=1,unit_id=1,label='1',route_id='r')],
                          schedule=[ScheduledStop(tr_id=1,target=s) for s in [a,b,later]], hints=[hint])
    e = Engine('http://ml'); e.set_live(context)
    assert e.request_for(1,T).target.id == 'a'  # подсказка ещё не получена
    assert e.request_for(1,T+timedelta(seconds=1)).target.id == 'b'
    assert e.request_for(1,T+timedelta(seconds=150)).target.id == 'later'  # b вышла из окна


def test_supplied_target_cannot_skip_an_earlier_plan_visit():
    from backend.engine import DelayHint
    hint = DelayHint(tr_id=1,observed_at=T,delay_s=90,target_stop_id='later',target_time_begin=stop('later',850).scheduled_at)
    with pytest.raises(ValueError,match='первым плановым'):
        LiveContext(vehicles=[VehicleConfig(tr_id=1,unit_id=1,label='1',route_id='r')],
                    schedule=[ScheduledStop(tr_id=1,target=stop('first',750)),
                              ScheduledStop(tr_id=1,target=stop('later',850))],hints=[hint])


def test_target_hint_requires_a_paired_id_time_and_strict_lower_boundary():
    from backend.engine import DelayHint
    with pytest.raises(ValueError,match='одновременно'):
        DelayHint(tr_id=1,observed_at=T,delay_s=90,target_stop_id='a')
    with pytest.raises(ValueError,match='плановом окне'):
        DelayHint(tr_id=1,observed_at=T,delay_s=90,target_stop_id='a',target_time_begin=T+timedelta(seconds=600))
