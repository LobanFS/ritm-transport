"""Постфактум reference не превращает будущее/неизвестное в доступный факт."""
from datetime import datetime,timedelta,timezone

from tools.audit_current_deviation import Visit,assess_prediction,hint_audit,latest_actual


T=datetime(2026,1,6,8,tzinfo=timezone.utc)


def visit(id,plan,actual):
    return Visit(id,1,T+timedelta(seconds=plan),
                 T+timedelta(seconds=actual) if actual is not None else None)


def prediction(id,arrived,received):
    return {'tr_id':1,'planned_stop_id':id,'arrived_at':(T+timedelta(seconds=arrived)).isoformat(),
            'received_at':(T+timedelta(seconds=received)).isoformat()}


def test_future_fact_can_explain_supplied_hint_but_is_never_last_actual_reference():
    visits=[visit('past',-90,-30),visit('future',0,120)]
    result=hint_audit(visits,T,120)
    assert result['reference'].id=='past' and result['reference'].delay_s==60
    assert result['actual_status']=='differs_from_last_actual'
    assert result['plan_status']=='matches_latest_plan_delay'
    assert result['all_matching_plan_facts_future'] is True


def test_equal_actual_times_stay_ambiguous_and_identical_visit_is_not_counted_twice():
    a=visit('a',-120,-30);b=visit('b',-60,-30)
    assert [v.id for v in latest_actual([a,a,b],T)]==['a','b']
    result=hint_audit([a,a,b],T,30)
    assert result['reference'] is None and result['actual_status']=='ambiguous_last_actual'


def test_no_actual_is_not_turned_into_zero_or_equal_to_initial_zero_hint():
    result=hint_audit([visit('future',60,120)],T,0)
    assert result['reference'] is None and result['actual_status']=='no_past_actual'
    assert result['plan_status']=='zero_before_first_plan'
    assert hint_audit([visit('unknown',-60,None)],T,0)['plan_status']=='latest_plan_reference_missing_fact'


def test_nonfinite_hints_never_participate_in_metrics_or_matches():
    visits=[visit('past',-60,-30)]
    for hint in ('nan','inf','-inf',''):
        result=hint_audit(visits,T,hint)
        assert result['hint'] is None and result['actual_status']=='hint_nonfinite'
        assert result['plan_status']=='hint_nonfinite' and not result['matching_plan_candidates']


def test_confirmation_availability_and_ttl_are_independent_and_boundary_is_inclusive():
    visits={s.id:s for s in [visit('old',-400,-300),visit('unconfirmed',-30,-10)]}
    rows=[prediction('old',-300,-290),prediction('unconfirmed',-10,1)]
    result=assess_prediction(rows,visits,T)
    assert result['status']=='fresh_detection' and result['prediction']['planned_stop_id']=='old'
    assert result['age_s']==300 and result['delay_s']==100
    expired=assess_prediction(rows[:1],visits,T+timedelta(seconds=1))
    assert expired['status']=='expired_detection'


def test_equal_latest_detection_times_remain_ambiguous_and_invalid_receipt_is_excluded():
    visits={s.id:s for s in [visit('a',-60,-30),visit('b',-90,-30)]}
    assert assess_prediction([prediction('a',-10,-5),prediction('b',-10,-5)],visits,T)['status']=='ambiguous_latest_detection'
    assert assess_prediction([prediction('a',-10,-20)],visits,T)['status']=='no_available_detection'
