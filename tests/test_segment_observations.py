"""Причинность и границы геометрического наблюдения; не валидация map matching."""
from datetime import datetime,timedelta,timezone
import math

import pytest

from backend.arrivals import CurrentDeviation
from backend.segment_observations import observe_segment
from common.contracts import StopTarget,Telemetry


T=datetime(2026,1,6,8,tzinfo=timezone.utc)
LAT,LON=55.75,37.6


def stop(id,x,seconds,y=0):
    return StopTarget(id=id,name=id,scheduled_at=T+timedelta(seconds=seconds),
        lat=LAT+y/111195,lon=LON+x/(111195*math.cos(math.radians(LAT))))


PLAN=[stop('origin',0,-120),stop('next',1000,180),stop('future',1500,780)]


def anchor(**changes):
    data=dict(delay_s=0,source='arrival',planned_stop_id='origin',
        observed_at=T-timedelta(seconds=120),received_at=T-timedelta(seconds=118),
        planned_at=PLAN[0].scheduled_at)
    data.update(changes)
    return CurrentDeviation(**data)


def gps(seconds,x,y=0,speed=24,heading=90,**changes):
    position=stop('gps',x,seconds,y)
    data=dict(tr_id=101,event_time=T+timedelta(seconds=seconds),
        received_at=T+timedelta(seconds=seconds),lat=position.lat,lon=position.lon,
        speed_kmh=speed,heading=heading)
    data.update(changes)
    return Telemetry(**data)


def observe(history,at=T,current=None,plan=PLAN):
    return observe_segment(history,at,plan,current or anchor())


def moving():
    return [gps(-45,100,speed=12),gps(-30,200,speed=24),gps(-15,300,speed=36),gps(0,400,speed=24)]


def test_mean_is_only_causal_unique_points_and_segment_is_next_visit_not_forecast_target():
    history=moving()
    history += [history[1],gps(15,500,speed=90),gps(-10,350,speed=90,received_at=T+timedelta(seconds=1))]
    result=observe(history)
    assert result.status=='available'
    assert result.from_stop_id=='origin' and result.to_stop_id=='next'
    assert result.to_stop_id != 'future'  # цель ML через13мин не активный перегон
    assert result.speed_mean_kmh==24
    assert result.selected_points==result.considered_points==4
    assert result.coverage_fraction==1
    assert result.observed_span_s==45
    assert result.current_idle_s is None
    assert result.method=='planned_chord'
    assert result.model_dump(mode='json')['evaluated_at'].startswith('2026-01-06T08:00:00')


def test_stopped_tail_is_observation_without_extrapolating_between_packets():
    history=[gps(-75,200),gps(-60,300)]+[gps(t,400,speed=0,heading=0) for t in (-45,-30,-15,0)]
    result=observe(history)
    assert result.status=='available'
    assert result.current_idle_s==45 and result.idle_points==4
    assert result.speed_mean_kmh==8
    assert observe(history,at=T+timedelta(seconds=20)).current_idle_s==45
    stale=observe(history,at=T+timedelta(seconds=31))
    assert stale.status=='unknown' and stale.current_idle_s is None
    assert stale.speed_mean_kmh is None and stale.reason=='stale_gps'


def test_stationary_points_alone_do_not_establish_segment_direction():
    result=observe([gps(t,400,speed=0) for t in (-60,-45,-30,-15,0)])
    assert result.status=='unknown' and result.reason=='direction_not_observed'
    assert result.speed_mean_kmh is None and result.current_idle_s is None


@pytest.mark.parametrize('x,y,reason',[(400,80,'outside_planned_chord'),(20,0,'inside_stop_zone'),
                                     (980,0,'inside_stop_zone'),(1100,0,'outside_planned_chord')])
def test_outside_chord_or_at_stop_is_not_segment_speed(x,y,reason):
    result=observe(moving()[:-1]+[gps(0,x,y=y)])
    assert result.status=='unknown' and result.reason==reason
    assert result.speed_mean_kmh is None


def test_reverse_or_perpendicular_movement_and_heading_are_rejected():
    reverse=observe([gps(-15,400,heading=None),gps(0,300,heading=None)])
    assert reverse.status=='unknown' and reverse.reason=='movement_disagrees_with_chord'
    sideways=observe([gps(-15,300,y=-30,heading=None),gps(0,300,y=30,heading=None)])
    assert sideways.status=='unknown' and sideways.reason=='movement_disagrees_with_chord'
    heading=observe([gps(-15,300),gps(0,400,heading=270)])
    assert heading.reason=='heading_disagrees_with_chord'


def test_speed_does_not_attribute_teleport_or_drift_to_the_segment():
    teleport=observe([gps(-1,100),gps(0,900)])
    assert teleport.reason=='implausible_displacement' and teleport.speed_mean_kmh is None
    drift=observe([gps(-15,300,speed=0),gps(0,350,speed=0)])
    assert drift.reason=='speed_position_conflict' and drift.current_idle_s is None


@pytest.mark.parametrize('change',[dict(location_valid=False),dict(speed_kmh=None),dict(speed_kmh=131),
                                  dict(lat=None),dict(received_at=T-timedelta(seconds=1))])
def test_invalid_latest_message_cannot_leave_old_numeric_observation(change):
    history=moving();history[-1]=history[-1].model_copy(update=change)
    result=observe(history)
    assert result.reason=='invalid_or_conflicting_gps' and result.speed_mean_kmh is None


def test_gap_or_invalid_old_point_uses_only_new_continuous_tail_with_explicit_coverage():
    history=[gps(-110,80),gps(-90,120),gps(-70,140),gps(-30,200),gps(-15,300),gps(0,400)]
    result=observe(history)
    assert result.status=='available' and result.selected_points==3
    assert result.considered_points==6 and result.coverage_fraction==.5
    assert result.observed_span_s==30
    history[3]=history[3].model_copy(update={'speed_kmh':None})
    result=observe(history)
    assert result.selected_points==2 and result.observed_span_s==15
    assert result.coverage_fraction==pytest.approx(1/3)


def test_conflicting_duplicate_stays_invalid_after_repeating_original_again():
    history=moving();history.extend([history[-1].model_copy(update={'speed_kmh':0}),history[-1]])
    result=observe(history)
    assert result.reason=='invalid_or_conflicting_gps' and result.speed_mean_kmh is None


def test_missing_or_snapshot_anchor_and_unavailable_confirmation_are_unknown():
    assert observe_segment(moving(),T,PLAN,None).reason=='no_arrival_anchor'
    assert observe(moving(),current=anchor(source='csv_snapshot')).reason=='no_arrival_anchor'
    assert observe(moving(),current=anchor(received_at=T+timedelta(seconds=1))).reason=='anchor_not_available'
    assert observe(moving(),current=anchor(valid_until=T-timedelta(seconds=1))).reason=='stale_arrival_anchor'
    assert observe(moving(),current=anchor(observed_at=T-timedelta(seconds=1000),delay_s=-880)).reason=='stale_arrival_anchor'
    assert observe(moving(),current=anchor(delay_s=100)).reason=='anchor_plan_mismatch'


def test_ambiguous_next_time_degenerate_geometry_and_wrong_vehicle_remain_unknown():
    assert observe(moving(),plan=[*PLAN,stop('tie',1500,180)]).reason=='ambiguous_next_visit'
    assert observe(moving(),plan=[PLAN[0],stop('same-place',0,180)]).reason=='unsupported_chord_geometry'
    assert observe(moving(),plan=[PLAN[0]]).reason=='no_next_visit'
    assert observe(moving()+[gps(-1,390,tr_id=999)]).reason=='mixed_vehicles'


def test_anchor_cutoff_excludes_old_movement_so_it_cannot_confirm_current_idle_direction():
    history=[gps(-140,200),gps(-125,300),gps(-15,400,speed=0),gps(0,400,speed=0)]
    result=observe(history)
    assert result.considered_points==2 and result.reason=='direction_not_observed'
