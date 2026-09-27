"""Фиксированные проверки причинности GPS-эвристики, не оценка на реальных данных.

Пороги заранее зафиксированы в GPSDetectorConfig; эти примеры проверяют контракт.
Качество отдельно оценивается на синтетической истине, скрытой от детектора.
"""
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import json

import pytest

from backend.gps_arrivals import GPSArrivalDetector, GPSDetectorConfig
from common.contracts import StopTarget, Telemetry

T = datetime(2026, 1, 6, 8, tzinfo=timezone.utc)


def stop(name='a', time=0, x=0, y=0):
    return StopTarget(id=name, name=name, scheduled_at=T+timedelta(seconds=time),
                      lat=55.75+y/111320, lon=37.6+x/(111320*0.5628049277))


def gps(time=0, receive=None, x=0, y=0, speed=0, **extra):
    target = stop(x=x, y=y)
    return Telemetry(tr_id=1, event_time=T+timedelta(seconds=time),
        received_at=T+timedelta(seconds=time if receive is None else receive),
        lat=target.lat, lon=target.lon, speed_kmh=speed, door_sensor_key=extra.pop('door_sensor_key', 'fixture:door1'), **extra)


def test_arrival_estimate_available_only_on_confirming_receipt_and_never_drifts():
    d = GPSArrivalDetector([stop(time=-30)])
    assert d.observe(gps(0, receive=3)) is None
    assert d.status()['state'] == 'candidate'
    assert d.observe(gps(4, receive=7)) is None
    arrival = d.observe(gps(5, receive=12))
    assert arrival.arrived_at == T
    assert arrival.received_at == T+timedelta(seconds=12)
    assert arrival.confirmation_span_s == 5
    assert arrival.uncertainty_s is None  # холодный старт: граница прибытия неизвестна
    for seconds in (6, 20, 200):
        assert d.observe(gps(seconds, receive=seconds+20)) is None
    assert d.status()['confirmed_arrivals'] == 1
    assert d.status()['last_estimated_arrival_at'] == T.isoformat()
    assert d.status()['last_confirmed_at'] == (T+timedelta(seconds=12)).isoformat()
    assert d.status()['confirmation_age_at_last_message_s'] == 208
    assert arrival.arrived_at == T
    json.dumps(d.status())


def test_fast_pass_and_brief_slow_pass_are_not_arrivals():
    fast = GPSArrivalDetector([stop()])
    for seconds in (0, 5, 10):
        assert fast.observe(gps(seconds, speed=25)) is None
    assert fast.status()['reason'] == 'moving'
    brief = GPSArrivalDetector([stop()])
    assert brief.observe(gps(0, x=-10, speed=3)) is None
    assert brief.observe(gps(4, x=20, speed=3)) is None
    assert brief.observe(gps(5, x=45, speed=3)) is None
    assert brief.status()['confirmed_arrivals'] == 0


def test_duplicate_and_late_packets_cannot_supply_confirmation_or_rewrite_arrival():
    d = GPSArrivalDetector([stop()])
    assert d.observe(gps(0)) is None
    assert d.observe(gps(0, receive=3)) is None
    assert d.observe(gps(-1, receive=4, x=200)) is None
    assert d.status()['candidate_points'] == 1
    arrival = d.observe(gps(5, receive=6))
    assert arrival.arrived_at == T
    assert d.observe(gps(4, receive=7)) is None
    assert d.status()['confirmed_arrivals'] == 1


def test_future_relative_to_receipt_and_reverse_receipt_order_are_ignored():
    d = GPSArrivalDetector([stop()])
    assert d.observe(gps(0, receive=2)) is None
    assert d.observe(gps(5, receive=4)) is None
    assert d.status()['last_event_time'] == T.isoformat()
    assert d.observe(gps(1, receive=1)) is None
    assert d.status()['reason'] == 'receipt_out_of_order'
    arrival = d.observe(gps(5, receive=7))
    assert arrival.received_at == T+timedelta(seconds=7)


def test_telemetry_gap_resets_candidate_and_observation_uncertainty():
    d = GPSArrivalDetector([stop()])
    assert d.observe(gps(0)) is None
    assert d.observe(gps(31)) is None
    assert d.status()['reason'] == 'candidate_after_gap'
    arrival = d.observe(gps(36))
    assert arrival.arrived_at == T+timedelta(seconds=31)
    assert arrival.uncertainty_s is None


@pytest.mark.parametrize('change', [dict(location_valid=False), dict(lat=None),
                                  dict(lon=None), dict(speed_kmh=None), dict(speed_kmh=131)])
def test_invalid_gps_breaks_candidate(change):
    d = GPSArrivalDetector([stop()])
    assert d.observe(gps(0)) is None
    assert d.observe(gps(3).model_copy(update=change)) is None
    assert d.status()['reason'] == 'invalid_gps'
    assert d.observe(gps(5)) is None
    arrival = d.observe(gps(10))
    assert arrival.arrived_at == T+timedelta(seconds=5)
    assert arrival.uncertainty_s is None


def test_invalid_messages_cannot_keep_old_evidence_alive_across_long_gap():
    d = GPSArrivalDetector([stop()])
    assert d.observe(gps(0)) is None
    for second in (10, 20, 30, 31, 35):
        assert d.observe(gps(second).model_copy(update={'location_valid': False})) is None
    assert d.status()['candidate_points'] == 0
    assert d.observe(gps(40)) is None
    assert d.observe(gps(45)).arrived_at == T+timedelta(seconds=40)


def test_hysteresis_and_order_separate_repeat_visits_and_do_not_duplicate_dwell():
    d = GPSArrivalDetector([stop('a', 0), stop('b', 60, x=200), stop('a-return', 120)])
    # Холодный старт в a неоднозначен из-за возврата, поэтому якорим b.
    assert d.observe(gps(60, x=200)) is None
    b = d.observe(gps(65, x=200))
    assert b.planned_stop_id == 'b' and b.skipped_visits == 1
    assert d.observe(gps(70, x=250)) is None  # <70 м от b: остаёмся у той же остановки
    assert d.status()['reason'] == 'at_confirmed_stop'
    assert d.observe(gps(80, x=100, speed=20)) is None  # подтверждён выход
    assert d.observe(gps(120)) is None
    a_return = d.observe(gps(125))
    assert a_return.planned_stop_id == 'a-return'
    assert d.observe(gps(140)) is None
    assert d.status()['confirmed_arrivals'] == 2


def test_cold_start_repeated_same_physical_stop_is_unknown_not_nearest_time():
    d = GPSArrivalDetector([stop('first', 0), stop('return', 600)])
    assert d.observe(gps(0)) is None
    assert d.observe(gps(5)) is None
    assert d.status()['reason'] == 'ambiguous_stop'
    assert d.status()['confirmed_arrivals'] == 0


def test_direction_resolves_opposing_visits_from_observed_approach():
    d = GPSArrivalDetector([stop('west', -180, x=-200), stop('center-eastbound', 0),
                           stop('east', 180, x=200), stop('center-westbound', 360)])
    assert d.observe(gps(-20, x=-100, speed=20)) is None
    assert d.observe(gps(-5, x=-50, speed=20)) is None
    assert d.observe(gps(0)) is None
    arrival = d.observe(gps(5))
    assert arrival.planned_stop_id == 'center-eastbound'
    assert arrival.uncertainty_s == 5


def test_same_direction_loop_stays_ambiguous_even_with_approach():
    d = GPSArrivalDetector([stop('west-1', -180, x=-200), stop('center-1', 0),
                           stop('west-2', 420, x=-200), stop('center-2', 600)])
    assert d.observe(gps(-20, x=-100, speed=20)) is None
    assert d.observe(gps(0)) is None
    assert d.observe(gps(5)) is None
    assert d.status()['reason'] == 'ambiguous_stop'


def test_unknown_direction_of_first_visit_cannot_be_silently_discarded():
    d = GPSArrivalDetector([stop('no-incoming-leg', 0), stop('west', 100, x=-200), stop('again', 200)])
    assert d.observe(gps(-20, x=-100, speed=20)) is None
    assert d.observe(gps(0)) is None
    assert d.observe(gps(5)) is None
    assert d.status()['reason'] == 'ambiguous_stop'


def test_skipped_visits_never_create_backfilled_arrivals():
    d = GPSArrivalDetector([stop('a', 0), stop('missed', 60, x=200), stop('c', 120, x=400)])
    d.observe(gps(0))
    assert d.observe(gps(5)).planned_stop_id == 'a'
    assert d.observe(gps(120, x=400)) is None
    arrival = d.observe(gps(125, x=400))
    assert arrival.planned_stop_id == 'c' and arrival.skipped_visits == 1
    assert d.status()['confirmed_arrivals'] == 2
    assert d.status()['skipped_visits'] == 1


def test_skip_is_rejected_before_missing_visit_expected_time():
    d = GPSArrivalDetector([stop('a', 0), stop('b', 600, x=200), stop('c', 700, x=400)])
    d.observe(gps(0))
    assert d.observe(gps(5)).planned_stop_id == 'a'
    assert d.observe(gps(20, x=400)) is None
    assert d.observe(gps(25, x=400)) is None
    assert d.status()['reason'] == 'skip_before_expected_time'


def test_adjacent_repeat_uses_plan_order_after_anchor_without_cold_start_guess():
    d = GPSArrivalDetector([stop('anchor', 0, x=-200), stop('outbound', 100),
                           stop('other', 200, x=200), stop('return', 300)])
    d.observe(gps(0, x=-200))
    assert d.observe(gps(5, x=-200)).planned_stop_id == 'anchor'
    assert d.observe(gps(25, x=-100, speed=20)) is None
    # Длинной паузы нет: следующий плановый визит однозначен по порядку.
    for sec in (45, 65, 85):
        assert d.observe(gps(sec, x=-100, speed=20)) is None
    assert d.observe(gps(100)) is None
    assert d.observe(gps(105)).planned_stop_id == 'outbound'


def test_limits_schedule_identity_vehicle_identity_and_empty_plan():
    with pytest.raises(ValueError, match='уникальны'):
        GPSArrivalDetector([stop('same'), stop('same', 180)])
    with pytest.raises(ValueError):
        GPSDetectorConfig(exit_radius_m=30)
    with pytest.raises(ValueError):
        GPSDetectorConfig(confirmation_s=31)
    d = GPSArrivalDetector([])
    assert d.observe(gps()) is None
    assert d.status()['reason'] == 'no_schedule'
    with pytest.raises(ValueError, match='одно ТС'):
        d.observe(gps(1).model_copy(update={'tr_id':2}))
    assert asdict(GPSDetectorConfig())['schedule_window_s'] == 900


def test_estimated_delay_cannot_walk_matching_window_to_an_earlier_loop():
    d = GPSArrivalDetector([stop('anchor', 0), stop('old-loop', 300, x=200),
                           stop('current-loop', 2100, x=200)])
    d.observe(gps(600))
    assert d.observe(gps(605)).planned_stop_id == 'anchor'  # estimate +600 s
    d.observe(gps(620, x=100, speed=20))
    # Relative to +600 this old loop is only 600 s away, but absolute delta
    # is +1200. Reusing the old loop would recursively amplify the wrong delay.
    assert d.observe(gps(1500, x=200)) is None
    assert d.observe(gps(1505, x=200)) is None
    assert d.observe(gps(2100, x=200)) is None
    assert d.observe(gps(2105, x=200)) is None
    # Два соседних плановых визита имеют одну физическую точку. Само
    # открытие временного окна ещё не доказывает новое прибытие после отстоя.
    assert d.observe(gps(2120, x=100, speed=20)) is None
    assert d.observe(gps(2140, x=200)) is None
    result = d.observe(gps(2145, x=200))
    assert result is not None and result.planned_stop_id == 'current-loop'
    assert (result.arrived_at-d.stops[2].scheduled_at).total_seconds() == 40


def test_delay_beyond_supported_window_is_unknown_not_clipped_to_boundary():
    d = GPSArrivalDetector([stop('anchor', 0), stop('too-old', 300, x=200)])
    d.observe(gps(600))
    assert d.observe(gps(605)).planned_stop_id == 'anchor'
    d.observe(gps(620, x=100, speed=20))
    for seconds in (1500, 1505, 1510):
        assert d.observe(gps(seconds, x=200)) is None
    assert d.status()['confirmed_arrivals'] == 1
    assert d.status()['reason'] == 'outside_stop_window'


def test_door_transition_confirms_at_observed_open_only_and_is_idempotent():
    d = GPSArrivalDetector([stop(time=-30)])
    assert d.observe(gps(0, x=-100, speed=25, doors_open=False)) is None
    result = d.observe(gps(15, receive=19, doors_open=True))
    assert result.source == 'door_estimate'
    assert result.arrived_at == T+timedelta(seconds=15)
    assert result.received_at == T+timedelta(seconds=19)
    assert result.confirmation_span_s == 0 and result.uncertainty_s == 15
    for seconds, opened in [(20, True), (25, False), (30, True)]:
        assert d.observe(gps(seconds, doors_open=opened)) is None
    assert d.status()['confirmed_arrivals'] == 1


@pytest.mark.parametrize('kind', ['cold_open', 'gap', 'invalid', 'fast', 'outside', 'ambiguous', 'late_receipt'])
def test_doors_never_bypass_gps_time_plan_or_ambiguity_guards(kind):
    stops = [stop()]+([stop('other', time=60)] if kind == 'ambiguous' else [])
    d = GPSArrivalDetector(stops)
    if kind != 'cold_open':
        assert d.observe(gps(0, x=-100, speed=20, doors_open=False)) is None
    if kind == 'invalid':
        assert d.observe(gps(5, location_valid=False)) is None
    opened = gps(35 if kind == 'gap' else 15,
        receive=14 if kind == 'late_receipt' else None,
        x=100 if kind == 'outside' else 0,
        speed=25 if kind == 'fast' else 0, doors_open=True)
    assert d.observe(opened) is None
    assert d.status()['confirmed_arrivals'] == 0


def test_unknown_door_sensor_keeps_two_point_gps_fallback():
    d = GPSArrivalDetector([stop()])
    assert d.observe(gps(0, x=-100, speed=25, doors_open=False)) is None
    assert d.observe(gps(15, doors_open=None)) is None
    result = d.observe(gps(30, doors_open=None))
    assert result.source == 'gps_estimate'
    assert result.arrived_at == T+timedelta(seconds=15)


@pytest.mark.parametrize('invalid', ['false', 'true', 1, 0, {}])
def test_normalized_door_field_rejects_ambiguous_non_boolean_values(invalid):
    with pytest.raises(ValueError):
        gps(doors_open=invalid)


def test_door_transition_after_impossible_jump_cannot_shortcut_gps_confirmation():
    d = GPSArrivalDetector([stop()])
    assert d.observe(gps(0, x=-37000, doors_open=False)) is None
    assert d.observe(gps(15, doors_open=True)) is None
    assert d.status()['confirmed_arrivals'] == 0


@pytest.mark.parametrize('sensor_key', [None, 'fixture:door2'])
def test_unknown_or_changed_sensor_set_does_not_create_opening_event(sensor_key):
    d = GPSArrivalDetector([stop()])
    assert d.observe(gps(0, x=-100, speed=25, doors_open=False)) is None
    assert d.observe(gps(15, doors_open=True, door_sensor_key=sensor_key)) is None
    assert d.status()['confirmed_arrivals'] == 0


@pytest.mark.parametrize('approach_x,expected', [(-100, 'recovered'), (100, None)])
def test_reanchor_releases_old_delay_only_with_observed_matching_approach(approach_x, expected):
    d = GPSArrivalDetector([stop('anchor', 0, x=-400), stop('missed', 700, x=-200),
                            stop('recovered', 800)])
    assert d.observe(gps(600, x=-400)) is None
    assert d.observe(gps(605, x=-400)).planned_stop_id == 'anchor'  # старая задержка +600с
    assert d.observe(gps(620, x=-300, speed=20)) is None
    # После перерыва старый delay ожидал бы missed только в1300с. Физически
    # согласованный подход к следующей зоне даёт независимый новый якорь.
    assert d.observe(gps(880, x=approach_x, speed=20)) is None
    assert d.observe(gps(900)) is None
    result = d.observe(gps(905))
    assert (result.planned_stop_id if result else None) == expected
    if result:
        assert result.arrived_at == T+timedelta(seconds=900)
        assert result.received_at == T+timedelta(seconds=905)
        assert result.skipped_visits == 1


def test_clock_window_invalid_packet_and_door_edge_do_not_invent_terminal_reentry():
    d = GPSArrivalDetector([stop('previous-terminal', -1800), stop('next-terminal', 600)])
    assert d.observe(gps(0, doors_open=False)) is None
    assert d.observe(gps(15, doors_open=True)) is None
    assert d.status()['reason'] == 'terminal_occupancy_without_reentry'
    assert d.observe(gps(20, location_valid=False)) is None
    assert d.observe(gps(25)) is None
    assert d.observe(gps(30)) is None
    assert d.status()['confirmed_arrivals'] == 0
    # Настоящий выход и возвращение дают новое физическое наблюдение.
    assert d.observe(gps(40, x=-100, speed=20, doors_open=False)) is None
    result = d.observe(gps(55, doors_open=True))
    assert result.planned_stop_id == 'next-terminal'
    assert result.source == 'door_estimate'
