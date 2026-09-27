"""Причинность, частота GPS, геометрия и отдельный проверочный эталон producer."""
from datetime import datetime, timedelta
import math

from fastapi.testclient import TestClient
import pytest

from backend.engine import LiveContext
from generator.app import create_app
from generator.scenarios import (ControlRequest, DURATION_S, FRAME_LIMIT, GeneratorSession,
                                 LOG_LIMIT, ResetRequest, START, segment_m)


def session(scenario='normal', speed=1, paused=False, **kwargs):
    return GeneratorSession(ResetRequest(scenario=scenario, speed=speed, paused=paused, **kwargs))


def times(frame):
    return datetime.fromisoformat(frame['clock_time'])


@pytest.mark.parametrize('scenario', ['normal', 'slow_segment', 'long_stop', 'gps_loss'])
def test_stream_never_exposes_arrivals_and_truth_contains_only_past_once(scenario):
    s = session(scenario)
    context = s.status(include_context=True)['context']
    LiveContext.model_validate(context)
    assert not context['hints']
    assert 'arrived_at' not in str(context)
    previous_truth = []
    for elapsed in range(0, 421):
        if elapsed:
            s.advance(1)
        frame = s.frames[-1]
        assert frame['seq'] == elapsed+1
        assert times(frame) == START+timedelta(seconds=elapsed)
        assert frame['arrivals'] == []
        for item in frame['telemetry']:
            assert datetime.fromisoformat(item['event_time']) == times(frame)
            assert datetime.fromisoformat(item['received_at']) == times(frame)
        truth = s.truth()['observed_truth_arrivals']
        assert truth[:len(previous_truth)] == previous_truth
        assert all(datetime.fromisoformat(item['arrived_at']) <= times(frame) for item in truth)
        assert len({(item['tr_id'], item['planned_stop_id']) for item in truth}) == len(truth)
        previous_truth = truth
    assert s.emitted_arrivals == len(previous_truth) > 20


@pytest.mark.parametrize('route_count', [1, 4, 20])
def test_network_counts_shared_stops_unique_visits_and_orthogonal_paths(route_count):
    s = session(route_count=route_count)
    assert len(s.context.routes) == route_count
    assert len(s.context.vehicles) == route_count*2
    LiveContext.model_validate(s.context)
    paths = set()
    stop_coordinates = {}
    shared = set(stop.id for stop in s.context.routes[0].stops)
    for route in s.context.routes:
        paths.add(tuple(route.path))
        assert len(route.path) > 10
        assert len(route.stops) >= 6
        shared &= {stop.id for stop in route.stops}
        for a, b in zip(route.path, route.path[1:]):
            assert (a[0] == b[0]) != (a[1] == b[1])
        for stop in route.stops:
            coordinate = (stop.lon, stop.lat)
            assert stop_coordinates.setdefault(stop.id, coordinate) == coordinate
            assert coordinate in route.path
    assert len(paths) == route_count
    assert 'grid-0-0' in shared
    # План содержит отдельный идентификатор каждого посещения, даже повторного.
    assert len({(item.tr_id, item.target.id) for item in s.context.schedule}) == len(s.context.schedule)


def test_position_follows_path_and_speed_including_turns_and_dwell():
    s = session(route_count=4, telemetry_interval_s=1)
    checked_motion = checked_stop = turns = 0
    routes = {route.route_id: route for route in s.context.routes}
    vehicle_routes = {v.tr_id: routes[v.route_id] for v in s.context.vehicles}
    for _ in range(500):
        before = {item['tr_id']: item for item in s.frames[-1]['telemetry']}
        s.advance(1)
        for b in s.frames[-1]['telemetry']:
            a = before[b['tr_id']]
            route = vehicle_routes[b['tr_id']]
            on_segment = lambda p, q: (
                min(p[0], q[0])-1e-10 <= b['lon'] <= max(p[0], q[0])+1e-10
                and min(p[1], q[1])-1e-10 <= b['lat'] <= max(p[1], q[1])+1e-10
                and (abs(p[0]-b['lon']) < 1e-10 if p[0] == q[0] else abs(p[1]-b['lat']) < 1e-10))
            assert any(on_segment(p, q) for p, q in zip(route.path, route.path[1:]))
            distance = segment_m((a['lon'], a['lat']), (b['lon'], b['lat']))
            if a['speed_kmh'] > 0 and b['speed_kmh'] > 0:
                if a['heading'] == b['heading']:
                    assert distance*3.6 == pytest.approx(b['speed_kmh'], rel=0.0001)
                    checked_motion += 1
                else:
                    # Хорда при повороте короче пройденного пути, но скачка нет.
                    assert distance*3.6 <= b['speed_kmh']*1.0001
                    turns += 1
            elif a['speed_kmh'] == b['speed_kmh'] == 0:
                assert distance == 0
                checked_stop += 1
    assert checked_motion > 100 and checked_stop > 10 and turns > 0


def test_scenario_delays_truth_only_after_arrival_and_never_injects_current_deviation():
    normal, slow, dwell = session(), session('slow_segment'), session('long_stop')
    for s in [normal, slow, dwell]:
        s.advance(510)
    def arrival(s, tr_id, index):
        return next(item for item in s.truth()['observed_truth_arrivals']
                    if item['tr_id'] == tr_id and item['planned_stop_id'] == f'synthetic-{tr_id}-{index}')
    for altered, tr_id in [(slow, 101), (dwell, 102)]:
        assert arrival(altered, tr_id, 5) == arrival(normal, tr_id, 5)
        changed, reference = arrival(altered, tr_id, 6), arrival(normal, tr_id, 6)
        assert datetime.fromisoformat(changed['arrived_at'])-datetime.fromisoformat(reference['arrived_at']) == timedelta(seconds=150)
        assert not altered.context.hints
        assert not any(frame['arrivals'] for frame in altered.frames)


@pytest.mark.parametrize('route_count, affected', [(1, 102), (4, 104)])
def test_gps_loss_hides_telemetry_but_not_evaluator_truth(route_count, affected):
    s = session('gps_loss', route_count=route_count)
    assert s.status()['scenario_vehicle_id'] == affected
    s.advance(210)
    for frame in s.frames:
        elapsed = (times(frame)-START).total_seconds()
        if 90 <= elapsed < 210:
            assert all(item['tr_id'] != affected for item in frame['telemetry'])
        assert not frame['arrivals']
    assert any(item['tr_id'] == affected for item in s.frames[-1]['telemetry'])
    assert any(item['tr_id'] == affected and START+timedelta(seconds=90) <= datetime.fromisoformat(item['arrived_at']) < START+timedelta(seconds=210)
               for item in s.truth()['observed_truth_arrivals'])


@pytest.mark.parametrize('interval', [1, 15, 30])
def test_cadence_is_simulation_time_independent_of_playback_speed(interval):
    s = session(speed=30, telemetry_interval_s=interval, route_count=1)
    s.advance(3)
    for frame in s.frames:
        elapsed = (times(frame)-START).total_seconds()
        assert len(frame['telemetry']) == (2 if elapsed % interval == 0 else 0)
    assert s.emitted_telemetry == 2*(90//interval+1)


def test_pause_speed_reproducibility_and_bounded_cursor():
    a, b = session('slow_segment', speed=30), session('slow_segment', speed=30)
    a.advance(4.5)
    b.advance(2)
    b.advance(2.5)
    assert a.session_id != b.session_id
    assert list(a.frames) == list(b.frames)
    assert a.truth()['observed_truth_arrivals'] == b.truth()['observed_truth_arrivals']
    assert a.clock_time == START+timedelta(seconds=135)
    a.control(ControlRequest(action='pause'))
    frozen = a.stream(0)
    a.advance(100)
    assert a.stream(0) == frozen
    a.control(ControlRequest(action='speed', speed=1))
    a.control(ControlRequest(action='resume'))
    a.advance(600)
    data = a.stream(0)
    assert len(data['frames']) == FRAME_LIMIT
    assert len(data['logs']) == LOG_LIMIT
    assert data['gap']
    assert not a.stream(data['first_seq']-1)['gap']
    assert not a.stream(data['last_seq'])['frames']
    with pytest.raises(ValueError):
        a.stream(data['last_seq']+1)
    a.advance(DURATION_S)
    assert a.finished and not a.running
    assert a.clock_time == START+timedelta(seconds=DURATION_S)
    with pytest.raises(ValueError):
        a.control(ControlRequest(action='resume'))


def test_http_reset_truth_control_validation_and_read_without_advancing_time():
    with TestClient(create_app(start_background=False)) as client:
        assert client.get('/health').json()['synthetic']
        config = dict(scenario='long_stop', seed=42, speed=10, paused=True, route_count=20, telemetry_interval_s=30)
        reset = client.post('/reset', json=config)
        assert reset.status_code == 200
        info = reset.json()
        assert info['clock_time'] == START.isoformat()
        assert info['route_count'] == 20 and info['telemetry_interval_s'] == 30
        cursor = {'session_id': info['session_id'], 'after': 0}
        first = client.get('/stream', params=cursor).json()
        assert len(first['frames']) == 1
        assert first['emitted_telemetry'] == 40
        assert not first['frames'][0]['arrivals']
        assert client.get('/stream', params=cursor).json() == first
        truth = client.get('/truth', params={'session_id':info['session_id']})
        assert truth.status_code == 200
        assert truth.json()['observed_truth_arrivals']
        assert client.post('/control', json={'action':'speed'}).status_code == 409
        for bad in [{'speed':31}, {'route_count':0}, {'route_count':21}, {'telemetry_interval_s':0}, {'telemetry_interval_s':31}]:
            assert client.post('/reset', json=bad).status_code == 422
        assert client.post('/reset', content='{"speed": Infinity}',
                           headers={'Content-Type':'application/json'}).status_code == 422
        changed = client.post('/reset', json=config).json()
        assert changed['session_id'] != info['session_id']
        assert client.get('/stream', params=cursor).status_code == 409
        assert client.get('/truth', params={'session_id':info['session_id']}).status_code == 409
        assert client.get('/stream', params={'session_id':changed['session_id'], 'after':2}).status_code == 409


def test_optional_door_sensor_is_current_state_without_arrival_identity():
    with_doors, without = session(door_sensors=True, route_count=1), session(route_count=1)
    states = set()
    for elapsed in range(301):
        if elapsed:
            with_doors.advance(1)
            without.advance(1)
        for received, baseline in zip(with_doors.frames[-1]['telemetry'], without.frames[-1]['telemetry'], strict=True):
            assert baseline['doors_open'] is None
            states.add(received['doors_open'])
            assert isinstance(received['doors_open'], bool)
            assert {k:v for k,v in received.items() if k not in ('doors_open','door_sensor_key')} == {k:v for k,v in baseline.items() if k not in ('doors_open','door_sensor_key')}
            if received['doors_open']:
                assert received['speed_kmh'] == 0
        assert with_doors.frames[-1]['arrivals'] == []
    assert states == {False, True}
