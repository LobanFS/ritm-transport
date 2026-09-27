"""Доступность CSV-событий, независимые часы и отсутствие подмешивания live."""
import asyncio
import csv
import hashlib
from datetime import datetime, timedelta, timezone
import json
import io
from pathlib import Path
import sys

from fastapi.testclient import TestClient
import httpx
import pytest

from backend.app import create_app
from backend.engine import Engine
from backend.replay import ReplayConfig, load_replay
from common.contracts import PredictionRequest, baseline

T = datetime(2026, 1, 6, 11, 30, tzinfo=timezone.utc)


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / 'validate'
    root.mkdir()
    def write(name, fields, rows):
        with (root / name).open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(fields.split(','))
            writer.writerows(rows)
    write('schedule_plan.csv', 'tr_id,tt_action_item_id,time_begin,geom,building_address', [
        [1, 'past', '2026-01-06 11:29:00', 'POINT (37.60 55.75)', 'Previous'],
        [1, 'target', '2026-01-06 11:42:00', 'POINT (37.61 55.76)', 'Target'],
        [1, 'next', '2026-01-06 11:52:00', 'POINT (37.62 55.77)', 'Next']])
    write('traffic.csv', 'tr_id,unit_id,event_time,receive_time,packet_id,lat,lon,speed,heading,location_valid', [
        [1, 2, '2026-01-06 11:29:59', '2026-01-06 11:29:59', 'a', 55.75, 37.60, 30, 0, True],
        # Поздно доставленное сообщение не доступно на T, даже если event_time в прошлом.
        [1, 2, '2026-01-06 11:29:58', '2026-01-06 11:30:10', 'b', 55.70, 37.50, 0, 0, True],
        # Неконсистентные часы: ждать и event_time, и receive_time.
        [1, 2, '2026-01-06 11:30:20', '2026-01-06 11:29:57', 'c', 55.76, 37.61, 20, 90, True]])
    write('points.csv', 'sample_id,tr_id,T,target_stop_id,target_time_begin,cur_dev_s', [
        ['snapshot-1', 1, '2026-01-06 11:30:00', 'target', '2026-01-06 11:42:00', 180],
        ['snapshot-2', 1, '2026-01-06 11:40:00', 'next', '2026-01-06 11:52:00', -30]])
    # Будущие факты намеренно некорректны: загрузчик не должен открывать их.
    (tmp_path / 'labels').mkdir()
    (tmp_path / 'labels' / 'labels_test.csv').write_text('DO NOT READ')
    return tmp_path


def replay(root, **kwargs):
    return load_replay(root, ReplayConfig(start=T, tr_ids=[1], duration_minutes=15, **kwargs))


def test_replay_delivers_only_available_rows_and_never_moves_position_back(dataset):
    e = Engine('http://ml')
    e.set_replay(replay(dataset))
    assert e.input_sequences[1] == 1
    assert e.deviation_at(1, T).delay_s == 180
    assert e.deviation_at(1, T).sample_id == 'snapshot-1'
    assert e.deviation_at(1, T).source == 'csv_snapshot'
    assert len(e.hints[1]) == 1  # будущий snapshot только в очереди проигрывателя
    assert not e.arrivals[1]
    assert e.request_for(1, T).features.valid_points_5m == 1
    e.clock = T + timedelta(seconds=10)
    e.replay.deliver(e)
    assert e.input_sequences[1] == 2
    assert e.state()['vehicles'][0]['lat'] == 55.75  # поздняя старая координата не отматывает автобус
    e.clock = T + timedelta(seconds=20)
    e.replay.deliver(e)
    assert e.input_sequences[1] == 3
    assert e.state()['vehicles'][0]['lat'] == 55.76
    assert all(max(x.event_time, x.received_at) <= e.clock for x in e.history[1])


def test_snapshots_expire_without_inventing_zero_and_next_snapshot_keeps_sign(dataset):
    e = Engine('http://ml')
    e.set_replay(replay(dataset))
    assert e.deviation_at(1, T+timedelta(seconds=300)).delay_s == 180
    assert e.deviation_at(1, T+timedelta(seconds=301)) is None
    e.clock = T+timedelta(minutes=10)
    e.replay.deliver(e)
    assert e.deviation_at(1, e.clock).delay_s == -30


def test_clock_pause_end_reset_and_clear_do_not_use_wall_time(dataset):
    e = Engine('http://ml')
    loaded = replay(dataset)
    e.set_replay(loaded)
    asyncio.run(e.tick(1))
    assert e.clock == T and e.state()['clock_time'] == T
    assert e.forecast_traces[1]['request']['issued_at'] == T.isoformat().replace('+00:00', 'Z')
    assert e.incidents[0]['lead_time_s'] == 720  # не сравнение с реальным сентябрём
    e.running = True
    asyncio.run(e.tick(1))
    assert e.clock == T+timedelta(seconds=10)
    e.clock = loaded.end-timedelta(seconds=1)
    asyncio.run(e.tick(1))
    assert e.clock == loaded.end and not e.running
    assert loaded.cursor == len(loaded.events)
    e.set_replay(loaded)
    assert e.clock == T and e.input_sequences[1] == 1 and not e.predictions
    e.set_live()
    assert e.replay is None and not e.history and not e.hints


def test_real_http_exchange_contract_uses_historical_cutoff_and_snapshot(dataset):
    async def run():
        def handler(request):
            return httpx.Response(200, json=[baseline(PredictionRequest.model_validate(r)).model_dump(mode='json')
                for r in json.loads(request.content)])
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            e = Engine('http://ml', client)
            e.set_replay(replay(dataset))
            await e.tick()
            trace = e.forecast_traces[1]
            assert trace['execution'] == 'ml_http'
            assert trace['request']['current_delay_s'] == trace['response']['predicted_delay_s'] == 180
            assert trace['request']['target']['id'] == 'target'
            assert trace['current_deviation']['sample_id'] == 'snapshot-1'
            await e.on_nav(2, {})  # живые NDTP не смешиваются с историей
            assert e.input_sequences[1] == 1
    asyncio.run(run())


def test_timezone_changes_interpretation_consistently(dataset):
    cfg = ReplayConfig(start=T.replace(tzinfo=timezone(timedelta(hours=3))), timezone='Europe/Moscow', tr_ids=[1])
    loaded = load_replay(dataset, cfg)
    e = Engine('http://ml')
    e.set_replay(loaded)
    assert e.request_for(1, e.clock).target.scheduled_at-e.clock == timedelta(minutes=12)
    assert e.state()['vehicles'][0]['age_s'] == 1
    with pytest.raises(ValueError):
        ReplayConfig(timezone='bad-zone')


def test_target_mismatch_fails_instead_of_silently_forecasting_wrong_stop(dataset):
    path = dataset/'validate'/'points.csv'
    path.write_text(path.read_text().replace('snapshot-1,1,2026-01-06 11:30:00,target,',
                                             'snapshot-1,1,2026-01-06 11:30:00,wrong,'))
    with pytest.raises(ValueError, match='Цель points.csv'):
        replay(dataset)


def test_api_load_control_failure_is_atomic_and_inputs_are_isolated(dataset, monkeypatch):
    monkeypatch.delenv('REPLAY_DATA_DIR', raising=False)
    with TestClient(create_app(start_background=False, enable_ndtp=False)) as c:
        body = ReplayConfig(start=T, tr_ids=[1]).model_dump(mode='json')
        assert c.post('/api/v1/replay/load', json=body).status_code == 409
        monkeypatch.setenv('REPLAY_DATA_DIR', str(dataset))
        assert c.post('/api/v1/replay/load', json=body).status_code == 200
        initial = c.get('/api/v1/state').json()
        assert initial['mode'] == 'replay' and initial['replay']['delivered'] == 2
        assert initial['vehicles'][0]['cur_dev_s'] == 180
        assert len(initial['replay']['sources_sha256']) == 3
        assert c.post('/api/v1/replay/load', json={**body, 'tr_ids':[999]}).status_code == 422
        assert c.get('/api/v1/state').json()['clock_time'] == initial['clock_time']
        assert c.post('/api/v1/replay/load', json={**body, 'path':'/etc/passwd'}).status_code == 422
        assert c.post('/api/v1/replay/control', json={'action':'speed'}).status_code == 422
        assert c.post('/api/v1/replay/control', json={'action':'resume'}).json()['replay']['running']
        assert not c.post('/api/v1/replay/control', json={'action':'pause'}).json()['replay']['running']
        assert c.post('/api/v1/demo/control', json={'action':'reset'}).status_code == 409
        assert c.post('/api/v1/arrivals', json={'tr_id':1, 'planned_stop_id':'past', 'arrived_at':T.isoformat()}).status_code == 409
        c.post('/api/v1/mode', json={'mode':'demo'})
        assert c.post('/api/v1/replay/control', json={'action':'resume'}).status_code == 409


def test_replay_honors_explicit_target_when_plan_times_are_tied(dataset):
    plan = dataset/'validate/schedule_plan.csv'
    with plan.open('a') as stream:
        stream.write('1,aaa,2026-01-06 11:42:00,POINT (37.6 55.7),Other same-time visit\n')
    loaded = replay(dataset)
    e = Engine('http://ml'); e.set_replay(loaded)
    assert e.request_for(1,T).target.id == 'target'  # points выбирает её, не лексикографическую aaa
    assert loaded.quality['target_checks'] == 2


def add_stop_observations(dataset):
    with (dataset/'validate/traffic.csv').open('a') as stream:
        stream.write('1,2,2026-01-06 11:29:30,2026-01-06 11:29:30,stop1,55.75,37.60,0,0,True\n')
        stream.write('1,2,2026-01-06 11:29:45,2026-01-06 11:29:45,stop2,55.75,37.60,0,0,True\n')


def test_gps_replay_never_opens_points_and_derives_current_delay_from_available_gps(dataset):
    add_stop_observations(dataset)
    (dataset/'validate/points.csv').unlink()  # GPS mode must work without any ready-made hint file.
    loaded = replay(dataset, deviation_source='gps')
    assert set(loaded.sources) == {'validate/traffic.csv', 'validate/schedule_plan.csv'}
    assert loaded.quality['snapshots'] == loaded.quality['target_checks'] == 0
    assert all(kind == 'gps' for _, kind, _ in loaded.events)
    e = Engine('http://ml'); e.set_replay(loaded)
    assert e.arrival_mode == 'gps' and e.gps_detectors and not any(e.hints.values())
    deviation = e.deviation_at(1, T)
    assert deviation.source == 'gps_estimate' and deviation.planned_stop_id == 'past'
    assert deviation.delay_s == 30  # observed 11:29:30 minus plan 11:29:00
    assert deviation.received_at <= T and deviation.observed_at <= T
    request = e.request_for(1, T)
    assert request.current_delay_s == 30 and request.current_delay_source == 'gps_estimate'
    assert request.telemetry_domain == 'historical_real'
    assert request.current_delay_detector_version == e.gps_detectors[1].version
    assert request.current_delay_detector_sha256 == hashlib.sha256(
        (Path(__file__).resolve().parents[1]/'backend/gps_arrivals.py').read_bytes()).hexdigest()
    assert e.input_sequences[1] == 3  # late and future messages still remain queued
    state = e.state()['replay']
    assert state['deviation_source'] == 'gps' and state['tr_ids'] == [1]
    assert state['warmup_minutes'] == 5 and state['running'] is False


def test_replay_source_switch_and_reset_cannot_mix_csv_hints_into_gps(dataset):
    add_stop_observations(dataset)
    e = Engine('http://ml')
    e.set_replay(replay(dataset))
    assert e.deviation_at(1,T).delay_s == 180
    e.set_replay(replay(dataset, deviation_source='gps'))
    assert e.deviation_at(1,T).delay_s == 30 and not any(e.hints.values())
    e.set_replay(e.replay)
    assert e.deviation_at(1,T).delay_s == 30 and len(e.arrivals[1]) == 1
    e.set_replay(replay(dataset))
    assert e.deviation_at(1,T).delay_s == 180 and e.gps_detectors == {} and not e.arrivals[1]
    assert e.request_for(1,T).current_delay_detector_version is None
    assert e.request_for(1,T).current_delay_detector_sha256 is None


def test_gps_warmup_is_explicit_and_zero_does_not_read_previous_observations(dataset):
    add_stop_observations(dataset)
    e = Engine('http://ml')
    e.set_replay(replay(dataset, deviation_source='gps', warmup_minutes=0))
    assert not e.history and e.deviation_at(1,T) is None
    assert e.replay.state(e).warmup_minutes == 0
    e.set_replay(replay(dataset, deviation_source='gps', warmup_minutes=30))
    assert e.deviation_at(1,T).delay_s == 30
    assert all(max(point.event_time,point.received_at) <= T for point in e.history[1])
    for minutes in (-1, 121):
        with pytest.raises(ValueError):
            replay(dataset, warmup_minutes=minutes)


def test_gps_replay_api_exposes_source_and_rejects_unknown_source_atomically(dataset, monkeypatch):
    add_stop_observations(dataset)
    monkeypatch.setenv('REPLAY_DATA_DIR', str(dataset))
    with TestClient(create_app(start_background=False, enable_ndtp=False)) as client:
        body = ReplayConfig(start=T, tr_ids=[1], deviation_source='gps', warmup_minutes=30).model_dump(mode='json')
        assert client.post('/api/v1/replay/load', json=body).status_code == 200
        state = client.get('/api/v1/state').json()
        assert state['replay']['deviation_source'] == 'gps' and state['replay']['tr_ids'] == [1]
        assert state['vehicles'][0]['cur_dev_s'] == 30
        assert client.post('/api/v1/replay/load', json={**body, 'deviation_source':'invented'}).status_code == 422
        assert client.get('/api/v1/state').json()['replay'] == state['replay']


def test_backend_telemetry_domain_cannot_confuse_synthetic_with_real_replay(dataset):
    add_stop_observations(dataset)
    e = Engine('http://ml')
    e.set_replay(replay(dataset, deviation_source='gps'))
    assert e.request_for(1,T).telemetry_domain == 'historical_real'
    # Источник отклонения одинаков, но режим генератора не становится реальной историей.
    e.mode = 'generator'
    assert e.request_for(1,T).telemetry_domain == 'synthetic'
    e.mode = 'live'
    assert e.request_for(1,T).telemetry_domain == 'live_unverified'
    e.reset_demo()
    assert e.request_for(101,e.clock).telemetry_domain == 'synthetic'


@pytest.fixture
def train_dataset(dataset):
    add_stop_observations(dataset)
    target = dataset/'train'
    target.mkdir()
    (target/'traffic.csv').write_bytes((dataset/'validate/traffic.csv').read_bytes())
    with (dataset/'validate/schedule_plan.csv').open() as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    with (target/'schedule.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=[*fields, 'time_fact_begin'])
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, 'time_fact_begin': 'FUTURE FACT MUST NOT BE PARSED'})
    return dataset


def test_train_replay_uses_only_its_two_files_and_observed_gps(train_dataset, monkeypatch):
    original_open = Path.open
    def guarded_open(path, *args, **kwargs):
        if path.is_relative_to(train_dataset):
            assert path.relative_to(train_dataset).as_posix() in ('train/traffic.csv', 'train/schedule.csv')
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded_open)
    loaded = replay(train_dataset, dataset_split='train', deviation_source='gps')
    engine = Engine('http://ml'); engine.set_replay(loaded)
    assert set(loaded.sources) == {'train/traffic.csv', 'train/schedule.csv'}
    assert loaded.quality['snapshots'] == 0 and not any(engine.hints.values())
    request = engine.request_for(1, T)
    assert request.current_delay_source == 'gps_estimate' and request.current_delay_s == 30
    assert request.telemetry_domain == 'historical_mixed'  # Приближённый перенос, не historical_real.
    assert all(max(point.event_time, point.received_at) <= T for point in engine.history[1])
    assert engine.input_sequences[1] == 3
    assert 'time_fact_begin' not in request.model_dump_json()
    state = engine.state()
    assert state['context']['source'] == 'train/schedule.csv'
    assert state['replay']['dataset_split'] == 'train'
    assert state['replay']['plan_file'] == 'train/schedule.csv' and state['replay']['points_file'] is None


def test_train_future_facts_cannot_change_plan_version_or_model_request(train_dataset):
    engine = Engine('http://ml')
    engine.set_replay(replay(train_dataset, dataset_split='train', deviation_source='gps'))
    before = engine.request_for(1,T).model_dump(mode='json')
    source_hash = engine.replay.sources['train/schedule.csv']
    path = train_dataset/'train/schedule.csv'
    path.write_text(path.read_text().replace('FUTURE FACT MUST NOT BE PARSED','2099-12-31 23:59:59'))
    engine.set_replay(replay(train_dataset, dataset_split='train', deviation_source='gps'))
    assert engine.request_for(1,T).model_dump(mode='json') == before
    assert engine.replay.sources['train/schedule.csv'] != source_hash  # Provenance отражает реальный файл.


def test_switching_train_and_validate_clears_hints_and_resets_domain(train_dataset):
    engine = Engine('http://ml')
    engine.set_replay(replay(train_dataset))
    assert engine.request_for(1,T).current_delay_s == 180
    engine.set_replay(replay(train_dataset, dataset_split='train', deviation_source='gps'))
    assert engine.request_for(1,T).current_delay_s == 30 and not any(engine.hints.values())
    engine.set_replay(engine.replay)
    assert engine.request_for(1,T).telemetry_domain == 'historical_mixed'
    engine.set_replay(replay(train_dataset, deviation_source='gps'))
    assert engine.request_for(1,T).telemetry_domain == 'historical_real'
    engine.set_replay(replay(train_dataset))
    assert engine.request_for(1,T).current_delay_s == 180 and not engine.arrivals[1]


def test_train_api_rejects_hint_mode_and_arbitrary_paths_without_changing_state(train_dataset, monkeypatch):
    monkeypatch.setenv('REPLAY_DATA_DIR', str(train_dataset))
    with TestClient(create_app(start_background=False, enable_ndtp=False)) as client:
        body = ReplayConfig(start=T,tr_ids=[1],dataset_split='train',deviation_source='gps').model_dump(mode='json')
        assert client.post('/api/v1/replay/load',json=body).status_code == 200
        initial = client.get('/api/v1/state').json()
        assert initial['replay']['dataset_split'] == 'train'
        for change in ({'deviation_source':'csv_snapshot'}, {'dataset_split':'../../train'}, {'dataset_split':'test'}):
            assert client.post('/api/v1/replay/load',json={**body,**change}).status_code == 422
            assert client.get('/api/v1/state').json()['replay'] == initial['replay']


def test_all_fleet_loads_39_vehicles_and_reports_missing_context(dataset):
    """Все — выбор по плану и доступности архива, не скрытый список из двух ID."""
    for name in ('schedule_plan.csv', 'traffic.csv'):
        path = dataset/'validate'/name
        with path.open() as stream:
            reader = csv.DictReader(stream)
            fields, original = reader.fieldnames, list(reader)
        rows = []
        for tr_id in range(1, 40):
            for row in original:
                row = {**row, 'tr_id':str(tr_id)}
                if name == 'traffic.csv':
                    row.update(unit_id=str(1000+tr_id), packet_id=f'{tr_id}-{row["packet_id"]}')
                rows.append(row)
        # План есть, но сообщений в срезе нет.
        if name == 'schedule_plan.csv':
            rows += [{**row, 'tr_id':'40'} for row in original]
        else:
            rows += [{**original[0], 'tr_id':'999', 'unit_id':'999'}]
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fields); writer.writeheader(); writer.writerows(rows)
    loaded = load_replay(dataset, ReplayConfig(start=T, deviation_source='gps'))
    engine = Engine('http://ml'); engine.set_replay(loaded)
    state = loaded.state(engine)
    assert state.tr_ids == list(range(1,40)) and len(engine.vehicles) == 39
    assert state.selection_mode == 'all_available'
    assert state.fleet == dict(planned_vehicles=40, loaded_vehicles=39,
                              without_telemetry=[40], telemetry_without_plan=[999])
    assert all(len(engine.history[i]) == 1 for i in state.tr_ids)
    manual = load_replay(dataset, ReplayConfig(start=T, deviation_source='gps', tr_ids=list(range(1,40))))
    assert len(manual.context.vehicles) == 39
    with pytest.raises(ValueError, match='40'):
        load_replay(dataset, ReplayConfig(start=T, deviation_source='gps', tr_ids=[1,40]))
    with pytest.raises(ValueError):
        ReplayConfig(tr_ids=list(range(1,130)))


def test_stream_diagnostics_explain_gap_and_end_without_feeding_future_to_model(dataset):
    path = dataset/'validate'/'traffic.csv'
    with path.open('a', newline='') as stream:
        csv.writer(stream).writerow([1,2,'2026-01-06 11:30:05','2026-01-06 11:30:05',
                                    'bad-speed',55.77,37.62,300,0,True])
    engine = Engine('http://ml'); engine.set_replay(replay(dataset, deviation_source='gps'))
    request_before = engine.request_for(1,T).model_dump(mode='json')
    stream = engine.replay.state(engine).streams[0]
    assert stream['delivered'] == 1 and stream['remaining'] == 3
    assert stream['last_delivery_at'] == T-timedelta(seconds=1)
    assert stream['next_delivery_at'] == T+timedelta(seconds=5)
    assert stream['next_position_delivery_at'] == T+timedelta(seconds=10)
    assert len(engine.history[1]) == 1
    assert engine.request_for(1,T).model_dump(mode='json') == request_before
    engine.clock = T+timedelta(seconds=20); engine.replay.deliver(engine)
    stream = engine.replay.state(engine).streams[0]
    assert stream['remaining'] == 0 and stream['next_delivery_at'] is None
    assert stream['next_position_delivery_at'] is None and stream['delivered'] == 4
    assert engine.replay.state(engine).finished and not engine.running  # пустой хвост не проигрывается
    engine.set_replay(engine.replay)
    assert engine.replay.state(engine).streams[0]['remaining'] == 3


def append_csv(root, name, rows):
    with (root/'validate'/name).open('a', newline='') as stream:
        csv.writer(stream).writerows(rows)


def test_duration_defaults_to_end_of_data_and_hint_can_be_last_delivery(dataset):
    assert ReplayConfig().duration_minutes is None
    assert ReplayConfig(duration_minutes=None).duration_minutes is None
    loaded = load_replay(dataset, ReplayConfig(start=T))
    assert loaded.end == T+timedelta(minutes=10)  # последний snapshot после GPS
    engine = Engine('http://ml'); engine.set_replay(loaded)
    assert loaded.state(engine).duration_minutes is None
    assert len(engine.hints[1]) == 1
    engine.running = True
    engine.clock = loaded.end-timedelta(seconds=1)
    asyncio.run(engine.tick())
    assert engine.clock == loaded.end and loaded.cursor == len(loaded.events)
    assert len(engine.hints[1]) == 2 and not engine.running
    engine.set_replay(loaded)
    assert engine.clock == T and len(engine.hints[1]) == 1 and not loaded.state(engine).finished


def test_full_archive_selects_late_vehicle_without_delivering_future_history(dataset):
    append_csv(dataset, 'schedule_plan.csv', [
        [3, 'late-past', '2026-01-06 12:35:00', 'POINT (37.6 55.75)', 'Late previous'],
        [3, 'late-next', '2026-01-06 12:50:00', 'POINT (37.61 55.76)', 'Late next'],
    ])
    append_csv(dataset, 'traffic.csv', [
        [1, 2, '2026-01-06 12:00:00', '2026-01-06 12:05:00', 'late-receive', 55.75, 37.60, 30, 0, True],
        [3, 4, '2026-01-06 12:40:00', '2026-01-06 12:39:00', 'future-event', 55.75, 37.60, 30, 0, True],
    ])
    loaded = load_replay(dataset, ReplayConfig(start=T, deviation_source='gps'))
    assert loaded.end == T+timedelta(minutes=70)
    assert [v.tr_id for v in loaded.context.vehicles] == [1,3]
    engine = Engine('http://ml'); engine.set_replay(loaded)
    assert not engine.history[3] and engine.input_sequences[1] == 1
    assert all(max(p.event_time,p.received_at) <= T for history in engine.history.values() for p in history)
    engine.clock = T+timedelta(minutes=69)
    loaded.deliver(engine)
    assert not engine.history[3]  # receive уже наступил, event ещё нет
    engine.running = True
    engine.clock = loaded.end-timedelta(seconds=1)
    asyncio.run(engine.tick())
    assert engine.clock == loaded.end and not engine.running and len(engine.history[3]) == 1
    assert loaded.cursor == len(loaded.events)
    engine.set_replay(loaded)
    assert loaded.end == T+timedelta(minutes=70) and not engine.history[3]
    manual = load_replay(dataset, ReplayConfig(start=T, deviation_source='gps', tr_ids=[1]))
    assert manual.end == T+timedelta(minutes=35)  # чужое более позднее ТС конец не продлевает
    bounded = load_replay(dataset, ReplayConfig(start=T, deviation_source='gps', duration_minutes=30))
    assert [v.tr_id for v in bounded.context.vehicles] == [1]
    assert bounded.end == T+timedelta(seconds=20)


def test_explicit_window_includes_every_final_event_and_no_later_deliveries(dataset):
    append_csv(dataset, 'traffic.csv', [
        [1, 2, '2026-01-06 11:30:59', '2026-01-06 11:31:00', 'final-receive', 55.75, 37.60, 30, 0, True],
        [1, 2, '2026-01-06 11:31:00', '2026-01-06 11:30:58', 'final-event', 55.75, 37.60, 30, 0, True],
        [1, 2, '2026-01-06 11:31:01', '2026-01-06 11:30:58', 'outside-event', 55.75, 37.60, 30, 0, True],
        [1, 2, '2026-01-06 11:30:58', '2026-01-06 11:31:01', 'outside-receive', 55.75, 37.60, 30, 0, True],
    ])
    loaded = load_replay(dataset, ReplayConfig(start=T, deviation_source='gps', duration_minutes=1))
    assert loaded.end == T+timedelta(minutes=1) and len(loaded.events) == 5
    engine = Engine('http://ml'); engine.set_replay(loaded)
    engine.running = True; engine.clock = loaded.end-timedelta(seconds=1)
    asyncio.run(engine.tick())
    assert engine.clock == loaded.end and loaded.state(engine).finished and not engine.running
    assert {p.event_id for p in engine.history[1]} == {'a','b','c','final-receive','final-event'}
    assert loaded.state(engine).streams[0]['remaining'] == 0


def test_hints_of_vehicle_without_telemetry_do_not_extend_archive(dataset):
    append_csv(dataset, 'schedule_plan.csv', [
        [3, 'orphan-past', '2026-01-06 12:35:00', 'POINT (37.6 55.75)', 'Previous'],
        [3, 'orphan-next', '2026-01-06 12:52:00', 'POINT (37.61 55.76)', 'Next'],
    ])
    append_csv(dataset, 'points.csv', [
        ['orphan-snapshot', 3, '2026-01-06 12:40:00', 'orphan-next', '2026-01-06 12:52:00', 180],
    ])
    loaded = load_replay(dataset, ReplayConfig(start=T))
    assert loaded.end == T+timedelta(minutes=10)
    assert loaded.fleet['without_telemetry'] == [3]
    assert all(payload.tr_id == 1 for _, _, payload in loaded.events)
    assert loaded.quality['snapshots'] == 2


def test_warmup_only_archive_is_finished_on_load_and_reset(dataset, monkeypatch):
    monkeypatch.setenv('REPLAY_DATA_DIR', str(dataset))
    with TestClient(create_app(start_background=False, enable_ndtp=False)) as client:
        body = dict(start=(T+timedelta(minutes=1)).isoformat(), deviation_source='gps',
                    duration_minutes=None, paused=False)
        response = client.post('/api/v1/replay/load', json=body)
        assert response.status_code == 200
        state = response.json()['replay']
        assert state['finished'] and not state['running'] and state['start'] == state['end']
        assert state['delivered'] == state['total'] == 3
        assert client.post('/api/v1/replay/control', json={'action':'resume'}).status_code == 409
        reset = client.post('/api/v1/replay/control', json={'action':'reset'}).json()['replay']
        assert reset['end'] == state['end'] and reset['finished'] and not reset['running']


def test_event_resource_limit_fails_explicitly_without_truncating(dataset, monkeypatch):
    from backend import replay as module
    monkeypatch.setattr(module, 'MAX_REPLAY_EVENTS', 2)
    with pytest.raises(ValueError, match='Более 2 событий'):
        load_replay(dataset, ReplayConfig(start=T, deviation_source='gps'))


@pytest.mark.parametrize('minutes', [-1,0,121])
def test_explicit_duration_remains_bounded_for_manual_checks(minutes):
    with pytest.raises(ValueError):
        ReplayConfig(duration_minutes=minutes)


@pytest.mark.parametrize('argv,expected', [([], None), (['--minutes','15'], 15)])
def test_load_replay_cli_sends_null_by_default_and_preserves_explicit_minutes(monkeypatch, capsys, argv, expected):
    from tools import load_replay as cli
    requests = []
    def fake_urlopen(request, **kwargs):
        requests.append(json.loads(request.data))
        return io.BytesIO(b'{"ok":true}')
    monkeypatch.setattr(cli, 'urlopen', fake_urlopen)
    monkeypatch.setattr(sys, 'argv', ['load_replay.py', *argv])
    cli.main()
    assert requests[0]['duration_minutes'] == expected
    assert requests[0]['tr_ids'] is None and requests[0]['paused'] is True
