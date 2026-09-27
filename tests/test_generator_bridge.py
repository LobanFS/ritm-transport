"""Причинность пачек, конкуренция запросов и реальная свежесть сценарного producer."""
import asyncio
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json

import httpx
import pytest

from backend.arrivals import ArrivalInput
from backend.engine import Engine, LiveContext
from backend.generator_bridge import GeneratorSession, ProducerState
from common.contracts import PredictionRequest, baseline


T = datetime(2026, 1, 6, 8, tzinfo=timezone.utc)


def setup(*, running=False):
    context = LiveContext.model_validate({
        'vehicles': [{'tr_id': 101, 'unit_id': 1001, 'label': 'ТС 101', 'route_id': 'route'}],
        'schedule': [dict(tr_id=101, target=dict(id=key, name=key, scheduled_at=at,
            lat=55.75, lon=37.6)) for key, at in [
                ('older', T-timedelta(seconds=300)), ('last', T-timedelta(seconds=200)),
                ('future', T+timedelta(minutes=12))]],
    })
    status = ProducerState(session_id='session-1', scenario='normal', seed=42,
                           clock_time=T, running=running, speed=10)
    engine = Engine('http://ml')
    engine.set_generator(context, status)
    wall = [0.0]
    engine.generator = GeneratorSession(status, monotonic=lambda: wall[0])
    return engine, wall


def frame(seq=1, *, at=T, arrivals=None):
    return dict(seq=seq, clock_time=at.isoformat(), telemetry=[dict(
        tr_id=101, unit_id=1001, event_time=at.isoformat(), received_at=at.isoformat(),
        lat=55.75, lon=37.6, speed_kmh=20, event_id=f'frame-{seq}')], arrivals=arrivals or [])


def stream(engine, frames, *, at=None, last_seq=None, running=None, finished=False):
    view = engine.generator.view
    return dict(session_id=view.session_id, scenario=view.scenario, seed=view.seed,
        speed=view.speed, clock_time=(at or engine.clock).isoformat(),
        running=view.running if running is None else running, finished=finished,
        first_seq=1, last_seq=last_seq if last_seq is not None else (
            frames[-1]['seq'] if frames else engine.generator.cursor), frames=frames, gap=False)


def arrival(stop='last', *, at=T):
    return dict(tr_id=101, planned_stop_id=stop, arrived_at=at.isoformat())


def test_concurrent_polls_use_updated_cursor_without_false_disconnect():
    async def exercise():
        engine, _ = setup()
        entered, release = asyncio.Event(), asyncio.Event()
        cursors = []

        async def handler(request):
            after = int(request.url.params['after'])
            cursors.append(after)
            if len(cursors) == 1:
                entered.set()
                await release.wait()
            return httpx.Response(200, json=stream(engine, [frame()] if after == 0 else [], last_seq=1))

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            engine.client = client
            first = asyncio.create_task(engine.poll_generator())
            await entered.wait()
            second = asyncio.create_task(engine.poll_generator())
            await asyncio.sleep(0)
            release.set()
            await asyncio.gather(first, second)
        assert cursors == [0, 1]
        assert engine.generator.view.connected
        assert engine.generator.cursor == 1 and engine.input_sequences[101] == 1
    asyncio.run(exercise())


def test_unknown_vehicle_in_later_frame_does_not_partially_ingest_batch():
    engine, _ = setup()
    second_time = T+timedelta(seconds=1)
    frames = [frame(), frame(2, at=second_time)]
    frames[1]['telemetry'][0]['unit_id'] = 9999
    before_metrics = dict(engine.metrics)
    with pytest.raises(ValueError, match='неизвестное ТС/терминал'):
        engine.generator.consume(engine, stream(engine, frames, at=second_time))
    assert engine.clock == T and engine.generator.cursor == 0
    assert not engine.history and not engine.arrivals and not engine.input_sequences
    assert engine.metrics == before_metrics


@pytest.mark.parametrize('future_kind', ['event', 'receipt', 'arrival'])
def test_future_observation_rejects_whole_batch(future_kind):
    engine, _ = setup()
    frames = [frame(), frame(2)]
    future = (T+timedelta(seconds=1)).isoformat()
    if future_kind == 'arrival':
        frames[1]['arrivals'] = [arrival(at=T+timedelta(seconds=1))]
    else:
        frames[1]['telemetry'][0]['event_time' if future_kind == 'event' else 'received_at'] = future
    with pytest.raises(ValueError, match='Будущ'):
        engine.generator.consume(engine, stream(engine, frames))
    assert engine.generator.cursor == 0 and engine.clock == T
    assert not engine.history and not engine.arrivals


def test_stalled_running_source_goes_unknown_and_recovers_only_on_progress():
    engine, wall = setup(running=True)
    engine.generator.consume(engine, stream(engine, [frame()]))
    # Только fixture watchdog: доверенный внешний факт не идёт через producer.
    engine.ingest_arrival(ArrivalInput.model_validate(arrival()), received_at=T)
    engine.predictions[101] = baseline(engine.request_for(101, T))
    assert engine.state()['vehicles'][0]['prediction']['risk'] == 'red'
    wall[0] = 5
    engine.generator.consume(engine, stream(engine, []))
    assert not engine.generator.view.connected
    assert '5 секунд' in engine.generator.view.error
    state = engine.state()
    assert state['health']['source_status'] == 'disconnected'
    assert state['vehicles'][0]['prediction']['risk'] == 'unknown'
    assert state['vehicles'][0]['status'] == 'stale'
    wall[0] = 6
    engine.generator.consume(engine, stream(engine, []))
    assert not engine.generator.view.connected
    advanced = T+timedelta(seconds=1)
    engine.generator.consume(engine, stream(engine, [frame(2, at=advanced)], at=advanced))
    assert engine.generator.view.connected and engine.clock == advanced
    assert engine.state()['vehicles'][0]['status'] == 'fresh'


@pytest.mark.parametrize('finished', [False, True])
def test_watchdog_ignores_intentional_pause_or_finished_session(finished):
    engine, wall = setup()
    engine.generator.consume(engine, stream(engine, [frame()], finished=finished))
    wall[0] = 600
    assert engine.generator.check_progress()
    engine.generator.consume(engine, stream(engine, [], finished=finished))
    assert engine.generator.view.connected


def test_resume_starts_new_watchdog_budget_after_long_pause():
    engine, wall = setup()
    engine.generator.consume(engine, stream(engine, [frame()]))
    wall[0] = 600
    engine.generator.consume(engine, stream(engine, [], running=True))
    assert engine.generator.check_progress()
    wall[0] = 604.9
    assert engine.generator.check_progress()
    wall[0] = 605
    assert not engine.generator.check_progress()


@pytest.mark.parametrize('response_fails', [False, True])
def test_old_poll_response_cannot_change_replacement_context(response_fails):
    async def exercise():
        engine, _ = setup()
        old_session = engine.generator
        old_payload = stream(engine, [frame()])
        entered, release = asyncio.Event(), asyncio.Event()

        async def handler(request):
            entered.set()
            await release.wait()
            return httpx.Response(503 if response_fails else 200, json=old_payload)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            engine.client = client
            task = asyncio.create_task(engine.poll_generator())
            await entered.wait()
            engine.set_live()
            replacement_clock = engine.clock
            release.set()
            await task
        assert engine.mode == 'live' and engine.generator is None
        assert engine.clock == replacement_clock
        assert not engine.history and not engine.arrivals
        assert old_session.cursor == 0 and old_session.view.connected
    asyncio.run(exercise())


@pytest.mark.parametrize('watchdog', [False, True])
def test_source_disconnected_during_ml_does_not_publish_prediction_or_incident(watchdog):
    async def exercise():
        engine, wall = setup(running=True)
        # Только fixture красного риска: отдельный операционный факт.
        engine.ingest_arrival(ArrivalInput.model_validate(arrival()), received_at=T)
        entered, release = asyncio.Event(), asyncio.Event()

        async def handler(request):
            if request.url.path == '/stream':
                return httpx.Response(200, json=stream(engine, [frame()]))
            entered.set()
            await release.wait()
            predictions = [baseline(PredictionRequest.model_validate(row)).model_dump(mode='json')
                           for row in json.loads(request.content)]
            assert predictions[0]['risk'] == 'red'
            return httpx.Response(200, json=predictions)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            engine.client = client
            task = asyncio.create_task(engine.tick(0))
            await entered.wait()
            if watchdog:
                wall[0] = 5
            else:
                engine.generator.fail('Источник отключён параллельным опросом')
            release.set()
            await task
        assert not engine.predictions and not engine.incidents and not engine.forecast_traces
        assert engine.ml_status == 'ok' and not engine.generator.view.connected
    asyncio.run(exercise())


@pytest.mark.parametrize('when', [T-timedelta(seconds=100), T])
def test_generator_truth_arrival_is_rejected_without_any_partial_gps_state(when):
    engine, _ = setup()
    after = T+timedelta(seconds=5)
    frames = [frame(), frame(2, at=after, arrivals=[arrival(at=when)])]
    frames[0]['telemetry'][0]['speed_kmh'] = 0
    before_status = engine.gps_detectors[101].status()
    before_metrics = dict(engine.metrics)
    with pytest.raises(ValueError, match='Истинные прибытия генератора запрещены'):
        engine.generator.consume(engine, stream(engine, frames, at=after))
    assert engine.gps_detectors[101].status() == before_status
    assert engine.metrics == before_metrics
    assert not engine.history and not engine.arrivals and not engine.input_sequences
    assert engine.clock == T and engine.generator.cursor == 0


def test_changed_producer_session_rejected_without_side_effects():
    engine, _ = setup()
    payload = deepcopy(stream(engine, [frame()]))
    payload['session_id'] = 'another-session'
    with pytest.raises(ValueError, match='сменил сессию'):
        engine.generator.consume(engine, payload)
    assert engine.clock == T and engine.generator.cursor == 0 and not engine.history
