"""HTTP-граница backend ↔ отдельный producer: изоляция, происхождение, гонки.

Оба настоящих ASGI-приложения работают без фонового времени и сокетов.
Задержки/обрывы добавляются только в HTTP-транспорт, не в бизнес-логику.
"""
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime

import httpx
import pytest

from backend.app import create_app as backend_app
from generator.app import create_app as producer_app


@asynccontextmanager
async def stack():
    producer = producer_app(start_background=False)
    backend = backend_app(start_background=False, enable_ndtp=False,
                          generator_url='http://producer', ml_url='http://producer/ml')
    async with producer.router.lifespan_context(producer):
        async with backend.router.lifespan_context(backend):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=producer),
                                         base_url='http://producer') as producer_client:
                backend.state.engine.client = producer_client
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=backend),
                                             base_url='http://backend') as client:
                    yield client, backend.state.engine, producer, producer_client


def test_start_controls_logs_and_source_provenance_through_real_apps():
    async def run():
        async with stack() as (client, engine, producer, _):
            assert (await client.post('/api/v1/generator/control', json={'action':'pause'})).status_code == 409
            assert (await client.get('/api/v1/generator/logs')).status_code == 409
            result = await client.post('/api/v1/generator/start', json={
                'scenario':'slow_segment', 'seed':42, 'speed':10, 'paused':True})
            assert result.status_code == 200, result.text
            session_id = result.json()['generator']['session_id']
            state = (await client.get('/api/v1/state')).json()
            assert state['mode'] == 'generator'
            assert state['generator']['scenario'] == 'slow_segment'
            assert state['generator']['connected'] and not state['generator']['running']
            assert state['generator']['received_frames'] == state['generator']['emitted_frames'] == 1
            assert state['generator']['received_telemetry'] == state['generator']['emitted_telemetry'] == 8
            assert state['generator']['route_count'] == 4
            assert state['generator']['telemetry_interval_s'] == 15
            assert {v['route_id'] for v in state['vehicles']} == {f'synthetic-{i}' for i in range(1, 5)}
            assert all(point.source == 'generator' for points in engine.history.values() for point in points)
            cutoff = datetime.fromisoformat(state['clock_time'])
            assert all(v['cur_dev_s'] is None and v['current_deviation'] is None for v in state['vehicles'])
            assert not any(engine.arrivals.values())
            logs = await client.get('/api/v1/generator/logs')
            assert logs.status_code == 200 and logs.headers['cache-control'] == 'no-store'
            assert logs.json()['session_id'] == session_id
            assert any(item['kind'] == 'telemetry' for item in logs.json()['logs'])
            assert all(datetime.fromisoformat(item['time']) <= cutoff for item in logs.json()['logs'])
            assert (await client.post('/api/v1/generator/control', json={'action':'speed'})).status_code == 422
            assert (await client.post('/api/v1/generator/control', json={'action':'resume'})).json()['generator']['running']
            producer.state.session.advance(18)
            paused = await client.post('/api/v1/generator/control', json={'action':'pause'})
            assert paused.status_code == 200, paused.text
            assert not paused.json()['generator']['running']
            assert paused.json()['generator']['received_frames'] >= 181
            state = (await client.get('/api/v1/state')).json()
            known = [v for v in state['vehicles'] if v['current_deviation'] is not None]
            assert known, 'Несколько GPS должны подтвердить хотя бы одно посещение'
            cutoff = datetime.fromisoformat(state['clock_time'])
            for vehicle in known:
                deviation = vehicle['current_deviation']
                assert deviation['source'] == 'gps_estimate'
                actual, received, planned = (datetime.fromisoformat(deviation[key])
                    for key in ('observed_at', 'received_at', 'planned_at'))
                assert actual < received <= cutoff
                assert vehicle['cur_dev_s'] == (actual-planned).total_seconds()
            assert all(arrival.source == 'gps_estimate' for records in engine.arrivals.values()
                       for arrival in records.values())
            at = paused.json()['generator']['clock_time']
            producer.state.session.advance(100)
            await engine.poll_generator()
            assert (await client.get('/api/v1/state')).json()['clock_time'] == at
            speed = await client.post('/api/v1/generator/control', json={'action':'speed','speed':30})
            assert speed.status_code == 200 and speed.json()['generator']['speed'] == 30
            assert speed.json()['generator']['session_id'] == session_id
            assert speed.json()['generator']['clock_time'] == at
    asyncio.run(run())


@pytest.mark.parametrize('failure', ['unreachable','invalid_json','unexpected_hints'])
def test_failed_start_preserves_existing_demo(failure):
    async def run():
        async with stack() as (client, engine, producer, _):
            assert (await client.post('/api/v1/mode', json={'mode':'demo'})).status_code == 200
            version = engine.version
            before = (await client.get('/api/v1/state')).json()
            def fail(request):
                if failure == 'unreachable':
                    raise httpx.ConnectError('Synthetic connection failure', request=request)
                if failure == 'invalid_json':
                    return httpx.Response(200, content=b'broken JSON')
                payload = producer.state.session.status(include_context=True)
                payload['context']['hints'] = [{'tr_id':101, 'observed_at':payload['clock_time'], 'delay_s':123}]
                return httpx.Response(200, json=payload)
            async with httpx.AsyncClient(transport=httpx.MockTransport(fail)) as broken:
                engine.client = broken
                response = await client.post('/api/v1/generator/start', json={})
            assert response.status_code == 502
            after = (await client.get('/api/v1/state')).json()
            assert engine.version == version
            assert after['mode'] == before['mode'] == 'demo'
            assert after['vehicles'] == before['vehicles']
            assert after['routes'] == before['routes']
            assert after['generator'] is None
    asyncio.run(run())


def test_route_count_twenty_has_forty_buses_and_only_gps_before_confirmation():
    async def run():
        async with stack() as (client, engine, _, _):
            result = await client.post('/api/v1/generator/start', json={
                'route_count':20, 'telemetry_interval_s':15, 'paused':True})
            assert result.status_code == 200, result.text
            state = (await client.get('/api/v1/state')).json()
            assert len(state['routes']) == 20 and len(state['vehicles']) == 40
            assert state['generator']['route_count'] == 20
            assert state['generator']['telemetry_interval_s'] == 15
            assert state['generator']['received_telemetry'] == 40
            assert all(v['cur_dev_s'] is None and v['current_deviation'] is None for v in state['vehicles'])
            assert not any(engine.arrivals.values())
            session_id = state['generator']['session_id']
            for invalid in ({'route_count':0}, {'route_count':21}, {'telemetry_interval_s':0}, {'telemetry_interval_s':31}):
                assert (await client.post('/api/v1/generator/start', json=invalid)).status_code == 422
                assert engine.generator.view.session_id == session_id
    asyncio.run(run())


def test_generator_rejects_live_inputs_and_quarantines_producer_failure():
    async def run():
        async with stack() as (client, engine, producer, _):
            assert (await client.post('/api/v1/generator/start', json={})).status_code == 200
            frame = producer.state.session.frames[-1]
            before_count = engine.state()['generator']['received_telemetry']
            assert (await client.post('/api/v1/telemetry', json=frame['telemetry'])).status_code == 409
            assert frame['arrivals'] == []
            manual = producer.state.session.truth()['observed_truth_arrivals'][0]
            assert (await client.post('/api/v1/arrivals', json=manual)).status_code == 409
            assert (await client.post('/api/v1/demo/control', json={'action':'resume'})).status_code == 409
            assert engine.state()['generator']['received_telemetry'] == before_count
            def fail(request):
                raise httpx.ConnectError('Producer disconnected', request=request)
            async with httpx.AsyncClient(transport=httpx.MockTransport(fail)) as broken:
                engine.client = broken
                control = await client.post('/api/v1/generator/control', json={'action':'resume'})
            assert control.status_code == 502
            state = (await client.get('/api/v1/state')).json()
            assert state['mode'] == 'generator'
            assert not state['generator']['connected']
            assert state['generator']['error']
            assert state['summary']['unknown'] == len(state['vehicles'])
            assert state['generator']['received_telemetry'] == before_count
    asyncio.run(run())


@pytest.mark.parametrize('operation', ['start','control'])
def test_mode_change_while_producer_http_is_pending_cannot_restore_old_context(operation):
    async def run():
        async with stack() as (client, engine, _, producer_client):
            if operation == 'control':
                assert (await client.post('/api/v1/generator/start', json={})).status_code == 200
            entered, release = asyncio.Event(), asyncio.Event()
            async def delayed(request):
                if request.url.path == ('/reset' if operation == 'start' else '/control'):
                    entered.set()
                    await release.wait()
                return await producer_client.request(request.method, request.url.path,
                    params=request.url.params, content=request.content, headers=request.headers)
            async with httpx.AsyncClient(transport=httpx.MockTransport(delayed)) as pending_transport:
                engine.client = pending_transport
                endpoint = '/api/v1/generator/start' if operation == 'start' else '/api/v1/generator/control'
                task = asyncio.create_task(client.post(endpoint, json={} if operation == 'start' else {'action':'resume'}))
                try:
                    await asyncio.wait_for(entered.wait(), timeout=1)
                    assert (await client.post('/api/v1/mode', json={'mode':'live'})).status_code == 200
                    replacement_version = engine.version
                    release.set()
                    response = await asyncio.wait_for(task, timeout=1)
                    assert response.status_code == 409, response.text
                    state = (await client.get('/api/v1/state')).json()
                    assert state['mode'] == 'live' and state['vehicles'] == []
                    assert state['generator'] is None and engine.version == replacement_version
                finally:
                    release.set()
                    if not task.done():
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
    asyncio.run(run())
