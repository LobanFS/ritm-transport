"""Большой парк проходит все пакеты ML и не публикует старый контекст."""
import asyncio
import json
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from backend.engine import Engine, LiveContext
from backend.ndtp import NDTPServer
from common.contracts import Features, PredictionRequest, StopTarget, baseline

T = datetime(2026, 1, 6, 12, tzinfo=timezone.utc)


def fleet(client, count=257):
    engine = Engine('http://ml', client)
    engine.set_live(LiveContext(vehicles=[dict(tr_id=i, unit_id=i, label=str(i), route_id='line') for i in range(1, count+1)]))
    engine.mode = 'demo'
    engine.running = False
    engine.clock = T
    def request(tr_id, now):
        return PredictionRequest(request_id=str(tr_id), tr_id=tr_id, issued_at=now,
            target=StopTarget(id=str(tr_id), name='Остановка', scheduled_at=now+timedelta(minutes=12), lat=55.7, lon=37.6),
            current_delay_s=10, features=Features(telemetry_age_s=0))
    engine.request_for = request
    return engine


@pytest.mark.parametrize('fail_second', [False, True])
def test_all_vehicles_are_published_across_ml_batches(fail_second):
    async def run():
        sizes = []
        async def handle(request):
            batch = json.loads(request.content)
            sizes.append(len(batch))
            assert len(batch) <= 128
            if fail_second and len(sizes) == 2:
                return httpx.Response(503)
            return httpx.Response(200, json=[baseline(PredictionRequest.model_validate(item)).model_dump(mode='json') for item in batch])
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            engine = fleet(client)
            await engine.tick(0)
            assert sizes == ([128, 128] if fail_second else [128, 128, 1])
            assert set(engine.predictions) == set(range(1, 258))
            assert len(engine.forecast_traces) == 257
            assert engine.ml_status == ('unavailable' if fail_second else 'ok')
            assert {p.method for p in engine.predictions.values()} == ({'fallback'} if fail_second else {'persistence'})
    asyncio.run(run())


def test_new_context_stops_remaining_batches_and_old_publication():
    async def run():
        calls = []
        async def handle(request):
            batch = json.loads(request.content)
            calls.append(batch)
            engine.set_live()
            return httpx.Response(200, json=[baseline(PredictionRequest.model_validate(item)).model_dump(mode='json') for item in batch])
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            engine = fleet(client)
            await engine.tick(0)
            assert len(calls) == 1
            assert not engine.predictions and not engine.vehicles and not engine.forecast_traces
    asyncio.run(run())


def test_default_ndtp_accepts_more_than_128_connections():
    async def run():
        async def receive(*args):
            pass
        errors = []
        server = NDTPServer(receive, errors.append, host='127.0.0.1', port=0)
        await server.start()
        writers = []
        try:
            for _ in range(129):
                _, writer = await asyncio.open_connection('127.0.0.1', server.port)
                writers.append(writer)
            await asyncio.sleep(0)
            assert server.connections == 129
            assert not errors
        finally:
            for writer in writers:
                writer.close()
            await asyncio.gather(*(writer.wait_closed() for writer in writers))
            await server.close()
    asyncio.run(run())
