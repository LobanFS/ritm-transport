"""Контракт прибытие → cur_dev_s → неизменный snapshot инференса."""
import asyncio
from datetime import datetime, timedelta, timezone
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.app import create_app
from backend.arrivals import ArrivalConflict, ArrivalInput
from backend.engine import Engine, LiveContext
from common.contracts import PredictionRequest, Telemetry, baseline

T = datetime(2026, 1, 6, 8, tzinfo=timezone.utc)


def context(at=T):
    def stop(name, delta):
        return dict(tr_id=1, target=dict(id=name, name=name, scheduled_at=at+timedelta(seconds=delta), lat=55.75, lon=37.6))
    return LiveContext(vehicles=[dict(tr_id=1, unit_id=2, label='Bus', route_id='r')],
        schedule=[stop('older', -400), stop('last', -180), stop('next', -60), stop('target', 720)])


def live():
    engine = Engine('http://ml')
    engine.set_live(context())
    return engine


def event(stop='last', when=T):
    return ArrivalInput(tr_id=1, planned_stop_id=stop, arrived_at=when)


def test_deviation_is_fact_minus_plan_and_does_not_grow_with_clock_or_gps():
    e = live()
    assert e.deviation_at(1, T) is None
    e.ingest(Telemetry(tr_id=1, event_time=T, received_at=T, lat=55.75, lon=37.6, speed_kmh=0))
    assert e.deviation_at(1, T) is None  # даже GPS точно на остановке не равен подтверждению
    e.ingest_arrival(event(), received_at=T)
    assert e.deviation_at(1, T).delay_s == 180
    assert e.deviation_at(1, T+timedelta(minutes=20)).delay_s == 180
    assert baseline(e.request_for(1, T)).predicted_delay_s == 180
    assert baseline(e.request_for(1, T+timedelta(seconds=30))).predicted_delay_s == 180
    e.ingest_arrival(event('next', T+timedelta(seconds=60)), received_at=T+timedelta(seconds=60))
    assert e.deviation_at(1, T+timedelta(seconds=59)).delay_s == 180
    assert e.deviation_at(1, T+timedelta(seconds=60)).delay_s == 120


def test_arrival_requires_available_fact_known_vehicle_and_planned_stop():
    e = live()
    with pytest.raises(ValueError):
        e.ingest_arrival(event(when=T+timedelta(seconds=1)), received_at=T)
    with pytest.raises(ValueError):
        e.ingest_arrival(event('missing'), received_at=T)
    with pytest.raises(ValueError):
        e.ingest_arrival(event().model_copy(update={'tr_id':999}), received_at=T)
    e.ingest_arrival(event(), received_at=T+timedelta(seconds=10))
    assert e.deviation_at(1, T+timedelta(seconds=9)) is None
    assert e.deviation_at(1, T+timedelta(seconds=10)).delay_s == 180


def test_duplicate_conflict_and_late_old_stop_do_not_rewrite_current_state():
    e = live()
    assert e.ingest_arrival(event(), received_at=T)
    assert not e.ingest_arrival(event(), received_at=T+timedelta(seconds=10))
    assert e.deviation_at(1, T).received_at == T
    with pytest.raises(ArrivalConflict):
        e.ingest_arrival(event(when=T-timedelta(seconds=1)), received_at=T)
    e.ingest_arrival(event('older', T-timedelta(seconds=100)), received_at=T+timedelta(seconds=20))
    assert e.deviation_at(1, T+timedelta(seconds=20)).planned_stop_id == 'last'
    e.set_live(context())
    assert e.deviation_at(1, T) is None


def test_early_arrival_retains_negative_sign_and_arrival_outweighs_hint():
    from backend.engine import DelayHint
    e = live()
    e.hints[1].append(DelayHint(tr_id=1, observed_at=T, delay_s=999))
    e.ingest_arrival(event(when=T-timedelta(seconds=200)), received_at=T)
    assert e.deviation_at(1, T).delay_s == -20
    assert e.deviation_at(1, T).source == 'arrival'


def test_demo_deviation_changes_only_on_delivered_arrival_and_gps_meets_stop():
    e = Engine('http://ml')
    value = e.deviation_at(101, e.clock).delay_s
    e.clock += timedelta(seconds=1)
    e.generate_demo()
    assert e.deviation_at(101, e.clock).delay_s == value
    next_i = e.demo_next_arrival[101]
    stop, actual = e.demo_timelines[101][next_i]
    e.clock = actual
    e.generate_demo()
    state = next(v for v in e.state()['vehicles'] if v['tr_id'] == 101)
    assert state['current_deviation']['planned_stop_id'] == stop.id
    assert state['cur_dev_s'] == (actual-stop.scheduled_at).total_seconds()
    assert state['lat'] == stop.lat and state['lon'] == stop.lon and state['speed_kmh'] == 0
    assert not e.hints[101]


def test_trace_freezes_actual_input_when_new_arrival_is_delivered_during_ml(monkeypatch):
    import backend.engine as module
    clock = T
    monkeypatch.setattr(module, 'utcnow', lambda: clock)
    async def exercise():
        async def handler(request):
            nonlocal clock
            values = [baseline(PredictionRequest.model_validate(r)).model_dump(mode='json') for r in json.loads(request.content)]
            clock = T+timedelta(seconds=30)
            e.ingest_arrival(event('next', clock), received_at=clock)
            return httpx.Response(200, json=values)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            e = live()
            e.client = client
            e.ingest_arrival(event(), received_at=T)
            await e.tick()
            trace = e.forecast_traces[1]
            assert trace['request']['current_delay_s'] == trace['response']['predicted_delay_s'] == 180
            assert trace['current_deviation']['planned_stop_id'] == 'last'
            assert trace['execution'] == 'ml_http'
            state = e.state()['vehicles'][0]
            assert state['cur_dev_s'] == 90 and state['prediction_current_delay_s'] == 180
            assert state['prediction']['predicted_delay_s'] == 180
            assert trace['current_deviation']['delay_s'] == 180  # не переписан свежим состоянием
    asyncio.run(exercise())


def test_arrival_http_and_trace_reproduce_formula_without_reloading_context():
    at = datetime.now(timezone.utc)
    with TestClient(create_app(start_background=False, enable_ndtp=False)) as client:
        client.post('/api/v1/live/context', json=context(at).model_dump(mode='json'))
        assert client.get('/api/v1/vehicles/1/forecast-trace').status_code == 404
        payload = event(when=at).model_dump(mode='json')
        response = client.post('/api/v1/arrivals', json=payload)
        assert response.status_code == 200 and response.json()['accepted'] is True
        assert response.json()['current_deviation']['delay_s'] == 180
        assert client.post('/api/v1/arrivals', json=payload).json()['accepted'] is False
        assert client.post('/api/v1/arrivals', json={**payload, 'received_at':at.isoformat()}).status_code == 422
        assert client.post('/api/v1/arrivals', json={**payload, 'arrived_at':(at-timedelta(seconds=1)).isoformat()}).status_code == 409
        assert client.post('/api/v1/arrivals', json={**payload, 'arrived_at':(at+timedelta(days=1)).isoformat()}).status_code == 422
        engine = client.app.state.engine
        engine.client = None
        asyncio.run(engine.tick())
        trace = client.get('/api/v1/vehicles/1/forecast-trace').json()
        assert trace['request']['current_delay_s'] == trace['response']['predicted_delay_s'] == 180
        assert trace['execution'] == 'backend_fallback'
        assert client.get('/api/v1/state').json()['vehicles'][0]['current_deviation']['source'] == 'arrival'
        client.post('/api/v1/mode', json={'mode':'demo'})
        assert client.post('/api/v1/arrivals', json=payload).status_code == 409
