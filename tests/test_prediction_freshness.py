"""Граница свежести между результатом ML, публикацией алерта и чтением UI state."""
import asyncio
from datetime import datetime, timedelta, timezone
import json

import httpx
import pytest

from backend.arrivals import ArrivalInput
from backend.engine import Engine, LiveContext
from common.contracts import Prediction, PredictionRequest, Telemetry, baseline

T = datetime(2026, 1, 6, 9, tzinfo=timezone.utc)


def configured(monkeypatch, *, issued_s=0, source='external_hint', client=None, probability_status='validated'):
    clock = {'at': T}
    monkeypatch.setattr('backend.engine.utcnow', lambda: clock['at'])
    engine = Engine('http://ml', client)
    target = dict(id='target', name='Цель', scheduled_at=T+timedelta(seconds=issued_s+750), lat=55.76, lon=37.61)
    last = dict(id='last', name='Последнее посещение', scheduled_at=T-timedelta(seconds=180), lat=55.75, lon=37.6)
    context = dict(vehicles=[dict(tr_id=101, unit_id=1, label='Тест', route_id='route')],
                   schedule=[dict(tr_id=101, target=last), dict(tr_id=101, target=target)], hints=[])
    if source != 'gps_estimate':
        context['hints'].append(dict(tr_id=101, observed_at=T, delay_s=180, source=source))
    engine.set_live(LiveContext.model_validate(context))
    if source == 'gps_estimate':
        engine.ingest_arrival(ArrivalInput(tr_id=101, planned_stop_id='last', arrived_at=T),
                              received_at=T, source='gps_estimate')
    issued = T+timedelta(seconds=issued_s)
    engine.predictions[101] = Prediction(
        request_id='fixture', tr_id=101, issued_at=issued, target=target, predicted_delay_s=180,
        model_version='test-regression', method='learned', risk='red', probability_late=.9,
        probability_status=probability_status, probability_note='Тестовая вероятность', reasons=[],
    )
    engine.forecast_traces[101] = {'request': {'current_delay_s': 180},
                                  'response': engine.predictions[101].model_dump(mode='json')}
    return engine, clock


def gps(engine, at):
    engine.ingest(Telemetry(tr_id=101, event_time=at, received_at=at,
                           lat=55.75, lon=37.6, speed_kmh=20))


@pytest.mark.parametrize('status',['validated','transferred'])
def test_reading_state_after_gps_expires_removes_probability_without_rewriting_ml_trace(monkeypatch,status):
    engine, clock = configured(monkeypatch,probability_status=status)
    gps(engine, T)
    clock['at'] = T+timedelta(seconds=60)
    assert engine.state()['vehicles'][0]['prediction']['probability_late'] == .9
    assert engine.state()['vehicles'][0]['prediction']['probability_status'] == status
    clock['at'] += timedelta(microseconds=1)
    state = engine.state()
    vehicle = state['vehicles'][0]
    assert vehicle['status'] == 'stale'
    assert vehicle['prediction']['risk'] == 'unknown'
    assert vehicle['prediction']['probability_late'] is None
    assert vehicle['prediction']['probability_status'] == 'unavailable'
    assert vehicle['prediction']['probability_note']
    assert vehicle['prediction']['predicted_delay_s'] == 180
    assert vehicle['prediction_current_delay_s'] == 180
    assert state['route_risks'][0]['risk'] == 'unknown'
    assert state['section_risks'][0]['risk'] == 'unknown'
    assert engine.forecast_traces[101]['response']['probability_late'] == .9
    assert engine.forecast_traces[101]['response']['probability_status'] == status
    assert engine.predictions[101].probability_late == .9


@pytest.mark.parametrize('source', ['external_hint', 'csv_snapshot', 'synthetic_hint', 'gps_estimate'])
def test_expired_current_deviation_cannot_keep_red_route_with_fresh_gps(monkeypatch, source):
    engine, clock = configured(monkeypatch, issued_s=290, source=source)
    gps(engine, T+timedelta(seconds=300))
    clock['at'] = T+timedelta(seconds=300)
    assert engine.state()['vehicles'][0]['prediction']['risk'] == 'red'
    clock['at'] += timedelta(microseconds=1)
    state = engine.state()
    vehicle = state['vehicles'][0]
    assert vehicle['status'] == 'fresh'
    assert vehicle['current_deviation'] is None and vehicle['cur_dev_s'] is None
    assert vehicle['prediction']['risk'] == 'unknown'
    assert vehicle['prediction']['probability_late'] is None
    assert vehicle['prediction']['predicted_delay_s'] == 180  # прежнее число с временем расчёта
    assert state['route_risks'][0]['unknown'] == 1
    assert state['route_risks'][0]['max_predicted_delay_s'] is None


def test_no_gps_hides_previous_probability_even_with_available_deviation(monkeypatch):
    engine, _ = configured(monkeypatch)
    vehicle = engine.state()['vehicles'][0]
    assert vehicle['status'] == 'no_data' and vehicle['cur_dev_s'] == 180
    assert vehicle['prediction']['risk'] == 'unknown'
    assert vehicle['prediction']['probability_late'] is None


@pytest.mark.parametrize('boundary,published_s,should_alert', [
    ('gps', 60, True), ('gps', 60.001, False),
    ('hint', 300, True), ('hint', 300.001, False),
])
def test_expiry_during_http_inference_gates_incident_at_publication(monkeypatch, boundary, published_s, should_alert):
    async def exercise():
        async def response(request):
            rows = []
            for raw in json.loads(request.content):
                parsed = PredictionRequest.model_validate(raw)
                assert parsed.current_delay_s == 180
                rows.append(baseline(parsed).model_copy(update={
                    'method':'learned', 'model_version':'test-regression',
                    'risk':'red', 'probability_late':.9, 'probability_note':'Тестовая вероятность',
                }).model_dump(mode='json'))
            clock['at'] = T+timedelta(seconds=published_s)
            return httpx.Response(200, json=rows)

        async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
            issued_s = 59 if boundary == 'gps' else 299
            engine, clock = configured(monkeypatch, issued_s=issued_s, client=client)
            gps(engine, T if boundary == 'gps' else T+timedelta(seconds=issued_s))
            clock['at'] = T+timedelta(seconds=issued_s)
            await engine.tick(0)
            assert engine.ml_status == 'ok'
            assert bool(engine.incidents) is should_alert
            prediction = engine.state()['vehicles'][0]['prediction']
            assert prediction['predicted_delay_s'] == 180
            assert prediction['risk'] == ('red' if should_alert else 'unknown')
            assert prediction['probability_late'] == (.9 if should_alert else None)
            assert engine.forecast_traces[101]['response']['probability_late'] == .9
    asyncio.run(exercise())


def test_known_target_visit_is_not_a_new_warning_and_ignores_late_confirmation(monkeypatch):
    engine, clock = configured(monkeypatch)
    gps(engine, T)
    # Искусственная крайняя ситуация: целевое посещение состоялось сильно раньше плана.
    engine.ingest_arrival(ArrivalInput(tr_id=101, planned_stop_id='target', arrived_at=T-timedelta(seconds=1)),
                          received_at=T+timedelta(seconds=5))
    assert not engine.target_reached_at(101, 'target', T)
    assert engine.state()['vehicles'][0]['prediction']['risk'] == 'red'
    clock['at'] = T+timedelta(seconds=5)
    assert engine.target_reached_at(101, 'target', clock['at'])
    prediction = engine.state()['vehicles'][0]['prediction']
    assert prediction['risk'] == 'unknown' and prediction['probability_late'] is None
    assert 'уже зарегистрировано' in prediction['probability_note']
    assert engine.target_reached_at(101, 'target', T+timedelta(hours=1))  # посещение не истекает с TTL


def test_arrival_during_http_prevents_alert_after_known_target_fact(monkeypatch):
    async def exercise():
        async def response(request):
            rows = [baseline(PredictionRequest.model_validate(raw)).model_dump(mode='json')
                    for raw in json.loads(request.content)]
            clock['at'] = T+timedelta(seconds=1)
            engine.ingest_arrival(ArrivalInput(tr_id=101, planned_stop_id='target', arrived_at=clock['at']),
                                  received_at=clock['at'])
            return httpx.Response(200, json=rows)
        async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
            engine, clock = configured(monkeypatch, client=client)
            gps(engine, T)
            await engine.tick(0)
            assert engine.forecast_traces[101]['response']['risk'] == 'red'
            assert not engine.incidents
            assert engine.state()['vehicles'][0]['prediction']['risk'] == 'unknown'
    asyncio.run(exercise())


@pytest.mark.parametrize('status',['validated','transferred'])
def test_incident_preserves_eta_probability_and_plan_reference(monkeypatch,status):
    async def exercise():
        async def response(request):
            return httpx.Response(200, json=[baseline(PredictionRequest.model_validate(raw)).model_copy(update={
                'probability_late':.8,'probability_status':status,'probability_note':'Тестовая вероятность',
            }).model_dump(mode='json') for raw in json.loads(request.content)])
        async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
            engine, clock = configured(monkeypatch, client=client)
            gps(engine, T)
            await engine.tick(0)
            from common.state import Incident
            first = Incident.model_validate(engine.incidents[0])
            assert first.estimated_arrival_at == first.target_time+timedelta(seconds=180)
            assert first.target_name == 'Цель'
            assert first.horizon_reference == 'scheduled_arrival'
            assert first.probability_late == .8 and first.lead_time_s == 750
            assert first.probability_status == status
            engine.predictions[101] = engine.predictions[101].model_copy(update={
                'probability_late':.1, 'probability_status':'unavailable'})
            assert engine.incidents[0]['probability_late'] == .8
            assert engine.incidents[0]['probability_status'] == status
    asyncio.run(exercise())
