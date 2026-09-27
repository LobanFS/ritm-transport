"""Критичные границы доступности данных и деградации; без обучения."""
import asyncio
from datetime import datetime, timedelta, timezone
import httpx
import pytest
from backend.engine import Engine, DelayHint, features_at
from common.contracts import PredictionRequest, StopTarget, Telemetry

T=datetime(2026,1,6,8,tzinfo=timezone.utc)
TARGET=StopTarget(id='test',name='Цель',scheduled_at=T+timedelta(minutes=12),lat=55.76,lon=37.61)

def point(**kwargs):
    return Telemetry(tr_id=101,event_time=T,received_at=T,lat=55.75,lon=37.60,speed_kmh=20,**kwargs)


def test_future_event_and_late_receive_are_not_features():
    valid=point()
    future=valid.model_copy(update={'event_time':T+timedelta(seconds=1),'speed_kmh':100})
    late=valid.model_copy(update={'event_time':T-timedelta(seconds=10),'received_at':T+timedelta(seconds=1),'speed_kmh':90})
    feats,latest=features_at([late,future,valid],T,TARGET)
    assert feats.valid_points_5m==1 and feats.speed_mean_5m==20
    assert latest is valid and feats.telemetry_age_s==0


def test_out_of_order_invalid_and_impossible_speed_not_current_position():
    valid=point()
    older=valid.model_copy(update={'event_time':T-timedelta(seconds=30),'lat':0})
    invalid=valid.model_copy(update={'location_valid':False,'speed_kmh':0})
    outlier=valid.model_copy(update={'speed_kmh':368})
    feats,latest=features_at([valid,older,invalid,outlier],T,TARGET)
    assert latest is valid and feats.valid_points_5m==2


def test_context_reset_and_future_hint():
    e=Engine('http://no-model')
    e.arrivals.clear()
    e.hints[101].clear()
    e.hints[101].append(DelayHint(tr_id=101,observed_at=e.clock+timedelta(seconds=1),delay_s=999))
    assert e.request_for(101,e.clock).current_delay_s is None
    e.set_live()
    assert not e.vehicles and not e.predictions and not e.incidents and not e.history


def test_stale_hint_is_unknown():
    e=Engine('http://no-model')
    e.arrivals.clear()
    e.hints[101].clear()
    e.hints[101].append(DelayHint(tr_id=101,observed_at=e.clock-timedelta(seconds=301),delay_s=123))
    assert e.request_for(101,e.clock).current_delay_s is None


def test_request_contains_only_causally_available_delay_history():
    e = Engine('http://no-model')
    e.arrivals.clear(); e.hints[101].clear()
    at = e.clock
    e.hints[101].extend([
        DelayHint(tr_id=101, observed_at=at-timedelta(minutes=91), delay_s=-1),
        DelayHint(tr_id=101, observed_at=at-timedelta(minutes=10), delay_s=10),
        DelayHint(tr_id=101, observed_at=at-timedelta(minutes=5), delay_s=20),
        DelayHint(tr_id=101, observed_at=at-timedelta(minutes=2), received_at=at+timedelta(seconds=1), delay_s=999),
        DelayHint(tr_id=101, observed_at=at, delay_s=30),
    ])
    request = e.request_for(101, at)
    assert request.current_delay_s == 30
    assert [(item.observed_at, item.delay_s) for item in request.delay_history] == [
        (at-timedelta(minutes=10), 10), (at-timedelta(minutes=5), 20),
    ]


def test_baseline_fallback_then_ml_recovers_and_incident_deduplicates():
    async def exercise():
        fail=True
        async def handler(request):
            import json
            from common.contracts import baseline
            if fail: return httpx.Response(503)
            values=[baseline(PredictionRequest.model_validate(p)).model_dump(mode='json') for p in json.loads(request.content)]
            return httpx.Response(200,json=values)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            e=Engine('http://ml',client)
            await e.tick(0)
            assert e.ml_status=='unavailable' and e.predictions[101].method=='fallback'
            first=len(e.incidents)
            fail=False
            await e.tick(0)
            assert e.ml_status=='ok' and e.predictions[101].method=='persistence'
            assert len(e.incidents)==first
            assert all(600<i['lead_time_s']<=900 for i in e.incidents)
            e.source_enabled=False
            for _ in range(8): await e.tick(1)
            assert e.state()['summary']['stale']==4
            assert all(v['prediction']['risk']=='unknown' for v in e.state()['vehicles'])
            e.source_enabled=True
            await e.tick(1)
            assert e.state()['summary']['stale']==0
    asyncio.run(exercise())


def test_mismatched_ml_response_falls_back():
    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _:httpx.Response(200,json=[]))) as client:
            e=Engine('http://ml',client)
            await e.tick(0)
            assert e.ml_status=='unavailable' and len(e.predictions)==4
    asyncio.run(exercise())


def test_ingest_deduplicates_and_bounds_history():
    e=Engine('http://ml')
    p=point(event_id='repeated')
    assert e.ingest(p) and not e.ingest(p)
    for i in range(2100): e.ingest(p.model_copy(update={'event_id':str(i)}))
    assert len(e.history[101])==2048 and len(e.seen[101])==2048


def test_exhausted_horizon_clears_prediction():
    async def exercise():
        e=Engine('http://ml')
        await e.tick(0)
        e.schedule.clear()
        await e.tick(0)
        assert not e.predictions
        assert all(v['target'] is None for v in e.state()['vehicles'])
    asyncio.run(exercise())


def test_inflight_prediction_cannot_repopulate_state_after_mode_change():
    async def exercise():
        import json
        from common.contracts import baseline
        async def handler(request):
            values=[baseline(PredictionRequest.model_validate(p)).model_dump(mode='json') for p in json.loads(request.content)]
            e.set_live()
            return httpx.Response(200,json=values)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            e=Engine('http://ml',client)
            await e.tick(0)
            assert e.mode=='live' and not e.predictions and not e.incidents
    asyncio.run(exercise())


def test_empty_live_still_checks_ml_health():
    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _:httpx.Response(503))) as client:
            e=Engine('http://ml',client)
            e.ml_status='ok'
            e.set_live()
            await e.tick(0)
            assert e.ml_status=='unavailable' and not e.predictions
    asyncio.run(exercise())


def test_prediction_watermark_does_not_acknowledge_packets_arriving_during_inference():
    """Иначе нагрузочная проверка ложно покажет обработку ещё не виденных данных."""
    async def exercise():
        import json
        from common.contracts import baseline
        async def handler(request):
            values = [baseline(PredictionRequest.model_validate(p)).model_dump(mode='json') for p in json.loads(request.content)]
            e.ingest(point(event_id='arrived-while-ml-running'))
            return httpx.Response(200, json=values)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            e = Engine('http://ml', client)
            e.running = False
            before = e.input_sequences[101]
            await e.tick(0)
            vehicle = next(v for v in e.state()['vehicles'] if v['tr_id'] == 101)
            assert vehicle['telemetry_sequence'] == before+1
            assert vehicle['prediction_input_sequence'] == before
            assert vehicle['prediction_published_at'] is not None
            await e.tick(0)
            assert e.prediction_sequences[101] == before+1
            assert e.input_sequences[101] == before+1  # повторный пакет — дубль
            assert e.state()['metrics']['pipeline_cycles'] == 2
            assert e.state()['metrics']['pipeline_p95_ms'] >= e.state()['metrics']['inference_p95_ms']
            e.set_live()
            assert not e.prediction_sequences and not e.input_sequences and not e.prediction_published_at
    asyncio.run(exercise())


def test_watermarks_do_not_survive_inflight_context_change():
    async def exercise():
        import json
        from common.contracts import baseline
        async def handler(request):
            values = [baseline(PredictionRequest.model_validate(p)).model_dump(mode='json') for p in json.loads(request.content)]
            e.reset_demo()
            return httpx.Response(200, json=values)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            e = Engine('http://ml', client)
            await e.tick(0)
            assert not e.prediction_sequences and not e.prediction_published_at
            assert all(v['prediction_input_sequence'] is None for v in e.state()['vehicles'])
    asyncio.run(exercise())


@pytest.mark.parametrize('inference_seconds, expected_alerts', [(.5, 1), (2, 0)])
def test_live_alert_horizon_is_checked_at_publication(monkeypatch, inference_seconds, expected_alerts):
    """Окно валидно при запросе, но может закрыться до ответа сервиса."""
    from backend.engine import LiveContext
    import backend.engine as engine_module
    clock = T
    monkeypatch.setattr(engine_module, 'utcnow', lambda: clock)
    async def exercise():
        import json
        from common.contracts import baseline
        async def handler(request):
            nonlocal clock
            values = [baseline(PredictionRequest.model_validate(p)).model_dump(mode='json') for p in json.loads(request.content)]
            clock = T+timedelta(seconds=inference_seconds)
            return httpx.Response(200, json=values)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            e = Engine('http://ml', client)
            e.set_live(LiveContext(vehicles=[dict(tr_id=101, unit_id=101, label='Test', route_id='test')],
                schedule=[dict(tr_id=101, target=TARGET.model_copy(update={'scheduled_at': T+timedelta(seconds=601)}))],
                hints=[dict(tr_id=101, observed_at=T, delay_s=180)]))
            e.ingest(point())
            await e.tick()
            assert len(e.incidents) == expected_alerts
            if expected_alerts:
                entry = e.incidents[0]
                assert entry['created_at'] == entry['published_at'] == clock
                assert entry['data_cutoff'] == T
                assert entry['lead_time_s'] == 601-inference_seconds
            else:
                assert e.state()['vehicles'][0]['prediction'] is None
    asyncio.run(exercise())
