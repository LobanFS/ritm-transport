"""Строгая проекция входов, причинность потока и знаменатель метрик алертов."""
import asyncio
from datetime import datetime, timedelta, timezone
import json

import httpx
import pandas as pd
import pytest

from backend.engine import LiveContext, ScheduledStop
from common.contracts import PredictionRequest, Telemetry, baseline
from tools.evaluate_stream_real import Inputs, Snapshot, evaluate, load_inputs, replay, POINT_COLUMNS

T = datetime(2026, 1, 6, 9, tzinfo=timezone.utc)


def tiny_inputs():
    target = dict(id='target', name='Stop', scheduled_at=T+timedelta(seconds=750), lat=55.75, lon=37.6, manual_fill=False)
    context = LiveContext.model_validate(dict(
        vehicles=[dict(tr_id=1, unit_id=11, label='Test', route_id='r')],
        schedule=[dict(tr_id=1, target=target)], plan_complete=True, plan_version='test', plan_timezone='UTC'))
    current = Telemetry(tr_id=1, event_time=T-timedelta(seconds=5), received_at=T-timedelta(seconds=4),
                        lat=55.75, lon=37.6, speed_kmh=20, event_id='current')
    late = current.model_copy(update={'event_time':T-timedelta(seconds=2), 'received_at':T+timedelta(seconds=20), 'speed_kmh':99, 'event_id':'late'})
    future = current.model_copy(update={'event_time':T+timedelta(seconds=10), 'received_at':T+timedelta(seconds=10), 'speed_kmh':98, 'event_id':'future'})
    return Inputs(context, [Snapshot('sample', 1, T, 'target', T+timedelta(seconds=750), 180)],
                  [current, future, late], {})


def test_future_and_late_gps_never_enter_engine_snapshot_or_http_features():
    async def exercise():
        inputs = tiny_inputs()
        calls = []
        async def handler(request):
            data = json.loads(request.content)
            calls.extend(data)
            predictions = [baseline(PredictionRequest.model_validate(row)).model_copy(update={
                'method':'learned', 'model_version':'fixture-model'}).model_dump(mode='json') for row in data]
            return httpx.Response(200, json=predictions)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await replay(inputs, client, 'http://test')
        assert result['gps_delivered'] == 1
        assert len(calls) == 1
        request = calls[0]
        assert request['features']['valid_points_5m'] == 1
        assert request['features']['speed_mean_5m'] == 20
        assert request['features']['telemetry_age_s'] == 5
        assert request['current_delay_s'] == 180 and request['current_delay_source'] == 'csv_snapshot'
        assert result['predictions'][0]['sample_id'] == 'sample'
        assert result['traces'][0]['current_deviation']['received_at'] == T.isoformat().replace('+00:00','Z')
        assert 'target_delay_s' not in json.dumps(calls)
        assert result['outside_planned_horizon'] == 0
    asyncio.run(exercise())


@pytest.mark.parametrize('source', ['csv_snapshot', 'gps'])
def test_input_loader_projects_label_file_and_does_not_load_factual_schedule(tmp_path, monkeypatch, source):
    for name in ('labels', 'test', 'validate'):
        (tmp_path/name).mkdir()
    pd.DataFrame([dict(sample_id='sample', tr_id=1, T=T.isoformat(), target_stop_id='target',
        target_time_begin=(T+timedelta(seconds=750)).isoformat(), cur_dev_s=180 if source=='csv_snapshot' else 'FORBIDDEN HINT',
        target_delay_s='FORBIDDEN FUTURE', target_class='FORBIDDEN FUTURE')]).to_csv(tmp_path/'labels/labels_test.csv', index=False)
    pd.DataFrame([dict(tt_action_item_id='target', tr_id=1, time_begin=(T+timedelta(seconds=750)).isoformat(),
        order_date='2026-01-06', manual_fill=False, geom='POINT (37.6 55.75)', building_address='Stop',
        time_fact_begin='FORBIDDEN FUTURE')]).to_csv(tmp_path/'validate/schedule_plan.csv', index=False)
    pd.DataFrame([dict(packet_id='one', tr_id=1, unit_id=11, event_time=T.isoformat(), receive_time=T.isoformat(),
        lat=55.75, lon=37.6, speed=20, heading=0, location_valid=True)]).to_csv(tmp_path/'test/traffic.csv', index=False)
    original = pd.read_csv
    reads = []
    def checked(path, **kwargs):
        assert kwargs.get('usecols')
        assert 'time_fact_begin' not in kwargs['usecols']
        assert 'target_delay_s' not in kwargs['usecols']
        assert 'target_class' not in kwargs['usecols']
        if source == 'gps':
            assert 'cur_dev_s' not in kwargs['usecols']
        reads.append(str(path))
        return original(path, **kwargs)
    monkeypatch.setattr(pd, 'read_csv', checked)
    result = load_inputs(tmp_path, deviation_source=source)
    assert result.snapshots[0].cur_dev_s == (180 if source == 'csv_snapshot' else None)
    assert result.context.arrival_mode == ('gps' if source == 'gps' else 'external')
    assert not result.context.hints
    assert result.context.plan_complete
    assert set(result.metadata['input_projection']) == set(POINT_COLUMNS)-({'cur_dev_s'} if source=='gps' else set())
    assert all(not path.endswith('/test/schedule.csv') for path in reads)
    assert len(reads) == 3


def test_stream_delivers_explicit_input_target_to_disambiguate_equal_plan_times():
    async def exercise():
        inputs = tiny_inputs()
        original = inputs.context.schedule[0]
        inputs.context.schedule.append(ScheduledStop(tr_id=1, target=original.target.model_copy(update={
            'id':'a-lower-id', 'name':'Другая остановка в то же время', 'lat':55.751})))
        seen = []
        async def handler(request):
            parsed = [PredictionRequest.model_validate(row) for row in json.loads(request.content)]
            seen.extend(row.target.id for row in parsed)
            return httpx.Response(200, json=[baseline(row).model_dump(mode='json') for row in parsed])
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await replay(inputs, client, 'http://test')
        assert seen == ['target']  # задан во входе, а не выбран по порядку ID
        assert result['failures'] == []
        assert len(result['predictions']) == 1
    asyncio.run(exercise())


def test_future_snapshot_is_injected_only_on_its_own_T():
    async def exercise():
        inputs = tiny_inputs()
        future_at = T+timedelta(seconds=300)
        original = inputs.context.schedule[0]
        target = original.target.model_copy(update={'id':'future-target', 'scheduled_at':future_at+timedelta(seconds=750)})
        inputs.context.schedule.append(ScheduledStop(tr_id=1, target=target))
        inputs.snapshots.append(Snapshot('future-sample',1,future_at,target.id,target.scheduled_at,999))
        calls = []
        async def handler(request):
            parsed = [PredictionRequest.model_validate(row) for row in json.loads(request.content)]
            calls.extend((row.issued_at,row.current_delay_s) for row in parsed)
            return httpx.Response(200,json=[baseline(row).model_dump(mode='json') for row in parsed])
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await replay(inputs,client,'http://test')
        assert calls == [(T,180),(future_at,999)]
    asyncio.run(exercise())


def test_unlabeled_alert_is_not_false_positive_and_actual_lead_uses_future_label_only_in_evaluation():
    result = dict(predictions=[dict(sample_id='sample', prediction=180, input_cur_dev_s=200,
                 method='learned', execution='ml_http', telemetry_age_s=5)],
        all_forecasts=1, outside_planned_horizon=0,
        alerts=[dict(id='1:target', tr_id=1, replay_published_at=T.isoformat(),
                     latency_adjusted_published_at=(T+timedelta(seconds=1)).isoformat()),
                dict(id='1:unlabeled', tr_id=1, replay_published_at=T.isoformat(),
                     latency_adjusted_published_at=T.isoformat())])
    oracle=[dict(sample_id='sample',tr_id=1,target_stop_id='target',target_time_begin=T+timedelta(seconds=750),target_delay_s=150)]
    metrics=evaluate(result,oracle,{'sample'})
    assert metrics['labeled_forecasts']['mae_s'] == 30
    assert metrics['alerts']['total'] == 2
    assert metrics['alerts']['matched'] == 1 and metrics['alerts']['unassessable'] == 1
    assert metrics['alerts']['label_coverage'] == .5 and metrics['alerts']['precision'] == 1
    assert metrics['alerts']['actual_lead_s']['min'] == 900
    assert metrics['alerts']['matched_details'][0]['latency_adjusted_actual_lead_s'] == 899
    assert metrics['alerts']['after_actual'] == 0
    assert metrics['alerts']['recall_on_labeled_targets'] == 1


def test_alert_at_or_after_actual_arrival_is_explicitly_counted():
    result=dict(predictions=[],all_forecasts=0,outside_planned_horizon=0,
        alerts=[dict(id='1:target',tr_id=1,replay_published_at=T.isoformat(),latency_adjusted_published_at=T.isoformat())])
    oracle=[dict(sample_id='s',tr_id=1,target_stop_id='target',target_time_begin=T+timedelta(seconds=750),target_delay_s=-750)]
    metrics=evaluate(result,oracle,{'s'})
    assert metrics['alerts']['after_actual'] == 1
    assert metrics['alerts']['after_actual_latency_adjusted'] == 1
    assert metrics['alerts']['precision'] == 0
    assert metrics['labeled_forecasts']['coverage'] == 0
    assert metrics['labeled_forecasts']['mae_s'] is None


def test_gps_pipeline_derives_deviation_ignores_supplied_hint_and_expires_it():
    async def exercise():
        inputs = tiny_inputs()
        inputs.context.arrival_mode = 'gps'
        previous = inputs.context.schedule[0].target.model_copy(update={
            'id':'previous', 'lat':55.74, 'scheduled_at':T-timedelta(seconds=90)})
        inputs.context.schedule.append(ScheduledStop(tr_id=1, target=previous))
        first = inputs.telemetry[0].model_copy(update={'event_time':T-timedelta(seconds=60),
            'received_at':T-timedelta(seconds=60), 'lat':55.74, 'speed_kmh':0, 'event_id':'arrival-first'})
        second = first.model_copy(update={'event_time':T-timedelta(seconds=45),
            'received_at':T-timedelta(seconds=45), 'event_id':'arrival-second'})
        inputs.telemetry = [first, second, *inputs.telemetry[1:]]
        later = T+timedelta(seconds=300)
        target = inputs.context.schedule[0].target.model_copy(update={
            'id':'later', 'scheduled_at':later+timedelta(seconds=750)})
        inputs.context.schedule.append(ScheduledStop(tr_id=1, target=target))
        inputs.snapshots.append(Snapshot('later-sample', 1, later, target.id, target.scheduled_at, 999))
        calls = []
        async def handler(request):
            parsed = [PredictionRequest.model_validate(row) for row in json.loads(request.content)]
            calls.extend(parsed)
            return httpx.Response(200, json=[baseline(row).model_dump(mode='json') for row in parsed])
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await replay(inputs, client, 'http://test', deviation_source='gps')
        assert calls[0].current_delay_s == 30  # observed T-60 minus scheduled T-90
        assert calls[0].current_delay_source == 'gps_estimate'
        assert calls[0].features.valid_points_5m == 2  # late/future points unavailable at T
        assert calls[1].current_delay_s is None  # estimate older than TTL 300; 999 hint ignored
        assert result['traces'][0]['current_deviation']['planned_stop_id'] == 'previous'
        assert [row['status'] for row in result['outcomes']] == ['predicted', 'unknown']
        assert result['predictions'][1]['prediction'] is None
        assert result['failures'] == []
    asyncio.run(exercise())


def test_gps_target_tie_remains_visible_without_fake_hint():
    async def exercise():
        inputs = tiny_inputs()
        inputs.context.arrival_mode = 'gps'
        original = inputs.context.schedule[0]
        inputs.context.schedule.append(ScheduledStop(tr_id=1, target=original.target.model_copy(update={'id':'a-lower-id'})))
        async def handler(request):
            parsed = [PredictionRequest.model_validate(row) for row in json.loads(request.content)]
            assert all(row.current_delay_s is None for row in parsed)
            return httpx.Response(200, json=[baseline(row).model_dump(mode='json') for row in parsed])
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await replay(inputs, client, 'http://test', deviation_source='gps')
        assert result['predictions'] == []
        assert result['outcomes'][0]['status'] == 'target_mismatch'
        assert result['failures'][0]['actual_target'] == 'a-lower-id'
    asyncio.run(exercise())


def test_paired_comparison_uses_only_common_predicted_samples(tmp_path):
    from tools.evaluate_stream_real import compare_reference
    base = dict(sample_id='known', tr_id=1, T=T.isoformat(), target_stop_id='target',
                target_time_begin=(T+timedelta(seconds=750)).isoformat(), model_version='frozen',
                input_source='csv_snapshot', method='learned', execution='ml_http',
                prediction=140, input_cur_dev_s=120)
    reference = [base, {**base, 'sample_id':'unknown', 'prediction':1000}]
    path = tmp_path/'predictions.jsonl'
    path.write_text('\n'.join(json.dumps(row) for row in reference))
    result = dict(predictions=[{**base, 'prediction':180, 'input_cur_dev_s':190, 'input_source':'gps_estimate'},
                               {**base, 'sample_id':'unknown', 'prediction':None, 'input_cur_dev_s':None}])
    oracle = [dict(sample_id='known', target_delay_s=150), dict(sample_id='unknown', target_delay_s=0)]
    compared = compare_reference(result, oracle, path)
    assert compared['matched_samples'] == 1
    assert compared['gps_model_mae_s'] == 30
    assert compared['supplied_model_mae_s'] == 10
    assert compared['gps_persistence_mae_s'] == 40
    reference[0]['model_version'] = 'different'
    path.write_text('\n'.join(json.dumps(row) for row in reference))
    with pytest.raises(ValueError, match='model_version'):
        compare_reference(result, oracle, path)
    reference[0]['model_version'] = 'frozen'
    reference[0]['input_source'] = 'gps_estimate'
    path.write_text('\n'.join(json.dumps(row) for row in reference))
    with pytest.raises(ValueError, match='CSV snapshot'):
        compare_reference(result, oracle, path)
