"""Причинный test replay через настоящий Engine и HTTP ML; без обучения.

Инференс на уникальных T выданных точек, GPS доставляются по max(event, receive).
В режиме csv_snapshot выданный cur_dev_s принимается как внешний snapshot на T.
В режиме gps эта колонка не читается: Engine сам распознаёт посещения и считает
отклонение. Oracle читается отдельно после прогнозов. Данные и артефакты
остаются локально вне Git. Это ускоренное event-time воспроизведение, без NDTP
перекодирования и браузера; runtime детектора и модели не меняется.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

POINT_COLUMNS = ('sample_id', 'tr_id', 'T', 'target_stop_id', 'target_time_begin', 'cur_dev_s')
PLAN_COLUMNS = ('tt_action_item_id', 'time_begin', 'order_date', 'manual_fill', 'tr_id', 'geom', 'building_address')
GPS_COLUMNS = ('packet_id', 'tr_id', 'unit_id', 'event_time', 'receive_time', 'lat', 'lon', 'speed', 'heading', 'location_valid')


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str)+'\n')


def fingerprint(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def sources():
    paths = [path for folder in ('backend', 'common') for path in sorted((ROOT/folder).glob('*.py'))]
    paths.append(Path(__file__).resolve())
    return {str(path.relative_to(ROOT)): fingerprint(path) for path in paths}


def distribution(values):
    import numpy as np
    return ({'count': len(values), 'min': float(min(values)), 'p05': float(np.quantile(values, .05)),
             'median': float(np.median(values)), 'p95': float(np.quantile(values, .95)), 'max': float(max(values))}
            if values else {'count': 0, 'min': None, 'p05': None, 'median': None, 'p95': None, 'max': None})


def parse_time(value, zone='UTC'):
    parsed = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    return parsed.replace(tzinfo=ZoneInfo(zone)) if parsed.tzinfo is None else parsed


@dataclass(frozen=True)
class Snapshot:
    sample_id: str
    tr_id: int
    at: datetime
    target_stop_id: str
    target_time_begin: datetime
    cur_dev_s: float | None


@dataclass
class Inputs:
    context: object
    snapshots: list[Snapshot]
    telemetry: list
    metadata: dict


def load_inputs(dataset: Path, zone='UTC', max_points=None, *, deviation_source='csv_snapshot') -> Inputs:
    """Проекция входов; не загружает target_delay_s, target_class или факты плана."""
    import pandas as pd
    from backend.engine import LiveContext, ScheduledStop, VehicleConfig
    from common.contracts import StopTarget, Telemetry, parse_manual_fill

    if deviation_source not in ('csv_snapshot', 'gps'):
        raise ValueError('Неизвестный источник отклонения')
    point_columns = [name for name in POINT_COLUMNS if name != 'cur_dev_s' or deviation_source == 'csv_snapshot']
    points = pd.read_csv(dataset/'labels/labels_test.csv', usecols=point_columns)
    points = points.sort_values(['T', 'tr_id', 'sample_id'], kind='stable')
    if max_points is not None:
        points = points.head(max_points)
    if points.empty or points.sample_id.duplicated().any():
        raise ValueError('Нужны непустые уникальные sample_id')
    snapshots = [Snapshot(str(row.sample_id), int(row.tr_id), parse_time(row.T, zone),
                         str(row.target_stop_id), parse_time(row.target_time_begin, zone),
                         float(row.cur_dev_s) if deviation_source == 'csv_snapshot' else None)
                 for row in points.itertuples(index=False)]
    if any((row.cur_dev_s is not None and not math.isfinite(row.cur_dev_s))
           or not 600 < (row.target_time_begin-row.at).total_seconds() <= 900 for row in snapshots):
        raise ValueError('Некорректный внешний snapshot или горизонт')
    ids = {row.tr_id for row in snapshots}
    plan_path = dataset/'validate/schedule_plan.csv'
    plan = pd.read_csv(plan_path, usecols=list(PLAN_COLUMNS))
    plan = plan[plan.tr_id.isin(ids)]
    scheduled = []
    for row in plan.itertuples(index=False):
        coords = re.fullmatch(r'POINT\s*\(\s*([-+\d.eE]+)\s+([-+\d.eE]+)\s*\)', str(row.geom))
        if coords is None:
            raise ValueError('Некорректный WKT плана')
        scheduled.append(ScheduledStop(tr_id=int(row.tr_id), target=StopTarget(
            id=str(row.tt_action_item_id), name=str(row.building_address)[:120], scheduled_at=parse_time(row.time_begin, zone),
            lon=float(coords[1]), lat=float(coords[2]), manual_fill=parse_manual_fill(str(row.manual_fill)))))
    gps = pd.read_csv(dataset/'test/traffic.csv', usecols=list(GPS_COLUMNS))
    gps = gps[gps.tr_id.isin(ids)]
    units = {}
    for tr_id, rows in gps.groupby('tr_id'):
        unique = rows.unit_id.unique()
        if len(unique) != 1:
            raise ValueError('Смена unit_id требует операционного маппинга')
        units[int(tr_id)] = int(unique[0])
    if set(units) != ids:
        raise ValueError('Нет GPS/маппинга для части выбранных ТС')
    telemetry = []
    quality = Counter()
    last = max(row.at for row in snapshots)
    for row in gps.itertuples(index=False):
        event, received = parse_time(row.event_time, zone), parse_time(row.receive_time, zone)
        if max(event, received) > last:
            quality['after_last_cutoff'] += 1
            continue
        quality['receive_before_event'] += int(received < event)
        def nullable(value):
            return None if pd.isna(value) else float(value)
        try:
            point = Telemetry(tr_id=int(row.tr_id), unit_id=int(row.unit_id),
                event_time=event, received_at=received, lat=nullable(row.lat), lon=nullable(row.lon),
                speed_kmh=nullable(row.speed), heading=nullable(row.heading),
                location_valid=str(row.location_valid).lower() == 'true', source='replay', event_id=str(row.packet_id))
        except ValueError:
            quality['invalid_rows'] += 1
            continue
        telemetry.append(point)
    telemetry.sort(key=lambda point: (max(point.event_time, point.received_at), point.event_time, point.event_id or ''))
    context = LiveContext(vehicles=[VehicleConfig(tr_id=tr, unit_id=units[tr], label=f'ТС {tr}', route_id=f'test-{tr}') for tr in sorted(ids)],
                          schedule=scheduled, arrival_mode='gps' if deviation_source == 'gps' else 'external', plan_complete=True,
                          plan_version=fingerprint(plan_path), plan_timezone=zone)
    return Inputs(context, snapshots, telemetry, dict(vehicles=len(ids), points=len(snapshots),
        unique_ticks=len({row.at for row in snapshots}), first_T=min(row.at for row in snapshots).isoformat(),
        last_T=last.isoformat(), plan_visits=len(scheduled), loaded_gps=len(telemetry), quality=dict(quality),
        input_projection=point_columns, plan_projection=list(PLAN_COLUMNS), deviation_source=deviation_source))


async def replay(inputs: Inputs, client, ml_url, *, budget_s=120, out: Path | None=None, deviation_source='csv_snapshot'):
    """Реальный Engine orchestration; виртуальные часы стоят во время HTTP."""
    from backend.engine import DelayHint, Engine
    started = time.perf_counter()
    engine = Engine(ml_url, client)
    if deviation_source == 'gps':
        if inputs.context.arrival_mode != 'gps' or inputs.context.hints:
            raise ValueError('GPS-проверка требует arrival_mode=gps и пустые hints')
    engine.set_live(inputs.context)
    initial_metrics = dict(engine.metrics)  # Engine constructor also creates demo data.
    # В Engine replay mode часы задаются датасетом. running=False не позволяет
    # _tick сдвигать их по реальному времени; вся доставка контролируется ниже.
    engine.mode = 'replay'
    engine.running = False
    by_time = defaultdict(list)
    for snapshot in inputs.snapshots:
        by_time[snapshot.at].append(snapshot)
    cursor = 0
    predictions, alerts, failures, ticks, traces, outcomes = [], [], [], [], [], []
    seen_alerts = set()
    all_forecasts = 0
    outside_horizon = 0
    for at in sorted(by_time):
        if time.perf_counter()-started > budget_s:
            raise TimeoutError('Исчерпан заранее объявленный бюджет потока')
        engine.clock = at
        delivered_now = 0
        while cursor < len(inputs.telemetry) and max(inputs.telemetry[cursor].event_time, inputs.telemetry[cursor].received_at) <= at:
            engine.clock = max(inputs.telemetry[cursor].event_time, inputs.telemetry[cursor].received_at)
            engine.ingest(inputs.telemetry[cursor])
            cursor += 1
            delivered_now += 1
        engine.clock = at
        if deviation_source == 'csv_snapshot':
            for snapshot in by_time[at]:
                engine.hints[snapshot.tr_id].append(DelayHint(tr_id=snapshot.tr_id, observed_at=at,
                    received_at=at, delay_s=snapshot.cur_dev_s, source='csv_snapshot', sample_id=snapshot.sample_id,
                    target_stop_id=snapshot.target_stop_id, target_time_begin=snapshot.target_time_begin))
        else:
            assert all(not hints for hints in engine.hints.values())
        # Ни очередь будущих GPS, ни будущие snapshots не находятся в Engine.
        assert all(max(point.event_time, point.received_at) <= at for rows in engine.history.values() for point in rows)
        before = time.perf_counter()
        await engine.tick(0)
        elapsed = time.perf_counter()-before
        ticks.append(dict(T=at.isoformat(), gps_delivered=delivered_now, wall_s=elapsed,
                          forecasts=len(engine.forecast_traces), ml_status=engine.ml_status))
        for tr_id, trace in engine.forecast_traces.items():
            request, response = trace['request'], trace['response']
            if deviation_source == 'gps':
                assert request['current_delay_source'] in (None, 'gps_estimate')
                current = trace['current_deviation']
                if current:
                    assert parse_time(current['received_at']) <= at
                    assert 0 <= (at-parse_time(current['observed_at'])).total_seconds() <= 300
            lead = (parse_time(response['target']['scheduled_at'])-at).total_seconds()
            outside_horizon += int(not 600 < lead <= 900)
            all_forecasts += 1
        for snapshot in by_time[at]:
            current = engine.deviation_at(snapshot.tr_id, at)
            detector = engine.gps_detectors.get(snapshot.tr_id)
            outcome = dict(sample_id=snapshot.sample_id, tr_id=snapshot.tr_id, T=at.isoformat(),
                current_deviation=current.model_dump(mode='json') if current else None,
                detector=detector.status() if detector else None)
            outcomes.append(outcome)
            trace = engine.forecast_traces.get(snapshot.tr_id)
            if trace is None:
                outcome['status'] = 'no_engine_forecast'
                failures.append(dict(sample_id=snapshot.sample_id, reason='no_engine_forecast'))
                continue
            request, response = trace['request'], trace['response']
            correct_target = request['target']['id'] == snapshot.target_stop_id and parse_time(request['target']['scheduled_at']) == snapshot.target_time_begin
            if not correct_target:
                outcome.update(status='target_mismatch', actual_target=request['target']['id'],
                               expected_target=snapshot.target_stop_id)
                failures.append(dict(sample_id=snapshot.sample_id, reason='target_mismatch', actual_target=request['target']['id']))
                continue
            if deviation_source == 'csv_snapshot':
                assert request['current_delay_s'] == snapshot.cur_dev_s
                assert request['current_delay_source'] == 'csv_snapshot'
                assert trace['current_deviation']['sample_id'] == snapshot.sample_id
            assert parse_time(request['issued_at']) == at
            outcome.update(status='predicted' if response['predicted_delay_s'] is not None else 'unknown',
                           method=response['method'], reasons=response['reasons'])
            predictions.append(dict(sample_id=snapshot.sample_id, tr_id=snapshot.tr_id, T=at.isoformat(),
                target_stop_id=snapshot.target_stop_id, target_time_begin=snapshot.target_time_begin.isoformat(),
                input_cur_dev_s=request['current_delay_s'], input_source=request['current_delay_source'],
                prediction=response['predicted_delay_s'],
                method=response['method'], risk=response['risk'], probability_late=response['probability_late'],
                execution=trace['execution'], model_version=response['model_version'],
                telemetry_age_s=request['features']['telemetry_age_s'], http_cycle_s=elapsed))
            compact = dict(request)
            full_plan = compact.pop('plan_context')
            compact['plan_context'] = {key: value for key, value in full_plan.items() if key != 'stops'}
            compact['plan_context']['stop_count'] = len(full_plan['stops'])
            traces.append(dict(sample_id=snapshot.sample_id, data_cutoff=at.isoformat(),
                request=compact, request_sha256=hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest(),
                response=response, current_deviation=trace['current_deviation'],
                telemetry_sequence=trace['telemetry_sequence'], execution=trace['execution']))
        for incident in engine.incidents:
            key = (incident['id'], str(incident['created_at']))
            if key not in seen_alerts:
                seen_alerts.add(key)
                # Event clock при paused replay не двигается во время сети.
                # Дополнительно считаем консервативное время + wall HTTP cycle.
                alerts.append({**incident, 'replay_published_at': at.isoformat(),
                               'latency_adjusted_published_at': (at+timedelta(seconds=elapsed)).isoformat()})
    result = dict(predictions=predictions, alerts=alerts, failures=failures, ticks=ticks, traces=traces,
                  outcomes=outcomes, engine_metrics={key: value-initial_metrics.get(key, 0)
                                                    for key, value in engine.metrics.items()},
                  detector_statuses={str(tr): detector.status() for tr, detector in engine.gps_detectors.items()},
                  gps_delivered=cursor, all_forecasts=all_forecasts, outside_planned_horizon=outside_horizon,
                  elapsed_s=time.perf_counter()-started)
    if out is not None:
        for name in ('predictions', 'alerts', 'traces', 'outcomes'):
            with (out/f'{name}.jsonl').open('w') as stream:
                for row in result[name]:
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False, default=str)+'\n')
    return result


def load_oracle(dataset: Path, zone='UTC'):
    """Вызывается только после полного replay; ни один oracle не передаётся Engine."""
    import pandas as pd
    frame = pd.read_csv(dataset/'labels/labels_test.csv', usecols=['sample_id','tr_id','target_stop_id','target_time_begin','target_delay_s'])
    return [dict(sample_id=str(row.sample_id), tr_id=int(row.tr_id), target_stop_id=str(row.target_stop_id),
                 target_time_begin=parse_time(row.target_time_begin, zone), target_delay_s=float(row.target_delay_s))
            for row in frame.itertuples(index=False)]


def evaluate(result, oracle, expected_samples):
    """Неизвестная метка никогда не заменяется отрицательным событием."""
    by_sample = {row['sample_id']: row for row in oracle}
    by_target = defaultdict(list)
    for row in oracle:
        by_target[(row['tr_id'], row['target_stop_id'])].append(row)
    eligible = [row for row in result['predictions'] if row['sample_id'] in by_sample and row['prediction'] is not None and math.isfinite(row['prediction'])]
    absolute = [abs(row['prediction']-by_sample[row['sample_id']]['target_delay_s']) for row in eligible]
    baseline = [abs(row['input_cur_dev_s']-by_sample[row['sample_id']]['target_delay_s']) for row in eligible
                if row['input_cur_dev_s'] is not None]
    matched_alerts, unknown = [], []
    for alert in result['alerts']:
        # Engine incident ID uses tr_id:target.id; target ids can contain colons.
        stop_id = alert['id'].split(':', 1)[1]
        candidates = by_target.get((alert['tr_id'], stop_id), [])
        unique = {(row['target_time_begin'], row['target_delay_s']) for row in candidates}
        if len(unique) != 1:
            unknown.append(dict(id=alert['id'], reason='no_label' if not unique else 'ambiguous_label'))
            continue
        target, delay = next(iter(unique))
        actual = target+timedelta(seconds=delay)
        published = parse_time(alert['replay_published_at'])
        adjusted = parse_time(alert['latency_adjusted_published_at'])
        matched_alerts.append(dict(id=alert['id'], tr_id=alert['tr_id'], target_stop_id=stop_id,
            target_delay_s=delay, actual_arrival_at=actual.isoformat(), planned_lead_s=(target-published).total_seconds(),
            actual_lead_s=(actual-published).total_seconds(), latency_adjusted_actual_lead_s=(actual-adjusted).total_seconds(),
            true_late=delay>120, after_actual=published>=actual, after_actual_adjusted=adjusted>=actual))
    true_alerts = [row for row in matched_alerts if row['true_late']]
    labeled_delayed_targets = {(row['tr_id'],row['target_stop_id']) for row in oracle
                               if row['sample_id'] in expected_samples and row['target_delay_s']>120}
    hit_targets = {(row['tr_id'],row['target_stop_id']) for row in true_alerts if not row['after_actual']}
    return dict(
        labeled_forecasts=dict(expected=len(expected_samples), predicted=len(eligible),
            coverage=len(eligible)/len(expected_samples) if expected_samples else None,
            learned=sum(row['method']=='learned' and row['execution']=='ml_http' for row in eligible),
            unavailable=sum(row['prediction'] is None for row in result['predictions']),
            unmatched_target=sum(row['reason']=='target_mismatch' for row in result.get('failures', [])),
            mae_s=sum(absolute)/len(absolute) if absolute else None,
            baseline_mae_s=sum(baseline)/len(baseline) if baseline else None,
            fresh_gps=sum(row['telemetry_age_s'] is not None and row['telemetry_age_s']<=60 for row in eligible)),
        planned_horizon=dict(forecasts=result['all_forecasts'], violations=result['outside_planned_horizon']),
        alerts=dict(total=len(result['alerts']), matched=len(matched_alerts), unassessable=len(unknown),
            label_coverage=len(matched_alerts)/len(result['alerts']) if result['alerts'] else None,
            precision=sum(row['true_late'] for row in matched_alerts)/len(matched_alerts) if matched_alerts else None,
            delayed_labeled_targets=len(labeled_delayed_targets), alerted_before_actual_targets=len(hit_targets & labeled_delayed_targets),
            recall_on_labeled_targets=len(hit_targets & labeled_delayed_targets)/len(labeled_delayed_targets) if labeled_delayed_targets else None,
            after_actual=sum(row['after_actual'] for row in matched_alerts),
            after_actual_latency_adjusted=sum(row['after_actual_adjusted'] for row in matched_alerts),
            actual_lead_s=distribution([row['actual_lead_s'] for row in matched_alerts]),
            true_late_actual_lead_s=distribution([row['actual_lead_s'] for row in true_alerts]),
            matched_details=matched_alerts, unassessable_details=unknown))


def compare_reference(result, oracle, path: Path):
    """Парное сравнение на тех же sample_id; вызывается только после прогнозов."""
    reference = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    by_id = {row['sample_id']: row for row in reference}
    if len(by_id) != len(reference):
        raise ValueError('Повтор sample_id в reference predictions')
    truth = {row['sample_id']: row['target_delay_s'] for row in oracle}
    rows = []
    for row in result['predictions']:
        other = by_id.get(row['sample_id'])
        if row['prediction'] is None or other is None or other['prediction'] is None:
            continue
        if row.get('input_source') != 'gps_estimate' or other.get('input_source') != 'csv_snapshot':
            raise ValueError('Парное сравнение требует GPS estimate против CSV snapshot')
        for item in (row, other):
            if item.get('method') != 'learned' or item.get('execution') != 'ml_http':
                raise ValueError('Парное сравнение требует настоящий learned HTTP в обоих прогонах')
            if any(not isinstance(item[key], (int, float)) or not math.isfinite(item[key])
                   for key in ('prediction', 'input_cur_dev_s')):
                raise ValueError('Нечисленный вход или прогноз в парном сравнении')
        for key in ('tr_id', 'T', 'target_stop_id', 'target_time_begin', 'model_version'):
            if row[key] != other[key]:
                raise ValueError(f'Reference не сопоставим по {key}: {row["sample_id"]}')
        label = truth[row['sample_id']]
        rows.append(dict(sample_id=row['sample_id'], target_delay_s=label,
            gps_input_s=row['input_cur_dev_s'], supplied_input_s=other['input_cur_dev_s'],
            gps_prediction_s=row['prediction'], supplied_prediction_s=other['prediction']))
    def mae(key):
        return sum(abs(row[key]-row['target_delay_s']) for row in rows)/len(rows) if rows else None
    return dict(reference_path=str(path.resolve()), reference_sha256=fingerprint(path),
        matched_samples=len(rows), sample_ids=[row['sample_id'] for row in rows],
        gps_model_mae_s=mae('gps_prediction_s'), supplied_model_mae_s=mae('supplied_prediction_s'),
        gps_persistence_mae_s=mae('gps_input_s'), supplied_persistence_mae_s=mae('supplied_input_s'),
        note='Одинаковые ID, T, цели и версия модели; MAE только на пересечении численных прогнозов.',
        rows=rows)


async def execute(args):
    import httpx
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    gps_mode = args.deviation_source == 'gps'
    contract = dict(started_at=datetime.now(timezone.utc).isoformat(), budget_s=args.budget_s,
        change='Источник текущего отклонения: GPS-детектор вместо CSV' if gps_mode else 'Контроль с выданным CSV snapshot',
        baseline='Зафиксированная модель и детектор; persistence на тех же известных входах; reference только на общих ID',
        deviation_source=args.deviation_source, training_or_tuning=False,
        population='Первые max_points по (T,tr_id,sample_id), иначе все 353 test points / 13 ТС',
        max_points=args.max_points, cadence='GPS по max(event_time,received_at); inference на уникальных T входных points',
        primary='Покрытие планового окна (600,900] секунд, MAE, HTTP learned coverage',
        secondary='Alert precision для target_delay_s>120 с label coverage; actual-arrival lead и alerts после факта',
        acceptance=('0 нарушений причинности/планового окна, HTTP без отказов. Coverage/MAE/неоднозначные цели — результат диагностики, без порога качества.' if gps_mode else
            '0 нарушений причинности/планового окна; 353/353 learned HTTP при полном population. Качество/late alerts сообщаются как измерения.'),
        oracle_policy='Только после завершения всех прогнозов; target_actual=target_time_begin+target_delay_s; фактические schedule не читаются',
        provided_hint_policy='cur_dev_s не читается; hints пусты; готовые arrivals не подаются' if gps_mode else
            'cur_dev_s разрешён раздачей на T, даже если его происхождение отличается от доступного GPS; reconstruction не утверждается',
        target_policy='Engine выбирает цель только из плана; выданный ID используется для сравнения, несовпадения учитываются отдельно' if gps_mode else
            'Переданный target_stop_id разрешает неоднозначность одинакового минимального time_begin в плановом окне; target не берётся из oracle',
        publication_clock='Engine replay event-time T; дополнительно actual lead с консервативной поправкой на реальную длительность HTTP цикла',
        timezone=args.timezone, url=args.url,
        input_allowlist=[name for name in POINT_COLUMNS if name != 'cur_dev_s' or not gps_mode], code_sha256=sources())
    write_json(out/'contract.json', contract)
    started = time.perf_counter()
    inputs = load_inputs(args.dataset, args.timezone, args.max_points, deviation_source=args.deviation_source)
    if args.max_points is None and (len(inputs.snapshots)!=353 or inputs.metadata['vehicles']!=13):
        raise ValueError('Полная зафиксированная population должна быть 353 точки / 13 ТС')
    write_json(out/'plan-input.json', inputs.context.model_dump(mode='json'))
    async with httpx.AsyncClient(timeout=15, trust_env=False) as client:
        card = (await client.get(args.url.rstrip('/')+'/model')).raise_for_status().json()
        from ml_service.learned import MODEL_SHA256
        if not card.get('trained') or card.get('provenance', {}).get('sha256') != MODEL_SHA256:
            raise RuntimeError('Для проверки нужен запущенный learned ML сервис')
        remaining = args.budget_s-(time.perf_counter()-started)
        if remaining<=0:
            raise TimeoutError('Бюджет закончился при подготовке входов')
        async with asyncio.timeout(remaining):
            result = await replay(inputs, client, args.url.rstrip('/'), budget_s=remaining, out=out,
                                  deviation_source=args.deviation_source)
    # Сначала сохранены все прогнозы/trace. Только теперь открываем oracle.
    oracle = load_oracle(args.dataset, args.timezone)
    metrics = evaluate(result, oracle, {row.sample_id for row in inputs.snapshots})
    unchanged = contract['code_sha256'] == sources()
    pipeline_passed = (metrics['planned_horizon']['violations']==0 and unchanged
                      and all(row['ml_status']=='ok' for row in result['ticks']))
    passed = pipeline_passed and (gps_mode or (not result['failures']
              and metrics['labeled_forecasts']['learned']==len(inputs.snapshots)))
    report = dict(status=('DIAGNOSTIC' if gps_mode else 'PASS') if passed else 'FAIL',
        pipeline_checks_passed=pipeline_passed, deviation_source=args.deviation_source,
        completed_at=datetime.now(timezone.utc).isoformat(),
        runtime_s=time.perf_counter()-started, population=inputs.metadata, model_card=card, metrics=metrics,
        failures=result['failures'], http_cycle_s=distribution([row['wall_s'] for row in result['ticks']]),
        gps_delivered=result['gps_delivered'], ticks=result['ticks'], source_unchanged=unchanged,
        current_deviation_known=sum(row['current_deviation'] is not None for row in result['outcomes']),
        outcome_counts=dict(Counter(row['status'] for row in result['outcomes'])),
        engine_metrics=result['engine_metrics'], detector_statuses=result['detector_statuses'],
        paired_reference=compare_reference(result, oracle, args.reference_predictions) if args.reference_predictions else None,
        files_sha256={name:fingerprint(args.dataset/name) for name in ('test/traffic.csv','validate/schedule_plan.csv','labels/labels_test.csv')},
        limitations=['Это повторный test, уже известный команде; не новый ML holdout и не выбор модели.',
            'Измеряется ранний прогноз по плановому окну раздачи, не независимое определение начала дорожного инцидента.',
            ('cur_dev_s рассчитан Engine из GPS с TTL 300 с; в истории нет датчиков дверей. MAE только на покрытой части.' if gps_mode else
             'Результат условен на выданном cur_dev_s; поток GPS сам это число не восстанавливает.'),
            '217 возможных точек виртуального времени, не inference каждую реальную секунду; точное число в population.',
            'Воспроизводятся Engine и настоящий HTTP ML; NDTP байты, backend HTTP ingress и UI этим прогоном не проверяются.',
            'Неразмеченные алерты исключены из precision и явно показаны в coverage.',
            'Фактическое прибытие восстановлено из выданной метки, не из независимого датчика.'])
    write_json(out/'report.json', report)
    print(json.dumps({key:value for key,value in report.items() if key in ('status','runtime_s','population','http_cycle_s')}, ensure_ascii=False))
    print(json.dumps(dict(forecasts=metrics['labeled_forecasts'], horizon=metrics['planned_horizon'],
                         outcome_counts=report['outcome_counts'], current_deviation_known=report['current_deviation_known']), ensure_ascii=False))
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--url', default='http://127.0.0.1:8001')
    parser.add_argument('--timezone', default='UTC')
    parser.add_argument('--max-points', type=int)
    parser.add_argument('--budget-s', type=float, default=120)
    parser.add_argument('--out', type=Path, default=ROOT/'artifacts/stream-real')
    parser.add_argument('--deviation-source', choices=('csv_snapshot', 'gps'), default='csv_snapshot')
    parser.add_argument('--reference-predictions', type=Path,
                        help='predictions.jsonl контрольного CSV-прогона; сравнение только после завершения GPS-прогнозов')
    args = parser.parse_args()
    if args.max_points is not None and args.max_points < 1:
        parser.error('--max-points должен быть положительным')
    if not 1 <= args.budget_s <= 120:
        parser.error('--budget-s должен быть в [1,120]')
    if args.reference_predictions and args.deviation_source != 'gps':
        parser.error('--reference-predictions используется только с --deviation-source gps')
    return 0 if asyncio.run(execute(args)) else 1


if __name__ == '__main__':
    raise SystemExit(main())
