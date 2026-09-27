"""Причинная проверка алертов: плановая цель ≠ начало нового сбоя.

Запуск из filipp: .venv/bin/python tools/evaluate_alerts.py
По умолчанию проверяется зафиксированный persistence baseline. --ml-url
подключает отдельный HTTP ML-сервис без изменения работающего backend.
GPS и план проходят через настоящий Engine; oracle читается лишь после
сохранения прогнозов. Синтетическая проверка не даёт балл жюри или DS MAE.
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
import statistics
import sys
import time
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.engine import Engine
from backend.generator_bridge import ProducerState
from common.contracts import PredictionRequest, baseline
from generator.scenarios import DURATION_S, GeneratorSession, ResetRequest, START

THRESHOLD_S = 120.0  # Факт задержки >2 минут; отдельная метка, не порог красного сигнала.


def dt(value: datetime | str) -> datetime:
    result = value if isinstance(value, datetime) else datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError('Время должно содержать часовой пояс')
    return result


def seconds(later: datetime | str, earlier: datetime | str) -> float:
    return (dt(later)-dt(earlier)).total_seconds()


def in_plan_window(lead: float) -> bool:
    return 600 < lead <= 900


def in_early_window(lead: float) -> bool:
    # PDF пишет 10–15 минут; строгая нижняя граница относится к DS-плану.
    return 600 <= lead <= 900


@dataclass(frozen=True)
class IncidentTruth:
    """Отдельная разметка точного начала события, никогда не вход модели."""
    event_id: str
    tr_id: int
    onset_at: datetime
    affected_visit_ids: tuple[str, ...]
    already_late: bool | None
    kind: str = 'synthetic_disruption'


def normalize_alert(item: dict[str, Any]) -> dict[str, Any]:
    tr_id = int(item['tr_id'])
    visit_id = item.get('planned_stop_id')
    if visit_id is None:
        prefix = f'{tr_id}:'
        if not str(item['id']).startswith(prefix):
            raise ValueError('Нет ID планового посещения в алерте')
        visit_id = str(item['id'])[len(prefix):]
    value = item.get('predicted_delay_s')
    if value is not None and not math.isfinite(float(value)):
        raise ValueError('Нечисловой прогноз')
    return dict(item, tr_id=tr_id, planned_stop_id=str(visit_id),
                created_at=dt(item['created_at']), data_cutoff=dt(item['data_cutoff']),
                target_time=dt(item['target_time']))


def evaluate_records(alerts: list[dict], plans: dict[tuple[int, str], datetime],
                     arrivals: dict[tuple[int, str], datetime], events: list[IncidentTruth],
                     *, started_at: datetime, ended_at: datetime,
                     threshold_s: float = THRESHOLD_S) -> dict:
    """Раздельные знаменатели: зрелые цели, точные новые события и proxy.

    Пустое/неполное окно не является отрицательным событием или 100% recall.
    Дедупликация повторов алерта на одно посещение: первая публикация.
    Для onset оцениваются только точные внешние события, не время прибытия.
    """
    started_at, ended_at = dt(started_at), dt(ended_at)
    if ended_at < started_at or not math.isfinite(threshold_s) or threshold_s < 0:
        raise ValueError('Неверный интервал или порог')
    plans = {k:dt(v) for k,v in plans.items()}
    arrivals = {k:dt(v) for k,v in arrivals.items()}
    normalized = sorted((normalize_alert(a) for a in alerts), key=lambda a:a['created_at'])
    unique = {}
    for alert in normalized:
        unique.setdefault((alert['tr_id'],alert['planned_stop_id']),alert)
    warnings = list(unique.values())
    invalid, outcomes = [], []
    valid = []
    for alert in warnings:
        key = alert['tr_id'],alert['planned_stop_id']
        published, cutoff = alert['created_at'],alert['data_cutoff']
        violation = None
        if key not in plans or plans[key] != alert['target_time']:
            violation = 'unknown_or_mismatched_plan'
        elif cutoff > published:
            violation = 'future_input_cutoff'
        elif not started_at <= published <= ended_at:
            violation = 'publication_outside_observation'
        elif not in_plan_window(seconds(alert['target_time'],published)):
            violation = 'publication_outside_plan_window'
        if violation:
            invalid.append(dict(id=alert.get('id'),reason=violation))
            continue
        valid.append(alert)
        actual = arrivals.get(key)
        if actual is None or actual > ended_at:
            # Отсутствие наступившего факта не говорит, что автобус не опоздает.
            outcomes.append(dict(id=alert.get('id'),status='unmatured_or_unlabeled'))
            continue
        delay = seconds(actual,plans[key])
        lead = seconds(actual,published)
        outcomes.append(dict(id=alert.get('id'),status='assessable',actual_delay_s=delay,
                             actual_arrival_lead_s=lead,late=delay > threshold_s,
                             actual_arrival_in_10_15m=in_early_window(lead),
                             after_actual_arrival=lead <= 0))
    known = [r for r in outcomes if r['status']=='assessable']
    late = [r for r in known if r['late']]
    event_rows = []
    used = set()
    for event in sorted(events,key=lambda e:e.onset_at):
        onset = dt(event.onset_at)
        exclusion = None
        if onset < started_at:
            exclusion = 'preexisting_event'
        elif onset > ended_at:
            exclusion = 'unobserved_event'
        elif event.already_late is None:
            exclusion = 'prior_delay_unknown'
        elif event.already_late:
            exclusion = 'already_late_before_disruption'
        elif onset-timedelta(seconds=900) < started_at:
            exclusion = 'left_censored_warning_window'
        row = dict(event_id=event.event_id,tr_id=event.tr_id,onset_at=onset,
                   kind=event.kind,eligible=exclusion is None,exclusion=exclusion,
                   matched_alert_id=None,lead_s=None)
        if exclusion is None:
            # Один алерт не засчитываем нескольким событиям. Цель обязана быть
            # среди явно размеченных затронутых посещений этого события.
            candidates = [(i,a) for i,a in enumerate(valid) if i not in used
                and a['tr_id']==event.tr_id and a['planned_stop_id'] in event.affected_visit_ids
                and in_early_window(seconds(onset,a['created_at']))]
            if candidates:
                index, matched = min(candidates,key=lambda pair:pair[1]['created_at'])
                used.add(index)
                row.update(matched_alert_id=matched.get('id'),lead_s=seconds(onset,matched['created_at']))
        event_rows.append(row)
    eligible = [r for r in event_rows if r['eligible']]
    matched_events = [r for r in eligible if r['matched_alert_id'] is not None]

    # Посещения дают лишь дискретный переход; точный onset между ними неизвестен.
    proxy = []
    by_vehicle = defaultdict(list)
    for key, planned in plans.items():
        by_vehicle[key[0]].append((planned,key))
    for tr_id, visits in by_vehicle.items():
        visits.sort()
        for (_,previous),(_,current) in zip(visits,visits[1:]):
            if previous not in arrivals or current not in arrivals:
                continue
            before, after = arrivals[previous],arrivals[current]
            if not before < after <= ended_at or after < started_at:
                continue
            previous_delay, delay = seconds(before,plans[previous]),seconds(after,plans[current])
            if previous_delay <= threshold_s < delay:
                proxy.append(dict(tr_id=tr_id,planned_stop_id=current[1],
                    previous_observed_at=before,first_late_visit_at=after,
                    previous_delay_s=previous_delay,delay_s=delay,
                    exact_onset_known=False))
    return dict(threshold_s=threshold_s,raw_alerts=len(alerts),unique_alerts=len(warnings),
        duplicate_alerts=len(alerts)-len(warnings),
        planned_horizon=dict(valid=len(valid),invalid=len(invalid),
            fraction=len(valid)/len(warnings) if warnings else None,violations=invalid),
        target_outcomes=dict(assessable=len(known),late=len(late),
            late_precision=len(late)/len(known) if known else None,
            unmatured_or_unlabeled=len(outcomes)-len(known),
            actual_arrival_10_15m=sum(r['actual_arrival_in_10_15m'] for r in known),
            after_actual_arrival=sum(r['after_actual_arrival'] for r in known),
            median_actual_arrival_lead_s=statistics.median(r['actual_arrival_lead_s'] for r in known) if known else None,
            caveat='Прибытие цели не является точным временем начала нового сбоя',rows=outcomes),
        new_event_onsets=dict(total=len(events),eligible=len(eligible),matched=len(matched_events),
            recall=len(matched_events)/len(eligible) if eligible else None,
            status='measured' if eligible else 'not_assessable_no_complete_new_event_windows',
            excluded=dict(Counter(r['exclusion'] for r in event_rows if r['exclusion'])),rows=event_rows),
        first_late_visit_proxies=proxy)


def synthetic_events(producer: GeneratorSession, plans, arrivals) -> list[IncidentTruth]:
    """Вызывается ПОСЛЕ прогноза: читает приватный oracle начала инъекции.

    Это время изменения движения, не разобранная из GPS причина. GPS-loss —
    событие качества данных; в recall задержек не включается.
    """
    if producer.scenario not in ('slow_segment','long_stop'):
        return []
    onset, tr_id, _ = producer._scenario_events[0]
    prior = max(((key,at) for key,at in arrivals.items() if key[0]==tr_id and at <= onset),
                key=lambda pair:pair[1],default=None)
    already_late = seconds(prior[1],plans[prior[0]]) > THRESHOLD_S if prior else None
    affected = tuple(key[1] for key,actual in arrivals.items() if key[0]==tr_id and actual > onset)
    return [IncidentTruth(event_id=f'{producer.scenario}-{tr_id}',tr_id=tr_id,onset_at=onset,
                          affected_visit_ids=affected,already_late=already_late)]


async def run_case(scenario: str, *, seed=42, route_count=4, cadence=15,
                   forecast_interval=15, ml_url=None) -> dict:
    if forecast_interval not in (1,5,15,30):
        raise ValueError('Интервал прогноза: 1, 5, 15 или 30 секунд')
    producer = GeneratorSession(ResetRequest(scenario=scenario,seed=seed,route_count=route_count,
                                            telemetry_interval_s=cadence,speed=1,paused=False))
    engine = Engine('http://evaluation-ml')
    engine.generator_url = 'http://evaluation-generator'
    engine.set_generator(producer.context,ProducerState.model_validate(producer.status()))
    snapshots, alerts, seen = [], [], set()
    ml_calls = 0
    remote_model = None
    async with httpx.AsyncClient(timeout=10,trust_env=False) as remote:
        if ml_url:
            metadata = await remote.get(ml_url.rstrip('/')+'/model')
            metadata.raise_for_status()
            remote_model = metadata.json()
        async def route(request):
            nonlocal ml_calls
            if request.url.host == 'evaluation-generator' and request.url.path == '/stream':
                payload = producer.stream(int(request.url.params['after']))
                payload['logs'] = []  # Даже уже наступившая истина не идёт в backend.
                return httpx.Response(200,json=payload)
            if request.url.host != 'evaluation-ml' or request.url.path != '/predict/batch':
                raise AssertionError(f'Неожиданный запрос: {request.url}')
            ml_calls += 1
            rows = json.loads(request.content)
            if ml_url:
                response = await remote.post(ml_url.rstrip('/')+'/predict/batch',json=rows)
                response.raise_for_status()
                return httpx.Response(200,json=response.json())
            predictions = [baseline(PredictionRequest.model_validate(row)).model_dump(mode='json') for row in rows]
            return httpx.Response(200,json=predictions)
        async with httpx.AsyncClient(transport=httpx.MockTransport(route)) as client:
            engine.client = client
            for tick in range(0,DURATION_S+1,forecast_interval):
                if tick:
                    producer.advance(forecast_interval)
                await engine.tick(0)
                for trace in engine.forecast_traces.values():
                    request,response = trace['request'],trace['response']
                    snapshot = dict(tr_id=request['tr_id'],planned_stop_id=request['target']['id'],
                        data_cutoff=request['issued_at'],created_at=engine.clock,
                        target_time=request['target']['scheduled_at'],
                        current_delay_s=request['current_delay_s'],
                        predicted_delay_s=response['predicted_delay_s'],risk=response['risk'],
                        method=response['method'],model_version=response['model_version'],
                        execution=trace['execution'])
                    snapshots.append(snapshot)
                for incident in reversed(engine.incidents):
                    if incident['id'] not in seen:
                        seen.add(incident['id'])
                        alerts.append(dict(incident))
                # Будущие планы разрешены. Подтверждения и телеметрия — только
                # доступные на cutoff; truth/приватный сценарий ещё не прочитаны.
                if any(max(p.event_time,p.received_at)>engine.clock for points in engine.history.values() for p in points):
                    raise AssertionError('Будущая телеметрия попала в Engine')
                if any(a.source != 'gps_estimate' for records in engine.arrivals.values() for a in records.values()):
                    raise AssertionError('Во входах оказался внешний факт прибытия')
    # Этот порядок — часть оценки: все прогнозы и алерты уже зафиксированы.
    frozen = json.dumps(dict(predictions=snapshots,alerts=alerts),default=str,sort_keys=True)
    frozen_hash = hashlib.sha256(frozen.encode()).hexdigest()
    truth = producer.truth()['observed_truth_arrivals']
    plans = {(entry.tr_id,entry.target.id):entry.target.scheduled_at for entry in producer.context.schedule}
    arrivals = {(entry['tr_id'],entry['planned_stop_id']):dt(entry['arrived_at']) for entry in truth}
    events = synthetic_events(producer,plans,arrivals)
    result = evaluate_records(alerts,plans,arrivals,events,started_at=START,ended_at=producer.clock_time)
    result.update(scenario=scenario,seed=seed,route_count=route_count,cadence_s=cadence,
                  forecast_interval_s=forecast_interval,predictions=len(snapshots),ml_calls=ml_calls,
                  model_versions=sorted({p['model_version'] for p in snapshots}),
                  execution=dict(Counter(p['execution'] for p in snapshots)),
                  requested_inference_succeeded=all(p['execution']=='ml_http' for p in snapshots),
                  remote_model=remote_model,
                  missing_current_delay=sum(p['current_delay_s'] is None for p in snapshots),
                  prediction_plan_window_violations=sum(not in_plan_window(seconds(p['target_time'],p['data_cutoff'])) for p in snapshots),
                  frozen_prediction_sha256=frozen_hash,
                  oracle_read_after_prediction_freeze=True,
                  predictions_log=snapshots,alerts_log=alerts)
    return result


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ml-url',help='HTTP ML-сервис; без опции фиксированный persistence baseline')
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts'/'alerts')
    parser.add_argument('--forecast-interval-s',type=int,choices=[1,5,15,30],default=15)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    from common.risk import RISK_POLICY
    contract = dict(change='Evaluate current alert policy against lateness >120s; no model fitting or threshold tuning',
        source='Synthetic generator seed42; known network; not an independent real-day test',
        threshold_s=THRESHOLD_S,forecast_interval_s=args.forecast_interval_s,
        risk_policy=RISK_POLICY.model_dump(),
        model='remote HTTP ML' if args.ml_url else 'fixed persistence-v1 baseline',
        plan_window='(600,900] seconds to planned target at publication',
        early_window='[600,900] seconds to separately annotated onset of a NEW disruption',
        onset_definition='Synthetic injection start; not delay-threshold crossing or actual stop arrival',
        matching='Same vehicle and affected planned visit; one alert per event and vice versa',
        exclusions='Ongoing lateness, unknown prior lateness, left-censored 15-minute window, unobserved event',
        acceptance='No future input/cutoff, no plan-window violations. Early recall remains null without eligible events.',
        budget_seconds=60,labels_read='Only after all predictions/alerts for the case are frozen',
        caveats=['Actual arrival is not precise onset of a new delay.',
                 'Missing future truth is unknown, never a negative.',
                 'Current sudden interventions have no 10–15-minute causal precursor.',
                 'In-memory virtual-time stream is not network latency or production load testing.'])
    (args.output/'contract.json').write_text(json.dumps(contract,ensure_ascii=False,indent=2)+'\n')
    (args.output/'report.json').write_text(json.dumps(dict(status='running',contract=contract),ensure_ascii=False,indent=2)+'\n')
    started = time.monotonic()
    cases = []
    for scenario in ('normal','slow_segment','long_stop','gps_loss'):
        remaining = max(.001,contract['budget_seconds']-(time.monotonic()-started))
        result = await asyncio.wait_for(run_case(scenario,forecast_interval=args.forecast_interval_s,
                                                ml_url=args.ml_url),timeout=remaining)
        for name in ('predictions','alerts'):
            rows = result.pop(name+'_log')
            filename = f'{scenario}-{name}.jsonl'
            (args.output/filename).write_text(''.join(json.dumps(row,ensure_ascii=False,default=str)+'\n' for row in rows))
        cases.append(result)
        print(json.dumps({key:result[key] for key in ('scenario','predictions','unique_alerts','new_event_onsets')},ensure_ascii=False,default=str))
        if time.monotonic()-started > contract['budget_seconds']:
            raise TimeoutError('Бюджет оценки превышен; частичные логи сохранены, успешный отчёт не записан')
    code = [*ROOT.joinpath('backend').glob('*.py'),*ROOT.joinpath('common').glob('*.py'),
            *ROOT.joinpath('generator').glob('*.py'),*ROOT.joinpath('ml_service').glob('*.py'),Path(__file__)]
    report = dict(status='complete',created_at=datetime.now(timezone.utc),elapsed_s=round(time.monotonic()-started,3),
        contract=contract,cases=cases,
        invariant_checks_passed=all(not c['prediction_plan_window_violations'] and not c['planned_horizon']['invalid'] for c in cases),
        requested_inference_succeeded=all(c['requested_inference_succeeded'] for c in cases),
        early_warning_claim='Not established by this scenario set; see eligible-event denominators',
        sources_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in code})
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str)+'\n')
    print(args.output/'report.json')


if __name__=='__main__':
    asyncio.run(main())
