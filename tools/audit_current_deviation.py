"""ТОЛЬКО постфактум-аудит выданного cur_dev_s и готовых GPS-прогнозов.

Сначала загружаются сохранённые predictions.json; затем test/schedule.csv с
фактами и labels_test используются как оценочный reference. Ничего из фактов
не передаётся детектору/модели/онлайн-сервису. Validate не открывается.
Результат не разрешает вычислять live-признак из будущего факта расписания.

Из filipp/: .venv/bin/python tools/audit_current_deviation.py \
    --data ../../data/raw/dataset
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import csv
from dataclasses import dataclass
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import time


def timestamp(value):
    parsed=datetime.fromisoformat(value)
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def finite_number(value):
    try:
        number=float(value)
    except (ValueError,TypeError):
        return None
    return number if math.isfinite(number) else None


@dataclass(frozen=True)
class Visit:
    id: str
    tr_id: int
    planned_at: datetime
    actual_at: datetime | None
    geom: str = ''
    address: str = ''

    @property
    def delay_s(self):
        return (self.actual_at-self.planned_at).total_seconds() if self.actual_at is not None else None


def latest_actual(visits, cutoff):
    """Уникальные ID последнего произошедшего визита; все tie остаются явными.

    Время получения операционного факта в раздаче неизвестно. Проверяется лишь
    необходимое условие actual_at<=T, не доступность уведомления на backend.
    """
    past={visit.id:visit for visit in visits if visit.actual_at is not None and visit.actual_at<=cutoff}
    latest=max((visit.actual_at for visit in past.values()),default=None)
    return sorted((visit for visit in past.values() if visit.actual_at==latest),key=lambda visit:visit.id)


def latest_plan(visits, cutoff):
    past={visit.id:visit for visit in visits if visit.planned_at<=cutoff}
    latest=max((visit.planned_at for visit in past.values()),default=None)
    return sorted((visit for visit in past.values() if visit.planned_at==latest),key=lambda visit:visit.id)


def hint_audit(visits, cutoff, hint):
    """Сравнить уже выданную подсказку с двумя reference; не создать ML-признак."""
    actual=latest_actual(visits,cutoff)
    planned=latest_plan(visits,cutoff)
    value=finite_number(hint)
    reference=actual[0] if len(actual)==1 else None
    actual_status=('no_past_actual' if not actual else 'ambiguous_last_actual' if len(actual)>1
                   else 'hint_nonfinite' if value is None else 'matches_last_actual' if value==reference.delay_s
                   else 'differs_from_last_actual')
    matches=[visit for visit in planned if value is not None and visit.delay_s==value]
    plan_status=('hint_nonfinite' if value is None else
                 'zero_before_first_plan' if not planned and value==0 else
                 'nonzero_before_first_plan' if not planned else
                 'latest_plan_reference_missing_fact' if all(v.actual_at is None for v in planned) else
                 'matches_latest_plan_delay' if matches else 'differs_from_latest_plan_delay')
    return {'hint':value,'actual_status':actual_status,'plan_status':plan_status,
            'last_actual_candidates':actual,'reference':reference,'latest_plan_candidates':planned,
            'matching_plan_candidates':matches,
            'all_matching_plan_facts_future':bool(matches) and all(v.actual_at>cutoff for v in matches)}


def assess_prediction(predictions, visits_by_id, cutoff, ttl_s=300):
    """Последняя GPS-оценка должна быть и произойти, и подтвердиться до cutoff."""
    available=[p for p in predictions if timestamp(p['arrived_at'])<=cutoff
               and timestamp(p['received_at'])<=cutoff
               and timestamp(p['received_at'])>=timestamp(p['arrived_at'])]
    newest=max((timestamp(p['arrived_at']) for p in available),default=None)
    candidates=[p for p in available if timestamp(p['arrived_at'])==newest]
    if not candidates:
        return {'status':'no_available_detection','prediction':None,'age_s':None,'delay_s':None}
    if len(candidates)!=1:
        return {'status':'ambiguous_latest_detection','prediction':None,'age_s':None,'delay_s':None}
    prediction=candidates[0]
    visit=visits_by_id.get(prediction['planned_stop_id'])
    if visit is None:
        return {'status':'detection_visit_not_in_plan','prediction':prediction,'age_s':None,'delay_s':None}
    age=(cutoff-newest).total_seconds()
    return {'status':'fresh_detection' if age<=ttl_s else 'expired_detection',
            'prediction':prediction,'age_s':age,'delay_s':(newest-visit.planned_at).total_seconds()}


def metrics(values):
    values=sorted(values)
    return {'n':len(values),'mae_s':statistics.fmean(values) if values else None,
            'median_absolute_error_s':statistics.median(values) if values else None,
            'p95_absolute_error_s':values[min(len(values)-1,int(.95*len(values)))] if values else None,
            'max_absolute_error_s':max(values,default=None)}


def visit_json(visit):
    return {'id':visit.id,'tr_id':visit.tr_id,'planned_at':visit.planned_at.isoformat(),
            'actual_at':visit.actual_at.isoformat() if visit.actual_at else None,
            'delay_s':visit.delay_s,'geom':visit.geom,'address':visit.address}


def read_rows(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:
        yield from csv.DictReader(stream)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(data, prediction_path, v1_path, output):
    started=time.monotonic()
    # Не запускаем детектор после прочтения evaluation-only фактов.
    predictions=json.loads(prediction_path.read_text())
    v1=json.loads(v1_path.read_text())
    files={'gps_v2_saved_predictions':prediction_path,'gps_v1_saved_predictions':v1_path,
           'test_schedule_evaluation_only':data/'test/schedule.csv',
           'test_labels_evaluation_only':data/'labels/labels_test.csv','audit_script':Path(__file__)}
    sources={name:{'path':str(path.resolve()),'sha256':sha(path)} for name,path in files.items()}
    counters=Counter();visits_by_id={};visits_by_tr=defaultdict(list)
    for row in read_rows(files['test_schedule_evaluation_only']):
        visit=Visit(row['tt_action_item_id'],int(row['tr_id']),timestamp(row['time_begin']),
                    timestamp(row['time_fact_begin']) if row['time_fact_begin'] else None,
                    row['geom'],row['building_address'])
        if visit.id in visits_by_id:
            if visits_by_id[visit.id]!=visit:
                raise ValueError(f'Conflicting duplicate plan visit: {visit.id}')
            counters['identical_duplicate_schedule_rows_deduplicated']+=1
            continue
        visits_by_id[visit.id]=visit;visits_by_tr[visit.tr_id].append(visit)
    predictions_by_tr=defaultdict(list);seen_predictions=set()
    for prediction in predictions:
        key=(prediction['tr_id'],prediction['planned_stop_id'])
        if key in seen_predictions:
            raise ValueError('Repeated GPS visit; resolve before audit')
        seen_predictions.add(key);predictions_by_tr[prediction['tr_id']].append(prediction)
    records=[];actual_counts=Counter();plan_counts=Counter();coverage=Counter();fresh_kind=Counter()
    hint_errors=[];reference_errors=[];same_visit_errors=[];hint_match_errors=[];hint_mismatch_errors=[]
    unique_reference_errors={};sample_ids=set()
    labels=list(read_rows(files['test_labels_evaluation_only']))
    labels.sort(key=lambda r:(timestamp(r['T']),r['sample_id']))
    for row in labels:
        if row['sample_id'] in sample_ids:
            raise ValueError('Duplicate sample_id')
        sample_ids.add(row['sample_id']);tr=int(row['tr_id']);at=timestamp(row['T'])
        state=hint_audit(visits_by_tr[tr],at,row['cur_dev_s']);reference=state['reference']
        prediction=assess_prediction(predictions_by_tr[tr],visits_by_id,at)
        actual_counts[state['actual_status']]+=1;plan_counts[state['plan_status']]+=1
        counters['finite_hint_snapshots' if state['hint'] is not None else 'nonfinite_hint_snapshots']+=1
        if state['hint']==0:
            counters['zero_hint_snapshots']+=1
            counters['zero_hint:'+state['actual_status']]+=1
        counters['latest_actual_time_tie_snapshots']+=len(state['last_actual_candidates'])>1
        counters['latest_plan_time_tie_snapshots']+=len(state['latest_plan_candidates'])>1
        counters['all_matching_latest_plan_facts_future']+=state['all_matching_plan_facts_future']
        category=prediction['status'];coverage[category]+=1
        record={'sample_id':row['sample_id'],'tr_id':tr,'T':at.isoformat(),'supplied_cur_dev_s':state['hint'],
                'hint_vs_last_actual':state['actual_status'],'hint_vs_latest_plan':state['plan_status'],
                'last_actual_candidates':[visit_json(v) for v in state['last_actual_candidates']],
                'latest_plan_candidates':[visit_json(v) for v in state['latest_plan_candidates']],
                'all_matching_latest_plan_facts_future':state['all_matching_plan_facts_future'],
                'gps_status':category,'gps_age_from_estimated_arrival_s':prediction['age_s'],
                'gps_estimated_delay_s':prediction['delay_s'],'gps_prediction':prediction['prediction']}
        if category in ('expired_detection','no_available_detection'):
            subtype=('no_or_ambiguous_actual_reference' if reference is None else
                     'last_actual_within_300s' if (at-reference.actual_at).total_seconds()<=300 else 'last_actual_older_300s')
            counters[category+':'+subtype]+=1
        if category=='fresh_detection':
            event=prediction['prediction'];visit=visits_by_id[event['planned_stop_id']]
            kind=('same_last_actual_visit' if reference and reference.id==visit.id else
                  'detected_visit_actual_is_future' if visit.actual_at and visit.actual_at>at else
                  'earlier_actual_visit' if visit.actual_at else 'detected_visit_without_fact')
            fresh_kind[kind]+=1;record['fresh_detection_relation']=kind
            if state['hint'] is not None:
                hint_errors.append(abs(prediction['delay_s']-state['hint']))
            if reference is not None:
                error=abs(prediction['delay_s']-reference.delay_s)
                reference_errors.append(error);record['error_vs_last_actual_s']=error
                # Одно первое свежее наблюдение на reference visit. Это устраняет
                # повторный вес визита, но не доказывает статистическую независимость.
                unique_reference_errors.setdefault((tr,reference.id),error)
                if kind=='same_last_actual_visit':same_visit_errors.append(error)
                if state['actual_status']=='matches_last_actual':hint_match_errors.append(error)
                elif state['actual_status']=='differs_from_last_actual':hint_mismatch_errors.append(error)
        records.append(record)

    # Наглядный ошибочный круг; реальные facts читаются лишь здесь в оценщике.
    example_id='53700641105';visit=visits_by_id[example_id]
    v1_event=next(p for p in v1 if p['planned_stop_id']==example_id)
    v2_event=next(p for p in predictions if p['planned_stop_id']==example_id)
    same_location=sorted((v for v in visits_by_tr[visit.tr_id] if v.geom==visit.geom
                         and abs((v.planned_at-visit.planned_at).total_seconds())<=7200),key=lambda v:v.planned_at)
    loop_example={'visit':visit_json(visit),'v1':v1_event,'v2':v2_event,
                  'v1_error_s':(timestamp(v1_event['arrived_at'])-visit.actual_at).total_seconds(),
                  'v2_error_s':(timestamp(v2_event['arrived_at'])-visit.actual_at).total_seconds(),
                  'other_visits_same_coordinates':[visit_json(v) for v in same_location]}
    terminal_id='53699018657';terminal=visits_by_id[terminal_id]
    terminal_example={'visit':visit_json(terminal),
        'v2':next(p for p in predictions if p['planned_stop_id']==terminal_id),
        'previous_same_location':[visit_json(v) for v in visits_by_tr[terminal.tr_id]
                                  if v.geom==terminal.geom and 0<(terminal.planned_at-v.planned_at).total_seconds()<=7200],
        'limitation':'Standing GPS at the terminal does not identify arrival vs layover vs next departure.'}
    report={'purpose':'Post-prediction reference audit only; do not use supplied future facts as online features',
        'created_at':datetime.now(timezone.utc).isoformat(),'elapsed_s':time.monotonic()-started,
        'sources':sources,'validate_opened':False,'detector_or_model_executed':False,'timezone_assumption':'UTC for all naive test timestamps',
        'unique_plan_visits':len(visits_by_id),'snapshot_count':len(records),
        'reference_semantics':'Latest UNIQUE visit by actual_at<=T; equal actual times remain explicit ambiguity. Receipt time of organizer fact is absent.',
        'unique_reference_visits_in_fresh_gps_metrics':len(unique_reference_errors),
        'same_day_statistical_independence_claimed':False,
        'hint_vs_last_actual_counts':dict(actual_counts),'hint_vs_latest_plan_counts':dict(plan_counts),
        'counts':dict(counters),'gps_coverage_ttl_300s':dict(coverage),'fresh_detection_relation':dict(fresh_kind),
        'metrics':{'fresh_gps_vs_supplied_hint':metrics(hint_errors),
                   'fresh_gps_vs_unambiguous_last_actual':metrics(reference_errors),
                   'fresh_same_visit_only':metrics(same_visit_errors),
                   'actual_reference_with_matching_hint':metrics(hint_match_errors),
                   'actual_reference_with_inconsistent_hint':metrics(hint_mismatch_errors),
                   'one_first_fresh_snapshot_per_unique_actual_visit':metrics(list(unique_reference_errors.values()))},
        'corrected_loop_example':loop_example,'remaining_terminal_ambiguity_example':terminal_example,
        'hint_future_fact_example':next(r for r in records if r['sample_id']=='131672_1767668400'),
        'interpretation':['353/353 compatibility with latest-plan delay plus initial zero is an observed data pattern, not knowledge of organizer generation code.',
            'The README says cur_dev_s is supplied and known on T. It remains a permitted dataset input; the pattern does not authorize reconstructing it from future schedule facts online.',
            'Disagreement with supplied cur_dev_s is not pure GPS arrival error when the reference uses a different temporal semantics.',
            'Low GPS coverage and premature terminal assignments remain actual limitations independently of hint semantics.',
            'Snapshots, unique visits and vehicles on one already inspected day are correlated; none is claimed as a new independent holdout.',
            'Fact occurrence time does not guarantee fact delivery time. No organizer receive timestamp exists for arrivals.'],
        'recommended_structural_change':{
            'name':'Separate physical-stop episodes from ordered planned-visit hypotheses',
            'mechanism':'Maintain a small set of trip-progress/visit candidates across multiple observed stop approaches and departures. Commit a visit only after its sequence/direction separates it from repeated-location alternatives; ambiguous terminal arrival/departure stays unknown.',
            'inputs':'GPS available by both timestamps, plan order/coordinates only; ideally operator trip_id, direction_id, physical stop ID and arrival/departure type.',
            'validation':'Freeze decoding policy on a calibration day; use an independent labeled day with physical arrival/departure labels. Score coverage, wrong-loop rate and availability delay; do not tune radius on inspected test.',
            'hackathon_value':'Improves real online input coverage in principle, but cannot reproduce future-derived supplied hints and does not itself improve an already-maxed CSV score. A documented operational current-deviation source is the reliable immediate integration route.',
            'recommended_now':'Keep organizer points.cur_dev_s in DS/replay because README explicitly provides it on T; keep live arrival API as the production contract and GPS as estimated/unknown. Ask organizer whether online grading expects this provided current-deviation input or requires independent GPS reconstruction, and what arrival vs departure means at terminals.',
            'do_not_do':'Do not read time_fact_begin for live features, label an artificial current-deviation source as real, silently change the frozen ML feature definition, or claim 26/26 from this audit.'},
        'rows':records}
    for name,path in files.items():
        if sha(path)!=sources[name]['sha256']:
            raise RuntimeError(f'Source changed: {name}')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('../../data/raw/dataset'))
    parser.add_argument('--predictions',type=Path,default=Path('artifacts/gps-diagnosis/v2-test/predictions.json'))
    parser.add_argument('--v1-predictions',type=Path,default=Path('artifacts/gps-real/predictions.json'))
    parser.add_argument('--out',type=Path,default=Path('artifacts/gps-diagnosis/current-deviation-audit.json'))
    args=parser.parse_args()
    report=audit(args.data,args.predictions,args.v1_predictions,args.out)
    print(json.dumps({key:report[key] for key in ('snapshot_count','hint_vs_last_actual_counts',
        'hint_vs_latest_plan_counts','gps_coverage_ttl_300s','fresh_detection_relation','metrics')},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
