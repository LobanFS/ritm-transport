"""Сравнить уже сохранённые GPS-прогнозы с фактами только ПОСЛЕ детекции.

Ничего не обучает, не вызывает backend, не читает validate. Признаки и оценки
не смешиваются: факты нужны исключительно этому post-hoc диагностическому коду.
Ранее просмотренный test не становится новым holdout от повторного запуска.

Пример из filipp/:
  .venv/bin/python tools/compare_gps_diagnostics.py \
    --data ../../data/raw/dataset --baseline artifacts/gps-review/baseline \
    --candidate artifacts/gps-review/independent-reanchor \
    --out artifacts/gps-review/independent-reanchor-comparison.json
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics


def timestamp(value):
    value = datetime.fromisoformat(value)
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def compare(data, baseline, candidate, out):
    # Predictions are loaded before the evaluation-only fact column.
    runs = {name: {'path': path, 'predictions': json.loads((path/'predictions.json').read_text()),
                   'report': json.loads((path/'report.json').read_text()),
                   'deviations': json.loads((path/'current-deviation.json').read_text())}
            for name, path in [('baseline', baseline), ('candidate', candidate)]}
    for key in ('traffic', 'plan', 'labels'):
        if len({run['report']['sources'][key]['sha256'] for run in runs.values()}) != 1:
            raise ValueError(f'Нельзя сравнивать разные входы: {key}')
    schedule_path = data/'test/schedule.csv'
    if digest(schedule_path) != runs['baseline']['report']['sources']['plan']['sha256']:
        raise ValueError('Evaluation schedule не соответствует источнику прогнозов')
    with schedule_path.open(encoding='utf-8-sig', newline='') as stream:
        schedule = {row['tt_action_item_id']: row for row in csv.DictReader(stream)}
    metrics, errors = {}, {}
    for name, run in runs.items():
        predictions = run['predictions']
        by_id = {row['planned_stop_id']: row for row in predictions}
        if len(by_id) != len(predictions):
            raise ValueError('Предположение уникального ID посещения нарушено')
        errors[name] = {row['planned_stop_id']:
            abs((timestamp(row['arrived_at']) - timestamp(schedule[row['planned_stop_id']]['time_fact_begin'])).total_seconds())
            for row in predictions if schedule[row['planned_stop_id']]['time_fact_begin']}
        fresh = [row for row in run['deviations'] if row['known']
            and 0 <= (timestamp(row['cutoff'])-timestamp(by_id[row['based_on_visit']]['arrived_at'])).total_seconds() <= 300]
        report = run['report']
        values = list(errors[name].values())
        metrics[name] = {
            'fresh_current_deviation_300s': len(fresh),
            'forecast_moments': len(run['deviations']),
            'full_schedule_assessable_visits': len(values),
            'full_schedule_mae_s': statistics.mean(values) if values else None,
            'full_schedule_errors_gt900s': sum(value > 900 for value in values),
            'labeled_visits': report['labeled_visits'],
            'detected_labeled_ids': report['detected_labeled_same_id'],
            'labeled_errors_gt900s': report['labeled_errors_over_900s'],
            'labeled_errors_within_s': report['labeled_matches_within_s'],
        }
    correct = {key for key, error in errors['baseline'].items() if error <= 60}
    retained = {key for key in correct if errors['candidate'].get(key, float('inf')) <= 60}
    base, new = metrics['baseline'], metrics['candidate']
    gates = {
        'fresh_coverage_increased': new['fresh_current_deviation_300s'] > base['fresh_current_deviation_300s'],
        'baseline_correct_ids_retained_98pct': len(retained)/len(correct) >= .98 if correct else False,
        'full_schedule_catastrophic_errors_not_increased': new['full_schedule_errors_gt900s'] <= base['full_schedule_errors_gt900s'],
        'labeled_catastrophic_errors_not_increased': new['labeled_errors_gt900s'] <= base['labeled_errors_gt900s'],
        'full_schedule_mae_within_5s_of_baseline': new['full_schedule_mae_s'] <= base['full_schedule_mae_s']+5,
    }
    source_files = [schedule_path, Path(__file__)]
    source_files += [run['path']/filename for run in runs.values()
                     for filename in ('predictions.json', 'report.json', 'current-deviation.json')]
    result = {'purpose': 'Диагностическое сравнение заранее сохранённых GPS-оценок; test уже просмотрен',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'metrics': metrics, 'predeclared_gates': gates,
        'gps_data_gates_passed': all(gates.values()),
        'baseline_correct_ids_retained': len(retained), 'baseline_correct_ids_total': len(correct),
        'lost_correct_ids': [{'id': key, 'before_error_s': errors['baseline'][key],
                             'after_error_s': errors['candidate'].get(key)} for key in sorted(correct-retained)],
        'sources_sha256': {str(path.resolve()): digest(path) for path in source_files},
        'limitations': ['Не новый holdout и не качество ML-модели.',
            'Выданные факты расписания не являются независимо проверенной операционной истиной.',
            'MAE считается только на оценённых посещениях; знаменатели разных вариантов различаются.',
            'cur_dev_s не используется ни для выбора варианта, ни как эталон текущего отставания.',
            'Synthetic acceptance и контрактные тесты проверяются отдельно; этот файл не заменяет их.']}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'metrics': metrics, 'gates': gates, 'passed': all(gates.values())}, ensure_ascii=False))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    compare(args.data, args.baseline, args.candidate, args.out)
