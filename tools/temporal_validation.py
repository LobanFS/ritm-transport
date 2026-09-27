"""Причинные expanding folds для следующего обучения; текущую модель не меняет.

Helper требует явное label_available_at. На CLI без файла доступности можно
лишь диагностировать нижнюю границу: план + известная TRAIN-метка = прибытие.
Это НЕ доказывает время получения/финализации ответа. Обучение не запускается.
"""
import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CausalFold:
    name: str
    validation_start: str
    validation_end: str
    train_sample_ids: tuple[str, ...]
    validation_sample_ids: tuple[str, ...]
    purged_sample_ids: tuple[str, ...]


def make_causal_temporal_folds(labels, *, n_splits=3, initial_train_fraction=.4):
    """Не разрезает одинаковые T; весь train timestamp исключается при поздней/неизвестной метке.

    Доступность ответа обязана быть строго раньше начала validation. Время
    финального исправления/получения должно входить в label_available_at.
    Это новый контракт, не воспроизведение старых frozen-fold метрик.
    """
    if n_splits < 1 or not .2 <= initial_train_fraction < .9:
        raise ValueError('Некорректные n_splits/initial_train_fraction')
    required = {'sample_id', 'T', 'label_available_at'}
    if not required <= set(labels.columns):
        raise ValueError('Нужны sample_id, T и явно заданное label_available_at')
    work = labels[list(required)].copy()
    if work['sample_id'].isna().any() or work['sample_id'].duplicated().any():
        raise ValueError('sample_id должен быть уникальным и известным')
    work['sample_id'] = work['sample_id'].astype(str)
    work['T'] = pd.to_datetime(work['T'], format='mixed', utc=True, errors='raise')
    work['label_available_at'] = pd.to_datetime(work['label_available_at'], format='mixed', utc=True, errors='raise')
    if work['T'].isna().any():
        raise ValueError('T должен быть известен')
    times = sorted(work['T'].unique())
    if len(times) < n_splits + 2:
        raise ValueError('Недостаточно уникальных T')
    first = max(1, math.ceil(len(times) * initial_train_fraction))
    if len(times) - first < n_splits:
        raise ValueError('Недостаточно уникальных T после initial_train_fraction')
    result = []
    for index, window in enumerate(np.array_split(np.array(times, dtype=object)[first:], n_splits), 1):
        start, end = pd.Timestamp(window[0]), pd.Timestamp(window[-1])
        candidate = work[work['T'] < start]
        immature = candidate['label_available_at'].isna() | (candidate['label_available_at'] >= start)
        # Консервативная атомарная группа: не сохраняем часть одного момента T.
        excluded_times = set(candidate.loc[immature, 'T'])
        purged = candidate[candidate['T'].isin(excluded_times)]
        train = candidate[~candidate['T'].isin(excluded_times)]
        valid = work[work['T'].between(start, end)]
        if train.empty or valid.empty:
            raise ValueError(f'causal_time_{index:02d}: пустая часть после purge')
        result.append(CausalFold(f'causal_time_{index:02d}', start.isoformat(), end.isoformat(),
            tuple(train['sample_id']), tuple(valid['sample_id']), tuple(purged['sample_id'])))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path, default=ROOT.parent.parent/'data/raw/dataset/labels/labels_train.csv')
    parser.add_argument('--frozen-folds', type=Path, default=ROOT.parent/'alexchist/artifacts/final_model/frozen_folds.json')
    parser.add_argument('--acceptance', type=Path, default=ROOT.parent/'alexchist/artifacts/final_model/acceptance.json')
    parser.add_argument('--label-availability-csv', type=Path,
                        help='sample_id,label_available_at — время фактической доступности ответа')
    parser.add_argument('--out', type=Path, default=ROOT/'artifacts/audit-fixes/temporal-validation')
    args = parser.parse_args()
    labels = pd.read_csv(args.labels, dtype={'sample_id': str})
    labels['T'] = pd.to_datetime(labels['T'], format='mixed', utc=True)
    actual = pd.to_datetime(labels['target_time_begin'], format='mixed', utc=True) + pd.to_timedelta(labels['target_delay_s'], unit='s')
    labels['earliest_possible_label_at'] = actual
    sources = [args.labels, args.frozen_folds, args.acceptance, Path(__file__)]
    if args.label_availability_csv:
        availability = pd.read_csv(args.label_availability_csv, dtype={'sample_id': str})
        if set(availability.columns) != {'sample_id', 'label_available_at'} or availability['sample_id'].duplicated().any():
            raise ValueError('Нужен уникальный sample_id и единственное поле label_available_at')
        if set(availability['sample_id']) != set(labels['sample_id']):
            raise ValueError('Доступность должна покрывать ровно переданные labels')
        labels = labels.merge(availability, on='sample_id', validate='one_to_one')
        labels['label_available_at'] = pd.to_datetime(labels['label_available_at'], format='mixed', utc=True)
        if (labels['label_available_at'] < labels['earliest_possible_label_at']).any():
            raise ValueError('Метка не может стать доступной раньше самого фактического прибытия')
        mode = 'explicit_label_availability'
        sources.append(args.label_availability_csv)
    else:
        labels['label_available_at'] = actual
        mode = 'arrival_time_lower_bound_only'
    frozen = json.loads(args.frozen_folds.read_text())
    causal = make_causal_temporal_folds(labels)
    indexed = labels.set_index('sample_id')
    audits = []
    for original, proposed in zip([fold for fold in frozen if fold['name'].startswith('time_')], causal, strict=True):
        train = indexed.loc[original['train_sample_ids']]
        valid = indexed.loc[original['validation_sample_ids']]
        start = valid['T'].min()
        assert set(proposed.validation_sample_ids) == set(original['validation_sample_ids'])
        assert set(proposed.train_sample_ids) | set(proposed.purged_sample_ids) == set(original['train_sample_ids'])
        late = train['earliest_possible_label_at'] >= start
        audits.append(dict(frozen_fold=original['name'], validation_start=start.isoformat(),
            old_train_rows=len(train), validation_rows=len(valid),
            arrival_at_or_after_validation_start=int(late.sum()),
            arrival_strictly_after_validation_start=int((train['earliest_possible_label_at'] > start).sum()),
            unknown_availability=int(train['label_available_at'].isna().sum()),
            proposed_train_rows=len(proposed.train_sample_ids), purged_rows=len(proposed.purged_sample_ids),
            purged_unique_T=int(indexed.loc[list(proposed.purged_sample_ids), 'T'].nunique()),
            examples=[dict(sample_id=identity, T=row['T'].isoformat(), arrival=row['earliest_possible_label_at'].isoformat())
                      for identity, row in train[late].head(3).iterrows()]))
    acceptance = json.loads(args.acceptance.read_text())
    report = dict(checked_at=datetime.now(timezone.utc).isoformat(), fit_performed=False, model_changed=False,
        availability_mode=mode, folds=audits, previous_six_fold_mean_mae_s=acceptance['mean_candidate_mae_s'],
        interpretation='Прежнее среднее шести frozen folds воспроизводимо, но не является причинной rolling-оценкой. '
            'Три temporal folds допускают метки после старта validation; три vehicle folds имеют другую постановку.',
        new_model_mae_s=None, limitations=[
            'Purge меняет train-популяцию. Без нового fit нельзя пересчитать или обещать улучшение MAE.',
            'Время прибытия из train target — только нижняя граница доступности ответа; время доставки/исправления не дано.',
            'Helper требует явное label_available_at; CLI lower-bound артефакт не доказывает причинность при задержанных метках.',
            'Все строки одного T исключаются вместе, если хотя бы одна метка в группе неизвестна или поздняя.',
            'Новый день/семейство синтетических данных и причинная доступность признаков этим split не проверяются.'],
        sources_sha256={str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources})
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    (args.out/'causal-folds.json').write_text(json.dumps(dict(availability_mode=mode,
        folds=[asdict(fold) for fold in causal]), ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(report=str(args.out/'report.json'), fit_performed=False,
        immature_rows=[row['arrival_at_or_after_validation_start'] for row in audits],
        purged_rows=[row['purged_rows'] for row in audits]), ensure_ascii=False))


if __name__ == '__main__':
    main()
