import pandas as pd
import pytest

from tools.temporal_validation import make_causal_temporal_folds


def labels():
    start = pd.Timestamp('2026-09-26T00:00:00Z')
    return pd.DataFrame([dict(sample_id=f'{minute}-{bus}', T=start+pd.Timedelta(minutes=minute),
        label_available_at=start+pd.Timedelta(minutes=minute, seconds=10))
        for minute in range(10) for bus in range(2)])


def test_equal_boundary_and_late_receipt_purge_whole_timestamp():
    rows = labels()
    rows.loc[rows.sample_id == '2-0', 'label_available_at'] = pd.Timestamp('2026-09-26T00:04:00Z')
    rows.loc[rows.sample_id == '3-1', 'label_available_at'] = pd.Timestamp('2026-09-26T00:06:00Z')
    fold = make_causal_temporal_folds(rows, n_splits=3)[0]
    assert set(fold.train_sample_ids) == {'0-0', '0-1', '1-0', '1-1'}
    assert set(fold.purged_sample_ids) == {'2-0', '2-1', '3-0', '3-1'}
    assert set(fold.validation_sample_ids) == {'4-0', '4-1', '5-0', '5-1'}


def test_unknown_availability_is_not_filled_as_immediate_or_zero():
    rows = labels()
    rows.loc[rows.sample_id == '1-0', 'label_available_at'] = pd.NaT
    folds = make_causal_temporal_folds(rows)
    assert all('1-0' not in fold.train_sample_ids and '1-1' not in fold.train_sample_ids for fold in folds)
    with pytest.raises(ValueError, match='явно заданное'):
        make_causal_temporal_folds(rows.drop(columns='label_available_at'))


def test_no_timestamp_is_split_between_train_purged_and_validation():
    rows = labels()
    lookup = rows.set_index('sample_id')
    for fold in make_causal_temporal_folds(rows):
        train = lookup.loc[list(fold.train_sample_ids)]
        valid = lookup.loc[list(fold.validation_sample_ids)]
        assert set(train['T']).isdisjoint(set(valid['T']))
        assert (train['label_available_at'] < valid['T'].min()).all()
        assert (train['T'] < valid['T'].min()).all()


def test_empty_train_after_purge_is_error_not_silent_fallback():
    rows = labels()
    rows['label_available_at'] = pd.NaT
    with pytest.raises(ValueError, match='пустая часть'):
        make_causal_temporal_folds(rows)
