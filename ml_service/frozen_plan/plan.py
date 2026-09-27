"""Exact selected plan-context r03 recipe, with audited identity fallback."""
from pathlib import Path
import numpy as np
import pandas as pd
from .features import build_features

PLAN_COLUMNS = ('tt_action_item_id', 'time_begin', 'order_date', 'manual_fill',
                'tr_id', 'geom', 'building_address')
PLAN_FEATURES = ('cur_dev_s', 'horizon_min', 'cur_x_horizon', 'cur_abs_s', 'cur_sign',
                 'target_rel_pos', 'state_rel_pos', 'stops_ahead', 'stops_ahead_log',
                 'plan_interval_s', 'route_len', 'plan_manual_target', 'plan_manual_rate')
PARAMETERS = dict(max_iter=160, learning_rate=0.035, max_leaf_nodes=12,
                  l2_regularization=30.0, min_samples_leaf=45, random_state=12012)


def load_plan(dataset: Path, split: str) -> pd.DataFrame:
    # There is deliberately no path to the factual test schedule here.
    paths = {'train': 'train/schedule.csv', 'validate': 'validate/schedule_plan.csv',
             'test': 'validate/schedule_plan.csv'}
    if split not in paths:
        raise ValueError('unsupported plan split')
    return pd.read_csv(dataset / paths[split], usecols=list(PLAN_COLUMNS), low_memory=False)


def build_plan_features(points: pd.DataFrame, plan: pd.DataFrame) -> pd.DataFrame:
    build_features(points)  # Preserve the existing point contract and horizon checks.
    if points.sample_id.isna().any() or points.sample_id.astype(str).duplicated().any():
        raise ValueError('sample_id must be non-null and unique')
    if set(plan.columns) != set(PLAN_COLUMNS):
        raise ValueError('plan must contain exactly the approved plan projection')
    if not np.isfinite(points.cur_dev_s.to_numpy(float)).all():
        raise ValueError('cur_dev_s must be finite')
    plan = plan.copy()
    plan['stop_key'] = plan.tt_action_item_id.astype(str)
    plan['tr_id'] = plan.tr_id.astype(str)
    plan['time_begin'] = pd.to_datetime(plan.time_begin, errors='raise')
    if plan[['tr_id', 'time_begin', 'stop_key']].isna().any().any():
        raise ValueError('plan keys must be non-null')
    plan = plan.sort_values(['tr_id', 'time_begin', 'stop_key'], kind='mergesort')
    routes = {}
    for tr, g in plan.groupby('tr_id', sort=False):
        times = g.time_begin.astype('int64').to_numpy() / 1e9
        manual = g.manual_fill.astype(bool).to_numpy()
        routes[tr] = times, manual, {k: i for i, k in enumerate(g.stop_key)}
    records = []
    for row in points.itertuples(index=False):
        t = pd.Timestamp(row.T); ts = t.value / 1e9
        h = (pd.Timestamp(row.target_time_begin) - t).total_seconds() / 60
        cur = float(row.cur_dev_s)
        record = dict(sample_id=str(row.sample_id), cur_dev_s=cur, horizon_s=h*60,
                      horizon_min=h, cur_x_horizon=cur*h, cur_abs_s=abs(cur), cur_sign=np.sign(cur))
        route = routes.get(str(row.tr_id))
        reason = 'missing_route' if route is None else ('missing_target' if str(row.target_stop_id) not in route[2] else '')
        record['fallback_reason'] = reason
        if reason:
            record.update({k: 0.0 for k in PLAN_FEATURES[5:]})
        else:
            times, manual, lookup = route; n = len(times); ti = lookup[str(row.target_stop_id)]
            state_i = int(np.searchsorted(times, ts, side='right') - 1)
            state_i = min(max(state_i, -1), n-1)
            assert state_i < 0 or times[state_i] <= ts
            ahead = max(ti-state_i, 0)
            record.update(target_rel_pos=ti/max(n-1,1), state_rel_pos=max(state_i,0)/max(n-1,1),
                          stops_ahead=ahead, stops_ahead_log=np.log1p(ahead),
                          plan_interval_s=max(times[ti]-(times[state_i] if state_i >= 0 else ts),0.0),
                          route_len=n, plan_manual_target=float(manual[ti]), plan_manual_rate=float(manual.mean()))
        records.append(record)
    result = pd.DataFrame(records)
    if not np.isfinite(result[list(PLAN_FEATURES)].to_numpy(float)).all():
        raise ValueError('plan features must be finite')
    return result


def fallback_audit(features: pd.DataFrame) -> dict:
    reasons = features.fallback_reason
    return dict(rows=len(features), fallback_rows=int(reasons.ne('').sum()),
                missing_route=int(reasons.eq('missing_route').sum()),
                missing_target=int(reasons.eq('missing_target').sum()))
