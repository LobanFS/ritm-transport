"""Point-in-time-safe feature generation.

The initial baseline uses only columns already known at forecast time T. The
feature-agent may add telemetry features here, but must cut every vehicle's
history at each row's T before aggregation.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .contracts import FEATURE_COLUMNS, FORBIDDEN_INFERENCE_COLUMNS, KEY_COLUMNS, POINT_COLUMNS


def build_features(points: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(POINT_COLUMNS) - set(points.columns))
    if missing:
        raise ValueError(f"features: missing point columns: {missing}")
    leaked = sorted(FORBIDDEN_INFERENCE_COLUMNS & set(points.columns))
    if leaked:
        raise ValueError(f"features: forbidden columns: {leaked}")

    t = pd.to_datetime(points["T"], errors="raise")
    target_t = pd.to_datetime(points["target_time_begin"], errors="raise")
    horizon_s = (target_t - t).dt.total_seconds()
    if not ((horizon_s > 600) & (horizon_s <= 900)).all():
        raise ValueError("features: forecast horizon must be in (10, 15] minutes")

    seconds_of_day = t.dt.hour * 3600 + t.dt.minute * 60 + t.dt.second
    angle = 2 * math.pi * seconds_of_day / 86400
    result = points.loc[:, list(KEY_COLUMNS)].copy()
    result["cur_dev_s"] = pd.to_numeric(points["cur_dev_s"], errors="raise")
    result["horizon_s"] = horizon_s
    result["t_hour_sin"] = np.sin(angle)
    result["t_hour_cos"] = np.cos(angle)
    result["t_weekday"] = t.dt.weekday
    if result[list(FEATURE_COLUMNS)].isna().any().any():
        raise ValueError("features: NaN values are not allowed")
    return result
