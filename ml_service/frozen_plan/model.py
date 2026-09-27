"""Model interface and the initial cur_dev_s baseline implementation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import __version__


@dataclass
class CurrentDeviationModel:
    model_type: str = "current_deviation"
    version: str = __version__
    fitted_rows: int = 0

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "CurrentDeviationModel":
        if len(features) != len(target):
            raise ValueError("model: features and target lengths differ")
        self.fitted_rows = len(features)
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if "cur_dev_s" not in features:
            raise ValueError("model: cur_dev_s feature is required")
        result = pd.to_numeric(features["cur_dev_s"], errors="raise").astype(float)
        result.name = "prediction"
        return result

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "model_type": self.model_type,
                    "version": self.version,
                    "fitted_rows": self.fitted_rows,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "CurrentDeviationModel":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("model_type") != "current_deviation":
            raise ValueError("model artifact has an unsupported model_type")
        return cls(**payload)


class PlanContextModel:
    """Fixed shortlist estimator. Deserialize only locally trusted artifacts."""
    def __init__(self):
        from sklearn.ensemble import HistGradientBoostingRegressor
        from .plan import PARAMETERS
        self.estimator = HistGradientBoostingRegressor(**PARAMETERS)

    def fit(self, features, target):
        from .plan import PLAN_FEATURES
        from threadpoolctl import threadpool_limits
        import numpy as np
        if len(features) != len(target) or not np.isfinite(target.to_numpy(float)).all():
            raise ValueError('invalid training target')
        if features.fallback_reason.ne('').any():
            raise ValueError('training plan incomplete; retain baseline')
        with threadpool_limits(limits=1):
            self.estimator.fit(features[list(PLAN_FEATURES)], target)
        return self

    def predict(self, features):
        from .plan import PLAN_FEATURES
        from threadpoolctl import threadpool_limits
        import numpy as np
        result = features.cur_dev_s.astype(float).copy()
        available = features.fallback_reason.eq('')
        if available.any():
            with threadpool_limits(limits=1):
                result.loc[available] = self.estimator.predict(features.loc[available, list(PLAN_FEATURES)])
        if not np.isfinite(result).all():
            raise ValueError('predictions must be finite')
        return result.rename('prediction')

    def save(self, path):
        import joblib
        from .plan import PLAN_FEATURES, PARAMETERS
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(dict(model=self.estimator, features=PLAN_FEATURES, parameters=PARAMETERS), path)

    @classmethod
    def load(cls, path):
        import joblib
        from .plan import PLAN_FEATURES, PARAMETERS
        payload = joblib.load(path)
        if payload['features'] != PLAN_FEATURES or payload['parameters'] != PARAMETERS:
            raise ValueError('model artifact does not match frozen recipe')
        result = cls(); result.estimator = payload['model']
        return result
