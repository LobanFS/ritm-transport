"""Точное групповое разложение frozen HGBR без новой ML-зависимости.

Это interventional Shapley относительно одного реального опорного train-примера.
Все семантические коалиции считаются одним batch-вызовом estimator.predict.
Производные признаки восстанавливаются после каждой подстановки, поэтому
``cur_abs_s``, ``cur_sign``, ``cur_x_horizon`` и ``stops_ahead_log`` остаются
согласованными. Результат объясняет число модели, но не физическую причину.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import json
import math
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from common.contracts import ForecastExplanation, ForecastFactor


@dataclass(frozen=True)
class FeatureGroup:
    code: str
    title: str
    features: tuple[str, ...]


PLAN_GROUPS = (
    FeatureGroup("current_state", "Текущее отклонение", ("cur_dev_s",)),
    FeatureGroup("forecast_horizon", "Горизонт прогноза", ("horizon_min",)),
    FeatureGroup("route_position", "Положение на маршруте", ("target_rel_pos", "state_rel_pos")),
    FeatureGroup("remaining_path", "Оставшийся путь", ("stops_ahead", "plan_interval_s")),
    FeatureGroup("plan_context", "Структура плана", ("route_len", "plan_manual_target", "plan_manual_rate")),
)
HYBRID_GROUPS = PLAN_GROUPS + (
    FeatureGroup("sequence_history", "Историческая динамика задержки", ("neural_prior",)),
)
PLAN_DERIVED = ("cur_x_horizon", "cur_abs_s", "cur_sign", "stops_ahead_log")


def restore_plan_relations(frame: pd.DataFrame) -> None:
    """Восстановить точные зависимости feature builder после подстановки."""
    frame["cur_abs_s"] = frame["cur_dev_s"].abs()
    frame["cur_sign"] = np.sign(frame["cur_dev_s"])
    frame["cur_x_horizon"] = frame["cur_dev_s"] * frame["horizon_min"]
    frame["stops_ahead_log"] = np.log1p(frame["stops_ahead"])


class GroupedShapleyExplainer:
    """Exact Shapley для небольшого числа осмысленных feature-групп."""

    def __init__(
        self,
        *,
        estimator,
        feature_names: Sequence[str],
        groups: Sequence[FeatureGroup],
        derived_features: Sequence[str],
        reference: dict[str, float],
        reference_description: str,
        repair: Callable[[pd.DataFrame], None] | None = None,
    ) -> None:
        self.estimator = estimator
        self.feature_names = tuple(feature_names)
        self.groups = tuple(groups)
        self.reference_description = reference_description
        self.repair = repair
        group_features = [name for group in groups for name in group.features]
        covered = group_features + list(derived_features)
        if len(group_features) != len(set(group_features)):
            raise ValueError("feature входит более чем в одну explainability-группу")
        if set(covered) != set(self.feature_names) or len(covered) != len(self.feature_names):
            raise ValueError("explainability-группы не покрывают frozen features ровно один раз")
        if tuple(reference) != self.feature_names:
            raise ValueError("порядок reference features не совпадает с frozen model")
        self.reference = np.asarray([reference[name] for name in self.feature_names], dtype=float)
        if not np.isfinite(self.reference).all():
            raise ValueError("reference содержит нечисловые значения")
        self._indices = {
            group.code: [self.feature_names.index(name) for name in group.features]
            for group in self.groups
        }

    def explain(self, frame: pd.DataFrame, current_delays: Sequence[float]) -> tuple[np.ndarray, list[ForecastExplanation]]:
        """Вернуть исходные predictions и их аддитивные объяснения."""
        if list(frame.columns) != list(self.feature_names):
            raise ValueError("explanation frame должен иметь точный frozen feature order")
        values = frame.to_numpy(float)
        current = np.asarray(current_delays, dtype=float)
        if len(values) != len(current) or not np.isfinite(values).all() or not np.isfinite(current).all():
            raise ValueError("невалидная матрица для объяснения")
        if not len(values):
            return np.empty(0), []

        coalition_frames = []
        coalition_count = 1 << len(self.groups)
        for mask in range(coalition_count):
            mixed = np.tile(self.reference, (len(values), 1))
            for index, group in enumerate(self.groups):
                if mask & (1 << index):
                    columns = self._indices[group.code]
                    mixed[:, columns] = values[:, columns]
            candidate = pd.DataFrame(mixed, columns=self.feature_names)
            if self.repair is not None:
                self.repair(candidate)
            coalition_frames.append(candidate)
        scores = np.asarray(
            self.estimator.predict(pd.concat(coalition_frames, ignore_index=True)), dtype=float
        ).reshape(coalition_count, len(values))
        if not np.isfinite(scores).all():
            raise ValueError("estimator вернул нечисловое объяснение")

        contributions = np.zeros((len(values), len(self.groups)), dtype=float)
        n = len(self.groups)
        denominator = math.factorial(n)
        for group_index in range(n):
            others = [index for index in range(n) if index != group_index]
            for size in range(n):
                weight = math.factorial(size) * math.factorial(n-size-1) / denominator
                for subset in combinations(others, size):
                    mask = sum(1 << index for index in subset)
                    contributions[:, group_index] += weight * (
                        scores[mask | (1 << group_index)] - scores[mask]
                    )

        predictions = scores[-1]
        baseline = scores[0]
        explanations = []
        for row in range(len(values)):
            factors = []
            for group, effect in zip(self.groups, contributions[row], strict=True):
                direction = "neutral" if abs(effect) < 1e-9 else "increases" if effect > 0 else "decreases"
                factors.append(ForecastFactor(
                    code=group.code, title=group.title,
                    effect_s=float(effect), direction=direction,
                ))
            reconstructed = float(baseline[row] + contributions[row].sum())
            explanations.append(ForecastExplanation(
                method="exact_grouped_shapley_reference_v1",
                reference=self.reference_description,
                reference_prediction_s=float(baseline[row]),
                current_delay_s=float(current[row]),
                prediction_s=float(predictions[row]),
                model_adjustment_s=float(predictions[row] - current[row]),
                reconstructed_prediction_s=reconstructed,
                reconstruction_error_s=float(predictions[row] - reconstructed),
                factors=factors,
            ))
        return predictions, explanations


def load_hybrid_explainer(estimator, path: Path, *, model_sha256: str,
                          encoder_sha256: str, feature_names: Sequence[str]) -> GroupedShapleyExplainer:
    """Загрузить привязанный к конкретным весам реальный reference-пример."""
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != "hybrid-explanation-reference-v1":
        raise ValueError("неподдерживаемая версия explanation reference")
    if payload.get("model_sha256") != model_sha256:
        raise ValueError("explanation reference относится к другим весам")
    if payload.get("encoder_sha256") != encoder_sha256:
        raise ValueError("explanation reference относится к другому encoder")
    if tuple(payload.get("feature_names", ())) != tuple(feature_names):
        raise ValueError("explanation reference не совпадает со схемой модели")
    return GroupedShapleyExplainer(
        estimator=estimator,
        feature_names=feature_names,
        groups=HYBRID_GROUPS,
        derived_features=PLAN_DERIVED,
        reference=payload["values"],
        reference_description=payload["description"],
        repair=restore_plan_relations,
    )
