"""Runtime адаптер frozen Swiss Transformer prior + Moscow HGBR."""
from __future__ import annotations

from datetime import timezone
from functools import lru_cache
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
from threading import Lock
from typing import Any

from common.contracts import (
    Prediction, PredictionRequest, TransformerAnalysis, TransformerHistoryImpact, baseline,
)

MODEL_VERSION = "swiss-transformer-hgbr-prior-v1"
MODEL_SHA256 = "bfed9d9608fd707a634605900ac75d81e863309c29105c9b5183491965289905"
ENCODER_SHA256 = "729174c663a6fe4c1a1f657543f16538b9a8ff7b7d33cdff1c9c4f96f4303a73"
PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl", "torch")


def fallback(request: PredictionRequest, reason: str) -> Prediction:
    result = baseline(request)
    return result.model_copy(update={
        "method": "unavailable" if request.current_delay_s is None else "fallback",
        "reasons": result.reasons + [f"Резервный прогноз: {reason}"],
        "fallback_reason": reason,
    })


class LearnedModel:
    """Read-only гибрид; оба артефакта проверяются до десериализации."""

    def __init__(self, path: Path, encoder_path: Path | None = None,
        manifest_path: Path | None = None):
        model_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if manifest_path is None and model_digest != MODEL_SHA256:
            raise ValueError("SHA-256 model.joblib не совпадает с frozen hybrid artifact")
        encoder_path = encoder_path or path.with_name("encoder.pt")
        encoder_digest = hashlib.sha256(encoder_path.read_bytes()).hexdigest()
        from ml_service.frozen_plan.plan import PLAN_FEATURES
        feature_names = tuple(PLAN_FEATURES) + ("neural_prior",)
        manifest = None
        if manifest_path is not None:
            from ml_service.artifacts import load_manifest
            manifest = load_manifest(manifest_path, model_path=path, encoder_path=encoder_path,
                                     feature_names=feature_names)
        else:
            if encoder_digest != ENCODER_SHA256:
                raise ValueError("SHA-256 encoder.pt не совпадает с frozen Transformer artifact")
        vendor = Path(__file__).parent / "frozen_plan"
        provenance = json.loads((vendor / "provenance.json").read_text())
        for name, metadata in provenance["files"].items():
            if hashlib.sha256((vendor / name).read_bytes()).hexdigest() != metadata["sha256"]:
                raise ValueError(f"Frozen builder изменён: {name}")
        self.versions = {package: version(package) for package in PACKAGES}
        if self.versions["scikit-learn"] != "1.8.0":
            raise ValueError("Артефакт требует scikit-learn==1.8.0")

        import joblib
        payload = joblib.load(path)
        if (not isinstance(payload, dict) or payload.get("variant") != "prior"
                or payload.get("projection") is not None
                or tuple(payload.get("features", ())) != tuple(PLAN_FEATURES)):
            raise ValueError("Неожиданная схема hybrid_model.joblib")
        self.estimator = payload["model"]
        if getattr(self.estimator, "n_features_in_", None) != len(PLAN_FEATURES) + 1:
            raise ValueError("HGBR ожидает другую схему признаков")
        from ml_service.hybrid import load_transformer
        self.transformer = load_transformer(encoder_path)
        self.manifest = manifest
        self.model_version = manifest["model_version"] if manifest else MODEL_VERSION
        self.sha256 = model_digest
        self.encoder_sha256 = encoder_digest
        self._predict_lock = Lock()
        from ml_service.explainability import load_hybrid_explainer
        self.feature_names = feature_names
        reference_path = (manifest_path.with_name("explanation_reference.json") if manifest_path
                          else path.with_name("explanation_reference.json"))
        self.explainer = load_hybrid_explainer(
            self.estimator, reference_path,
            model_sha256=model_digest, encoder_sha256=encoder_digest,
            feature_names=self.feature_names,
        )
        from ml_service.probability import load_probability
        self.probability, self.probability_error = load_probability(model_digest, expected_encoder_sha256=encoder_digest)

    @staticmethod
    def input_frames(request: PredictionRequest):
        import pandas as pd
        from ml_service.frozen_plan.plan import PLAN_COLUMNS
        context = request.plan_context
        if context is None:
            return None, None, "Нет полного контекста плана"
        if not context.complete:
            return None, None, "Передан фрагмент плана"
        if any(stop.manual_fill is None for stop in context.stops):
            return None, None, "В плане неизвестен manual_fill"
        matches = [stop for stop in context.stops if stop.id == request.target.id]
        if len(matches) != 1 or matches[0].scheduled_at != request.target.scheduled_at:
            return None, None, "Цель отсутствует или не совпадает с версией плана"
        plan_rows = []
        for stop in context.stops:
            planned = stop.scheduled_at.astimezone(timezone.utc)
            plan_rows.append({"tt_action_item_id": stop.id, "time_begin": planned,
                "order_date": planned.date().isoformat(), "manual_fill": stop.manual_fill,
                "tr_id": request.tr_id, "geom": f"POINT ({stop.lon} {stop.lat})",
                "building_address": stop.name})
        point = {"sample_id": request.request_id, "tr_id": request.tr_id,
            "T": request.issued_at.astimezone(timezone.utc), "target_stop_id": request.target.id,
            "target_time_begin": request.target.scheduled_at.astimezone(timezone.utc),
            "cur_dev_s": request.current_delay_s}
        return pd.DataFrame([point]), pd.DataFrame(plan_rows, columns=PLAN_COLUMNS), None

    def features(self, request: PredictionRequest):
        from ml_service.frozen_plan.plan import build_plan_features
        if request.current_delay_s is None:
            return None, "Текущее отклонение неизвестно"
        points, plan, reason = self.input_frames(request)
        if reason:
            return None, reason
        result = build_plan_features(points, plan)
        reason = str(result.iloc[0].fallback_reason)
        return result, reason or None

    @staticmethod
    def sequence_inputs(requests, plan_matrix):
        import numpy as np
        from ml_service.hybrid import context_row, sequence_row
        sequences, contexts, counts, spans = [], [], [], []
        current_only_sequences, current_only_contexts = [], []
        for request, (_, features) in zip(requests, plan_matrix.iterrows(), strict=True):
            history = request.delay_history
            times = [item.observed_at for item in history] + [request.issued_at]
            delays = [item.delay_s for item in history] + [request.current_delay_s]
            sequence, count, span = sequence_row(times, delays, request.issued_at)
            sequences.append(sequence); contexts.append(context_row(features, count))
            counts.append(count); spans.append(span)
            only, _, _ = sequence_row([request.issued_at], [request.current_delay_s], request.issued_at)
            current_only_sequences.append(only); current_only_contexts.append(context_row(features, 1))
        return (np.stack(sequences), np.stack(contexts), np.asarray(counts), np.asarray(spans),
                np.stack(current_only_sequences), np.stack(current_only_contexts))

    @staticmethod
    def history_counterfactual_inputs(requests, plan_matrix):
        """Пересобрать sequence после удаления каждого прошлого наблюдения."""
        import numpy as np
        from ml_service.hybrid import CHANNELS, CONTEXT_FEATURES, STEPS, context_row, sequence_row
        sequences, contexts, currents, metadata = [], [], [], []
        for request_index, (request, (_, features)) in enumerate(
                zip(requests, plan_matrix.iterrows(), strict=True)):
            effective = [item for item in request.delay_history
                         if (request.issued_at-item.observed_at).total_seconds() <= 90*60]
            for history_index, removed in enumerate(effective):
                kept = effective[:history_index] + effective[history_index+1:]
                times = [item.observed_at for item in kept] + [request.issued_at]
                delays = [item.delay_s for item in kept] + [request.current_delay_s]
                sequence, count, _ = sequence_row(times, delays, request.issued_at)
                sequences.append(sequence); contexts.append(context_row(features, count))
                currents.append(request.current_delay_s)
                metadata.append((request_index, removed))
        if not sequences:
            return (np.empty((0, STEPS, CHANNELS), dtype=np.float32),
                    np.empty((0, CONTEXT_FEATURES), dtype=np.float32),
                    np.empty(0, dtype=float), metadata)
        return np.stack(sequences), np.stack(contexts), np.asarray(currents, dtype=float), metadata

    @staticmethod
    def transformer_analyses(requests, prior, current_only_prior, counterfactual_prior, metadata):
        """Собрать человекочитаемый анализ Transformer из честных counterfactuals."""
        import numpy as np
        impacts = [[] for _ in requests]
        for without_value, (request_index, observation) in zip(counterfactual_prior, metadata, strict=True):
            effect = float(prior[request_index] - without_value)
            direction = "neutral" if abs(effect) < 1e-9 else "increases" if effect > 0 else "decreases"
            request = requests[request_index]
            impacts[request_index].append(TransformerHistoryImpact(
                observed_at=observation.observed_at, delay_s=observation.delay_s,
                age_s=float((request.issued_at-observation.observed_at).total_seconds()),
                prior_effect_s=effect, direction=direction,
            ))
        analyses = []
        for index, request in enumerate(requests):
            effective = [item for item in request.delay_history
                         if (request.issued_at-item.observed_at).total_seconds() <= 90*60]
            if effective:
                values = np.asarray([item.delay_s for item in effective] + [request.current_delay_s], dtype=float)
                trend = float(values[-1]-values[0])
                recent = float(values[-1]-values[-2])
                volatility = float(np.mean(np.abs(np.diff(values))))
                history_effect = float(prior[index]-current_only_prior[index])
                if trend > 15 and recent > 15:
                    movement = (f"Опоздание растёт: с прошлого наблюдения +{recent:.0f} с, "
                                f"за доступную историю +{trend:.0f} с.")
                elif trend > 15 and recent < -15:
                    movement = (f"Опоздание начало сокращаться: с прошлого наблюдения {recent:.0f} с, "
                                f"но за доступную историю оно выросло на {trend:.0f} с.")
                elif trend < -15 and recent < -15:
                    movement = (f"Автобус сокращает опоздание: с прошлого наблюдения {recent:.0f} с, "
                                f"за доступную историю {trend:.0f} с.")
                elif trend < -15 and recent > 15:
                    movement = (f"После улучшения опоздание снова выросло: с прошлого наблюдения "
                                f"+{recent:.0f} с.")
                elif volatility > 30:
                    movement = "Опоздание меняется нестабильно между соседними наблюдениями."
                else:
                    movement = "Опоздание остаётся примерно на одном уровне."
                if abs(history_effect) < 1:
                    model_signal = "Модель опирается в основном на текущее отклонение."
                elif history_effect > 0:
                    model_signal = "Прошлая динамика усиливает прогноз задержки."
                else:
                    model_signal = "Прошлая динамика ослабляет прогноз задержки."
                summary = f"{movement} {model_signal}"
            else:
                trend = recent = volatility = None
                summary = "Недостаточно истории: прогноз рассчитан по текущему отклонению и плану."
            influential = sorted(impacts[index], key=lambda item: abs(item.prior_effect_s), reverse=True)[:3]
            analyses.append(TransformerAnalysis(
                method="leave_one_history_observation_out_v1", summary=summary,
                delay_trend_s=trend, recent_change_s=recent, step_volatility_s=volatility,
                influential_history=influential,
            ))
        return analyses

    def predict(self, request: PredictionRequest) -> Prediction:
        return self.predict_batch([request])[0]

    def predict_batch(self, requests: list[PredictionRequest]) -> list[Prediction]:
        import numpy as np
        import pandas as pd
        from ml_service.frozen_plan.plan import PLAN_FEATURES
        from ml_service.hybrid import transformer_prior
        result: list[Prediction | None] = [None] * len(requests)
        frames, indices = [], []
        for index, request in enumerate(requests):
            features, reason = self.features(request)
            if reason:
                result[index] = fallback(request, reason)
            else:
                frames.append(features); indices.append(index)
        if frames:
            selected = [requests[index] for index in indices]
            plan = pd.concat(frames, ignore_index=True).loc[:, list(PLAN_FEATURES)]
            seq, ctx, counts, spans, only_seq, only_ctx = self.sequence_inputs(selected, plan)
            cf_seq, cf_ctx, cf_current, cf_metadata = self.history_counterfactual_inputs(selected, plan)
            current = np.asarray([request.current_delay_s for request in selected], dtype=float)
            with self._predict_lock:
                prior = transformer_prior(self.transformer, seq, ctx, current)
                current_only_prior = transformer_prior(self.transformer, only_seq, only_ctx, current)
                cf_prior = (transformer_prior(self.transformer, cf_seq, cf_ctx, cf_current)
                            if len(cf_seq) else np.empty(0, dtype=float))
                hybrid = plan.copy(); hybrid["neural_prior"] = prior
                values, explanations = self.explainer.explain(hybrid, current)
            transformer_analyses = self.transformer_analyses(
                selected, prior, current_only_prior, cf_prior, cf_metadata)
            for local, (index, value, explanation) in enumerate(zip(indices, values, explanations, strict=True)):
                explanation = explanation.model_copy(update={
                    "history_points": int(counts[local]), "history_span_s": float(spans[local]),
                    "neural_prior_s": float(prior[local]),
                    "current_only_neural_prior_s": float(current_only_prior[local]),
                    "history_effect_on_neural_prior_s": float(prior[local] - current_only_prior[local]),
                    "transformer_analysis": transformer_analyses[local],
                })
                result[index] = self.response(requests[index], float(value), explanation)
        return result

    def response(self, request: PredictionRequest, value: float, explanation) -> Prediction:
        risk = "red" if value > 120 else "amber" if value > 60 else "green"
        reasons = ["Прогноз HGBR по плану и prior причинного Transformer"]
        if request.features.telemetry_age_s is None or request.features.telemetry_age_s > 60:
            risk = "unknown"; reasons.append("Нет свежей валидной телеметрии")
        probability = None
        probability_status = 'unavailable'
        note = "Калибратор вероятности не подключён для этой версии гибридной регрессии"
        if self.probability is not None:
            source_status = self.probability.source_status(request.current_delay_source,
                telemetry_domain=request.telemetry_domain,
                detector_version=request.current_delay_detector_version,
                detector_sha256=request.current_delay_detector_sha256)
            if source_status != 'unavailable':
                probability = self.probability.predict(value, current_delay_s=request.current_delay_s,
                                                       telemetry_age_s=request.features.telemetry_age_s)
                if probability is None:
                    note = "Нет свежих входных данных для оценки вероятности"
                elif source_status == 'transferred':
                    probability_status = source_status
                    note = ("Приближённая оценка: перенос калибровки на смешанный train; "
                            "качество здесь не проверено.")
                elif request.current_delay_source == 'gps_estimate':
                    probability_status = source_status
                    note = ("Оценка P(задержка >2 мин). Вторичная проверка на доступной реальной "
                            "GPS-истории; новый день и live-поток не проверены.")
                else:
                    probability_status = source_status
                    note = ("Оценка P(задержка >2 мин) по истории для этой версии Transformer + HGBR. "
                            "Калибратор обучен на vehicle-OOF, вторично проверен на выданном test; новый день не проверен.")
            else:
                note = self.probability.unavailable_source_note(
                    request.current_delay_source, telemetry_domain=request.telemetry_domain)
        elif self.probability_error:
            note = "Калибратор вероятности не прошёл проверку; прогноз секунд остаётся доступным"
        return Prediction(request_id=request.request_id, tr_id=request.tr_id, issued_at=request.issued_at,
            target=request.target, predicted_delay_s=value, model_version=self.model_version,
            method="learned", risk=risk, probability_late=probability,
            probability_status=probability_status, probability_note=note,
            reasons=reasons, forecast_explanation=explanation)


@lru_cache(maxsize=1)
def runtime() -> tuple[LearnedModel | None, str | None]:
    path = os.environ.get("MODEL_PATH")
    if not path:
        return None, None
    try:
        encoder = os.environ.get("ENCODER_PATH")
        manifest = os.environ.get("MODEL_MANIFEST_PATH")
        return LearnedModel(Path(path), Path(encoder) if encoder else None,
                            Path(manifest) if manifest else None), None
    except Exception as error:
        return None, f"{type(error).__name__}: {error}"
