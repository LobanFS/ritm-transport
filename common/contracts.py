"""Общий версионируемый контракт backend ↔ ML; все времена с явным UTC offset."""
from datetime import datetime
from typing import Literal
from zoneinfo import ZoneInfo
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, FiniteFloat, StrictBool, model_validator
from common.risk import risk_for_delay


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


ProbabilityStatus = Literal['validated', 'transferred', 'unavailable']


def parse_manual_fill(value: str | None) -> bool | None:
    """Явная семантика CSV: строка 'False' не превращается в bool('False') == True."""
    if value is None or not value.strip():
        return None
    normalized = value.strip().lower()
    if normalized in ('true', '1'):
        return True
    if normalized in ('false', '0'):
        return False
    raise ValueError('manual_fill должен быть True/False/1/0 или отсутствовать')


class Telemetry(Contract):
    tr_id: int = Field(gt=0)
    unit_id: int | None = Field(default=None, ge=0)
    event_time: AwareDatetime
    received_at: AwareDatetime
    lat: float | None = Field(default=None, ge=-90, le=90)
    lon: float | None = Field(default=None, ge=-180, le=180)
    speed_kmh: float | None = Field(default=None, ge=0, le=400)
    heading: float | None = Field(default=None, ge=0, le=360)
    location_valid: bool = True
    doors_open: StrictBool | None = Field(default=None, description='Есть открытая дверь по присутствующим датчикам; null — датчик/состояние неизвестны. Не готовое событие прибытия.')
    door_sensor_key: str | None = Field(default=None, min_length=1, max_length=120, description='Идентификатор набора присутствующих датчиков. Переход closed→open используется только при одинаковом известном наборе.')
    source: Literal["demo", "ndtp", "replay", "http", "generator"] = "http"
    event_id: str | None = Field(default=None, max_length=120)


class StopTarget(Contract):
    id: str = Field(min_length=1, max_length=120)
    name: str = Field(min_length=1, max_length=120)
    scheduled_at: AwareDatetime
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    manual_fill: bool | None = Field(default=None, description='Плановое поле; null означает неизвестное, а не false')


class ModelPlanContext(Contract):
    """Полный разрешённый план одного ТС для frozen plan-context builder.

    Все строки — план, включая будущие плановые посещения, без фактов прибытия.
    complete подтверждает полноту исходного плана, а не наличие GPS для всех остановок.
    """
    version: str = Field(min_length=1, max_length=160)
    timezone: str = Field(min_length=1, max_length=80)
    complete: bool
    stops: list[StopTarget] = Field(min_length=1)

    @model_validator(mode='after')
    def valid_plan(self):
        try:
            ZoneInfo(self.timezone)
        except (ValueError, KeyError) as exc:
            raise ValueError('Неизвестный часовой пояс плана IANA') from exc
        if len({s.id for s in self.stops}) != len(self.stops):
            raise ValueError('Повтор ID посещения в плане модели')
        return self


class Features(Contract):
    telemetry_age_s: float | None = Field(default=None, ge=0)
    valid_points_5m: int = Field(default=0, ge=0)
    speed_mean_5m: float | None = Field(default=None, ge=0, le=150)
    stopped_share_5m: float | None = Field(default=None, ge=0, le=1)
    distance_to_target_m: float | None = Field(default=None, ge=0)


class DelayObservation(Contract):
    """Наблюдение известного тогда отклонения, строго предшествующее issued_at."""
    observed_at: AwareDatetime
    delay_s: FiniteFloat


class PredictionRequest(Contract):
    request_id: str = Field(min_length=1, max_length=200)
    tr_id: int = Field(gt=0)
    issued_at: AwareDatetime
    target: StopTarget
    current_delay_s: FiniteFloat | None = None
    current_delay_source: Literal['arrival','demo_arrival','generator_arrival','gps_estimate','door_estimate','external_hint','csv_snapshot','synthetic_hint'] | None = None
    current_delay_detector_version: str | None = Field(default=None, max_length=120)
    current_delay_detector_sha256: str | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')
    telemetry_domain: Literal['historical_real', 'historical_mixed', 'live_unverified', 'synthetic', 'unknown'] = Field(
        default='unknown', description='Происхождение входов, выставленное backend; область проверки вероятности не переносится на синтетику автоматически')
    features: Features = Field(default_factory=Features)
    plan_context: ModelPlanContext | None = None
    delay_history: list[DelayObservation] = Field(
        default_factory=list, max_length=11,
        description='До 11 причинно доступных наблюдений перед issued_at; текущий cur_dev_s добавляет ML-модуль.',
    )

    @model_validator(mode="after")
    def valid_horizon(self):
        seconds = (self.target.scheduled_at - self.issued_at).total_seconds()
        if not 600 < seconds <= 900:
            raise ValueError("Цель должна находиться в окне (T+10, T+15] минут")
        if self.plan_context is not None:
            planned = next((s for s in self.plan_context.stops if s.id == self.target.id), None)
            if planned is None or planned != self.target:
                raise ValueError('Цель должна совпадать с посещением в плане модели')
        times = [item.observed_at for item in self.delay_history]
        if any(value >= self.issued_at for value in times):
            raise ValueError('История отклонения должна быть строго раньше issued_at')
        if times != sorted(times) or len(times) != len(set(times)):
            raise ValueError('История отклонения должна быть строго упорядочена без повторов')
        return self


class ForecastFactor(Contract):
    """Локальный вклад группы признаков в численный прогноз.

    Вклад объясняет поведение модели относительно зафиксированного опорного
    примера. Он не устанавливает физическую причину задержки.
    """
    code: Literal[
        "current_state", "forecast_horizon", "route_position",
        "remaining_path", "plan_context", "sequence_history",
    ]
    title: str
    effect_s: FiniteFloat
    direction: Literal["increases", "decreases", "neutral"]


class TransformerHistoryImpact(Contract):
    """Counterfactual-вклад одного прошлого наблюдения во внутренний prior."""
    observed_at: AwareDatetime
    delay_s: FiniteFloat
    age_s: FiniteFloat = Field(ge=0)
    prior_effect_s: FiniteFloat
    direction: Literal["increases", "decreases", "neutral"]


class TransformerAnalysis(Contract):
    """Анализ временного сигнала Transformer, без заявления о физической причине."""
    method: Literal["leave_one_history_observation_out_v1"]
    summary: str
    delay_trend_s: FiniteFloat | None = None
    recent_change_s: FiniteFloat | None = None
    step_volatility_s: FiniteFloat | None = Field(default=None, ge=0)
    influential_history: list[TransformerHistoryImpact] = Field(default_factory=list, max_length=3)
    interpretation: str = (
        "Это чувствительность Transformer к истории cur_dev_s, а не доказательство "
        "пробки, ДТП, посадки или другой физической причины."
    )


class ForecastExplanation(Contract):
    """Проверяемое аддитивное разложение результата регрессии."""
    method: Literal["exact_grouped_shapley_reference_v1"]
    reference: str
    reference_prediction_s: FiniteFloat
    current_delay_s: FiniteFloat
    prediction_s: FiniteFloat
    model_adjustment_s: FiniteFloat
    reconstructed_prediction_s: FiniteFloat
    reconstruction_error_s: FiniteFloat
    factors: list[ForecastFactor]
    history_points: int = Field(default=1, ge=1, le=12)
    history_span_s: FiniteFloat = Field(default=0, ge=0)
    neural_prior_s: FiniteFloat | None = None
    current_only_neural_prior_s: FiniteFloat | None = None
    history_effect_on_neural_prior_s: FiniteFloat | None = None
    transformer_analysis: TransformerAnalysis | None = None
    interpretation: str = (
        "Вклады описывают чувствительность модели относительно опорного "
        "train-примера и не являются доказанными причинами задержки."
    )


class Prediction(Contract):
    request_id: str
    tr_id: int
    issued_at: AwareDatetime
    target: StopTarget
    predicted_delay_s: FiniteFloat | None
    model_version: str
    method: Literal["persistence", "fallback", "unavailable", "learned"]
    risk: Literal["green", "amber", "red", "unknown"]
    probability_late: float | None = Field(default=None, ge=0, le=1)
    probability_status: ProbabilityStatus = Field(default='unavailable', description='validated — внутри проверенной области; transferred — приближённый перенос на смешанный train; unavailable — оценки нет')
    probability_note: str | None = None
    fallback_reason: str | None = None
    reasons: list[str]
    forecast_explanation: ForecastExplanation | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


def baseline(request: PredictionRequest, *, fallback: bool = False) -> Prediction:
    """Перенос выданного текущего отставания; не обученная модель и не вероятность."""
    value = request.current_delay_s
    risk = risk_for_delay(value)
    reasons = ["Текущее отставание не известно"] if value is None else ["Перенос текущего отклонения от расписания"]
    if request.features.telemetry_age_s is None or request.features.telemetry_age_s > 60:
        risk = "unknown"
        reasons.append("Нет свежей валидной телеметрии")
    if fallback:
        reasons.append("ML-сервис недоступен: резервный baseline")
    return Prediction(request_id=request.request_id, tr_id=request.tr_id,
        issued_at=request.issued_at, target=request.target, predicted_delay_s=value,
        model_version="persistence-v1", method="unavailable" if value is None else "fallback" if fallback else "persistence",
        risk=risk, probability_late=None, reasons=reasons)
