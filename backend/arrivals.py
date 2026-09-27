"""Факт посещения плановой остановки и происхождение текущего отклонения.

NDTP Nav00 не содержит событие прибытия. ArrivalInput подаёт отдельный
операционный источник; определение прибытия по GPS здесь не подменяется эвристикой.
"""
from typing import Literal
from pydantic import AwareDatetime, Field, FiniteFloat
from common.contracts import Contract


class ArrivalInput(Contract):
    """Одно подтверждённое прибытие; время получения назначает backend."""
    tr_id: int = Field(gt=0)
    planned_stop_id: str = Field(min_length=1, max_length=120)
    arrived_at: AwareDatetime


class CurrentDeviation(Contract):
    """Наблюдение последнего известного отклонения, не прогноз будущей остановки."""
    delay_s: FiniteFloat
    source: Literal['arrival', 'demo_arrival', 'generator_arrival', 'gps_estimate', 'door_estimate', 'external_hint', 'csv_snapshot', 'synthetic_hint']
    sample_id: str | None = None
    observed_at: AwareDatetime
    received_at: AwareDatetime
    planned_stop_id: str | None = None
    stop_name: str | None = None
    planned_at: AwareDatetime | None = None
    valid_until: AwareDatetime | None = None
    uncertainty_s: float | None = Field(default=None, ge=0, description='Разрешение оценки по GPS, не калиброванный доверительный интервал')
    confirmation_span_s: float | None = Field(default=None, ge=0)


class ArrivalConflict(ValueError):
    """Прибытие на этот плановый пункт уже зарегистрировано с другим временем."""
