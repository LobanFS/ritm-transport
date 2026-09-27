"""Ответ read-only сценария подачи автобуса с соседнего плана."""
from typing import Literal

from pydantic import AwareDatetime, Field

from common.contracts import Contract, StopTarget


class TransferVehicle(Contract):
    tr_id: int
    label: str
    route_id: str
    route_name: str


class TransferScenario(Contract):
    distance_m: int = Field(ge=0, description='Расстояние по прямой от GPS до остановки подачи')
    relocation_s: int = Field(ge=0, description='Сценарий: 1.4 × расстояние, 20 км/ч и 120 с на подготовку; не дорожный ETA')
    donor_arrival_at: AwareDatetime
    service_at: AwareDatetime = Field(description='Обслуживание не раньше планового времени целевой остановки')
    current_bus_arrival_at: AwareDatetime
    earlier_by_s: int = Field(ge=0, description='Насколько раньше дополнительный автобус может обслужить эту остановку; прогноз задержки исходного ТС не меняется')


class DonorImpact(Contract):
    next_stop_name: str | None
    next_stop_at: AwareDatetime | None
    planned_stops_during_transfer: int = Field(ge=0)
    plan_conflict: bool
    availability: Literal['requires_dispatcher_confirmation'] = 'requires_dispatcher_confirmation'
    note: str


class TransferAdvice(Contract):
    status: Literal['ready', 'not_needed', 'unavailable']
    reason: str
    calculated_at: AwareDatetime
    context_version: int
    algorithm_version: str = 'nearby-plan-scenario-v1'
    target: TransferVehicle
    donor: TransferVehicle | None = None
    meeting_stop: StopTarget | None = None
    scenario: TransferScenario | None = None
    donor_impact: DonorImpact | None = None
    assumptions: list[str] = Field(default_factory=list)
    candidates_considered: int = 0
