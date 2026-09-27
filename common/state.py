"""Типизированное состояние дашборда для OpenAPI и интеграции второго frontend."""
from typing import Literal
from pydantic import AwareDatetime, Field, StrictBool
from common.contracts import Contract, Prediction, PredictionRequest, ProbabilityStatus, StopTarget
from backend.arrivals import CurrentDeviation
from backend.engine import Route
from backend.diagnostics import Explanation
from backend.segment_observations import SegmentObservation
from backend.replay import ReplayState
from backend.generator_bridge import GeneratorView


class DemoState(Contract):
    running: bool
    source_enabled: bool
    speed: float


class HealthState(Contract):
    ml: Literal['ok','unavailable','starting']
    ndtp_connections: int
    source_status: str


class Summary(Contract):
    vehicles: int
    red: int
    amber: int
    stale: int
    unknown: int
    events: int


class Metrics(Contract):
    received: int
    duplicates: int
    invalid: int
    ndtp_errors: int
    unknown_units: int
    ml_failures: int
    learning_store_failures: int
    inference_p95_ms: float | None
    pipeline_p95_ms: float | None
    pipeline_cycles: int


class PredictionAvailability(Contract):
    code: Literal['ready', 'source_unavailable', 'no_target', 'telemetry_missing',
                  'telemetry_stale', 'deviation_expired', 'deviation_missing',
                  'target_reached', 'prediction_pending', 'model_input_unavailable']
    message: str = Field(description='Причина доступности прогноза на текущий момент; не физическая причина задержки')


class VehicleView(Contract):
    tr_id: int
    unit_id: int
    label: str
    route_id: str
    lat: float | None
    lon: float | None
    speed_kmh: float | None
    heading: float | None = Field(default=None, ge=0, lt=360,
        description='Курс в градусах от севера по часовой стрелке из того же пригодного GPS, что lat/lon; null — не передан')
    doors_open: StrictBool | None = None
    event_time: AwareDatetime | None
    age_s: float | None
    status: Literal['fresh','stale','no_data']
    cur_dev_s: float | None
    current_deviation: CurrentDeviation | None
    gps_detector: dict | None = None
    prediction_current_delay_s: float | None
    target: StopTarget | None
    prediction: Prediction | None
    prediction_availability: PredictionAvailability | None = None
    telemetry_sequence: int = Field(description='Число принятых недублирующихся сообщений ТС в текущем контексте')
    prediction_input_sequence: int | None = Field(description='Граница входа при расчёте опубликованного прогноза; не число валидных GPS и не признак модели')
    prediction_published_at: AwareDatetime | None = Field(description='Реальное серверное время публикации; issued_at — время отсечения признаков')
    reasons: list[str]
    recommendation: str
    section: str
    explanation: Explanation
    segment_observation: SegmentObservation | None = None


class Incident(Contract):
    id: str
    tr_id: int
    route_id: str
    created_at: AwareDatetime
    data_cutoff: AwareDatetime = Field(description='Момент отсечения данных, по которым рассчитан прогноз')
    published_at: AwareDatetime = Field(description='Время публикации по реальным часам сервера; created_at использует часы сценария в demo/replay')
    target_time: AwareDatetime
    target_name: str
    horizon_reference: Literal['scheduled_arrival'] = 'scheduled_arrival'
    estimated_arrival_at: AwareDatetime = Field(description='Плановое прибытие плюс прогноз задержки; это оценка, не факт')
    probability_late: float | None = Field(default=None, ge=0, le=1)
    probability_status: ProbabilityStatus = 'unavailable'
    probability_note: str | None = None
    risk: Literal['red']
    predicted_delay_s: float
    lead_time_s: float
    estimated_lead_time_s: float | None = Field(default=None, description='Оценённое прибытие минус время выдачи; не измеренный lead до факта')
    reason: str
    section: str
    model_version: str
    method: str
    explanation: Explanation


class ForecastTrace(Contract):
    """Сохранённый реальный обмен backend ↔ ML, не реконструкция из нового состояния."""
    context_version: int
    execution: Literal['ml_http', 'backend_fallback']
    current_deviation: CurrentDeviation | None
    request: PredictionRequest
    response: Prediction
    telemetry_sequence: int
    published_at: AwareDatetime


class IncidentExport(Contract):
    exported_at: AwareDatetime
    mode: Literal['demo','live','replay','generator']
    horizon_reference: Literal['scheduled_arrival'] = 'scheduled_arrival'
    note: str
    incidents: list[Incident]


class RiskAggregate(Contract):
    risk: Literal['green','amber','red','unknown']
    vehicles: int
    evaluated: int
    red: int
    amber: int
    unknown: int
    max_predicted_delay_s: float | None
    tr_ids: list[int]


class RouteRisk(RiskAggregate):
    route_id: str
    name: str


class SectionRisk(RiskAggregate):
    route_id: str
    section: str
    path: list[tuple[float,float]]
    interpretation: str


class DashboardState(Contract):
    server_time: AwareDatetime
    clock_time: AwareDatetime
    mode: Literal['demo','live','replay','generator']
    context: dict = Field(default_factory=dict)
    gps_arrivals: list[dict] = Field(default_factory=list)
    route_risks: list[RouteRisk] = Field(default_factory=list)
    section_risks: list[SectionRisk] = Field(default_factory=list)
    replay: ReplayState | None = None
    generator: GeneratorView | None = None
    demo: DemoState
    health: HealthState
    summary: Summary
    routes: list[Route]
    line_memberships: dict[str, list[str]] = Field(default_factory=dict,
        description='Для каждого плана — непосредственно совпадающие линии по ID или геометрии; без транзитивного объединения пересечений')
    vehicles: list[VehicleView]
    incidents: list[Incident]
    metrics: Metrics
