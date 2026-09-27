"""Граница отдельного сценарного producer: атомарная проверка кадров до публикации.

Декодированные GPS идут по HTTP. Официальный NDTP остаётся независимым live-входом.
Сценарные причины/логи никогда не попадают в признаки или объяснение прогноза.
"""
from __future__ import annotations

import asyncio
import time
from typing import Literal
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StrictBool, model_validator

from backend.arrivals import ArrivalInput
from common.contracts import Contract, Telemetry

Scenario = Literal['normal', 'slow_segment', 'long_stop', 'gps_loss']


class GeneratorStart(Contract):
    scenario: Scenario = 'normal'
    seed: int = Field(default=42, ge=0, le=2147483647)
    speed: float = Field(default=10, ge=1, le=30)
    paused: bool = True
    route_count: int = Field(default=4, ge=1, le=20)
    telemetry_interval_s: int = Field(default=15, ge=1, le=30)
    door_sensors: StrictBool = False


class GeneratorControl(Contract):
    action: Literal['pause', 'resume', 'speed']
    speed: float | None = Field(default=None, ge=1, le=30)


class GeneratorView(Contract):
    session_id: str
    scenario: Scenario
    seed: int
    clock_time: AwareDatetime
    running: bool
    speed: float
    route_count: int = 4
    telemetry_interval_s: int = 15
    door_sensors: StrictBool = False
    scenario_vehicle_id: int | None = None
    finished: bool = False
    connected: bool = True
    received_frames: int = 0
    emitted_frames: int = 0
    received_telemetry: int = 0
    emitted_telemetry: int = 0
    emitted_arrivals: int = 0
    gap: bool = False
    error: str | None = None
    logs: list[dict] = Field(default_factory=list, max_length=100)
    transport: Literal['HTTP decoded telemetry'] = 'HTTP decoded telemetry'


class ProducerState(BaseModel):
    """Расширяемую метаинформацию producer игнорируем, данные кадров строги."""
    model_config = ConfigDict(extra='ignore', allow_inf_nan=False)
    session_id: str = Field(min_length=1, max_length=120)
    scenario: Scenario
    seed: int
    clock_time: AwareDatetime
    running: bool
    speed: float = Field(ge=1, le=30)
    route_count: int = Field(default=4, ge=1, le=20)
    telemetry_interval_s: int = Field(default=15, ge=1, le=30)
    door_sensors: StrictBool = False
    scenario_vehicle_id: int | None = None
    finished: bool = False
    emitted_telemetry: int = Field(default=0, ge=0)
    emitted_arrivals: int = Field(default=0, ge=0)
    logs: list[dict] = Field(default_factory=list, max_length=100)


class GeneratorFrame(Contract):
    seq: int = Field(ge=1)
    clock_time: AwareDatetime
    telemetry: list[Telemetry] = Field(default_factory=list, max_length=128)
    arrivals: list[ArrivalInput] = Field(default_factory=list, max_length=128)

    @model_validator(mode='after')
    def causal(self):
        if any(max(p.event_time, p.received_at) > self.clock_time for p in self.telemetry):
            raise ValueError('Будущая телеметрия в кадре генератора')
        if any(a.arrived_at > self.clock_time for a in self.arrivals):
            raise ValueError('Будущий факт прибытия в кадре генератора')
        return self


class GeneratorStream(ProducerState):
    frames: list[GeneratorFrame] = Field(max_length=600)
    first_seq: int = Field(ge=0)
    last_seq: int = Field(ge=0)
    gap: bool


class GeneratorSession:
    def __init__(self, status: ProducerState, *, monotonic=None):
        self.cursor = 0
        self.view = GeneratorView(**status.model_dump(exclude={'logs'}), logs=status.logs)
        self.poll_lock = asyncio.Lock()
        self._monotonic = monotonic or time.monotonic
        self._last_progress_at = self._monotonic()
        self._expect_progress = status.running and not status.finished

    def consume(self, engine, payload):
        """Сначала проверить весь пакет; при разрыве последовательности нужен новый сценарий."""
        stream = GeneratorStream.model_validate(payload)
        if stream.session_id != self.view.session_id:
            raise ValueError('Генератор сменил сессию; запустите сценарий заново')
        if stream.gap or stream.first_seq > self.cursor+1:
            self.view.gap = True
            raise ValueError('Пропущены кадры генератора; запустите сценарий заново')
        if stream.last_seq < self.cursor or stream.clock_time < engine.clock:
            raise ValueError('Генератор откатил время или счётчик')
        expected, at = self.cursor+1, engine.clock
        for frame in stream.frames:
            if frame.arrivals:
                raise ValueError('Истинные прибытия генератора запрещены во входном потоке; используйте GPS')
            if frame.seq != expected or not at <= frame.clock_time <= stream.clock_time:
                raise ValueError('Нарушен порядок кадров генератора')
            for point in frame.telemetry:
                vehicle = engine.vehicles.get(point.tr_id)
                if vehicle is None or point.unit_id != vehicle.unit_id:
                    raise ValueError('Генератор прислал неизвестное ТС/терминал')
            expected += 1
            at = frame.clock_time
        if expected-1 != stream.last_seq:
            raise ValueError('Генератор не вернул все кадры до last_seq')
        expects_progress = stream.running and not stream.finished
        if stream.clock_time > self.view.clock_time or not expects_progress or not self._expect_progress:
            self._last_progress_at = self._monotonic()
        self._expect_progress = expects_progress
        for frame in stream.frames:
            engine.clock = frame.clock_time
            for point in frame.telemetry:
                engine.ingest(point.model_copy(update={'source':'generator', 'received_at':frame.clock_time}))
            self.cursor = frame.seq
        engine.clock = stream.clock_time
        engine.running = stream.running
        engine.speed = stream.speed
        self.view = GeneratorView(**stream.model_dump(include=set(ProducerState.model_fields)),
            received_frames=self.cursor, emitted_frames=stream.last_seq,
            received_telemetry=sum(engine.input_sequences.values()), connected=True)
        self.check_progress()

    def check_progress(self):
        """Живой HTTP не означает движение сценария: watchdog использует реальные секунды."""
        if self._expect_progress and self.view.running and not self.view.finished:
            if self._monotonic()-self._last_progress_at >= 5:
                self.fail('Часы запущенного генератора не движутся более 5 секунд; показано последнее состояние.')
        return self.view.connected

    def fail(self, message):
        self.view.connected = False
        self.view.error = message
