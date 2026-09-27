"""Ограниченное состояние потока, причинные признаки и оркестрация инференса."""
from __future__ import annotations

import math
import hashlib
import sqlite3
import time
from collections import OrderedDict, defaultdict, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import httpx
from pydantic import AwareDatetime, Field, FiniteFloat, model_validator

from common.contracts import Contract, DelayObservation, Features, ModelPlanContext, Prediction, PredictionRequest, StopTarget, Telemetry, baseline
from common.risk import RISK_POLICY, risk_for_delay
from backend.diagnostics import explain, usable_history
from backend.arrivals import ArrivalConflict, ArrivalInput, CurrentDeviation


def utcnow():
    return datetime.now(timezone.utc)


class VehicleConfig(Contract):
    tr_id: int = Field(gt=0)
    unit_id: int = Field(ge=0, le=2147483647)
    label: str = Field(min_length=1, max_length=80)
    route_id: str = Field(min_length=1, max_length=80)


class RouteStop(Contract):
    id: str
    name: str
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


class Route(Contract):
    route_id: str
    name: str
    color: str = Field(pattern=r"^#[0-9a-fA-F]{6}$")
    path: list[tuple[float, float]] = Field(min_length=2)
    stops: list[RouteStop] = Field(default_factory=list)

    @model_validator(mode="after")
    def coordinates(self):
        if any(not (-180 <= lon <= 180 and -90 <= lat <= 90) for lon, lat in self.path):
            raise ValueError("Координаты path должны быть [lon,lat]")
        return self


class ScheduledStop(Contract):
    tr_id: int
    target: StopTarget


class DelayHint(Contract):
    tr_id: int
    observed_at: AwareDatetime
    delay_s: FiniteFloat
    received_at: AwareDatetime | None = None
    source: Literal['external_hint', 'csv_snapshot', 'synthetic_hint'] = 'external_hint'
    sample_id: str | None = None
    target_stop_id: str | None = None
    target_time_begin: AwareDatetime | None = None

    @model_validator(mode='after')
    def supplied_target(self):
        if (self.target_stop_id is None) != (self.target_time_begin is None):
            raise ValueError('Цель подсказки требует одновременно ID и плановое время')
        if self.target_time_begin and not 600 < (self.target_time_begin-self.observed_at).total_seconds() <= 900:
            raise ValueError('Выданная цель должна быть в плановом окне (10,15] минут на момент snapshot')
        return self


class LiveContext(Contract):
    plan_version: str | None = Field(default=None, min_length=1, max_length=160)
    plan_timezone: str | None = Field(default=None, min_length=1, max_length=80)
    plan_complete: bool = Field(default=False, description='Передан весь доступный исходный план выбранных ТС, без обрезки по окну replay')
    arrival_mode: Literal['gps', 'external'] = Field(default='external', description='gps — экспериментальная оценка, не проверенный операционный источник; generator включает её отдельно')
    vehicles: list[VehicleConfig] = Field(min_length=1)
    routes: list[Route] = Field(default_factory=list)
    schedule: list[ScheduledStop] = Field(default_factory=list)
    hints: list[DelayHint] = Field(default_factory=list)

    @model_validator(mode="after")
    def references(self):
        if self.plan_complete and (not self.plan_version or not self.plan_timezone):
            raise ValueError('Для полного плана нужны версия и часовой пояс')
        if self.plan_timezone:
            from zoneinfo import ZoneInfo
            try:
                ZoneInfo(self.plan_timezone)
            except (ValueError, KeyError) as exc:
                raise ValueError('Неизвестный часовой пояс плана') from exc
        ids = {v.tr_id for v in self.vehicles}
        if len(ids) != len(self.vehicles) or len({v.unit_id for v in self.vehicles}) != len(ids):
            raise ValueError("tr_id и unit_id должны быть уникальны")
        if len({r.route_id for r in self.routes}) != len(self.routes):
            raise ValueError("Повтор route_id")
        if any(s.tr_id not in ids for s in self.schedule) or any(h.tr_id not in ids for h in self.hints):
            raise ValueError("Расписание/подсказки ссылаются на неизвестное ТС")
        if len({(s.tr_id, s.target.id) for s in self.schedule}) != len(self.schedule):
            raise ValueError("Повтор ID планового прибытия")
        for hint in self.hints:
            if hint.target_stop_id is not None:
                targets = [s.target for s in self.schedule if s.tr_id == hint.tr_id
                           and 600 < (s.target.scheduled_at-hint.observed_at).total_seconds() <= 900]
                first_at = min((s.scheduled_at for s in targets), default=None)
                if not any(s.id == hint.target_stop_id and s.scheduled_at == hint.target_time_begin == first_at
                           for s in targets):
                    raise ValueError('Выданная цель не принадлежит первым плановым посещениям в окне')
        return self


def features_at(history, at: datetime, target: StopTarget) -> tuple[Features, Telemetry | None]:
    """Оба времени ≤ T. Поздние сообщения и будущие события не входят в признаки."""
    valid = usable_history(history, at)
    latest = max(valid, key=lambda x: x.event_time, default=None)
    recent = [x for x in valid if x.event_time >= at-timedelta(minutes=5)]
    speeds = [x.speed_kmh for x in recent if x.speed_kmh is not None]
    distance = None
    if latest:
        lat1, lat2 = math.radians(latest.lat), math.radians(target.lat)
        dlat, dlon = lat2-lat1, math.radians(target.lon-latest.lon)
        a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
        distance = 6371000 * 2 * math.asin(min(1, math.sqrt(a)))
    return Features(telemetry_age_s=(at-latest.event_time).total_seconds() if latest else None,
                    valid_points_5m=len(recent), speed_mean_5m=sum(speeds)/len(speeds) if speeds else None,
                    stopped_share_5m=sum(s < 1 for s in speeds)/len(speeds) if speeds else None,
                    distance_to_target_m=distance), latest


def gate_prediction(prediction: Prediction, *, telemetry_age_s: float | None,
                    has_current_deviation: bool, source_available: bool = True,
                    target_reached: bool = False) -> Prediction:
    """Актуальность на момент показа/алерта, отдельно от набора входов ML.

Сохраняем число и время прежнего расчёта, но просроченные входы не должны
поддерживать текущую вероятность или уровень риска до следующего инференса.
Исходный ответ в forecast trace остаётся неизменным.
"""
    reasons = []
    retained = '; показан ранее рассчитанный прогноз' if prediction.predicted_delay_s is not None else ''
    if not source_available:
        reasons.append('Нет связи с источником'+retained)
    if telemetry_age_s is None or telemetry_age_s > 60:
        reasons.append('Нет свежей валидной телеметрии')
    if not has_current_deviation:
        reasons.append('Текущее отклонение недоступно'+retained)
    if target_reached:
        reasons.append('Посещение целевой остановки уже зарегистрировано; прогноз не является ранним предупреждением')
    if not reasons:
        # Не доверяем цвету старой ML-реплики, но сохраняем её отказ от оценки.
        risk = 'unknown' if prediction.risk == 'unknown' else risk_for_delay(prediction.predicted_delay_s)
        return prediction if risk == prediction.risk else prediction.model_copy(update={'risk': risk})
    return prediction.model_copy(update={
        'risk':'unknown', 'probability_late':None, 'probability_status':'unavailable',
        'probability_note':('Целевое посещение уже зарегистрировано' if target_reached
                            else 'Вероятность недоступна: входные данные отсутствуют или устарели'),
        'reasons':prediction.reasons + [reason for reason in reasons if reason not in prediction.reasons],
    })


class Engine:
    """Один worker, без потолка числа ТС; история и журнал хранятся в кольцевых буферах."""
    def __init__(self, ml_url: str, client: httpx.AsyncClient | None = None, *, learning_store=None):
        self.ml_url = ml_url.rstrip("/")
        self.client = client
        self.learning_store = learning_store
        self.mode: Literal["demo", "live", "replay", "generator"] = "demo"
        self.generator_url = 'http://127.0.0.1:8002'
        self.version = 0
        self.speed = 10.0
        self.running = True
        self.source_enabled = True
        self.ndtp_connections = 0
        self.ml_status = "starting"
        self.last_error = None
        self.metrics = {k: 0 for k in ["received", "duplicates", "invalid", "ndtp_errors", "unknown_units", "ml_failures", "learning_store_failures"]}
        self.latencies = deque(maxlen=200)
        self.pipeline_latencies = deque(maxlen=200)
        self.pipeline_cycles = 0
        self.reset_demo()

    def clear(self):
        self.version += 1
        self.replay = None
        self.generator = None
        self.history = defaultdict(lambda: deque(maxlen=2048))
        self.seen = defaultdict(set)
        self.predictions = {}
        self.input_sequences = defaultdict(int)
        self.prediction_sequences = {}
        self.prediction_published_at = {}
        self.forecast_traces = {}
        self.incidents = deque(maxlen=200)
        self.incident_keys = deque(maxlen=200)
        self.vehicles = {}
        self.routes = []
        self._line_memberships = None
        self.schedule = defaultdict(list)
        self.model_plans = {}
        self.plan_version = None
        self.plan_timezone = None
        self.plan_complete = False
        self.hints = defaultdict(lambda: deque(maxlen=100))
        self.arrivals = defaultdict(OrderedDict)
        self.arrival_revisions = defaultdict(lambda: deque(maxlen=100))
        self.demo_timelines = {}
        self.demo_next_arrival = {}
        self.gps_detectors = {}
        self.gps_detector_sha256 = None
        self.gps_pending = defaultdict(lambda: deque(maxlen=2048))
        self.gps_events = deque(maxlen=200)
        self.arrival_mode = 'external'
        self.context_loaded_at = utcnow()

    def reset_demo(self):
        """Полностью искусственный сценарий; не импортирует данные организаторов."""
        self.clear()
        self.mode = "demo"
        self.running = True
        self.source_enabled = True
        self.clock = datetime(2026, 1, 6, 8, 0, tzinfo=timezone.utc)
        self.started_at = self.clock
        paths = [
            [(37.586,55.759),(37.595,55.763),(37.604,55.768),(37.614,55.773),(37.625,55.776),(37.638,55.779)],
            [(37.587,55.781),(37.599,55.778),(37.612,55.770),(37.624,55.764),(37.636,55.759),(37.650,55.754)],
        ]
        for idx, path in enumerate(paths):
            route_id = f"demo-{idx+1}"
            stops = [RouteStop(id=f"{route_id}-{j}", name=f"Демо-остановка {idx+1}.{j+1}", lon=x, lat=y) for j,(x,y) in enumerate(path)]
            self.routes.append(Route(route_id=route_id, name=f"Демо {['А','Б'][idx]}", color=["#4169e1","#16a085"][idx], path=path, stops=stops))
        for n in range(4):
            v = VehicleConfig(tr_id=101+n, unit_id=1166336+n, label=f"Автобус {101+n}", route_id=f"demo-{n//2+1}")
            self.vehicles[v.tr_id] = v
            route = self.routes[n//2]
            for j in range(720):
                leg = j % (2*(len(route.stops)-1))
                stop = route.stops[min(leg, 2*(len(route.stops)-1)-leg)]
                target = StopTarget(id=f"demo-{v.tr_id}-{j}", name=stop.name,
                    scheduled_at=self.clock+timedelta(minutes=-6+j*3), lat=stop.lat, lon=stop.lon)
                self.schedule[v.tr_id].append(target)
            # Истина синтетического генератора остаётся в demo, в ML передаётся
            # только уже доставленное событие. Отклонение меняется на остановках.
            base = [185, 82, 15, -35][n]
            offsets = [0, 0, 60, 60, -60, 0]
            self.demo_timelines[v.tr_id] = [(s, s.scheduled_at+timedelta(seconds=base+offsets[j % 6]))
                for j, s in enumerate(self.schedule[v.tr_id])]
            self.demo_next_arrival[v.tr_id] = 0
        self.generate_demo()
        # Полный план искусственного сценария известен; manual_fill задан его автором.
        self.schedule = defaultdict(list, {tr: [s.model_copy(update={'manual_fill':False}) for s in stops]
                                          for tr, stops in self.schedule.items()})
        self.configure_model_plans('builtin-demo-v1', 'UTC', True)

    def configure_model_plans(self, version, timezone_name, complete):
        """Одно неизменяемое состояние полного плана на ТС; факты в нём отсутствуют."""
        self.plan_version, self.plan_timezone, self.plan_complete = version, timezone_name, complete
        self.model_plans = {tr: ModelPlanContext(version=version, timezone=timezone_name,
            complete=complete, stops=stops) for tr, stops in self.schedule.items() if stops} if version and timezone_name else {}

    def set_live(self, context: LiveContext | None = None):
        self.clear()
        self.mode = "live"
        self.clock = utcnow()
        self.running = True
        self.source_enabled = True
        if context:
            self.vehicles = {v.tr_id:v for v in context.vehicles}
            self.routes = context.routes
            for s in context.schedule:
                self.schedule[s.tr_id].append(s.target)
            for items in self.schedule.values():
                items.sort(key=lambda x: (x.scheduled_at, x.id))
            self.configure_model_plans(context.plan_version, context.plan_timezone, context.plan_complete)
            for hint in sorted(context.hints, key=lambda x:x.observed_at):
                self.hints[hint.tr_id].append(hint.model_copy(update={'received_at':self.clock}))
            self.arrival_mode = context.arrival_mode
            if self.arrival_mode == 'gps':
                from backend.gps_arrivals import GPSArrivalDetector
                self.gps_detectors = {tr: GPSArrivalDetector(self.schedule[tr]) for tr in self.vehicles}
                try:
                    self.gps_detector_sha256 = hashlib.sha256(Path(__file__).with_name('gps_arrivals.py').read_bytes()).hexdigest()
                except OSError:
                    # Отсутствие provenance запрещает проверенную вероятность,
                    # но не должно обрывать сам приём телеметрии.
                    self.gps_detector_sha256 = None

    def generate_demo(self):
        for n, vehicle in enumerate(self.vehicles.values()):
            timeline = self.demo_timelines[vehicle.tr_id]
            next_i = self.demo_next_arrival[vehicle.tr_id]
            while next_i < len(timeline) and timeline[next_i][1] <= self.clock:
                stop, actual = timeline[next_i]
                self.ingest_arrival(ArrivalInput(tr_id=vehicle.tr_id, planned_stop_id=stop.id, arrived_at=actual),
                                    received_at=self.clock, source='demo_arrival')
                next_i += 1
            self.demo_next_arrival[vehicle.tr_id] = next_i
            previous, arrived = timeline[max(0, next_i-1)]
            upcoming, next_time = timeline[min(next_i, len(timeline)-1)]
            gap = (next_time-arrived).total_seconds()
            distance_m = math.hypot((upcoming.lon-previous.lon)*111320*math.cos(math.radians((previous.lat+upcoming.lat)/2)),
                                    (upcoming.lat-previous.lat)*111320)
            dwell = min(120 if n == 0 else 20, max(0, gap-max(20, distance_m*3.6/100)))
            travel_s = max(1, gap-dwell)
            frac = min(1, max(0, (self.clock-arrived).total_seconds()-dwell)/travel_s)
            stopped = frac == 0 or next_i == len(timeline)
            lon = previous.lon*(1-frac)+upcoming.lon*frac
            lat = previous.lat*(1-frac)+upcoming.lat*frac
            # Скорость соответствует движению по прямому демонстрационному перегону.
            self.ingest(Telemetry(tr_id=vehicle.tr_id, unit_id=vehicle.unit_id,
                event_time=self.clock, received_at=self.clock, lat=lat, lon=lon,
                speed_kmh=0 if stopped else distance_m/travel_s*3.6, source="demo",
                event_id=f"demo-{vehicle.tr_id}-{self.clock.isoformat()}"))

    def set_replay(self, replay):
        """Смена режима очищает live/demo. Прогрев содержит только уже доступные события."""
        self.set_live(replay.context)
        self.mode = 'replay'
        # Источники выбираются явно; готовые snapshots и GPS-оценки не смешиваются.
        if replay.config.deviation_source == 'csv_snapshot':
            self.gps_detectors = {}
            self.arrival_mode = 'csv_snapshot'
        self.replay = replay
        replay.cursor = 0
        self.clock = replay.config.start
        self.speed = replay.config.speed
        self.running = not replay.config.paused
        self.source_enabled = True
        replay.deliver(self)

    def set_generator(self, context, status):
        from backend.generator_bridge import GeneratorSession
        if context.hints:
            raise ValueError('Генератор передаёт только GPS и план; hints запрещены')
        self.set_live(context.model_copy(update={'arrival_mode':'gps'}))
        self.mode = 'generator'
        self.generator = GeneratorSession(status)
        self.clock = status.clock_time
        self.running = status.running
        self.speed = status.speed

    async def poll_generator(self):
        """Обновления режима во время await не могут подмешать кадры старой сессии."""
        session, version = self.generator, self.version
        if session is None:
            return
        # Control API и фоновый tick могут запросить поток одновременно.
        # Cursor читается только после захвата lock, иначе второй ответ устареет.
        async with session.poll_lock:
            if version != self.version or self.generator is not session:
                return
            try:
                if self.client is None:
                    raise RuntimeError('Generator client not configured')
                response = await self.client.get(self.generator_url+'/stream',
                    params={'session_id':session.view.session_id, 'after':session.cursor})
                response.raise_for_status()
                if version == self.version and self.generator is session:
                    session.consume(self, response.json())
            except (httpx.HTTPError, ValueError, RuntimeError, KeyError, TypeError):
                if version == self.version and self.generator is session:
                    session.fail('Нет согласованного потока генератора. При смене сессии/потере кадров запустите сценарий заново.')

    def ingest_arrival(self, event: ArrivalInput, *, received_at: datetime, source='arrival', uncertainty_s=None, confirmation_span_s=None) -> bool:
        """Идемпотентный факт: будущие события/неизвестные ссылки отклоняются.

        Храним последние 100 по времени факта, а не по порядку доставки.
        Поздняя доставка старого прибытия не откатывает текущее отклонение.
        """
        if event.tr_id not in self.vehicles:
            raise ValueError('Неизвестное ТС')
        stop = next((s for s in self.schedule[event.tr_id] if s.id == event.planned_stop_id), None)
        if stop is None:
            raise ValueError('Нет такого планового прибытия у данного ТС')
        if event.arrived_at > received_at:
            raise ValueError('Нельзя подтвердить ещё не наступившее прибытие')
        records = self.arrivals[event.tr_id]
        previous = records.get(stop.id)
        if previous:
            # Операционный факт может уточнить нашу GPS-оценку, но не наоборот.
            if source in ('gps_estimate', 'door_estimate'):
                return False
            if previous.source in ('gps_estimate', 'door_estimate') and source == 'arrival':
                self.arrival_revisions[event.tr_id].append(previous)
                previous = None
        if previous:
            if previous.observed_at != event.arrived_at:
                raise ArrivalConflict('Это прибытие уже зарегистрировано с другим фактическим временем')
            return False
        records[stop.id] = CurrentDeviation(delay_s=(event.arrived_at-stop.scheduled_at).total_seconds(),
            source=source, observed_at=event.arrived_at, received_at=received_at,
            planned_stop_id=stop.id, stop_name=stop.name, planned_at=stop.scheduled_at,
            uncertainty_s=uncertainty_s, confirmation_span_s=confirmation_span_s)
        while len(records) > 100:
            del records[min(records, key=lambda key: records[key].observed_at)]
        if self.mode == 'live' and source == 'arrival' and self.learning_store is not None:
            try:
                self.learning_store.record_arrival(event.tr_id, records[stop.id])
            except (OSError, ValueError, sqlite3.Error) as exc:
                self.metrics['learning_store_failures'] += 1
                self.last_error = f"Learning store arrival: {exc}"[:300]
        return True

    def latest_arrival_at(self, tr_id, at):
        """Последнее известное к T посещение, включая доступные версии уточнений."""
        visits = {}
        for a in [*self.arrival_revisions[tr_id], *self.arrivals[tr_id].values()]:
            if a.observed_at <= at and a.received_at <= at:
                older = visits.get(a.planned_stop_id)
                if older is None or (a.received_at, a.source == 'arrival') >= (older.received_at, older.source == 'arrival'):
                    visits[a.planned_stop_id] = a
        return max(visits.values(), key=lambda a: (a.observed_at, a.planned_stop_id), default=None)

    def deviation_at(self, tr_id, at):
        """Последняя доступная остановка; hint — только запасной внешний snapshot.

        Операционный факт остаётся последним известным. Для экспериментальной
        GPS-оценки и внешнего snapshot одинаковая политика свежести: TTL 300 с.
        Историческое GPS-посещение сохраняется в журнале после истечения TTL.
        """
        latest = self.latest_arrival_at(tr_id, at)
        if latest is not None:
            if latest.source not in ('gps_estimate', 'door_estimate'):
                return latest
            if (at-latest.observed_at).total_seconds() <= 300:
                return latest.model_copy(update={'valid_until':latest.observed_at+timedelta(seconds=300)})
        hints = [h for h in self.hints[tr_id] if 0 <= (at-h.observed_at).total_seconds() <= 300
                 and (h.received_at or h.observed_at) <= at]
        hint = max(hints, key=lambda h: h.observed_at, default=None)
        return CurrentDeviation(delay_s=hint.delay_s, source=hint.source, sample_id=hint.sample_id, observed_at=hint.observed_at,
            received_at=hint.received_at or hint.observed_at, valid_until=hint.observed_at+timedelta(seconds=300)) if hint else None

    def prediction_availability_at(self, tr_id, at, *, target, prediction,
                                   telemetry_age_s, source_available=True):
        """Главная причина доступности прогноза, отдельно от риска/вероятности.

        Не меняет входы, TTL или результат ML. Истёкшая оценка учитывается только
        после обоих времён её доступности; будущие уточнения не меняют прошлое.
        Полные причины и GPS-диагностика остаются в соседних полях состояния.
        """
        def result(code, message):
            return dict(code=code, message=message)

        if not source_available:
            return result('source_unavailable', 'Нет связи с источником телеметрии')
        if target is None:
            return result('no_target', 'Нет плановой остановки в окне 10–15 минут')
        if telemetry_age_s is None:
            return result('telemetry_missing', 'Нет валидной телеметрии')
        if telemetry_age_s > 60:
            return result('telemetry_stale', f'Телеметрия устарела: {round(telemetry_age_s)} с без обновления')
        if self.deviation_at(tr_id, at) is None:
            latest = self.latest_arrival_at(tr_id, at)
            observations = [latest] if latest is not None else []
            observations.extend(h for h in self.hints[tr_id]
                if h.observed_at <= at and (h.received_at or h.observed_at) <= at)
            if observations:
                age = int((at-max(observations, key=lambda item: item.observed_at).observed_at).total_seconds())
                return result('deviation_expired',
                    f'Отклонение устарело: последняя оценка {age//60} мин {age%60} с назад (срок 5 мин)')
            return result('deviation_missing', 'Нет подтверждённого прибытия или готового отклонения')
        if self.target_reached_at(tr_id, target.id, at):
            return result('target_reached', 'Целевая остановка уже посещена; ранний прогноз неприменим')
        if prediction is None:
            return result('prediction_pending', 'Входы готовы; ожидается прогноз для текущей цели')
        if prediction.predicted_delay_s is None or prediction.risk == 'unknown':
            return result('model_input_unavailable', prediction.fallback_reason or
                'Последний расчёт не дал актуальной оценки риска для этих входов')
        return result('ready', 'Прогноз доступен')

    def delay_history_at(self, tr_id, at):
        """Причинная история состояния до T для sequence-модели.

        CSV snapshots уже имеют нужную семантику и временную сетку. Для событий
        прибытия временем знания служит received_at: поздно доставленный факт не
        может задним числом попасть в историю модели.
        """
        cutoff = at - timedelta(minutes=90)
        candidates = {}
        for hint in self.hints[tr_id]:
            known_at = hint.observed_at
            if cutoff <= known_at < at and (hint.received_at or known_at) <= at:
                candidates[known_at] = (0, float(hint.delay_s))
        for arrival in [*self.arrival_revisions[tr_id], *self.arrivals[tr_id].values()]:
            known_at = arrival.received_at
            if cutoff <= known_at < at and arrival.observed_at <= at:
                previous = candidates.get(known_at)
                priority = 2 if arrival.source == 'arrival' else 1
                if previous is None or priority >= previous[0]:
                    candidates[known_at] = (priority, float(arrival.delay_s))
        selected = sorted(candidates.items())[-11:]
        return [DelayObservation(observed_at=known_at, delay_s=value[1])
                for known_at, value in selected]

    def target_reached_at(self, tr_id, target_id, at):
        """Только уже полученное подтверждение/оценка, без будущих фактов плана.

        TTL отклонения не отменяет сам факт ранее зарегистрированного посещения.
        Поздно полученное подтверждение не влияет на состояние до его получения.
        """
        return any(a.planned_stop_id == target_id and a.observed_at <= at and a.received_at <= at
                   for a in [*self.arrival_revisions[tr_id], *self.arrivals[tr_id].values()])

    def ingest(self, point: Telemetry) -> bool:
        if point.tr_id not in self.vehicles:
            self.metrics["unknown_units"] += 1
            return False
        key = (point.event_time, point.event_id, point.source)
        seen, history = self.seen[point.tr_id], self.history[point.tr_id]
        if key in seen:
            self.metrics["duplicates"] += 1
            return False
        if len(history) == history.maxlen:
            old = history[0]
            seen.discard((old.event_time, old.event_id, old.source))
        history.append(point)
        seen.add(key)
        self.input_sequences[point.tr_id] += 1
        self.metrics["received"] += 1
        if point.tr_id in self.gps_detectors:
            cutoff = utcnow() if self.mode == 'live' else self.clock
            if max(point.event_time, point.received_at) <= cutoff:
                self.detect_arrival(point)
            else:
                self.gps_pending[point.tr_id].append(point)
        return True

    def detect_arrival(self, point):
        result = self.gps_detectors[point.tr_id].observe(point)
        if result is None:
            return
        accepted = self.ingest_arrival(ArrivalInput(tr_id=point.tr_id, planned_stop_id=result.planned_stop_id,
            arrived_at=result.arrived_at), received_at=result.received_at, source=result.source,
            uncertainty_s=result.uncertainty_s, confirmation_span_s=result.confirmation_span_s)
        if accepted:
            self.gps_events.appendleft(dict(tr_id=point.tr_id, planned_stop_id=result.planned_stop_id,
                source=result.source,
                estimated_arrived_at=result.arrived_at.isoformat(), available_at=result.received_at.isoformat(),
                delay_s=self.arrivals[point.tr_id][result.planned_stop_id].delay_s,
                uncertainty_s=result.uncertainty_s, skipped_visits=result.skipped_visits))

    def process_pending_gps(self, at):
        for tr, pending in self.gps_pending.items():
            ready = sorted((p for p in pending if max(p.event_time,p.received_at) <= at),
                           key=lambda p: (max(p.event_time,p.received_at),p.event_time))
            self.gps_pending[tr] = deque((p for p in pending if max(p.event_time,p.received_at) > at), maxlen=2048)
            for point in ready:
                self.detect_arrival(point)

    async def on_nav(self, unit_id, nav):
        if self.mode != "live":
            return  # демо не смешивается с реальными пакетами
        vehicle = next((v for v in self.vehicles.values() if v.unit_id == unit_id), None)
        if vehicle is None:
            self.metrics["unknown_units"] += 1
            return
        try:
            self.ingest(Telemetry(tr_id=vehicle.tr_id, unit_id=unit_id, received_at=utcnow(), source="ndtp", **nav))
        except ValueError:
            self.metrics["invalid"] += 1

    def ndtp_error(self, error):
        self.metrics["ndtp_errors"] += 1
        self.last_error = str(error)[:300]

    def request_for(self, tr_id, at):
        target = next((s for s in self.schedule[tr_id] if 600 < (s.scheduled_at-at).total_seconds() <= 900), None)
        if target is None:
            return None
        # При одинаковом плановом времени раздача явно задаёт ID посещения.
        # Он является разрешённым входом, а не выбирается по будущему факту.
        hints = [h for h in self.hints[tr_id] if h.target_stop_id is not None
                 and 0 <= (at-h.observed_at).total_seconds() <= 300
                 and (h.received_at or h.observed_at) <= at]
        hint = max(hints, key=lambda h:h.observed_at, default=None)
        if hint and hint.target_time_begin == target.scheduled_at:
            supplied = next((s for s in self.schedule[tr_id] if s.id == hint.target_stop_id
                             and s.scheduled_at == target.scheduled_at), None)
            if supplied is not None:
                target = supplied
        features, _ = features_at(self.history[tr_id], at, target)
        deviation = self.deviation_at(tr_id, at)
        detector = self.gps_detectors.get(tr_id) if deviation and deviation.source in ('gps_estimate','door_estimate') else None
        # Train смешивает реальные и искусственные ТС без явного флага строки.
        # Передаём смешанное происхождение явно: перенос калибровки не становится
        # проверенной вероятностью только потому, что поток воспроизводится из CSV.
        replay_domain = ('historical_mixed' if self.replay and self.replay.config.dataset_split == 'train'
                         else 'unknown' if self.replay and self.replay.config.dataset_split == 'custom'
                         else 'historical_real')
        return PredictionRequest(request_id=f"{tr_id}:{target.id}:{at.isoformat()}", tr_id=tr_id,
            issued_at=at, target=target, current_delay_s=deviation.delay_s if deviation else None, features=features,
            current_delay_source=deviation.source if deviation else None,
            current_delay_detector_version=detector.version if detector else None,
            current_delay_detector_sha256=self.gps_detector_sha256 if detector else None,
            telemetry_domain=(replay_domain if self.mode == 'replay' else
                              'synthetic' if self.mode in ('demo','generator') else 'live_unverified'),
            plan_context=self.model_plans.get(tr_id), delay_history=self.delay_history_at(tr_id, at))

    def planned_section(self, tr_id, target):
        """Последний плановый перегон перед целью; не результат GPS map matching."""
        if target is None:
            return "Целевая остановка не определена"
        previous = max((s for s in self.schedule[tr_id] if s.scheduled_at < target.scheduled_at),
                       key=lambda s:s.scheduled_at, default=None)
        return f"{previous.name} → {target.name}" if previous else f"До остановки «{target.name}»"

    async def tick(self, elapsed=1.0):
        """Измеряет полный цикл: признаки → HTTP ML/fallback → публикация.

        Ожидание следующего цикла и опрос UI сюда не входят; это не end-to-end latency.
        """
        started = time.perf_counter()
        await self._tick(elapsed)
        self.pipeline_latencies.append((time.perf_counter()-started)*1000)
        self.pipeline_cycles += 1

    async def _tick(self, elapsed):
        if self.mode == 'generator':
            version = self.version
            await self.poll_generator()
            if version != self.version:
                return
            if not self.generator.view.connected:
                return  # не выпускаем новые алерты по замершим часам источника
        if self.mode == "demo":
            if self.running:
                self.clock += timedelta(seconds=min(elapsed,5)*self.speed)
                if self.source_enabled:
                    self.generate_demo()
        elif self.mode == 'replay':
            if self.running:
                self.clock = min(self.replay.end, self.clock + timedelta(seconds=min(elapsed, 5)*self.speed))
                self.replay.deliver(self)
                if self.clock >= self.replay.end:
                    self.running = False
        elif self.mode != 'generator':
            self.clock = utcnow()
        self.process_pending_gps(self.clock)
        requests = [r for v in self.vehicles if (r := self.request_for(v, self.clock))]
        # Захватываем границу входа до await: пакет, пришедший во время HTTP,
        # нельзя объявить обработанным опубликованным прогнозом.
        sequences = {r.tr_id: self.input_sequences[r.tr_id] for r in requests}
        deviations = {r.tr_id: self.deviation_at(r.tr_id, self.clock) for r in requests}
        version = self.version
        if not requests:
            # В пустом live-режиме состояние ML тоже должно обновляться.
            try:
                if self.client is None:
                    raise RuntimeError("ML client not configured")
                health = await self.client.get(self.ml_url+"/health")
                health.raise_for_status()
                status = "ok" if health.json().get("status") == "ok" else "unavailable"
            except (httpx.HTTPError, ValueError, RuntimeError, AttributeError):
                status = "unavailable"
            if version == self.version:
                self.predictions = {}
                self.prediction_sequences = {}
                self.prediction_published_at = {}
                self.forecast_traces = {}
                self.ml_status = status
            return
        started = time.perf_counter()
        try:
            if self.client is None:
                raise RuntimeError("ML client not configured")
            # Ограничение ML относится к одному HTTP-пакету, а не размеру парка.
            predictions = []
            for start in range(0, len(requests), 128):
                response = await self.client.post(self.ml_url+"/predict/batch",
                    json=[r.model_dump(mode="json") for r in requests[start:start+128]])
                response.raise_for_status()
                predictions.extend(Prediction.model_validate(p) for p in response.json())
                if version != self.version:
                    return  # Не посылаем остальные старые пакеты после смены контекста.
            if len(predictions) != len(requests) or any(p.request_id != r.request_id or p.tr_id != r.tr_id or p.target != r.target or p.issued_at != r.issued_at for p,r in zip(predictions,requests)):
                raise ValueError("ML response does not match request")
            ml_status = "ok"
        except (httpx.HTTPError, ValueError, RuntimeError, TypeError, KeyError) as exc:
            predictions = [baseline(r, fallback=True) for r in requests]
            self.metrics["ml_failures"] += 1
            self.last_error = str(exc)[:300]
            ml_status = "unavailable"
        self.latencies.append((time.perf_counter()-started)*1000)
        if version != self.version:
            return  # запрос завершился уже после переключения сценария
        self.ml_status = ml_status
        if self.mode == 'generator' and not self.generator.check_progress():
            return  # связь могла пропасть во время HTTP ML: новый алерт недопустим
        self.predictions = {p.tr_id:p for p in predictions}
        self.prediction_sequences = sequences
        published_at = utcnow()
        publication_clock = self.clock if self.mode in ('demo', 'replay', 'generator') else published_at
        self.prediction_published_at = {p.tr_id: published_at for p in predictions}
        self.forecast_traces = {r.tr_id: dict(context_version=version, execution='ml_http' if ml_status == 'ok' else 'backend_fallback',
            current_deviation=deviations[r.tr_id].model_dump(mode='json') if deviations[r.tr_id] else None,
            request=r.model_dump(mode='json'), response=p.model_dump(mode='json'),
            telemetry_sequence=sequences[r.tr_id], published_at=published_at)
            for r, p in zip(requests, predictions)}
        if self.mode == 'live' and ml_status == 'ok' and self.learning_store is not None:
            for request, prediction in zip(requests, predictions, strict=True):
                try:
                    self.learning_store.record_prediction(request, prediction, published_at=published_at)
                except (OSError, ValueError, sqlite3.Error) as exc:
                    self.metrics['learning_store_failures'] += 1
                    self.last_error = f"Learning store prediction: {exc}"[:300]
        for p in predictions:
            key = f"{p.tr_id}:{p.target.id}"
            lead = (p.target.scheduled_at-publication_clock).total_seconds()
            # За время HTTP GPS/hint могли пересечь TTL. Свежесть на issued_at
            # не даёт права выпустить новый алерт после истечения входов.
            publication_features, _ = features_at(self.history[p.tr_id], publication_clock, p.target)
            current = gate_prediction(p, telemetry_age_s=publication_features.telemetry_age_s,
                                      has_current_deviation=self.deviation_at(p.tr_id, publication_clock) is not None,
                                      target_reached=self.target_reached_at(p.tr_id, p.target.id, publication_clock))
            # Медленный ответ ML не должен порождать алерт задним числом.
            if current.risk == "red" and key not in self.incident_keys and 600 < lead <= 900:
                explanation = explain(self.history[p.tr_id], p.issued_at,
                    current_deviation=self.deviation_at(p.tr_id, p.issued_at),
                    deviation_history=[*self.arrival_revisions[p.tr_id], *self.arrivals[p.tr_id].values()],
                    schedule=self.schedule[p.tr_id])
                self.incident_keys.append(key)
                self.incidents.appendleft(dict(id=key,tr_id=p.tr_id,route_id=self.vehicles[p.tr_id].route_id,
                    created_at=publication_clock,data_cutoff=p.issued_at,published_at=published_at,
                    target_time=p.target.scheduled_at,risk=current.risk,
                    target_name=p.target.name,horizon_reference='scheduled_arrival',
                    estimated_arrival_at=p.target.scheduled_at+timedelta(seconds=p.predicted_delay_s),
                    estimated_lead_time_s=lead+p.predicted_delay_s,
                    probability_late=p.probability_late,probability_status=p.probability_status,
                    probability_note=p.probability_note,
                    predicted_delay_s=p.predicted_delay_s,lead_time_s=lead,
                    reason=explanation.possible_cause, section=self.planned_section(p.tr_id,p.target),
                    model_version=p.model_version, method=p.method,
                    explanation=explanation.model_dump(mode="json")))

    def state(self):
        if self._line_memberships is None:
            from backend.line_identity import line_memberships
            self._line_memberships = line_memberships([r.model_dump(mode='json') for r in self.routes])
        now = self.clock if self.mode in ('demo', 'replay', 'generator') else utcnow()
        producer_unavailable = self.generator is not None and not self.generator.check_progress()
        vehicles = []
        for tr_id, cfg in self.vehicles.items():
            deviation = self.deviation_at(tr_id, now)
            request = self.request_for(tr_id, now)
            target = request.target if request else None
            # Координаты доступны и при отсутствии остановки в горизонте.
            dummy = target or StopTarget(id="position", name="Положение", scheduled_at=now+timedelta(minutes=12), lat=0,lon=0)
            features, latest = features_at(self.history[tr_id], now, dummy)
            explanation = explain(self.history[tr_id], now, current_deviation=deviation,
                deviation_history=[*self.arrival_revisions[tr_id], *self.arrivals[tr_id].values()],
                schedule=self.schedule[tr_id])
            from backend.segment_observations import observe_segment
            segment_observation = observe_segment(self.history[tr_id], now, self.schedule[tr_id], deviation)
            age = features.telemetry_age_s
            status = "no_data" if age is None else "stale" if age>60 else "fresh"
            if producer_unavailable:
                status = 'stale' if latest else 'no_data'
            p = self.predictions.get(tr_id)
            if p and (target is None or p.target.id != target.id or not 600 < (p.target.scheduled_at-now).total_seconds() <= 900):
                p = None
            if p:
                p = gate_prediction(p, telemetry_age_s=features.telemetry_age_s,
                                    has_current_deviation=deviation is not None,
                                    source_available=not producer_unavailable,
                                    target_reached=self.target_reached_at(tr_id, p.target.id, now))
            reasons = list(p.reasons) if p else ["Нет остановки в окне 10–15 минут" if target is None else "Ожидается прогноз"]
            if status != "fresh" and "Нет свежей валидной телеметрии" not in reasons:
                reasons.append("Нет свежей валидной телеметрии")
            if producer_unavailable:
                reasons.append('Нет связи с генератором; показано последнее состояние')
            recommendation = explanation.recommendation if status != "fresh" or (p and p.risk in ("red", "amber")) else "Продолжить наблюдение"
            vehicles.append(dict(**cfg.model_dump(),lat=latest.lat if latest else None,lon=latest.lon if latest else None,
                speed_kmh=latest.speed_kmh if latest else None,event_time=latest.event_time if latest else None,
                heading=latest.heading % 360 if latest and latest.heading is not None else None,
                doors_open=latest.doors_open if latest else None,
                age_s=age,status=status,cur_dev_s=deviation.delay_s if deviation else None,
                current_deviation=deviation.model_dump(mode='json') if deviation else None,
                gps_detector=self.gps_detectors[tr_id].status() if tr_id in self.gps_detectors else None,
                prediction_current_delay_s=self.forecast_traces.get(tr_id, {}).get('request', {}).get('current_delay_s') if p else None,
                target=target.model_dump(mode="json") if target else None,
                prediction=p.model_dump(mode="json") if p else None,reasons=reasons,recommendation=recommendation,
                prediction_availability=self.prediction_availability_at(tr_id, now, target=target,
                    prediction=p, telemetry_age_s=age, source_available=not producer_unavailable),
                telemetry_sequence=self.input_sequences[tr_id],
                prediction_input_sequence=self.prediction_sequences.get(tr_id) if p else None,
                prediction_published_at=self.prediction_published_at.get(tr_id) if p else None,
                section=self.planned_section(tr_id,target),explanation=explanation.model_dump(mode="json"),
                segment_observation=segment_observation.model_dump(mode='json')))
        risks = [v["prediction"]["risk"] if v["prediction"] else "unknown" for v in vehicles]
        p95 = sorted(self.latencies)[min(len(self.latencies)-1,math.ceil(len(self.latencies)*.95)-1)] if self.latencies else None
        pipeline_p95 = sorted(self.pipeline_latencies)[math.ceil(len(self.pipeline_latencies)*.95)-1] if self.pipeline_latencies else None
        from backend.risk_summary import summarize
        route_risks, section_risks = summarize(vehicles, self.schedule, {r.route_id:r.name for r in self.routes})
        return dict(server_time=utcnow(),clock_time=now,mode=self.mode,
            risk_policy=RISK_POLICY.model_dump(mode='json'),
            context=dict(version=self.version, loaded_at=self.context_loaded_at, arrival_mode=self.arrival_mode,
                         plan_version=self.plan_version, plan_timezone=self.plan_timezone, plan_complete=self.plan_complete,
                         vehicles=len(self.vehicles), planned_visits=sum(map(len,self.schedule.values())),
                         routes=len(self.routes), source=(self.replay.config.plan_file if self.mode == 'replay' and self.replay else
                             {'generator':'synthetic_generator','replay':'validate/schedule_plan.csv','demo':'builtin_demo','live':'live_context'}[self.mode])),
            gps_arrivals=list(self.gps_events),
            route_risks=route_risks, section_risks=section_risks,
            generator=self.generator.view.model_dump(mode='json') if self.generator else None,
            replay=self.replay.state(self).model_dump(mode='json') if self.replay else None,
            demo=dict(running=self.running,source_enabled=self.source_enabled,speed=self.speed),
            health=dict(ml=self.ml_status,ndtp_connections=self.ndtp_connections,
                source_status="disconnected" if producer_unavailable else "receiving" if any(v["status"]=="fresh" for v in vehicles) else "awaiting"),
            summary=dict(vehicles=len(vehicles),red=risks.count("red"),amber=risks.count("amber"),
                stale=sum(v["status"]!="fresh" for v in vehicles),unknown=risks.count("unknown"),events=len(self.incidents)),
            routes=[r.model_dump(mode="json") for r in self.routes],line_memberships=self._line_memberships,
            vehicles=vehicles,incidents=list(self.incidents),
            metrics={**self.metrics,"inference_p95_ms":round(p95,2) if p95 is not None else None,
                "pipeline_p95_ms":round(pipeline_p95,2) if pipeline_p95 is not None else None,
                "pipeline_cycles":self.pipeline_cycles})
