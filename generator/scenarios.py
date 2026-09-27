"""Искусственная маршрутная сеть, причинная GPS-телеметрия и закрытая истина.

Это условная общая сетка улиц, не маршруты Москвы. Движение следует ломаной
с изменением курса на поворотах. План известен заранее; наступившие истинные
прибытия доступны только evaluator через /truth, а не через поток backend.
"""
from __future__ import annotations

import math
import random
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal

from pydantic import Field, StrictBool

from backend.arrivals import ArrivalInput
from backend.engine import LiveContext, Route, RouteStop, ScheduledStop, VehicleConfig
from common.contracts import Contract, StopTarget, Telemetry

SCENARIOS = Literal['normal', 'slow_segment', 'long_stop', 'gps_loss']
START = datetime(2026, 1, 6, 8, tzinfo=timezone.utc)
DURATION_S = 1800
FRAME_LIMIT = 600
LOG_LIMIT = 100
GRID_M = 300.0
ORIGIN = (37.61, 55.75)
COLORS = ['#4169e1', '#16a085', '#b45bd2', '#d97524', '#cf4369', '#2584a0', '#817628', '#734dbc']


class ResetRequest(Contract):
    scenario: SCENARIOS = 'normal'
    seed: int = Field(default=42, ge=0, le=2147483647)
    speed: float = Field(default=10, ge=1, le=30)
    paused: bool = True
    route_count: int = Field(default=4, ge=1, le=20)
    telemetry_interval_s: int = Field(default=15, ge=1, le=30)
    door_sensors: StrictBool = False


class ControlRequest(Contract):
    action: Literal['pause', 'resume', 'speed']
    speed: float | None = Field(default=None, ge=1, le=30)


@dataclass(frozen=True)
class Visit:
    """Приватный факт и геометрия подхода; не выдаются вместе с планом."""
    target: StopTarget
    arrival: datetime
    departure: datetime
    approach: tuple[tuple[float, float], ...]


def segment_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot((b[0]-a[0])*111320*math.cos(math.radians((a[1]+b[1])/2)),
                      (b[1]-a[1])*111320)


def grid_coordinate(point: tuple[int, int]) -> tuple[float, float]:
    return (ORIGIN[0]+point[0]*GRID_M/(111320*math.cos(math.radians(ORIGIN[1]))),
            ORIGIN[1]+point[1]*GRID_M/111320)


def expand_path(corners: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Разбить ортогональную улицу на одинаковые блоки без срезания углов."""
    result = [corners[0]]
    for destination in corners[1:]:
        x, y = result[-1]
        tx, ty = destination
        if x != tx and y != ty:
            raise ValueError('Улицы условной сети должны быть ортогональными')
        while (x, y) != destination:
            x += (tx > x)-(tx < x)
            y += (ty > y)-(ty < y)
            result.append((x, y))
    return result


def route_grid(index: int) -> list[tuple[int, int]]:
    """20 разных линий через общие узлы, с поворотами и обратным ходом."""
    band, variant = divmod(index, 5)
    left, right = 5+variant, 4+(variant+2)%5
    south, north = 2+variant%3, 2+(variant+1)%3
    patterns = [
        [(-left, -south), (-2, -south), (-2, 0), (0, 0), (0, north), (right, north)],
        [(-left, north), (0, north), (0, 0), (2, 0), (2, -south), (right, -south)],
        [(-south, -left), (-south, -2), (0, -2), (0, 0), (north, 0), (north, right)],
        [(-north, left), (-north, 0), (0, 0), (0, -2), (south, -2), (south, -right)],
    ]
    return expand_path(patterns[band])


class GeneratorSession:
    """30 минут, кадр каждую симсекунду, GPS по отдельной заданной частоте.

    ``advance(real_seconds)`` не пропускает такты при ускорении. GET не двигает
    часы. Буфер 600 кадров ограничен; старый consumer обязан начать новую сессию.
    Истина прибытий хранится отдельно от GPS даже при потере связи.
    """

    def __init__(self, config: ResetRequest):
        self.session_id = str(uuid.uuid4())
        self.scenario, self.seed, self.speed = config.scenario, config.seed, config.speed
        self.route_count = config.route_count
        self.telemetry_interval_s = config.telemetry_interval_s
        self.door_sensors = config.door_sensors
        self.scenario_vehicle_id = ({'slow_segment': 101, 'long_stop': 102,
                                     'gps_loss': min(104, 100+self.route_count*2)}.get(self.scenario))
        self.running, self.finished = not config.paused, False
        self.clock_time = START
        self._remainder = 0.0
        self.frames: deque[dict] = deque(maxlen=FRAME_LIMIT)
        self.logs: deque[dict] = deque(maxlen=LOG_LIMIT)
        self._seq = self._log_seq = 0
        self.emitted_telemetry = self.emitted_arrivals = 0
        self._visits: dict[int, list[Visit]] = {}
        self._truth_arrivals: list[dict] = []
        self._next_arrival: dict[int, int] = {}
        self._scenario_events: list[tuple[datetime, int, str]] = []
        self._next_scenario = 0
        self.context = self._build_context()
        self._log('session', 'Условная сеть создана. Backend получает GPS и, при включении, датчики дверей; истина прибытий отделена для проверки.')
        self._emit_frame()

    def _build_context(self) -> LiveContext:
        routes, vehicles, schedule = [], [], []
        route_nodes, route_stop_indices = [], []
        for i in range(self.route_count):
            nodes = route_grid(i)
            # Остановки через два квартала и общий пересадочный узел. Поворот
            # внутри перегона остаётся в path, поэтому движение не режет угол.
            stop_indices = sorted(set(range(0, len(nodes), 2)) | {len(nodes)-1, nodes.index((0, 0))})
            route_id = f'synthetic-{i+1}'
            stops = []
            for j in stop_indices:
                x, y = nodes[j]
                lon, lat = grid_coordinate((x, y))
                stops.append(RouteStop(id=f'grid-{x}-{y}',
                    name='Пересадка · Центр' if (x, y) == (0, 0) else f'Узел {x:+d}/{y:+d}', lon=lon, lat=lat))
            route_nodes.append([grid_coordinate(node) for node in nodes])
            route_stop_indices.append(stop_indices)
            routes.append(Route(route_id=route_id, name=f'Условный маршрут {i+1}',
                color=COLORS[i % len(COLORS)], path=route_nodes[-1], stops=stops))
        rng = random.Random(self.seed)
        for n in range(self.route_count*2):
            tr_id = 101+n
            route = routes[n//2]
            nodes, indices = route_nodes[n//2], route_stop_indices[n//2]
            vehicles.append(VehicleConfig(tr_id=tr_id, unit_id=1166336+n,
                label=f'Синт. автобус {tr_id}', route_id=route.route_id))
            base_delay = [150, 90, 30, -20][n % 4]+rng.randint(-5, 5)
            phase = n % 4*45+(n//4 % 3)*10
            visits: list[Visit] = []
            previous_index = None
            for j in range(30):
                # Конечная не повторяется мгновенно; после неё обратный ход.
                period = 2*(len(route.stops)-1)
                leg = j % period
                stop_index = min(leg, period-leg)
                stop, node_index = route.stops[stop_index], indices[stop_index]
                planned = START+timedelta(seconds=-900+phase+j*180)
                target = StopTarget(id=f'synthetic-{tr_id}-{j}', name=stop.name,
                    lat=stop.lat, lon=stop.lon, scheduled_at=planned, manual_fill=False)
                if previous_index is None:
                    approach = (nodes[node_index],)
                elif previous_index < node_index:
                    approach = tuple(nodes[previous_index:node_index+1])
                else:
                    approach = tuple(reversed(nodes[node_index:previous_index+1]))
                slow_leg = self.scenario == 'slow_segment' and tr_id == 101 and j == 6
                travel_s = 300 if slow_leg else 150
                actual = (visits[-1].departure+timedelta(seconds=travel_s)
                          if visits else planned+timedelta(seconds=base_delay))
                long_dwell = self.scenario == 'long_stop' and tr_id == 102 and j == 5
                departure = actual+timedelta(seconds=180 if long_dwell else 30)
                visits.append(Visit(target, actual, departure, approach))
                schedule.append(ScheduledStop(tr_id=tr_id, target=target))
                if slow_leg:
                    self._scenario_events.extend([
                        (visits[-2].departure, tr_id, 'Истина сценария: начался медленный перегон, 300 с вместо 150 с.'),
                        (actual, tr_id, 'Истина сценария: медленный перегон завершён.'),
                    ])
                if long_dwell:
                    self._scenario_events.extend([
                        (actual, tr_id, 'Истина сценария: началась долгая стоянка, 180 с вместо 30 с.'),
                        (departure, tr_id, 'Истина сценария: долгая стоянка завершена.'),
                    ])
                previous_index = node_index
            self._visits[tr_id] = visits
            self._next_arrival[tr_id] = 0
        if self.scenario == 'gps_loss':
            self._scenario_events.extend([
                (START+timedelta(seconds=90), self.scenario_vehicle_id, 'Истина сценария: GPS не доставляется 120 с. Прибытия не передаются backend и при нормальной связи.'),
                (START+timedelta(seconds=210), self.scenario_vehicle_id, 'Истина сценария: связь восстановлена, GPS возобновится на следующем такте выдачи.'),
            ])
        self._scenario_events.sort(key=lambda event: event[0])
        schedule.sort(key=lambda item: (item.target.scheduled_at, item.tr_id))
        return LiveContext(vehicles=vehicles, routes=routes, schedule=schedule, hints=[],
            plan_version=f'synthetic-grid-v2:{self.session_id}', plan_timezone='UTC', plan_complete=True)

    def _log(self, kind: str, message: str, tr_id: int | None = None, **fields):
        self._log_seq += 1
        self.logs.append(dict(seq=self._log_seq, frame_seq=self._seq, time=self.clock_time.isoformat(),
                              kind=kind, tr_id=tr_id, message=message, **fields))

    def _connected(self, tr_id: int) -> bool:
        elapsed = (self.clock_time-START).total_seconds()
        return not (self.scenario == 'gps_loss' and tr_id == self.scenario_vehicle_id and 90 <= elapsed < 210)

    def _position(self, tr_id: int) -> tuple[float, float, float, float]:
        visits = self._visits[tr_id]
        i = max(j for j, visit in enumerate(visits) if visit.arrival <= self.clock_time)
        previous = visits[i]
        if self.clock_time <= previous.departure or i == len(visits)-1:
            return previous.target.lon, previous.target.lat, 0.0, 0.0
        following = visits[i+1]
        duration = (following.arrival-previous.departure).total_seconds()
        lengths = [segment_m(a, b) for a, b in zip(following.approach, following.approach[1:])]
        total = sum(lengths)
        remaining = total*(self.clock_time-previous.departure).total_seconds()/duration
        for index, length in enumerate(lengths):
            if remaining <= length or index == len(lengths)-1:
                a, b = following.approach[index:index+2]
                fraction = min(1.0, remaining/length)
                heading = math.degrees(math.atan2((b[0]-a[0])*math.cos(math.radians(a[1])), b[1]-a[1])) % 360
                return (a[0]+(b[0]-a[0])*fraction, a[1]+(b[1]-a[1])*fraction, total/duration*3.6, heading)
            remaining -= length
        raise RuntimeError('Пустой перегон')

    def _emit_frame(self):
        self._seq += 1
        telemetry = []
        while (self._next_scenario < len(self._scenario_events)
               and self._scenario_events[self._next_scenario][0] <= self.clock_time):
            at, tr_id, message = self._scenario_events[self._next_scenario]
            self._log('scenario', message, tr_id, event_time=at.isoformat())
            self._next_scenario += 1
        emit_gps = int((self.clock_time-START).total_seconds()) % self.telemetry_interval_s == 0
        for vehicle in self.context.vehicles:
            tr_id = vehicle.tr_id
            visits = self._visits[tr_id]
            cursor = self._next_arrival[tr_id]
            while cursor < len(visits) and visits[cursor].arrival <= self.clock_time:
                visit = visits[cursor]
                arrival = ArrivalInput(tr_id=tr_id, planned_stop_id=visit.target.id, arrived_at=visit.arrival)
                self._truth_arrivals.append(arrival.model_dump(mode='json'))
                self._log('truth_arrival', 'Истина синтетики: прибытие только для проверки, не передаётся детектору.', tr_id,
                          planned_stop_id=visit.target.id, arrived_at=visit.arrival.isoformat(),
                          scheduled_at=visit.target.scheduled_at.isoformat())
                cursor += 1
                self.emitted_arrivals += 1
            self._next_arrival[tr_id] = cursor
            if not emit_gps or not self._connected(tr_id):
                continue
            lon, lat, speed, heading = self._position(tr_id)
            # Текущее состояние датчика, без ID остановки и времени прибытия.
            # На отправлении двери уже закрыты; частота выдачи та же, что у GPS.
            doors_open = (any(v.arrival <= self.clock_time < v.departure for v in visits)
                          if self.door_sensors else None)
            message = Telemetry(tr_id=tr_id, unit_id=vehicle.unit_id,
                event_time=self.clock_time, received_at=self.clock_time, lat=lat, lon=lon,
                speed_kmh=speed, heading=heading, doors_open=doors_open,
                door_sensor_key='synthetic:door1' if self.door_sensors else None, source='generator',
                event_id=f'synthetic-{self.seed}-{tr_id}-{self.clock_time.isoformat()}')
            telemetry.append(message.model_dump(mode='json'))
            self._log('telemetry', 'Сформирован пакет синтетической GPS-телеметрии для потока.', tr_id,
                      event_time=self.clock_time.isoformat(), lat=lat, lon=lon, speed_kmh=round(speed, 2),
                      doors_open=doors_open)
        self.emitted_telemetry += len(telemetry)
        self.frames.append(dict(seq=self._seq, clock_time=self.clock_time.isoformat(),
                                telemetry=telemetry, arrivals=[]))

    def advance(self, real_seconds: float):
        """Продвинуть только запущенную сессию, сохраняя cadence в виртуальном времени."""
        if not math.isfinite(real_seconds) or real_seconds < 0:
            raise ValueError('Прошедшее время должно быть конечным и неотрицательным')
        if not self.running or self.finished:
            return
        self._remainder += real_seconds*self.speed
        count = min(int(self._remainder), DURATION_S-int((self.clock_time-START).total_seconds()))
        self._remainder -= count
        for _ in range(count):
            self.clock_time += timedelta(seconds=1)
            self._emit_frame()
        if self.clock_time >= START+timedelta(seconds=DURATION_S):
            self.running, self.finished, self._remainder = False, True, 0
            self._log('control', '30-минутный сценарий завершён. Для повторения сбросьте сессию.')

    def control(self, request: ControlRequest) -> dict:
        if request.action == 'resume':
            if self.finished:
                raise ValueError('Сценарий завершён; сначала сбросьте сессию')
            self.running = True
        elif request.action == 'pause':
            self.running = False
        else:
            if request.speed is None:
                raise ValueError('Для action=speed требуется speed')
            self.speed = request.speed
        self._log('control', f'Управление: {request.action}; скорость ×{self.speed:g}.')
        return self.status()

    def status(self, *, include_context=False) -> dict:
        result = dict(session_id=self.session_id, clock_time=self.clock_time.isoformat(),
                      running=self.running, speed=self.speed, scenario=self.scenario,
                      seed=self.seed, finished=self.finished, start_time=START.isoformat(),
                      end_time=(START+timedelta(seconds=DURATION_S)).isoformat(), duration_s=DURATION_S,
                      route_count=self.route_count, telemetry_interval_s=self.telemetry_interval_s,
                      door_sensors=self.door_sensors,
                      scenario_vehicle_id=self.scenario_vehicle_id,
                      emitted_telemetry=self.emitted_telemetry, emitted_arrivals=self.emitted_arrivals)
        if include_context:
            result['context'] = self.context.model_dump(mode='json')
        return result

    def stream(self, after: int) -> dict:
        if after < 0 or after > self._seq:
            raise ValueError('after должен быть между 0 и последним seq сессии')
        first = self.frames[0]['seq']
        return dict(**self.status(), frames=[frame for frame in self.frames if frame['seq'] > after],
                    first_seq=first, last_seq=self._seq, gap=after < first-1, logs=list(self.logs))

    def truth(self) -> dict:
        """Уже наступивший эталон только для evaluator, недоступный признакам."""
        return dict(**self.status(), observed_truth_arrivals=[dict(item) for item in self._truth_arrivals])
