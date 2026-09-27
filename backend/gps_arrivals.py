"""Причинный детектор посещений остановок по GPS, без ML и истинных прибытий.

Один экземпляр соответствует одному ТС и неизменному упорядоченному плану.
Пороги — зафиксированная инженерная гипотеза, не результат калибровки
на организаторских данных. Вызвавший код обязан подавать только уже доступные
сообщения (event_time и received_at <= его текущих часов).

Без датчиков дверей нужны два медленных измерения в зоне остановки.
При свежем переходе дверей «закрыты → открыты» достаточно одной медленной
точки в однозначной плановой зоне; оценка доступна только при получении пакета.
GPS без дверей не отличает посадку от светофора/затора рядом с остановкой.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from datetime import datetime
import math

from common.contracts import StopTarget, Telemetry


@dataclass(frozen=True)
class GPSDetectorConfig:
    """Фиксированные пороги; evaluator обязан записывать их вместе с результатом."""
    enter_radius_m: float = 35.0
    exit_radius_m: float = 70.0
    max_stop_speed_kmh: float = 5.0
    confirmation_s: float = 5.0
    min_points: int = 2
    max_gap_s: float = 30.0
    # Ограничивает и абсолютное расхождение с планом, и изменение относительно
    # последней оценки. При большей задержке нужен внешний якорь/другой метод;
    # нельзя позволять собственной ошибке сдвигать окно на всё более ранний круг.
    schedule_window_s: float = 900.0

    def __post_init__(self):
        values = (self.enter_radius_m, self.exit_radius_m, self.max_stop_speed_kmh,
                  self.confirmation_s, self.max_gap_s, self.schedule_window_s)
        if any(not math.isfinite(x) or x <= 0 for x in values):
            raise ValueError('Пороги GPS-детектора должны быть конечными и положительными')
        if self.exit_radius_m <= self.enter_radius_m:
            raise ValueError('Радиус выхода должен превышать радиус входа')
        if self.min_points < 2 or not isinstance(self.min_points, int):
            raise ValueError('Нужно минимум два разных измерения GPS')
        if self.confirmation_s > self.max_gap_s:
            raise ValueError('Подтверждение не должно превышать допустимый разрыв')


@dataclass(frozen=True)
class GPSArrival:
    """Оценка посещения, а не подтверждённое диспетчером событие.

    uncertainty_s — расстояние по времени от предыдущей пригодной точки до
    первого наблюдения в зоне; это дискретность наблюдений, НЕ доверительный
    интервал и НЕ верхняя граница ошибки. При холодном старте/разрыве — None.
    skipped_visits — число пропущенных в плане записей без созданных прибытий.
    """
    tr_id: int
    planned_stop_id: str
    arrived_at: datetime
    received_at: datetime
    confirmation_span_s: float
    uncertainty_s: float | None
    skipped_visits: int = 0
    source: str = 'gps_estimate'


@dataclass
class _Candidate:
    index: int
    first: Telemetry
    count: int
    uncertainty_s: float | None
    skipped: int


def distance_m(lat: float, lon: float, stop: StopTarget) -> float:
    """Гаверсин; координаты остановки берутся из входного плана."""
    lat1, lat2 = math.radians(lat), math.radians(stop.lat)
    a = (math.sin((lat2-lat1)/2)**2
         + math.cos(lat1)*math.cos(lat2)*math.sin(math.radians(stop.lon-lon)/2)**2)
    return 6371000 * 2 * math.asin(min(1.0, math.sqrt(a)))


class GPSArrivalDetector:
    """План → кандидат → подтверждение → выход с гистерезисом.

    Холодный старт и восстановление после пропуска требуют единственного
    пространственного/временного соответствия. Одинаковые координаты разных
    посещений внутри окна не разрешаются выбором ближайшего времени: unknown.
    После якоря используются только следующие посещения, по времени с учётом
    последнего оценённого отклонения. Нет обратного заполнения пропущенных фактов.
    """
    version = 'gps-arrivals-v6-terminal-reentry'

    def __init__(self, stops: list[StopTarget], config: GPSDetectorConfig | None = None):
        self.config = config or GPSDetectorConfig()
        if len({stop.id for stop in stops}) != len(stops):
            raise ValueError('ID конкретного планового посещения должны быть уникальны')
        self.stops = sorted(stops, key=lambda stop: stop.scheduled_at)
        self._tr_id = None
        self._last_event = None
        self._last_received = None
        self._last_valid = None
        self._candidate = None
        self._confirmed_index = None
        self._locked_index = None
        self._next_index = 0
        self._delay_s = 0.0
        self._reason = 'awaiting_gps' if self.stops else 'no_schedule'
        self._state = 'unknown'
        self._distance = None
        self._confirmed_count = 0
        self._skipped_count = 0
        self._approach = deque(maxlen=120)
        self._needs_anchor = True
        self._last_arrival: GPSArrival | None = None

    def status(self) -> dict:
        """Компактная JSON-совместимая диагностика без доступа к истине генератора."""
        return {
            'version': self.version,
            'state': self._state,
            'reason': self._reason,
            'candidate_stop_id': self.stops[self._candidate.index].id if self._candidate else None,
            'confirmed_stop_id': self.stops[self._confirmed_index].id if self._confirmed_index is not None else None,
            'next_stop_id': self.stops[self._next_index].id if self._next_index < len(self.stops) else None,
            'last_event_time': self._last_event.isoformat() if self._last_event else None,
            'last_received_at': self._last_received.isoformat() if self._last_received else None,
            'distance_to_candidate_m': round(self._distance, 1) if self._distance is not None else None,
            'candidate_points': self._candidate.count if self._candidate else 0,
            'confirmed_arrivals': self._confirmed_count,
            'skipped_visits': self._skipped_count,
            'last_estimated_arrival_at': self._last_arrival.arrived_at.isoformat() if self._last_arrival else None,
            'last_confirmed_at': self._last_arrival.received_at.isoformat() if self._last_arrival else None,
            'confirmation_age_at_last_message_s': max(0.0, (self._last_received-self._last_arrival.received_at).total_seconds())
                if self._last_arrival and self._last_received else None,
        }

    def _reject(self, reason: str, *, reset: bool = True):
        self._reason = reason
        if reset:
            self._candidate = None
            self._state = 'at_stop' if self._locked_index is not None else 'unknown'
            self._distance = None
        return None

    def _select(self, point: Telemetry) -> int | None:
        matches = []
        nearby = False
        for index in range(self._next_index, len(self.stops)):
            stop = self.stops[index]
            raw_delta = (point.event_time-stop.scheduled_at).total_seconds()
            expected_delta = raw_delta-self._delay_s
            if abs(raw_delta) > self.config.schedule_window_s:
                continue
            if distance_m(point.lat, point.lon, stop) <= self.config.enter_radius_m:
                # После потери якоря старое отклонение не является временем
                # нового рейса. Ослабить его временной фильтр можно только при
                # положительном наблюдаемом подходе в направлении этого визита.
                # Одна стоящая точка/будущее плановое время такого права не дают.
                recovering = self._needs_anchor and self._approach_matches(point, index) is True
                if abs(expected_delta) > self.config.schedule_window_s and not recovering:
                    continue
                nearby = True
                # Пропуск следующего посещения допустим лишь после его ожидаемого
                # времени; отдельный факт для него не создаётся.
                if self._confirmed_index is not None and index > self._next_index:
                    previous = self.stops[index-1]
                    if (point.event_time-previous.scheduled_at).total_seconds() < self._delay_s and not recovering:
                        continue
                matches.append(index)
        # После якоря порядок посещений важнее повторяющихся координат. После
        # разрыва эту привилегию теряем: ТС могло пропустить несколько остановок.
        if not self._needs_anchor and self._next_index in matches:
            return self._next_index
        if len(matches) > 1:
            directions = [(index, self._approach_matches(point, index)) for index in matches]
            directional = [index for index, matched in directions if matched is True]
            if len(directional) == 1 and all(matched is not None for _, matched in directions):
                return directional[0]
        if len(matches) != 1:
            self._reject('ambiguous_stop' if len(matches) > 1 else
                         'skip_before_expected_time' if nearby else 'outside_stop_window')
            return None
        return matches[0]

    def _approach_matches(self, point: Telemetry, index: int) -> bool | None:
        """Разрешить встречные посещения только по наблюдаемому подходу.

        Сравнивается последний GPS за пределами зоны выхода (не старше 90 с)
        с направлением предыдущая плановая остановка → текущая. Это простой
        локальный ориентир; на сложном дорожном перегоне нужен map matching.
        Одинаковые круги одного направления остаются неоднозначными.
        """
        if index == 0:
            return None
        stop, previous = self.stops[index], self.stops[index-1]
        if distance_m(previous.lat, previous.lon, stop) < self.config.exit_radius_m:
            return None
        approach = next((p for p in reversed(self._approach)
            if 0 < (point.event_time-p.event_time).total_seconds() <= 90
            and self.config.exit_radius_m <= distance_m(p.lat, p.lon, stop) <= 300), None)
        if approach is None:
            return None
        coslat = math.cos(math.radians(stop.lat))
        ax, ay = (stop.lon-approach.lon)*coslat, stop.lat-approach.lat
        bx, by = (stop.lon-previous.lon)*coslat, stop.lat-previous.lat
        cosine = (ax*bx+ay*by)/(math.hypot(ax, ay)*math.hypot(bx, by))
        return cosine >= math.cos(math.radians(45))

    def observe(self, point: Telemetry) -> GPSArrival | None:
        """Принять один доступный GPS; запоздавшие события не меняют оценённый факт."""
        if self._tr_id is None:
            self._tr_id = point.tr_id
        elif self._tr_id != point.tr_id:
            raise ValueError('GPSArrivalDetector должен обслуживать одно ТС')
        if point.received_at < point.event_time:
            return self._reject('event_after_receipt', reset=False)
        if self._last_event is not None and point.event_time <= self._last_event:
            return self._reject('late_or_duplicate', reset=False)
        if self._last_received is not None and point.received_at < self._last_received:
            return self._reject('receipt_out_of_order', reset=False)
        gap = None if self._last_event is None else (point.event_time-self._last_event).total_seconds()
        previous = self._last_valid
        self._last_event = point.event_time
        self._last_received = point.received_at
        if (not point.location_valid or point.lat is None or point.lon is None
                or point.speed_kmh is None or point.speed_kmh > 130):
            self._last_valid = None
            self._approach.clear()
            self._needs_anchor = True
            return self._reject('invalid_gps')
        self._last_valid = point
        after_gap = gap is not None and gap > self.config.max_gap_s
        if after_gap:
            self._candidate = None
            previous = None
            self._approach.clear()
            self._needs_anchor = True
        self._approach.append(point)
        if not self.stops:
            return self._reject('no_schedule')
        if self._locked_index is not None:
            locked_distance = distance_m(point.lat, point.lon, self.stops[self._locked_index])
            if locked_distance < self.config.exit_radius_m:
                self._state = 'at_stop'
                return self._reject('at_confirmed_stop', reset=False)
            self._locked_index = None
            self._state = 'between_stops'
        if self._next_index >= len(self.stops):
            return self._reject('plan_finished')
        if point.speed_kmh > self.config.max_stop_speed_kmh:
            return self._reject('moving')
        # Не закрепляем неоднозначного кандидата: если временное окно стало
        # включать другой повтор той же остановки, подтверждение отменяется.
        selected = self._select(point)
        if selected is None:
            return None
        stop = self.stops[selected]
        if selected > 0:
            preceding = self.stops[selected-1]
            if (preceding.lat == stop.lat and preceding.lon == stop.lon
                    and not any(0 < (point.event_time-p.event_time).total_seconds() <= 90
                        and distance_m(p.lat,p.lon,stop) >= self.config.exit_radius_m
                        for p in self._approach)):
                # Два плановых визита одной конечной могут обрамлять отстой.
                # Проверка после _select не убирает конкурирующий ID, чтобы
                # не превратить неоднозначное совпадение в ложную уникальность.
                return self._reject('terminal_occupancy_without_reentry')
        self._distance = distance_m(point.lat, point.lon, self.stops[selected])
        if (point.doors_open is True and previous is not None
                and previous.doors_open is False
                and point.door_sensor_key is not None and point.door_sensor_key == previous.door_sensor_key
                and 0 < (point.event_time-previous.event_time).total_seconds() <= self.config.max_gap_s
                and distance_m(previous.lat, previous.lon, point)
                    <= 130/3.6*(point.event_time-previous.event_time).total_seconds()):
            # Открытие — дополнительное наблюдение, не готовый ID/факт прибытия.
            # Все проверки времени, скорости, плана и неоднозначности выше
            # действуют и здесь. Холодный open/разрыв не создаёт переход.
            # Первая GPS-точка могла быть светофором: используем время открытия.
            candidate = _Candidate(selected, point, 1,
                (point.event_time-previous.event_time).total_seconds(), selected-self._next_index)
            return self._confirm(candidate, point, source='door_estimate')
        if self._candidate is None or self._candidate.index != selected:
            uncertainty = None if previous is None else (point.event_time-previous.event_time).total_seconds()
            self._candidate = _Candidate(selected, point, 1, uncertainty, selected-self._next_index)
            self._state = 'candidate'
            self._reason = 'candidate_after_gap' if after_gap else 'confirming_stop'
            return None
        candidate = self._candidate
        candidate.count += 1
        span = (point.event_time-candidate.first.event_time).total_seconds()
        if candidate.count < self.config.min_points or span < self.config.confirmation_s:
            self._state = 'candidate'
            self._reason = 'confirming_stop'
            return None
        return self._confirm(candidate, point)

    def _confirm(self, candidate: _Candidate, point: Telemetry, *, source='gps_estimate') -> GPSArrival:
        selected = candidate.index
        span = (point.event_time-candidate.first.event_time).total_seconds()
        stop = self.stops[selected]
        arrival = GPSArrival(tr_id=point.tr_id, planned_stop_id=stop.id,
            arrived_at=candidate.first.event_time, received_at=point.received_at,
            confirmation_span_s=span, uncertainty_s=candidate.uncertainty_s,
            skipped_visits=candidate.skipped, source=source)
        self._confirmed_index = self._locked_index = selected
        self._last_arrival = arrival
        self._needs_anchor = False
        self._next_index = selected+1
        self._delay_s = (arrival.arrived_at-stop.scheduled_at).total_seconds()
        self._confirmed_count += 1
        self._skipped_count += candidate.skipped
        self._candidate = None
        self._distance = None
        self._state = 'at_stop'
        self._reason = ('confirmed_by_doors' if source == 'door_estimate' else
                        'confirmed_after_skipped_visits' if candidate.skipped else 'confirmed')
        return arrival
