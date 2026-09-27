"""Условная подача дополнительного автобуса по свежему срезу, без команд ТС.

Это пространственный what-if, а не оптимизатор перевозок. Нагрузка, резерв,
смена водителя и возможность снять автобус с другой линии неизвестны.
В расчёте участвуют только текущий GPS, доступный прогноз и плановые времена.
"""
from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Mapping

from common.transfer import DonorImpact, TransferAdvice, TransferScenario, TransferVehicle

NEEDED_DELAY_S = 150
MAX_DONOR_DELAY_S = 60
MAX_DISTANCE_M = 5000
MAX_DATA_AGE_S = 60
MIN_BENEFIT_S = 60


def _time(value):
    return value if isinstance(value, datetime) else datetime.fromisoformat(value.replace('Z', '+00:00'))


def _distance(a, b):
    """Гаверсин; пары передаются как (lon, lat)."""
    lon1, lat1, lon2, lat2 = map(math.radians, (*a, *b))
    h = math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
    return 6371000 * 2 * math.asin(min(1, math.sqrt(h)))


def _geometry(route):
    stops = route.get('stops') or []
    raw = [(s['lon'], s['lat']) for s in stops] if stops else route.get('path', [])
    points = list(dict.fromkeys((round(p[0], 5), round(p[1], 5)) for p in raw))
    # Ограничение работы на custom-планах с тысячами промежуточных вершин.
    return points if len(points) <= 256 else [points[round(i*(len(points)-1)/255)] for i in range(256)]


def same_line(a, b):
    """ID плана одного ТС не доказывает другую линию.

    Совпадающие/вложенные геометрии считаем одной линией в обе стороны.
    Один общий пересадочный узел не объединяет разные линии.
    """
    if a.get('route_id') == b.get('route_id'):
        return True
    left, right = _geometry(a), _geometry(b)
    if not left or not right:
        return True  # Другую линию нельзя подтвердить без геометрии.
    if len(left) > len(right):
        left, right = right, left
    return _same_geometry(tuple(left), tuple(right))


@lru_cache(maxsize=256)
def _same_geometry(left, right):
    # Географическая сетка отсекает далёкие остановки до гаверсина. Масштаб
    # долготы взят по крайней широте: ячейки не мельче радиуса сопоставления.
    lat_scale = 6371000 * math.pi / 180 / 80
    lon_scale = lat_scale * max(.00001, math.cos(math.radians(max(abs(p[1]) for p in (*left, *right)))))
    origin = left[0][0]
    def cell(point):
        longitude = (point[0]-origin+180) % 360-180
        return math.floor(longitude*lon_scale), math.floor(point[1]*lat_scale)
    grid = defaultdict(list)
    for point in right:
        grid[cell(point)].append(point)
    matches = 0
    for point in left:
        x, y = cell(point)
        if any(_distance(point, q) <= 80 for dx in (-1, 0, 1) for dy in (-1, 0, 1)
               for q in grid.get((x+dx, y+dy), ())):
            matches += 1
    return matches >= min(2, len(left)) and matches / len(left) >= .6


def _ready(vehicle, now):
    prediction = vehicle.get('prediction') or {}
    age = vehicle.get('age_s')
    if (vehicle.get('status') != 'fresh' or age is None or not 0 <= age <= MAX_DATA_AGE_S
            or vehicle.get('lat') is None or vehicle.get('lon') is None
            or (vehicle.get('prediction_availability') or {}).get('code') != 'ready'
            or prediction.get('risk') == 'unknown' or prediction.get('predicted_delay_s') is None):
        return False
    issued_at = prediction.get('issued_at')
    event_time = vehicle.get('event_time')
    target = prediction.get('target') or {}
    return bool(issued_at and event_time and target.get('scheduled_at')
                and 0 <= (now-_time(issued_at)).total_seconds() <= MAX_DATA_AGE_S
                and 0 <= (now-_time(event_time)).total_seconds() <= MAX_DATA_AGE_S
                and 600 < (_time(target['scheduled_at'])-now).total_seconds() <= 900)


def _vehicle(vehicle, routes):
    return TransferVehicle(tr_id=vehicle['tr_id'], label=vehicle['label'], route_id=vehicle['route_id'],
                           route_name=routes.get(vehicle['route_id'], {}).get('name', vehicle['route_id']))


def advise_transfer(snapshot: Mapping, schedule: Mapping, tr_id: int) -> TransferAdvice:
    """Чистый расчёт над атомарным state и известным планом, без чтения replay.

    schedule содержит только StopTarget. Не читаем ни будущую GPS-очередь, ни
    фактические времена расписания. Отсутствующий автобус вызывает KeyError.
    """
    now = _time(snapshot['clock_time'])
    routes = {r['route_id']: r for r in snapshot['routes']}
    buses = {v['tr_id']: v for v in snapshot['vehicles']}
    target = buses[tr_id]
    base = dict(calculated_at=now, context_version=snapshot['context']['version'],
                target=_vehicle(target, routes))
    if not _ready(target, now):
        return TransferAdvice(**base, status='unavailable', reason='Для сценария нужен актуальный прогноз и свежая позиция выбранного автобуса')
    forecast = target['prediction']
    if forecast['predicted_delay_s'] < NEEDED_DELAY_S:
        return TransferAdvice(**base, status='not_needed', reason='Перевод с другой линии сейчас не требуется: прогноз задержки меньше 2,5 минут')
    target_route = routes.get(target['route_id'])
    if not target_route:
        return TransferAdvice(**base, status='unavailable', reason='Не загружена геометрия целевой линии')
    stop = forecast['target']
    planned_at = _time(stop['scheduled_at'])
    current_arrival = planned_at + timedelta(seconds=forecast['predicted_delay_s'])
    choices, considered = [], 0
    for candidate in buses.values():
        if candidate['tr_id'] == tr_id or not _ready(candidate, now):
            continue
        if candidate['prediction']['predicted_delay_s'] > MAX_DONOR_DELAY_S:
            continue
        distance = _distance((candidate['lon'], candidate['lat']), (stop['lon'], stop['lat']))
        if distance > MAX_DISTANCE_M:
            continue
        route = routes.get(candidate['route_id'])
        if not route or same_line(target_route, route):
            continue
        considered += 1
        relocation = math.ceil(120 + distance * 1.4 / (20 / 3.6))
        arrival = now + timedelta(seconds=relocation)
        service = max(arrival, planned_at)
        benefit = math.floor((current_arrival-service).total_seconds())
        if benefit >= MIN_BENEFIT_S:
            choices.append((distance, candidate['tr_id'], candidate, relocation, arrival, service, benefit))
    if not choices:
        return TransferAdvice(**base, status='unavailable', candidates_considered=considered,
                              reason='Поблизости нет подходящего автобуса с другой линии, который по сценарию подойдёт хотя бы на минуту раньше')
    distance, _, donor, relocation, arrival, service, benefit = min(choices, key=lambda c: (c[0], c[1]))
    future_plan = sorted((s for s in schedule.get(donor['tr_id'], []) if s.scheduled_at > now),
                         key=lambda s: s.scheduled_at)
    next_stop = future_plan[0] if future_plan else None
    # Подача дополнительного автобуса не заканчивается возвращением на свою
    # линию. Считаем задания до обслуживания цели; дальнейший ущерб неизвестен.
    conflicts = sum(s.scheduled_at <= service for s in future_plan)
    note = ('Перевод затрагивает ближайшие плановые остановки донора; последующие рейсы требуют перепланирования'
            if conflicts else 'Время возвращения на свою линию и последующие рейсы не рассчитаны')
    if not next_stop:
        note = 'В загруженном плане нет будущих остановок; это не подтверждает доступность автобуса'
    return TransferAdvice(**base, status='ready', reason='Рассмотреть подачу ближайшего подходящего автобуса с другой линии',
        donor=_vehicle(donor, routes), meeting_stop=stop,
        scenario=TransferScenario(distance_m=round(distance), relocation_s=relocation,
                                  donor_arrival_at=arrival, service_at=service,
                                  current_bus_arrival_at=current_arrival, earlier_by_s=benefit),
        donor_impact=DonorImpact(next_stop_name=next_stop.name if next_stop else None,
                                next_stop_at=next_stop.scheduled_at if next_stop else None,
                                planned_stops_during_transfer=conflicts, plan_conflict=bool(conflicts), note=note),
        assumptions=[
            'Если автобус можно снять с линии; загрузку и резерв подтверждает диспетчер',
            'Подача: расстояние по прямой × 1,4, скорость 20 км/ч, подготовка 2 минуты; дороги и пробки не проверены',
            'Дополнительный автобус обслуживает остановку не раньше плана; прогноз исходного автобуса не меняется',
            'Сходные планы объединяются по геометрии; идентификаторы реальных маршрутов в CSV не заданы',
        ], candidates_considered=considered)
