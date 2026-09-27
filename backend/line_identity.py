"""Сопоставление планов одной линии, когда CSV выдаёт отдельный ID каждому ТС.

Общий route_id считается явной связью. Иначе используем геометрию: не менее
80% остановок каждого плана должны совпадать с допуском 80 м. Это приближение
для планов без номера маршрута, а не восстановленный официальный маршрут.
"""
from __future__ import annotations

import math
from collections import defaultdict
from functools import lru_cache
from typing import Iterable, Mapping

MATCH_RADIUS_M = 80
MIN_COVERAGE = .8
EARTH_RADIUS_M = 6_371_000


def _distance(a, b):
    lon1, lat1, lon2, lat2 = map(math.radians, (*a, *b))
    h = math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
    return EARTH_RADIUS_M * 2 * math.asin(min(1, math.sqrt(h)))


def _geometry(route):
    """Все уникальные координаты; повторы рейсов и сдвиг начала не важны."""
    stops = route.get('stops') or []
    raw = [(s.get('lon'), s.get('lat')) for s in stops] if stops else route.get('path') or []
    points = set()
    for point in raw:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            continue
        try:
            lon, lat = map(float, point)
        except (TypeError, ValueError):
            continue
        if math.isfinite(lon) and math.isfinite(lat) and -180 <= lon <= 180 and -90 <= lat <= 90:
            points.add((round(lon, 6), round(lat, 6)))
    return tuple(sorted(points))


def has_line_geometry(route: Mapping) -> bool:
    """Есть хотя бы две точки: одного узла недостаточно для выбора донора."""
    return len(_geometry(route)) >= 2


def _separate_stops(points):
    # Несколько платформ одного пересадочного узла не доказывают общую линию.
    if len(points) < 2:
        return False
    first = points[0]
    far = max(points, key=lambda point: _distance(first, point))
    return any(_distance(far, point) > MATCH_RADIUS_M for point in points)


@lru_cache(maxsize=4096)
def _same_geometry(left, right):
    if len(left) < 2 or len(right) < 2:
        return False
    if left == right:
        return _separate_stops(left)
    # Географическая сетка оставляет только соседние точки для гаверсина.
    # Масштаб долготы не даёт ячейке стать меньше радиуса сопоставления.
    lat_scale = EARTH_RADIUS_M * math.pi / 180 / MATCH_RADIUS_M
    lon_scale = lat_scale * max(.00001, math.cos(math.radians(max(abs(p[1]) for p in (*left, *right)))))
    origin = left[0][0]

    def cell(point):
        longitude = (point[0]-origin+180) % 360-180
        return math.floor(longitude*lon_scale), math.floor(point[1]*lat_scale)

    grid = defaultdict(list)
    for i, point in enumerate(right):
        grid[cell(point)].append((i, point))
    matched_left, matched_right = [], set()
    for point in left:
        x, y = cell(point)
        matches = [i for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                   for i, other in grid.get((x+dx, y+dy), ())
                   if _distance(point, other) <= MATCH_RADIUS_M]
        if matches:
            matched_left.append(point)
            matched_right.update(matches)
    return (len(matched_left) / len(left) >= MIN_COVERAGE
            and len(matched_right) / len(right) >= MIN_COVERAGE
            and _separate_stops(matched_left)
            and _separate_stops([right[i] for i in matched_right]))


def same_line(a: Mapping, b: Mapping) -> bool:
    """Прямая связь двух планов, без транзитивного объединения коридоров."""
    if a.get('route_id') and a.get('route_id') == b.get('route_id'):
        return True
    left, right = sorted((_geometry(a), _geometry(b)))
    return _same_geometry(left, right)


def line_memberships(routes: Iterable[Mapping]) -> dict[str, list[str]]:
    """Для каждого плана вернуть себя и непосредственно сходные планы.

    Вызывать при загрузке/смене контекста. Кэш зависит от координат, поэтому
    повторное использование route_id в новом архиве не сохраняет старую связь.
    A≈B и B≈C не означают A≈C: списки не являются компонентами связности.
    """
    plans = {route['route_id']: _geometry(route) for route in routes if route.get('route_id')}
    members = {identifier: [identifier] for identifier in plans}
    entries = list(plans.items())
    for i, (left_id, left) in enumerate(entries):
        for right_id, right in entries[i+1:]:
            if _same_geometry(*sorted((left, right))):
                members[left_id].append(right_id)
                members[right_id].append(left_id)
    return members
