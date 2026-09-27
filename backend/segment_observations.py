"""Операционные наблюдения в коридоре прямой между плановыми остановками.

НЕ дорожный map matching, не признаки frozen ML и не определение причины.
Пороги фиксированы как инженерная политика, без подбора по данным:
окно 300 с после уже доступного arrival-якоря, возраст якоря <=900 с;
свежесть последнего GPS и разрыв <=30 с; хорда длиной 70..10000 м;
коридор 50 м, зоны самих остановок 35 м исключены; допустимое направление
<=60° от хорды, наблюдаемый шаг >=10 м подтверждает движение по перегону.
Невозможная скорость >130 км/ч, обратное движение, противоречивые повторы,
неизвестная скорость и невалидные координаты разрывают пригодный хвост.

Средняя скорость — арифметическое среднее сообщённых GPS speed по уникальным
пригодным моментам хвоста, НЕ длина перегона / время проезда. coverage_fraction
— доля этих моментов среди всех доступных уникальных GPS timestamps окна,
НЕ процент пройденной дороги. Простой — наблюдаемый непрерывный хвост <=1 км/ч
в круге 15 м, >=2 точек и >=5 с; это нижняя оценка длительности, без прибавления
времени от последнего GPS до текущих часов. Причина простоя не устанавливается.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime, timedelta
import math
from statistics import fmean
from typing import Literal

from pydantic import AwareDatetime, Field

from backend.arrivals import CurrentDeviation
from common.contracts import Contract, StopTarget, Telemetry


class SegmentObservation(Contract):
    method: Literal['planned_chord'] = 'planned_chord'
    status: Literal['available', 'unknown'] = 'unknown'
    reason: str
    evaluated_at: AwareDatetime
    from_stop_id: str | None = None
    from_stop_name: str | None = None
    to_stop_id: str | None = None
    to_stop_name: str | None = None
    anchor_source: str | None = None
    speed_mean_kmh: float | None = Field(default=None, ge=0, le=130)
    selected_points: int = Field(default=0, ge=0)
    considered_points: int = Field(default=0, ge=0)
    coverage_fraction: float | None = Field(default=None, ge=0, le=1)
    coverage_basis: Literal['available_unique_timestamps'] = 'available_unique_timestamps'
    observed_span_s: float | None = Field(default=None, ge=0)
    current_idle_s: float | None = Field(default=None, ge=0)
    idle_points: int = Field(default=0, ge=0)
    idle_duration_kind: Literal['observed_span_lower_bound'] = 'observed_span_lower_bound'
    latest_event_at: AwareDatetime | None = None
    window_start: AwareDatetime | None = None
    chord_length_m: float | None = Field(default=None, ge=0)
    corridor_width_m: float = 50.0
    rule_version: str = 'planned-chord-observations-v1'


def _xy(lat, lon, origin: StopTarget):
    """Локальная метрическая проекция; используется только для короткой хорды."""
    return ((lon-origin.lon)*111195*math.cos(math.radians(origin.lat)),
            (lat-origin.lat)*111195)


def _heading_difference(a, b):
    return abs((a-b+180) % 360-180)


def observe_segment(history: Iterable[Telemetry], at: datetime,
                    schedule: Sequence[StopTarget],
                    current_deviation: CurrentDeviation | None) -> SegmentObservation:
    """Наблюдать активный плановый перегон на доступных к ``at`` сообщениях.

    Вход history должен относиться к одному ТС; schedule — его план. Функция
    ничего не записывает и не использует прогнозную цель: следующий визит
    берётся непосредственно после последнего подтверждённого/оценённого прибытия.
    Без однозначной геометрии и наблюдаемого направления значения — None.
    """
    result = SegmentObservation(evaluated_at=at, reason='no_arrival_anchor')
    anchor = current_deviation
    if (anchor is None or anchor.source in ('external_hint', 'csv_snapshot')
            or anchor.planned_stop_id is None):
        return result
    result.anchor_source = anchor.source
    if (anchor.observed_at > at or anchor.received_at > at
            or anchor.received_at < anchor.observed_at):
        return result.model_copy(update={'reason':'anchor_not_available'})
    if ((at-anchor.observed_at).total_seconds() > 900
            or (anchor.valid_until is not None and at > anchor.valid_until)):
        return result.model_copy(update={'reason':'stale_arrival_anchor'})
    if len({stop.id for stop in schedule}) != len(schedule):
        return result.model_copy(update={'reason':'ambiguous_plan'})
    plan = sorted(schedule,key=lambda stop:stop.scheduled_at)
    index = next((i for i,stop in enumerate(plan) if stop.id == anchor.planned_stop_id),None)
    if index is None:
        return result.model_copy(update={'reason':'anchor_not_in_plan'})
    origin = plan[index]
    if (anchor.planned_at is not None and origin.scheduled_at != anchor.planned_at
            or abs((anchor.observed_at-origin.scheduled_at).total_seconds()-anchor.delay_s)>1e-6):
        return result.model_copy(update={'reason':'anchor_plan_mismatch'})
    result.from_stop_id, result.from_stop_name = origin.id, origin.name
    if index+1 >= len(plan):
        return result.model_copy(update={'reason':'no_next_visit'})
    destination = plan[index+1]
    if (destination.scheduled_at <= origin.scheduled_at
            or sum(stop.scheduled_at == destination.scheduled_at for stop in plan)>1):
        return result.model_copy(update={'reason':'ambiguous_next_visit'})
    result.to_stop_id, result.to_stop_name = destination.id, destination.name
    dx,dy = _xy(destination.lat,destination.lon,origin)
    length = math.hypot(dx,dy)
    result.chord_length_m = round(length,2)
    if not 70 < length <= 10000:
        return result.model_copy(update={'reason':'unsupported_chord_geometry'})
    ux,uy = dx/length,dy/length
    bearing = math.degrees(math.atan2(dx,dy)) % 360
    start = max(anchor.observed_at,at-timedelta(seconds=300))
    result.window_start = start
    available = [point for point in history if start <= point.event_time <= at and point.received_at <= at]
    if len({point.tr_id for point in available})>1:
        return result.model_copy(update={'reason':'mixed_vehicles'})
    # Дубликат не прибавляет вес; конфликт остаётся конфликтом при любых
    # последующих повторах, в том числе если последний снова похож на первый.
    by_time = {}
    for point in available:
        signature = (point.lat,point.lon,point.speed_kmh,point.heading,point.location_valid,
                     point.received_at >= point.event_time)
        if point.event_time not in by_time:
            by_time[point.event_time] = (point,signature)
        elif by_time[point.event_time][1] != signature:
            by_time[point.event_time] = (None,None)
    points = sorted(by_time.items())
    result.considered_points = len(points)
    if not points:
        return result.model_copy(update={'reason':'no_available_gps'})
    result.latest_event_at = points[-1][0]
    if (at-points[-1][0]).total_seconds()>30:
        return result.model_copy(update={'reason':'stale_gps'})

    selected = []
    reason = 'insufficient_continuous_points'
    has_direction = False
    for timestamp,(point,_) in reversed(points):
        if (point is None or not point.location_valid or point.lat is None or point.lon is None
                or point.speed_kmh is None or point.speed_kmh>130 or point.received_at<point.event_time):
            reason='invalid_or_conflicting_gps';break
        x,y = _xy(point.lat,point.lon,origin)
        along,cross = x*ux+y*uy,abs(x*uy-y*ux)
        if not 0 <= along <= length or cross>50:
            reason='outside_planned_chord';break
        if math.hypot(x,y)<=35 or math.hypot(x-dx,y-dy)<=35:
            reason='inside_stop_zone';break
        if point.speed_kmh>5 and point.heading is not None and _heading_difference(point.heading,bearing)>60:
            reason='heading_disagrees_with_chord';break
        if selected:
            newer,nx,ny = selected[-1]
            elapsed = (newer.event_time-timestamp).total_seconds()
            if elapsed>30:
                reason='gps_gap';break
            sx,sy = nx-x,ny-y
            step = math.hypot(sx,sy)
            if step/elapsed*3.6>130:
                reason='implausible_displacement';break
            if step>=10:
                cosine = (sx*ux+sy*uy)/step
                if cosine < .5:
                    reason='movement_disagrees_with_chord';break
                if point.speed_kmh<=1 and newer.speed_kmh<=1 and step>15:
                    reason='speed_position_conflict';break
                has_direction=True
        selected.append((point,x,y))
    # Две неподвижные точки внутри коридора без наблюдения подхода не доказывают,
    # что автобус находится на активном направлении данного перегона.
    if len(selected)<2 or not has_direction:
        return result.model_copy(update={'reason':reason if len(selected)<2 else 'direction_not_observed'})
    result.status='available'
    result.reason='observed_in_planned_chord'
    result.selected_points=len(selected)
    result.coverage_fraction=len(selected)/len(points)
    result.speed_mean_kmh=fmean(point.speed_kmh for point,_,_ in selected)
    result.observed_span_s=(selected[0][0].event_time-selected[-1][0].event_time).total_seconds()
    idle=[]
    latest,x0,y0=selected[0]
    for point,x,y in selected:
        if point.speed_kmh>1 or math.hypot(x-x0,y-y0)>15:
            break
        idle.append(point)
    if len(idle)>=2:
        duration=(idle[0].event_time-idle[-1].event_time).total_seconds()
        if duration>=5:
            result.current_idle_s=duration
            result.idle_points=len(idle)
    return result
