"""Наблюдаемые паттерны движения. Это эвристики, не объяснение весов модели."""
from datetime import datetime, timedelta
from statistics import median
from typing import Literal
import math

from backend.arrivals import CurrentDeviation

from pydantic import AwareDatetime
from common.contracts import Contract, StopTarget, Telemetry


class Observation(Contract):
    code: str
    title: str
    evidence: str
    kind: Literal['movement', 'data_quality', 'schedule']


class Explanation(Contract):
    evaluated_at: AwareDatetime
    cause_status: Literal['hypothesis', 'unknown']
    possible_cause: str
    observations: list[Observation]
    recommendation: str
    summary: str = 'Недостаточно наблюдений'
    observation_status: Literal['observed', 'no_pattern', 'insufficient_data'] = 'insufficient_data'
    rule_version: str = 'telemetry-rules-v2'


def usable_history(history, at: datetime) -> list[Telemetry]:
    """Одинаковая фильтрация для признаков и объяснений: оба времени ≤ T."""
    return [p for p in history if p.event_time <= at and p.received_at <= at
            and p.location_valid and p.lat is not None and p.lon is not None
            and (p.speed_kmh is None or p.speed_kmh <= 130)]


def _distance_m(left, right):
    lat1, lat2 = math.radians(left.lat), math.radians(right.lat)
    a = math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(math.radians(right.lon-left.lon)/2)**2
    return 6371000*2*math.asin(min(1, math.sqrt(a)))


def _available_deviation(value, at):
    return (value is not None and value.observed_at <= at and value.received_at <= at
            and value.received_at >= value.observed_at
            and (value.valid_until is None or at <= value.valid_until))


def _schedule_observations(result, current, history, at):
    """Динамика между разными уже известными посещениями, не физическая причина."""
    if not _available_deviation(current, at):
        return None
    sign = '+' if current.delay_s > 0 else ''
    result.observations.append(Observation(code='current_deviation', title='Последнее известное отклонение',
        evidence=f'{sign}{current.delay_s:.0f} с относительно плана; наблюдение {(at-current.observed_at).total_seconds():.0f} с назад',
        kind='schedule'))
    visits = {}
    for value in history:
        # valid_until не применяется к историческому факту: сравниваются сами
        # состоявшиеся наблюдения. Будущие версии одного визита исключаются.
        if (value.observed_at > at or value.received_at > at or value.received_at < value.observed_at
                or value.planned_stop_id is None):
            continue
        old = visits.get(value.planned_stop_id)
        if old is None or (value.received_at, value.source == 'arrival') >= (old.received_at, old.source == 'arrival'):
            visits[value.planned_stop_id] = value
    previous = max((value for value in visits.values()
        if value.planned_stop_id != current.planned_stop_id
        and 0 < (current.observed_at-value.observed_at).total_seconds() <= 900),
        key=lambda value: value.observed_at, default=None)
    if previous is None or current.planned_stop_id is None:
        return None
    change = current.delay_s-previous.delay_s
    if abs(change) >= 30:
        grew = change > 0
        result.observations.append(Observation(code='deviation_increased' if grew else 'deviation_decreased',
            title='Оценка отклонения выросла' if grew else 'Оценка отклонения снизилась',
            evidence=f'{previous.delay_s:+.0f} → {current.delay_s:+.0f} с между двумя посещениями; изменение {change:+.0f} с. Это динамика отклонения, а не установленная причина.',
            kind='schedule'))
        return change
    return None


def explain(history, at: datetime, *, current_deviation: CurrentDeviation | None = None,
            deviation_history=(), schedule: list[StopTarget] | tuple[StopTarget, ...] = ()) -> Explanation:
    """Наблюдаемые движение и отклонение отдельно от гипотезы о причине.

    Сохранены пороги длинной стоянки/падения скорости. Дополнительно показываем
    короткий наблюдаемый простой (>=30 с), медленное движение (медиана <=10 км/ч
    за >=90 с) и доступную динамику отклонения между разными посещениями.
    Для гипотезы о вкладе движения требуется известное положительное отклонение.
    Стоянка у плановой остановки сама по себе не объявляется причиной: это может
    быть штатная посадка или отстой. Пробка/ДТП/погода из GPS не определяются.
    """
    history = list(history)
    valid = [p for p in usable_history(history, at) if p.received_at >= p.event_time]
    latest = max(valid, key=lambda p:p.event_time, default=None)
    result = Explanation(evaluated_at=at, cause_status='unknown',
        possible_cause='Физическая причина отклонения не установлена', observations=[],
        recommendation='Продолжить наблюдение; при росте отклонения уточнить ситуацию у водителя')
    if latest is None or (at-latest.event_time).total_seconds() > 60:
        age = round((at-latest.event_time).total_seconds()) if latest else None
        result.summary='Нет свежей телеметрии'
        result.observations.append(Observation(code='stale',title=result.summary,
            evidence=f'Возраст последней точки: {age} с' if latest else 'Валидные точки на этот момент отсутствуют',kind='data_quality'))
        result.recommendation='Проверить связь с ТС; причина задержки по этим данным не определяется'
        _schedule_observations(result, current_deviation, deviation_history, at)
        return result

    # Конфликт любого повтора остаётся конфликтом; повтор исходного пакета
    # не восстанавливает пригодность. Координаты также входят в идентичность.
    by_time = {}
    for p in valid:
        if p.event_time < at-timedelta(minutes=5):
            continue
        signature=(p.speed_kmh,p.lat,p.lon)
        if p.event_time not in by_time:
            by_time[p.event_time]=(p,signature)
        elif by_time[p.event_time][1] != signature:
            by_time[p.event_time]=(None,None)
    continuous=[]
    points=sorted(by_time.items())
    if points and (at-points[-1][0]).total_seconds() <= 30:
        for t,(point,_) in reversed(points):
            if point is None or point.speed_kmh is None or (continuous and (continuous[-1].event_time-t).total_seconds()>30):
                break
            continuous.append(point)
        continuous.reverse()
    stopped=[]
    for point in reversed(continuous):
        if point.speed_kmh > 1 or _distance_m(point, continuous[-1]) > 35:
            break
        stopped.append(point)
    stop_span=(stopped[0].event_time-stopped[-1].event_time).total_seconds() if stopped else 0
    span=(continuous[-1].event_time-continuous[0].event_time).total_seconds() if continuous else 0
    enough=len(continuous)>=6 and span>=90
    known_delay=current_deviation.delay_s if _available_deviation(current_deviation, at) else None
    nearby=[stop for stop in schedule if _distance_m(latest,stop)<=35]
    physical={(stop.lat,stop.lon) for stop in nearby}
    stop_name=nearby[0].name if len(physical)==1 else None
    movement_code=None
    if len(stopped)>=3 and stop_span>=30:
        long_stop=len(stopped)>=4 and stop_span>=90
        movement_code='long_stop' if long_stop else 'observed_stop'
        result.summary=f'Почти не движется не менее {round(stop_span)} с'
        if stop_name:
            result.summary+=f' · у остановки «{stop_name}»'
        result.observations.append(Observation(code=movement_code,title=result.summary,
            evidence=f'Скорость ≤1 км/ч на протяжении {round(stop_span)} с; {len(stopped)} измерений, разрывы не более 30 с. Это наблюдаемый минимум; назначение стоянки неизвестно.',kind='movement'))
        if long_stop and known_delay is not None and known_delay>60 and schedule and not nearby:
            result.possible_cause='Продолжительная остановка вне плановых остановок может усиливать отставание; причина остановки неизвестна'
            result.cause_status='hypothesis'
            result.recommendation='Уточнить причину остановки у водителя и проверить необходимость помощи'
    else:
        recent=[p.speed_kmh for p in continuous if p.event_time>=at-timedelta(seconds=60)]
        previous=[p.speed_kmh for p in continuous if at-timedelta(seconds=180)<=p.event_time<at-timedelta(seconds=60)]
        recent_times=[p.event_time for p in continuous if p.event_time>=at-timedelta(seconds=60)]
        previous_times=[p.event_time for p in continuous if at-timedelta(seconds=180)<=p.event_time<at-timedelta(seconds=60)]
        if len(recent)>=3 and len(previous)>=4 and (recent_times[-1]-recent_times[0]).total_seconds()>=20 and (previous_times[-1]-previous_times[0]).total_seconds()>=30:
            old,new=median(previous),median(recent)
            if old>=15 and new<=10 and new<=old*.5:
                movement_code='speed_drop'
                result.summary=f'Скорость снизилась: {old:g} → {new:g} км/ч'
                result.observations.append(Observation(code=movement_code,title=result.summary,
                    evidence='Медианы предыдущих 2 минут и последней минуты; только доступные непрерывные GPS.',kind='movement'))
        if movement_code is None and enough and 1<median(p.speed_kmh for p in continuous)<=10:
            movement_code='slow_movement'
            result.summary=f'Медленное движение · медиана {median(p.speed_kmh for p in continuous):g} км/ч'
            result.observations.append(Observation(code=movement_code,title=result.summary,
                evidence=f'{len(continuous)} измерений за {round(span)} с; разрывы не более 30 с. Норма скорости этого участка неизвестна.',kind='movement'))
        if movement_code in ('speed_drop','slow_movement') and known_delay is not None and known_delay>60 and not nearby:
            result.cause_status='hypothesis'
            result.possible_cause='Замедление движения может поддерживать отставание; дорожная причина не подтверждена'
            result.recommendation='Проверить динамику отклонения и уточнить ситуацию у водителя'
    if movement_code:
        result.observation_status='observed'
    else:
        result.observation_status='no_pattern' if enough else 'insufficient_data'
        result.summary=(f'Текущая скорость {latest.speed_kmh:g} км/ч; выраженного паттерна не видно'
            if enough else 'Недостаточно непрерывной истории для вывода')
        result.observations.append(Observation(code='no_pattern' if enough else 'short_history',
            title='Выраженный паттерн не обнаружен' if enough else 'Недостаточно непрерывной истории',
            evidence=f'{len(continuous)} последовательных точек за {round(span)} с; правила стоянки и снижения скорости не сработали' if enough else 'Для динамики нужны валидные измерения без разрывов более 30 с',kind='data_quality'))
    change=_schedule_observations(result, current_deviation, deviation_history, at)
    if change is not None:
        result.observation_status='observed'
        if movement_code is None:
            result.summary=f'Оценка отклонения {"выросла" if change>0 else "снизилась"} на {abs(change):.0f} с'
    return result
