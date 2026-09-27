"""Причинное воспроизведение validate/train: телеметрия и только план расписания.

Будущие события лежат в очереди проигрывателя, а не в истории Engine/ML.
Доставка GPS — max(event_time, receive_time). В csv_snapshot наблюдения cur_dev_s
доступны с T; в gps они не читаются и отклонение восстанавливает Engine.
Часовой пояс naive CSV задаётся явно; UTC по умолчанию согласуется с sample_id,
но не является подтверждением часового пояса со стороны организаторов.
"""
from __future__ import annotations

import csv
from bisect import bisect_right
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import AwareDatetime, Field, field_validator, model_validator

from common.contracts import Contract, Telemetry, parse_manual_fill

MAX_REPLAY_EVENTS = 500000


class ReplayConfig(Contract):
    dataset_split: Literal['validate', 'train'] = 'validate'
    start: AwareDatetime = datetime(2026, 1, 6, 11, 30, tzinfo=timezone.utc)
    duration_minutes: int | None = Field(default=None, ge=1, le=120,
        description='null: до последнего доступного события архива; число: верхняя граница окна для короткой проверки')
    timezone: str = 'UTC'
    tr_ids: list[int] | None = Field(default=None, min_length=1, max_length=128,
        description='null: все ТС с планом и сообщениями в срезе с предысторией; список: ручной выбор')
    speed: float = Field(default=10, ge=1, le=30)
    paused: bool = True
    deviation_source: Literal['csv_snapshot', 'gps'] = 'csv_snapshot'
    warmup_minutes: int = Field(default=5, ge=0, le=120,
                                description='Сколько доступной истории перед start доставить для прогрева; будущие события остаются в очереди')

    @model_validator(mode='after')
    def available_source(self):
        if self.dataset_split == 'train' and self.deviation_source != 'gps':
            raise ValueError('Для train доступен только расчёт по GPS: points.csv нет, labels не используются')
        return self

    @property
    def plan_file(self):
        return 'train/schedule.csv' if self.dataset_split == 'train' else 'validate/schedule_plan.csv'

    @property
    def points_file(self):
        return 'validate/points.csv' if self.deviation_source == 'csv_snapshot' else None

    @field_validator('timezone')
    @classmethod
    def known_zone(cls, value):
        try:
            ZoneInfo(value)
        except (KeyError, ValueError) as exc:
            raise ValueError('Неизвестный часовой пояс IANA') from exc
        return value

    @field_validator('tr_ids')
    @classmethod
    def unique_ids(cls, value):
        if value is None:
            return value
        if any(i <= 0 for i in value) or len(set(value)) != len(value):
            raise ValueError('Нужны уникальные положительные tr_id')
        return value


class ReplayState(Contract):
    dataset_split: Literal['validate', 'train']
    plan_file: str
    points_file: str | None
    start: AwareDatetime
    end: AwareDatetime
    duration_minutes: int | None
    timezone: str
    running: bool
    finished: bool
    speed: float
    tr_ids: list[int]
    selection_mode: Literal['all_available', 'manual']
    fleet: dict
    streams: list[dict] = Field(description='Диагностика архива для проигрывателя, не онлайн-признаки модели')
    deviation_source: Literal['csv_snapshot', 'gps']
    warmup_minutes: int
    delivered: int
    total: int
    sources_sha256: dict[str, str]
    quality: dict[str, int]
    note: str


class ReplayControl(Contract):
    action: Literal['pause', 'resume', 'reset', 'speed']
    speed: float | None = Field(default=None, ge=1, le=30)


@dataclass
class LoadedReplay:
    config: ReplayConfig
    context: object
    events: list
    sources: dict
    quality: dict
    cursor: int = 0
    fleet: dict = field(default_factory=dict)
    telemetry_times: dict[int, list[datetime]] = field(default_factory=dict)
    position_times: dict[int, list[datetime]] = field(default_factory=dict)

    @property
    def end(self):
        # События уже отфильтрованы по выбранным ТС/окну и отсортированы.
        # План и пустой хвост явного окна не продлевают проигрывание.
        return max(self.config.start, self.events[-1][0]) if self.events else self.config.start

    def deliver(self, engine):
        while self.cursor < len(self.events) and self.events[self.cursor][0] <= engine.clock:
            _, kind, payload = self.events[self.cursor]
            if kind == 'gps':
                engine.ingest(payload)
            else:
                engine.hints[payload.tr_id].append(payload)
            self.cursor += 1
        if self.cursor == len(self.events) and engine.clock >= self.end:
            engine.running = False

    def state(self, engine):
        streams = []
        for vehicle in self.context.vehicles:
            times = self.telemetry_times.get(vehicle.tr_id, [])
            positions = self.position_times.get(vehicle.tr_id, [])
            index, position_index = bisect_right(times, engine.clock), bisect_right(positions, engine.clock)
            streams.append(dict(tr_id=vehicle.tr_id, delivered=index, remaining=len(times)-index,
                last_delivery_at=times[index-1] if index else None,
                next_delivery_at=times[index] if index < len(times) else None,
                next_position_delivery_at=positions[position_index] if position_index < len(positions) else None))
        source_note = ('Отклонение рассчитывается из GPS и плана; points.csv не используется. '
                       'До распознавания посещения отклонение может быть неизвестно. '
                       if self.config.deviation_source == 'gps' else
                       'Готовое отклонение из points.csv доступно с T. ')
        return ReplayState(dataset_split=self.config.dataset_split, plan_file=self.config.plan_file,
            points_file=self.config.points_file, start=self.config.start, end=self.end,
            duration_minutes=self.config.duration_minutes, timezone=self.config.timezone,
            running=engine.running, finished=engine.clock >= self.end, speed=engine.speed,
            tr_ids=[v.tr_id for v in self.context.vehicles],
            selection_mode='all_available' if self.config.tr_ids is None else 'manual',
            fleet=self.fleet, streams=streams, deviation_source=self.config.deviation_source,
            warmup_minutes=self.config.warmup_minutes,
            delivered=self.cursor, total=len(self.events), sources_sha256=self.sources, quality=self.quality,
            note=(f'Исторические {self.config.dataset_split} CSV. '+source_note+
                  ('Train содержит реальную и синтетическую телеметрию; перенос вероятности на этот набор не проверен. '
                   'Фактические прибытия schedule.csv и labels не используются. ' if self.config.dataset_split == 'train' else '')+
                  'TTL 300 с — правило приложения. '
                  'Часовой пояс naive CSV — допущение. Линии соединяют плановые остановки, '
                  'не являются дорожной геометрией. MAE здесь не измеряется.'))


def load_replay(root: Path, config: ReplayConfig) -> LoadedReplay:
    """Читает выбранный срез, проверяет связи; labels не открывает.

    Только фиксированные имена в каталоге из REPLAY_DATA_DIR. В режиме gps
    points.csv вообще не открывается. API не принимает
    пути. Загрузка атомарна: при ошибке текущий режим остаётся прежним.
    """
    from backend.engine import DelayHint, LiveContext, Route, RouteStop, ScheduledStop, VehicleConfig
    from backend.diagnostics import usable_history
    from common.contracts import StopTarget

    zone = ZoneInfo(config.timezone)
    def dt(value):
        parsed = datetime.fromisoformat(value)
        return parsed.replace(tzinfo=zone) if parsed.tzinfo is None else parsed

    def rows(relative_path, columns=None):
        with (root / relative_path).open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                # В train schedule есть будущие факты. За границу чтения строки
                # проходят только явно разрешённые плановые поля.
                yield {name: row.get(name, '') for name in columns} if columns else row

    sources = {}
    traffic_file = config.dataset_split+'/traffic.csv'
    file_names = [traffic_file, config.plan_file]
    if config.points_file:
        file_names.append(config.points_file)
    for name in file_names:
        with (root / name).open('rb') as stream:
            sources[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
    requested_ids = set(config.tr_ids) if config.tr_ids is not None else None
    warmup = config.start - timedelta(minutes=config.warmup_minutes)
    window_end = (config.start + timedelta(minutes=config.duration_minutes)
                  if config.duration_minutes is not None else None)
    plan, planned_ids = [], set()
    for row in rows(config.plan_file, ('tr_id', 'time_begin', 'tt_action_item_id',
                                     'geom', 'building_address', 'manual_fill')):
        tr_id = int(row['tr_id'])
        planned_ids.add(tr_id)
        if requested_ids is not None and tr_id not in requested_ids:
            continue
        at = dt(row['time_begin'])
        # Frozen builder использует длину/позиции всего плана ТС. Время ограничивает
        # доставку телеметрии, а не заранее известные плановые посещения.
        match = re.fullmatch(r'POINT\s*\(\s*([-+\d.eE]+)\s+([-+\d.eE]+)\s*\)', row['geom'])
        if not match:
            raise ValueError('Некорректный WKT плановой остановки')
        lon, lat = map(float, match.groups())
        plan.append(ScheduledStop(tr_id=tr_id, target=StopTarget(id=row['tt_action_item_id'],
            name=row['building_address'] or row['tt_action_item_id'], scheduled_at=at, lon=lon, lat=lat,
            manual_fill=parse_manual_fill(row.get('manual_fill')))))
        if len(plan) > 20000:
            raise ValueError('Слишком большой срез расписания')
    plan.sort(key=lambda s: (s.target.scheduled_at, s.target.id))
    ids = requested_ids if requested_ids is not None else planned_ids
    if len(ids) > 128:
        raise ValueError('Более 128 ТС; выберите автобусы вручную')
    events, units = [], {}
    telemetry_without_plan = set()
    quality = dict(telemetry=0, snapshots=0, invalid_rows=0, invalid_locations=0,
                   receive_before_event=0, target_checks=0)
    for row in rows(traffic_file):
        tr_id = int(row['tr_id'])
        if tr_id not in planned_ids:
            telemetry_without_plan.add(tr_id)
        if tr_id not in ids:
            continue
        event, received = dt(row['event_time']), dt(row['receive_time'])
        available = max(event, received)
        if available < warmup or (window_end is not None and available > window_end):
            continue
        unit = int(row['unit_id'])
        if tr_id in units and units[tr_id] != unit:
            raise ValueError('Смена unit_id внутри среза требует отдельного маппинга')
        try:
            point = Telemetry(tr_id=tr_id, unit_id=unit, event_time=event, received_at=received,
                lat=float(row['lat']) if row['lat'] else None, lon=float(row['lon']) if row['lon'] else None,
                speed_kmh=float(row['speed']) if row['speed'] else None,
                heading=float(row['heading']) if row['heading'] else None,
                location_valid=row['location_valid'].lower() == 'true', source='replay', event_id=row['packet_id'])
        except ValueError:
            quality['invalid_rows'] += 1
            continue
        units[tr_id] = unit
        quality['telemetry'] += 1
        quality['invalid_locations'] += int(not point.location_valid or point.lat is None or point.lon is None)
        quality['receive_before_event'] += int(received < event)
        events.append((available, 'gps', point))
        if len(events) > MAX_REPLAY_EVENTS:
            raise ValueError(f'Более {MAX_REPLAY_EVENTS} событий; выберите ТС или задайте duration_minutes явно')
    for row in (rows(config.points_file) if config.points_file else ()):
        tr_id, at = int(row['tr_id']), dt(row['T'])
        # Context содержит только ТС с принятой телеметрией. Подсказки остальных
        # не становятся сиротскими входами и не продлевают конец replay.
        if tr_id not in units or at < warmup or (window_end is not None and at > window_end):
            continue
        if at >= config.start:
            targets = [s.target for s in plan if s.tr_id == tr_id and 600 < (s.target.scheduled_at-at).total_seconds() <= 900]
            first_at = min((s.scheduled_at for s in targets), default=None)
            if not any(s.id == row['target_stop_id'] and s.scheduled_at == dt(row['target_time_begin']) == first_at
                       for s in targets):
                raise ValueError('Цель points.csv не совпала с первым плановым пунктом в окне 10–15 минут')
            quality['target_checks'] += 1
        if row['cur_dev_s']:
            events.append((at, 'hint', DelayHint(tr_id=tr_id, observed_at=at, received_at=at,
                delay_s=float(row['cur_dev_s']), source='csv_snapshot', sample_id=row['sample_id'],
                target_stop_id=row['target_stop_id'], target_time_begin=dt(row['target_time_begin']))))
            quality['snapshots'] += 1
            if len(events) > MAX_REPLAY_EVENTS:
                raise ValueError(f'Более {MAX_REPLAY_EVENTS} событий; выберите ТС или задайте duration_minutes явно')
    missing_ids = sorted(ids-set(units))
    if not quality['telemetry'] or (requested_ids is not None and missing_ids):
        raise ValueError('В срезе нет телеметрии для выбранных ТС: '+', '.join(map(str, missing_ids)))
    selected_ids = config.tr_ids if requested_ids is not None else sorted(units)
    if requested_ids is None:
        plan = [s for s in plan if s.tr_id in units]
    events.sort(key=lambda e: e[0])
    end = max(config.start, events[-1][0])
    routes, vehicles = [], []
    for n, tr_id in enumerate(selected_ids):
        stops = [s.target for s in plan if s.tr_id == tr_id]
        if len(stops) < 2:
            raise ValueError('Недостаточно плановых остановок для выбранного ТС')
        route_id = f'replay-{tr_id}'
        # Отображение ограничено окном replay; ML выше получил весь исходный план.
        display = [s for s in stops if warmup <= s.scheduled_at <= end+timedelta(minutes=15)]
        if len(display) < 2:
            display = stops[:2]
        physical = {(s.lon,s.lat):s for s in display}
        routes.append(Route(route_id=route_id, name=f'План ТС {tr_id}', color=['#4169e1', '#16a085', '#b7764a', '#885eb5'][n % 4],
            path=[(s.lon, s.lat) for s in display],
            stops=[RouteStop(id=s.id, name=s.name, lon=s.lon, lat=s.lat) for s in physical.values()]))
        vehicles.append(VehicleConfig(tr_id=tr_id, unit_id=units[tr_id], label=f'Автобус {tr_id}', route_id=route_id))
    telemetry_times, position_times = {}, {}
    for available, kind, point in events:
        if kind != 'gps':
            continue
        telemetry_times.setdefault(point.tr_id, []).append(available)
        if usable_history([point], available):
            position_times.setdefault(point.tr_id, []).append(available)
    # Хеш исходного файла остаётся в отчёте происхождения. Версия плана train,
    # передаваемая ML, зависит лишь от разрешённых полей, не time_fact_begin.
    plan_version = (hashlib.sha256(json.dumps([s.model_dump(mode='json') for s in plan],
                    sort_keys=True).encode()).hexdigest() if config.dataset_split == 'train'
                    else sources[config.plan_file])
    return LoadedReplay(config=config, context=LiveContext(vehicles=vehicles, routes=routes, schedule=plan,
                        arrival_mode='gps' if config.deviation_source == 'gps' else 'external',
                        plan_version=plan_version, plan_timezone=config.timezone, plan_complete=True),
                        events=events, sources=sources, quality=quality,
                        fleet=dict(planned_vehicles=len(planned_ids), loaded_vehicles=len(vehicles),
                            without_telemetry=missing_ids, telemetry_without_plan=sorted(telemetry_without_plan)),
                        telemetry_times=telemetry_times, position_times=position_times)
