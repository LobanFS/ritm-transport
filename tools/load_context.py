"""Проверить/подготовить live-контекст; отправка в backend только с --apply.

Примеры (из корня репозитория, с установленными зависимостями):
  .venv/bin/python tools/load_context.py --schedule-plan /path/schedule_plan.csv \
      --vehicles /path/vehicles.json --timezone UTC --out context.json
  .venv/bin/python tools/load_context.py context.json
  .venv/bin/python tools/load_context.py context.json --apply

vehicles.json: {"vehicles": [{"tr_id": 1, "unit_id": 2,
"label": "Автобус 1", "route_id": "route-1"}]}. Маппинг задаёт оператор;
по GPS он не угадывается. CSV даёт плановые посещения, но не дорожную
геометрию: routes остаётся пустым. Полный JSON может содержать routes/path.
CSV создаёт arrival_mode="external": требуется подтверждённый источник прибытий.
Экспериментальный GPS можно включить только явно в JSON: arrival_mode="gps";
его точность на реальной истории пока непригодна для штатной работы.
Применение заменяет контекст и очищает историю. Январский план нельзя
использовать для сентябрьского потока без соответствующего нового расписания.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from datetime import datetime, timezone as dt_timezone
import json
from pathlib import Path
import re
import sys
from urllib.error import URLError
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

# Позволяет запускать скрипт из любой директории без установки собственного пакета.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.engine import LiveContext, ScheduledStop  # noqa: E402
from common.contracts import StopTarget, parse_manual_fill  # noqa: E402


PLAN_COLUMNS = {'tr_id', 'tt_action_item_id', 'time_begin', 'geom', 'building_address'}
UNUSED_PLAN_COLUMNS = {'order_date', 'manual_fill'}
NUMBER = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
POINT = re.compile(rf'POINT\s*\(\s*({NUMBER})\s+({NUMBER})\s*\)', re.IGNORECASE)


def _planned_time(value: str, zone: ZoneInfo | None) -> datetime:
    if 'T' not in value and ' ' not in value:
        raise ValueError('time_begin должен содержать дату и время')
    result = datetime.fromisoformat(value)
    if result.tzinfo is not None:
        return result
    if zone is None:
        raise ValueError('Для time_begin без offset явно укажите --timezone IANA')
    # При переходах DST один wall-clock может не существовать или повторяться.
    # Для таких строк требуется явный offset, чтобы не выбирать время молча.
    options = [result.replace(tzinfo=zone, fold=fold) for fold in (0, 1)]
    if (options[0].utcoffset() != options[1].utcoffset()
            or options[0].astimezone(dt_timezone.utc).astimezone(zone).replace(tzinfo=None) != result):
        raise ValueError('Неоднозначное/несуществующее время DST: укажите offset в CSV')
    return options[0]


def build_context(plan_path: Path, mapping_path: Path, timezone: str | None = None, *, complete: bool = False) -> LiveContext:
    """Полный план выбранных ТС, без фактов, labels, hints и выдуманных дорог.

    Пять обязательных столбцов плюс order_date/manual_fill из фактической раздачи.
    manual_fill сохраняется для frozen ML; order_date не используется builder.
    Полноту плана оператор подтверждает параметром complete. Необязательный time_fact_begin
    допустим только целиком пустым: факт у любого ТС отклоняет весь файл.
    Маппинг — объект с единственным ключом vehicles; каждый ТС должен иметь план.
    """
    try:
        zone = ZoneInfo(timezone) if timezone is not None else None
    except (ValueError, KeyError) as exc:
        raise ValueError('Неизвестный часовой пояс IANA') from exc
    with Path(mapping_path).open(encoding='utf-8-sig') as stream:
        mapping = json.load(stream)
    if not isinstance(mapping, dict) or set(mapping) != {'vehicles'}:
        raise ValueError('Маппинг должен быть объектом с единственным ключом vehicles')
    context = LiveContext.model_validate(mapping)
    vehicle_ids = {vehicle.tr_id for vehicle in context.vehicles}
    schedule = []
    covered = set()
    with Path(plan_path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        if (len(set(columns)) != len(columns) or not PLAN_COLUMNS <= set(columns)
                or set(columns) - PLAN_COLUMNS - UNUSED_PLAN_COLUMNS - {'time_fact_begin'}):
            raise ValueError('CSV требует tr_id,tt_action_item_id,time_begin,geom,building_address; '
                             'дополнительно допускаются order_date,manual_fill и пустой time_fact_begin')
        for line_number, row in enumerate(reader, 2):
            try:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError('Число полей строки не совпадает с заголовком')
                if row.get('time_fact_begin', '').strip():
                    raise ValueError('Фактические прибытия запрещены: нужен план без фактов')
                tr_id = int(row['tr_id'])
                if tr_id not in vehicle_ids:
                    continue
                point = POINT.fullmatch(row['geom'].strip())
                if point is None:
                    raise ValueError('geom должен быть WKT POINT (lon lat)')
                lon, lat = (float(value) for value in point.groups())
                target = StopTarget(id=row['tt_action_item_id'],
                    name=row['building_address'] or row['tt_action_item_id'],
                    scheduled_at=_planned_time(row['time_begin'], zone), lon=lon, lat=lat,
                    manual_fill=parse_manual_fill(row.get('manual_fill')))
                schedule.append(ScheduledStop(tr_id=tr_id, target=target))
                covered.add(tr_id)
            except (ValueError, TypeError) as exc:
                raise ValueError(f'Строка CSV {line_number}: {exc}') from exc
    missing = sorted(vehicle_ids - covered)
    if missing:
        raise ValueError(f'У настроенных ТС нет плановых посещений: {missing}')
    schedule.sort(key=lambda item: (item.tr_id, item.target.scheduled_at, item.target.id))
    return LiveContext(vehicles=context.vehicles, schedule=schedule, routes=[], hints=[], arrival_mode='external',
        plan_version=hashlib.sha256(Path(plan_path).read_bytes()).hexdigest(), plan_timezone=timezone or 'UTC',
        plan_complete=complete)


def load_context(path: Path) -> LiveContext:
    """Проверяет готовый JSON по тому же контракту, который принимает backend."""
    return LiveContext.model_validate_json(Path(path).read_text(encoding='utf-8-sig'))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('context', type=Path, nargs='?', help='Готовый JSON: проверить, при --apply применить')
    parser.add_argument('--schedule-plan', type=Path, help='CSV плана без фактов (разделитель запятая)')
    parser.add_argument('--vehicles', type=Path, help='JSON соответствий unit_id ↔ tr_id')
    parser.add_argument('--timezone', help='Явный IANA timezone для времени CSV без offset')
    parser.add_argument('--complete-plan', action='store_true', help='Подтвердить, что CSV содержит весь исходный план выбранных ТС, а не временной срез')
    parser.add_argument('--out', type=Path, help='Куда сохранить проверенный JSON; обязательно при сборке CSV')
    parser.add_argument('--apply', action='store_true', help='ЗАМЕНИТЬ live-контекст backend и очистить его историю')
    parser.add_argument('--backend', default='http://127.0.0.1:8000')
    parser.add_argument('--timeout', type=float, default=120, help='Время ожидания HTTP-ответа в секундах')
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error('--timeout должен быть положительным')
    building = args.schedule_plan is not None
    if (args.context is None) == (not building):
        parser.error('Выберите готовый context.json ИЛИ --schedule-plan с --vehicles и --out')
    if building and (args.vehicles is None or args.out is None):
        parser.error('Для CSV нужны --vehicles и --out')
    if not building and (args.vehicles is not None or args.timezone is not None or args.complete_plan):
        parser.error('--vehicles/--timezone применяются только с --schedule-plan')
    if building and args.out.resolve() in {args.schedule_plan.resolve(), args.vehicles.resolve()}:
        parser.error('--out не должен перезаписывать исходный CSV или маппинг')
    try:
        context = build_context(args.schedule_plan, args.vehicles, args.timezone, complete=args.complete_plan) if building else load_context(args.context)
        payload = context.model_dump_json(indent=2)
        if args.out:
            args.out.write_text(payload+'\n', encoding='utf-8')
        summary = {'valid': True, 'vehicles': len(context.vehicles), 'planned_visits': len(context.schedule),
                   'routes': len(context.routes), 'applied': False}
        if args.out:
            summary['output'] = str(args.out.resolve())
        if args.apply:
            request = Request(args.backend.rstrip('/')+'/api/v1/live/context', data=payload.encode('utf-8'),
                              headers={'Content-Type': 'application/json'}, method='POST')
            with urlopen(request, timeout=args.timeout) as response:
                summary['backend_response'] = json.load(response)
            summary['applied'] = True
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print('Контекст заменён, предыдущая история очищена.' if args.apply else
              'Backend не изменён. Для применения выполните ту же команду с --apply.')
        return 0
    except (OSError, ValueError, URLError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    raise SystemExit(main())
