"""Загрузить исторический срез в работающий backend; файлы читает сам backend."""
import argparse
import json
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', default='http://127.0.0.1:8000')
    parser.add_argument('--start', default='2026-01-06T11:30:00+00:00', help='RFC3339 с часовым поясом')
    parser.add_argument('--timezone', default='UTC', help='Интерпретация naive времён CSV (допущение)')
    parser.add_argument('--minutes', type=int, default=None,
                        help='Верхняя граница окна в минутах; без флага — до последнего события архива')
    parser.add_argument('--vehicles', type=int, nargs='+', default=None,
                        help='ID вручную; без флага — все ТС с планом и сообщениями в срезе')
    parser.add_argument('--speed', type=float, default=10)
    parser.add_argument('--timeout', type=float, default=120, help='Время ожидания HTTP-ответа в секундах')
    parser.add_argument('--dataset-split', choices=('validate', 'train'), default='validate',
                        help='Набор истории: validate (по умолчанию) либо train; train включает синтетику')
    parser.add_argument('--deviation-source', choices=('csv_snapshot', 'gps'), default=None,
                        help='Источник отклонения: по умолчанию CSV для validate, GPS для train')
    parser.add_argument('--warmup-minutes', type=int, default=None,
                        help='Предыстория перед start: по умолчанию 30 минут для GPS, 5 для CSV')
    parser.add_argument('--play', action='store_true', help='Иначе загрузить на паузе')
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('--timeout должен быть положительным')
    if args.deviation_source is None:
        args.deviation_source = 'gps' if args.dataset_split == 'train' else 'csv_snapshot'
    if args.dataset_split == 'train' and args.deviation_source != 'gps':
        parser.error('Для train доступен только --deviation-source gps: points.csv в этом наборе нет.')
    payload = dict(start=args.start, timezone=args.timezone, duration_minutes=args.minutes,
                   tr_ids=args.vehicles, speed=args.speed, paused=not args.play,
                   dataset_split=args.dataset_split,
                   deviation_source=args.deviation_source,
                   warmup_minutes=args.warmup_minutes if args.warmup_minutes is not None else
                       (30 if args.deviation_source == 'gps' else 5))
    request = Request(args.backend+'/api/v1/replay/load', data=json.dumps(payload).encode(),
                      headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=args.timeout) as response:
        print(json.dumps(json.load(response), ensure_ascii=False, indent=2))
    print(f'Откройте http://127.0.0.1:8080 — replay {args.dataset_split}, источник {args.deviation_source}. '
          'Продолжить / Пауза / Сброс в панели воспроизведения.')


if __name__ == '__main__':
    main()
