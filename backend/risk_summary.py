"""Агрегация прогнозов по маршрутам и плановым перегонам, не детектор пробок."""
from collections import defaultdict
from datetime import datetime


def summarize(vehicles, schedule, route_names):
    routes, sections = defaultdict(list), defaultdict(list)
    geometry = {}
    for vehicle in vehicles:
        route = vehicle['route_id']
        routes[route].append(vehicle)
        target = vehicle.get('target')
        if not target:
            continue
        target_time = datetime.fromisoformat(target['scheduled_at'].replace('Z','+00:00'))
        previous = max((s for s in schedule[vehicle['tr_id']] if s.scheduled_at < target_time),
                       key=lambda s: s.scheduled_at, default=None)
        if previous is None:
            continue
        coords = (round(previous.lon,6), round(previous.lat,6), round(target['lon'],6), round(target['lat'],6))
        key = (route, coords)
        sections[key].append(vehicle)
        geometry[key] = dict(section=f"{previous.name} → {target['name']}", path=[coords[:2], coords[2:]])

    def aggregate(items):
        predictions = [v.get('prediction') for v in items]
        risks = [p['risk'] if p and v['status'] == 'fresh' else 'unknown' for v,p in zip(items,predictions)]
        risk = next(level for level in ('red','amber','unknown','green') if level in risks)
        values = [p['predicted_delay_s'] for p,r in zip(predictions,risks) if p and r != 'unknown' and p['predicted_delay_s'] is not None]
        return dict(risk=risk, vehicles=len(items), evaluated=len(items)-risks.count('unknown'),
                    red=risks.count('red'), amber=risks.count('amber'), unknown=risks.count('unknown'),
                    max_predicted_delay_s=max(values) if values else None,
                    tr_ids=[v['tr_id'] for v in items])

    return ([dict(route_id=key, name=route_names.get(key,key), **aggregate(items)) for key,items in routes.items()],
            [dict(route_id=key[0], **geometry[key], **aggregate(items),
                  interpretation='Прогнозы на плановом перегоне; дорожная причина не установлена')
             for key,items in sections.items()])
