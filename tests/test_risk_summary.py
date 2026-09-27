"""Риск маршрута не скрывает неизвестные/устаревшие данные за зелёным цветом."""
from backend.risk_summary import summarize


def vehicle(n, risk='green', status='fresh'):
    return dict(tr_id=n, route_id='r', status=status, target=None,
                prediction=dict(risk=risk,predicted_delay_s={'red':151, 'amber':130, 'unknown':None}.get(risk,0)))


def test_stale_red_does_not_become_current_red_or_green():
    routes, _ = summarize([vehicle(1),vehicle(2,'red','stale')], {}, {})
    assert routes[0]['risk'] == 'unknown'
    assert routes[0]['evaluated'] == 1 and routes[0]['red'] == 0
    assert routes[0]['max_predicted_delay_s'] == 0


def test_partial_coverage_kept_with_current_alert():
    routes, _ = summarize([vehicle(1,'red'),vehicle(2,'unknown')], {}, {})
    assert routes[0]['risk'] == 'red'
    assert routes[0]['unknown'] == 1
    assert routes[0]['tr_ids'] == [1,2]


def test_amber_does_not_become_red_or_hide_unknown_coverage():
    routes, _ = summarize([vehicle(1,'amber'),vehicle(2,'unknown'),vehicle(3,'red','stale')], {}, {})
    assert routes[0]['risk'] == 'amber'
    assert routes[0]['amber'] == 1 and routes[0]['red'] == 0
    assert routes[0]['unknown'] == 2 and routes[0]['evaluated'] == 1
    assert routes[0]['max_predicted_delay_s'] == 130
