"""Линия — сходный план, а не один пересадочный узел или цепочка пересечений."""
from backend.line_identity import has_line_geometry, line_memberships, same_line


def route(identifier, points):
    return {'route_id': identifier, 'path': points}


def corridor(start=0, count=5, latitude=55.75):
    return [(37.6 + .005 * i, latitude) for i in range(start, start + count)]


def test_shared_explicit_id_is_authoritative_even_without_coordinates():
    assert same_line({'route_id': '17'}, {'route_id': '17'})
    assert not same_line({}, {})


def test_shifted_start_reversed_direction_and_different_stop_ids_are_same_line():
    points = corridor()
    a = route('a', points)
    b = {'route_id': 'b', 'stops': [
        {'id': f'new-{i}', 'lon': lon, 'lat': lat}
        for i, (lon, lat) in enumerate(points[2:] + points[:2])
    ]}
    c = route('c', list(reversed(points)))
    assert line_memberships([a, b, c]) == {'a': ['a', 'b', 'c'], 'b': ['b', 'a', 'c'], 'c': ['c', 'a', 'b']}


def test_small_coordinate_jitter_matches_but_nearby_parallel_line_does_not():
    a = route('a', corridor())
    b = route('b', [(x+.0001, y+.0001) for x, y in corridor()])
    c = route('c', corridor(latitude=55.751))
    assert same_line(a, b) and same_line(b, a)
    assert not same_line(a, c)


def test_one_interchange_does_not_join_crossing_lines():
    horizontal = route('horizontal', corridor())
    vertical = route('vertical', [(37.6, 55.75 + .005 * i) for i in range(5)])
    assert not same_line(horizontal, vertical)


def test_several_platforms_at_one_interchange_are_not_a_line():
    a = route('a', [(37.6, 55.75), (37.6001, 55.7501)])
    b = route('b', list(reversed(a['path'])))
    assert not same_line(a, b)


def test_subset_requires_coverage_of_both_plans():
    assert not same_line(route('a', corridor(count=3)), route('b', corridor(count=10)))
    assert same_line(route('a', corridor(count=4)), route('b', corridor(count=5)))


def test_direct_membership_does_not_join_via_shared_corridor():
    a, b, c = [route(name, corridor(start=i)) for i, name in enumerate('abc')]
    assert same_line(a, b) and same_line(b, c) and not same_line(a, c)
    assert line_memberships([a, b, c]) == {'a': ['a', 'b'], 'b': ['b', 'a', 'c'], 'c': ['c', 'b']}


def test_missing_or_single_point_geometry_does_not_highlight_unrelated_bus():
    a = route('a', corridor())
    empty, single = route('empty', []), route('single', corridor(count=1))
    assert not has_line_geometry(empty) and not has_line_geometry(single)
    assert line_memberships([a, empty, single]) == {'a': ['a'], 'empty': ['empty'], 'single': ['single']}


def test_invalid_coordinates_do_not_produce_membership():
    invalid = route('bad', [[None, 55.75], [37.6, float('nan')], [181, 55.75], ['bad', 55.75], [37.6]])
    assert not has_line_geometry(invalid)
    assert not same_line(route('a', corridor()), invalid)


def test_reusing_route_ids_with_new_geometry_invalidates_geometric_match():
    a, b = route('a', corridor()), route('b', corridor())
    assert line_memberships([a, b])['a'] == ['a', 'b']
    b['path'] = corridor(latitude=55.8)
    assert line_memberships([a, b])['a'] == ['a']


def test_long_custom_plans_keep_all_stops_without_sampling_or_truncation():
    a = route('a', corridor(count=400))
    b = route('b', corridor(count=400))
    assert same_line(a, b)
    b['path'] = corridor(count=300) + corridor(start=400, count=100, latitude=55.8)
    assert not same_line(a, b)


def test_repeated_visits_do_not_inflate_shared_stop_coverage():
    a = route('a', corridor())
    b = route('b', corridor(count=1) * 100 + corridor(start=10, count=5))
    assert not same_line(a, b)
