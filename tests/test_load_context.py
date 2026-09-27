"""Импорт реального плана: явный маппинг, отсутствие фактов и неоднозначного времени."""
import csv
import json

import pytest

from tools import load_context as importer


COLUMNS = ['tr_id', 'tt_action_item_id', 'time_begin', 'geom', 'building_address']


@pytest.fixture
def sources(tmp_path):
    plan = tmp_path / 'schedule_plan.csv'
    mapping = tmp_path / 'vehicles.json'
    mapping.write_text(json.dumps({'vehicles': [
        {'tr_id': 1, 'unit_id': 901, 'label': 'Автобус 1', 'route_id': 'operational-A'}
    ]}), encoding='utf-8')
    def write(rows=None, columns=None):
        with plan.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(columns or COLUMNS)
            writer.writerows(rows or [
                [1, 'visit-after', '2026-01-06 11:14:00', 'POINT (37.62 55.77)', 'Вторая'],
                [1, 'visit-before', '2026-01-06 11:00:00', 'POINT (37.60 55.75)', 'Первая'],
                [2, 'other-bus', '2026-01-06 11:00:00', 'POINT (37.63 55.74)', 'Другая'],
            ])
        return plan, mapping
    write()
    return plan, mapping, write


def test_full_selected_plan_sorted_without_invented_routes_or_deviations(sources):
    plan, mapping, _ = sources
    context = importer.build_context(plan, mapping, 'Europe/Moscow')
    assert [entry.target.id for entry in context.schedule] == ['visit-before', 'visit-after']
    assert context.schedule[0].target.scheduled_at.isoformat() == '2026-01-06T11:00:00+03:00'
    assert context.schedule[0].target.lon == 37.60
    assert context.vehicles[0].unit_id == 901
    assert context.routes == [] and context.hints == []
    assert context.arrival_mode == 'external'


def test_official_manual_flag_preserved_without_facts_and_completeness_explicit(sources):
    plan, mapping, write = sources
    write([['a', '2026-01-06 11:00:00', '2026-01-06', 'False', 1, 'POINT (37.6 55.75)', 'A']],
          ['tt_action_item_id', 'time_begin', 'order_date', 'manual_fill', 'tr_id', 'geom', 'building_address'])
    context = importer.build_context(plan, mapping, 'UTC')
    assert context.schedule[0].target.id == 'a'
    assert context.schedule[0].target.manual_fill is False
    assert context.plan_complete is False
    assert importer.build_context(plan, mapping, 'UTC', complete=True).plan_complete
    assert 'order_date' not in context.model_dump_json()


def test_naive_requires_explicit_timezone_and_aware_keeps_offset(sources):
    plan, mapping, write = sources
    with pytest.raises(ValueError, match='timezone'):
        importer.build_context(plan, mapping)
    with pytest.raises(ValueError, match='IANA'):
        importer.build_context(plan, mapping, 'Imaginary/Zone')
    write([[1, 'a', '2026-01-06T11:00:00+03:00', 'POINT (37.6 55.75)', 'A']])
    assert importer.build_context(plan, mapping).schedule[0].target.scheduled_at.isoformat().endswith('+03:00')


@pytest.mark.parametrize('at', ['2026-10-25 02:30:00', '2026-03-29 02:30:00'])
def test_dst_ambiguity_requires_explicit_offset(sources, at):
    plan, mapping, write = sources
    write([[1, 'a', at, 'POINT (37.6 55.75)', 'A']])
    with pytest.raises(ValueError, match='DST'):
        importer.build_context(plan, mapping, 'Europe/Berlin')


@pytest.mark.parametrize('geom', ['LINESTRING (37.6 55.75, 37.61 55.76)', 'POINT (NaN 55.75)',
                                  'POINT (37.6 91)', 'POINT (37.6 55.75); extra'])
def test_bad_or_out_of_range_geometry_rejected(sources, geom):
    plan, mapping, write = sources
    write([[1, 'a', '2026-01-06T11:00:00Z', geom, 'A']])
    with pytest.raises(ValueError):
        importer.build_context(plan, mapping)


def test_fact_on_unselected_vehicle_still_rejects_entire_file(sources):
    plan, mapping, write = sources
    write([[1, 'a', '2026-01-06T11:00:00Z', 'POINT (37.6 55.75)', 'A', ''],
           [2, 'b', '2026-01-06T11:00:00Z', 'POINT (37.6 55.75)', 'B', '2026-01-06T11:05:00Z']],
          COLUMNS + ['time_fact_begin'])
    with pytest.raises(ValueError, match='Фактические прибытия'):
        importer.build_context(plan, mapping)
    write([[1, 'a', '2026-01-06T11:00:00Z', 'POINT (37.6 55.75)', 'A', '']], COLUMNS+['time_fact_begin'])
    assert len(importer.build_context(plan, mapping).schedule) == 1


def test_duplicate_visits_and_unmapped_configured_vehicle_fail(sources):
    plan, mapping, write = sources
    row = [1, 'a', '2026-01-06T11:00:00Z', 'POINT (37.6 55.75)', 'A']
    write([row, row])
    with pytest.raises(ValueError, match='Повтор ID'):
        importer.build_context(plan, mapping)
    write([[2, *row[1:]]])
    with pytest.raises(ValueError, match='нет плановых посещений'):
        importer.build_context(plan, mapping)


@pytest.mark.parametrize('change', ['missing_unit', 'duplicate_unit', 'unexpected_field'])
def test_mapping_contract_is_strict(sources, change):
    plan, mapping, _ = sources
    payload = json.loads(mapping.read_text())
    if change == 'missing_unit':
        del payload['vehicles'][0]['unit_id']
    elif change == 'duplicate_unit':
        payload['vehicles'].append({**payload['vehicles'][0], 'tr_id': 2})
    else:
        payload['future_facts'] = []
    mapping.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        importer.build_context(plan, mapping, 'UTC')


@pytest.mark.parametrize('column', ['target_delay_s', 'cur_dev_s', 'future_fact'])
def test_unknown_or_label_columns_rejected(sources, column):
    plan, mapping, write = sources
    write([[1, 'a', '2026-01-06T11:00:00Z', 'POINT (37.6 55.75)', 'A', '0']], COLUMNS+[column])
    with pytest.raises(ValueError, match='CSV требует'):
        importer.build_context(plan, mapping)


def test_default_cli_only_writes_reviewable_json_and_apply_is_explicit(sources, tmp_path, monkeypatch):
    plan, mapping, _ = sources
    out = tmp_path / 'context.json'
    calls = []
    def fake_open(request, timeout):
        calls.append(json.loads(request.data))
        from io import StringIO
        return StringIO('{"ok": true}')
    monkeypatch.setattr(importer, 'urlopen', fake_open)
    assert importer.main(['--schedule-plan', str(plan), '--vehicles', str(mapping),
                          '--timezone', 'UTC', '--out', str(out)]) == 0
    assert calls == []
    assert importer.main([str(out)]) == 0
    assert calls == []
    assert importer.main([str(out), '--apply']) == 0
    assert len(calls) == 1 and len(calls[0]['schedule']) == 2


def test_failed_validation_neither_writes_nor_applies(sources, tmp_path, monkeypatch):
    plan, mapping, _ = sources
    out = tmp_path / 'context.json'
    def forbidden(*args, **kwargs):
        raise AssertionError('Не должно быть сетевого запроса')
    monkeypatch.setattr(importer, 'urlopen', forbidden)
    with pytest.raises(SystemExit):
        importer.main(['--schedule-plan', str(plan), '--vehicles', str(mapping), '--out', str(out), '--apply'])
    assert not out.exists()


def test_existing_json_uses_backend_contract_without_allowing_unknown_future_fields(tmp_path):
    context = tmp_path / 'context.json'
    context.write_text(json.dumps({'vehicles': [{'tr_id': 1, 'unit_id': 2, 'label': 'A', 'route_id': 'R'}],
                                  'time_fact_begin': '2026-01-06T13:00:00Z'}))
    with pytest.raises(ValueError):
        importer.load_context(context)


def test_live_plan_and_model_contract_accept_more_than_20000_visits(sources):
    from datetime import datetime, timedelta, timezone
    from common.contracts import ModelPlanContext
    plan, mapping, write = sources
    start = datetime(2030, 1, 1, tzinfo=timezone.utc)
    write([[1, f'visit-{index}', (start + timedelta(minutes=index)).isoformat(),
            'POINT (37.6 55.75)', 'Остановка'] for index in range(20001)])
    context = importer.build_context(plan, mapping, 'UTC', complete=True)
    model_plan = ModelPlanContext(version=context.plan_version, timezone='UTC', complete=True,
                                  stops=[entry.target for entry in context.schedule])
    assert len(context.schedule) == len(model_plan.stops) == 20001
    assert model_plan.stops[-1].id == 'visit-20000'
