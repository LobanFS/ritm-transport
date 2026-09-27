import hashlib
import json

import pytest

from tools.prepare_route_geometry import prepare


def asset(legs, gps=False):
    return json.dumps({"version": 4, "vehicles": {"1": {"gps_used": gps, "legs": legs}}}).encode()


def test_plan_road_cache_retains_direction_and_provenance():
    a, b = [37.0, 55.0], [37.01, 55.01]
    raw = asset([{"start": a, "end": b, "path": [a, [37.005, 55.004], b]},
                 {"start": b, "end": a, "path": [b, [37.003, 55.006], a]}])
    result = prepare(raw)
    assert len(result["segments"]) == 2
    assert result["source"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["gps_used"] is False
    assert prepare(raw) == result


def test_zero_length_snapped_leg_keeps_plan_fallback():
    a, b = [37.0, 55.0], [37.0001, 55.0001]
    result = prepare(asset([{"start": a, "end": b, "path": [a, a]}]))
    assert result["segments"] == []
    assert result["skipped_degenerate_legs"] == 1


def test_telemetry_asset_is_rejected():
    with pytest.raises(ValueError, match="исключать GPS"):
        prepare(asset([], gps=True))


def test_invalid_road_coordinate_is_rejected():
    with pytest.raises(ValueError, match="геометрия"):
        prepare(asset([{"start": [37, 55], "end": [37.1, 55], "path": [[None, 55], [37.1, 55]]}]))
