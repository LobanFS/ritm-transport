"""Упаковать дорожные сегменты OSRM для карты; GPS и ML не используются."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

SOURCE_COMMIT = "a8069898a663b7f98a052a8989bb3a69b1c773b0"
SOURCE_PATH = "irina/dashboard/data/route-network.json"


def valid_point(value):
    return (isinstance(value, list) and len(value) == 2
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in value)
            and abs(value[0]) <= 180 and abs(value[1]) <= 85.0511)


def metres(a, b):
    lat1, lat2 = map(math.radians, (a[1], b[1]))
    longitude = math.radians(a[0] - b[0])
    return 12_742_000 * math.asin(min(1, math.sqrt(math.sin((lat1-lat2)/2)**2
           + math.cos(lat1)*math.cos(lat2)*math.sin(longitude/2)**2)))


def prepare(raw: bytes) -> dict:
    source = json.loads(raw)
    if source.get("version") != 4 or not isinstance(source.get("vehicles"), dict):
        raise ValueError("Ожидается road-only route-network.json версии 4")
    segments = {}
    skipped = 0
    for vehicle in source["vehicles"].values():
        if vehicle.get("gps_used") is not False:
            raise ValueError("Источник должен явно исключать GPS")
        for leg in vehicle.get("legs", []):
            start, end, path = leg.get("start"), leg.get("end"), leg.get("path")
            if (not valid_point(start) or not valid_point(end) or not isinstance(path, list)
                    or not all(valid_point(p) for p in path)):
                raise ValueError("Некорректная дорожная геометрия")
            # Some OSRM waypoints snap to the very same road node. Such a
            # zero-length leg cannot stand in for the pair of actual stops.
            if (len(path) < 2 or not any(p != path[0] for p in path)
                    or metres(start, path[0]) > 150 or metres(end, path[-1]) > 150):
                skipped += 1
                continue
            key = ";".join(",".join(f"{n:.6f}" for n in p) for p in (start, end))
            segments.setdefault(key, {"start": start, "end": end, "path": path})
    return {
        "version": 1, "gps_used": False,
        "source": {"repository": "LobanFS/mos-transport-hack", "commit": SOURCE_COMMIT,
                   "path": SOURCE_PATH, "sha256": hashlib.sha256(raw).hexdigest(),
                   "method": "OSRM driving routes through planned stops; display only",
                   "attribution": "© OpenStreetMap contributors",
                   "license": "ODbL-1.0", "license_url": "https://opendatacommons.org/licenses/odbl/1-0/"},
        "skipped_degenerate_legs": skipped,
        "segments": [segments[key] for key in sorted(segments)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Исходный route-network.json с дорожными сегментами OSRM")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1]/"dashboard/assets/road-network.json")
    args = parser.parse_args()
    payload = prepare(args.source.read_bytes())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":"))+"\n", encoding="utf-8")
    print(f"{len(payload['segments'])} дорожных сегментов; {args.output.stat().st_size} байт")


if __name__ == "__main__":
    main()
