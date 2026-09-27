"use strict";

// Display-only road geometry. No telemetry, forecast inputs or network routing.
// All coordinates use GeoJSON order: [longitude, latitude].
(() => {
  const point = value => Array.isArray(value) && value.length === 2 && value.every(Number.isFinite)
    && Math.abs(value[0]) <= 180 && Math.abs(value[1]) <= 85.0511;
  const pointKey = value => value.map(number => number.toFixed(6)).join(",");
  const segmentKey = (start, end) => `${pointKey(start)};${pointKey(end)}`;
  const metres = (a, b) => {
    const radians = Math.PI / 180;
    const latitude = (a[1] - b[1]) * radians, longitude = (a[0] - b[0]) * radians;
    return 12742000 * Math.asin(Math.min(1, Math.sqrt(Math.sin(latitude / 2) ** 2
      + Math.cos(a[1] * radians) * Math.cos(b[1] * radians) * Math.sin(longitude / 2) ** 2)));
  };

  function create({ fetcher = globalThis.fetch?.bind(globalThis), url = "/assets/road-network.json" } = {}) {
    let roads = new Map(), revision = 0, loading;
    const cache = new Map();

    function accept(payload) {
      if (payload?.version !== 1 || payload.gps_used !== false || !Array.isArray(payload.segments)
        || payload.segments.length > 2500) throw new Error("Invalid road asset");
      const next = new Map();
      let count = 0;
      for (const segment of payload.segments) {
        const { start, end, path } = segment || {};
        if (!point(start) || !point(end) || !Array.isArray(path) || path.length < 2
          || !path.every(point) || (count += path.length) > 100000
          || metres(start, path[0]) > 150 || metres(end, path.at(-1)) > 150) {
          throw new Error("Invalid road segment");
        }
        const key = segmentKey(start, end);
        if (next.has(key)) throw new Error("Duplicate road segment");
        next.set(key, path);
      }
      if (!next.size) throw new Error("Empty road asset");
      roads = next;
      revision += 1;
      cache.clear();
      return true;
    }

    async function load() {
      // A failed optional asset must not delay telemetry or repeat every poll.
      if (!loading) loading = (async () => {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 6000);
        try {
          if (!fetcher) return false;
          const response = await fetcher(url, { cache: "force-cache", signal: controller.signal });
          if (!response.ok || Number(response.headers?.get("content-length")) > 2000000) return false;
          const source = await response.text();
          if (source.length > 2000000) return false;
          return accept(JSON.parse(source));
        } catch (_) {
          return false;
        } finally {
          clearTimeout(timeout);
        }
      })();
      return loading;
    }

    function paths(route) {
      const planned = Array.isArray(route?.path) ? route.path : [];
      const signature = JSON.stringify(planned);
      if (cache.has(signature)) return cache.get(signature);
      const result = { paths: [], roadPaths: [], planPaths: [], roadSegments: 0, planSegments: 0 };
      // Match directed neighbours, not vehicle IDs or a globally deduplicated
      // stop list: A → B → A → C must keep the return leg and its direction.
      for (let index = 1; index < planned.length; index += 1) {
        const start = planned[index - 1], end = planned[index];
        if (!point(start) || !point(end) || pointKey(start) === pointKey(end)) continue;
        const road = roads.get(segmentKey(start, end));
        const path = road || [start, end];
        result.paths.push(path);
        result[road ? "roadPaths" : "planPaths"].push(path);
        result[road ? "roadSegments" : "planSegments"] += 1;
      }
      // Each segment remains separate. In particular, no invented connector
      // bridges differently snapped road endpoints or missing plan geometry.
      if (cache.size >= 48) cache.delete(cache.keys().next().value);
      cache.set(signature, result);
      return result;
    }

    return { load, paths, get revision() { return revision; } };
  }

  const instance = create();
  const api = { create, load: instance.load, paths: instance.paths };
  Object.defineProperty(api, "revision", { get: () => instance.revision });
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else globalThis.RouteGeometry = api;
})();
