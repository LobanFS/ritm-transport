"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { create } = require("../dashboard/route-geometry.js");

const A = [37, 55], B = [37.01, 55.01], C = [37.02, 55];
const AB = [A, [37.004, 55.007], B], BA = [B, [37.007, 55.002], A];
const payload = { version: 1, gps_used: false, segments: [
  { start: A, end: B, path: AB }, { start: B, end: A, path: BA },
] };
const response = value => ({ ok: true, text: async () => JSON.stringify(value) });
let passed = 0;
async function check(name, run) {
  await run();
  passed += 1;
  console.log(`ok ${passed} - ${name}`);
}

(async () => {
  await check("missing asset preserves the original plan and is fetched once", async () => {
    let calls = 0;
    const geometry = create({ fetcher: async () => { calls += 1; throw Error("offline"); } });
    assert.deepEqual(await Promise.all([geometry.load(), geometry.load()]), [false, false]);
    assert.equal(await geometry.load(), false);
    assert.equal(calls, 1);
    assert.deepEqual(geometry.paths({ path: [A, B] }).planPaths, [[A, B]]);
  });
  await check("road asset is loaded once and clears earlier fallback", async () => {
    let calls = 0;
    const geometry = create({ fetcher: async () => { calls += 1; return response(payload); } });
    assert.equal(geometry.paths({ path: [A, B] }).roadSegments, 0);
    assert.deepEqual(await Promise.all([geometry.load(), geometry.load()]), [true, true]);
    assert.equal(calls, 1);
    assert.equal(geometry.revision, 1);
    assert.deepEqual(geometry.paths({ path: [A, B] }).roadPaths, [AB]);
  });
  const geometry = create({ fetcher: async () => response(payload) });
  await geometry.load();
  await check("out-and-back loops retain order and asymmetric direction", () => {
    const display = geometry.paths({ path: [A, B, A, B, C] });
    assert.deepEqual(display.paths, [AB, BA, AB, [B, C]]);
    assert.equal(display.roadSegments, 3);
    assert.equal(display.planSegments, 1);
  });
  await check("unknown directed legs are not reversed from the opposite road", async () => {
    const oneWay = create({ fetcher: async () => response({ ...payload, segments: payload.segments.slice(0, 1) }) });
    await oneWay.load();
    assert.deepEqual(oneWay.paths({ path: [B, A] }).planPaths, [[B, A]]);
  });
  await check("geometry matches plan coordinates, never a reused bus identifier", () => {
    assert.equal(geometry.paths({ route_id: "replay-122048", path: [A, B] }).roadSegments, 1);
    assert.equal(geometry.paths({ route_id: "replay-122048", path: [A, C] }).roadSegments, 0);
    assert.equal(geometry.paths({ route_id: "custom-new-id", path: [A, B] }).roadSegments, 1);
  });
  await check("an invalid point breaks the path instead of making a shortcut", () => {
    assert.deepEqual(geometry.paths({ path: [A, null, B, C] }).paths, [[B, C]]);
    assert.equal(geometry.paths({ path: [A, [null, 55], B] }).paths.length, 0);
  });
  await check("consecutive duplicate stops do not remove later return visits", () => {
    assert.deepEqual(geometry.paths({ path: [A, A, B, B, A] }).paths, [AB, BA]);
  });
  await check("empty and invalid routes have no geometry", () => {
    for (const route of [null, {}, { path: [] }, { path: [A] }, { path: [A, [181, 55]] }]) {
      assert.equal(geometry.paths(route).paths.length, 0);
    }
  });
  await check("cached drawing is reused for unchanged planned coordinates", () => {
    assert.equal(geometry.paths({ path: [A, B] }), geometry.paths({ path: structuredClone([A, B]) }));
  });
  await check("GPS-derived or malformed assets are rejected atomically", async () => {
    for (const bad of [{ ...payload, gps_used: true }, { ...payload, version: 99 },
      { ...payload, segments: [payload.segments[0], { start: B, end: A, path: [[null, 55], A] }] },
      { ...payload, segments: [payload.segments[0], payload.segments[0]] }]) {
      const rejected = create({ fetcher: async () => response(bad) });
      assert.equal(await rejected.load(), false);
      assert.equal(rejected.paths({ path: [A, B] }).roadSegments, 0);
    }
  });
  await check("road snapped far away from a stop is rejected", async () => {
    const rejected = create({ fetcher: async () => response({ ...payload,
      segments: [{ start: A, end: B, path: [C, B] }] }) });
    assert.equal(await rejected.load(), false);
  });
  await check("non-JSON, HTTP error and oversized assets fall back safely", async () => {
    for (const fetcher of [async () => ({ ok: false }),
      async () => ({ ok: true, text: async () => "not json" }),
      async () => ({ ok: true, headers: { get: () => "2000001" } })]) {
      assert.equal(await create({ fetcher }).load(), false);
    }
  });
  await check("packaged asset loads without OSRM or telemetry", async () => {
    const text = fs.readFileSync(path.join(__dirname, "../dashboard/assets/road-network.json"), "utf8");
    const packaged = create({ fetcher: async url => {
      assert.equal(url, "/assets/road-network.json");
      return { ok: true, text: async () => text };
    } });
    assert.equal(await packaged.load(), true);
    const asset = JSON.parse(text);
    assert.equal(asset.gps_used, false);
    assert.equal(asset.source.license, "ODbL-1.0");
    assert.ok(asset.segments.length > 800);
    for (const leg of asset.segments) {
      assert.deepEqual(packaged.paths({ path: [leg.start, leg.end] }).roadPaths, [leg.path]);
    }
  });
  console.log(`${passed} route geometry checks passed`);
})().catch(error => { console.error(error); process.exitCode = 1; });
