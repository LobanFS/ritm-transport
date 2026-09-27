// Run with Node: node tests/transfer_ui.test.cjs. No browser or packages required.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../dashboard/app.js'), 'utf8');
const bands = source.slice(source.indexOf('  const RISK ='), source.indexOf('  const $ ='));
const gate = source.slice(source.indexOf('  const effectiveRisk ='), source.indexOf('  const availabilityLabel ='));
const transfer = source.slice(source.indexOf('  const transferMatches ='), source.indexOf('  function createTableRow'));
assert.ok(transfer.includes('function showTransferDonor'), 'extract the actual transfer implementation');

const bus = (id, delay = 240) => ({
  tr_id: id, label: `Автобус ${id}`, route_id: `route-${id}`, status: 'fresh',
  lat: 55.75, lon: 37.62, prediction_availability: {code: 'ready'},
  prediction: {risk: delay > 150 ? 'red' : 'green', predicted_delay_s: delay,
    target: {id: `stop-${id}`, name: `Остановка ${id}`, scheduled_at: '2026-09-27T12:12:00Z'}},
});
const state = () => ({context: {version: 1, loaded_at: '2026-09-27T12:00:00Z'},
  clock_time: '2026-09-27T12:00:00Z', vehicles: [bus('A'), bus('B'), bus('D', 20)]});
function advice(snapshot, targetId = 'A') {
  const target = snapshot.vehicles.find(v => v.tr_id === targetId);
  return {status: 'ready', context_version: snapshot.context.version,
    target: {tr_id: targetId}, donor: {tr_id: 'D', route_name: 'Линия D'},
    meeting_stop: {...target.prediction.target},
    scenario: {distance_m: 750, relocation_s: 160, earlier_by_s: 80},
    donor_impact: {next_stop_at: '2026-09-27T12:03:00Z'}, assumptions: ['Проверяет диспетчер']};
}
function harness() {
  const elements = new Map(), requests = [], centres = [], toasts = [];
  const el = id => {
    if (!elements.has(id)) elements.set(id, {hidden: false, textContent: '', dataset: {}, checked: false, value: ''});
    return elements.get(id);
  };
  const scope = {
    snapshot: state(), selectedId: 'A', disconnected: false,
    transferAdvice: null, transferKey: '', transferPending: '', transferCheckedAt: 0,
    transferClock: null, transferSerial: 0,
    $: el, text: (id, value) => { el(id).textContent = String(value); },
    finite: Number.isFinite, list: value => Array.isArray(value) ? value : [],
    idOf: value => String(value ?? ''), timestamp: value => value ? Date.parse(value) : null,
    duration: value => `${value} с`, time: value => value,
    vehicleName: value => value.label, markers: new Map(),
    map: {getZoom: () => 12}, latLng: (lon, lat) => Number.isFinite(lon) && Number.isFinite(lat) ? [lat, lon] : null,
    setFollow: () => {}, renderMap: () => {}, renderTable: () => {}, filteredVehicles: () => [],
    centreVehicle: (...args) => centres.push(args), toast: value => toasts.push(value), node: () => ({}),
    request: url => new Promise((resolve, reject) => requests.push({url, resolve, reject})),
  };
  scope.apiStale = () => scope.disconnected;
  scope.chosenVehicle = () => scope.snapshot.vehicles.find(v => v.tr_id === scope.selectedId);
  const api = vm.runInNewContext(bands + gate + transfer + '\n;({transferMatches, renderTransfer, paintTransfer, showTransferDonor});', scope);
  return {scope, api, el, requests, centres, toasts};
}
const flush = () => new Promise(resolve => setImmediate(resolve));
let checks = 0;
async function check(name, test) {
  await test(); checks++; process.stdout.write(`ok ${checks} - ${name}\n`);
}

(async () => {
  await check('late A response cannot replace the already selected B recommendation', async () => {
    const h = harness(); h.api.renderTransfer();
    const a = advice(h.scope.snapshot);
    h.scope.selectedId = 'B'; h.api.renderTransfer();
    h.requests[1].resolve(advice(h.scope.snapshot, 'B')); await flush();
    assert.equal(h.scope.transferAdvice.target.tr_id, 'B');
    h.requests[0].resolve(a); await flush();
    assert.equal(h.scope.transferAdvice.target.tr_id, 'B');
    assert.match(h.el('transfer-detail').textContent, /Остановка B/);
  });
  await check('archive replacement invalidates its pending request', async () => {
    const h = harness(); h.api.renderTransfer(); const old = advice(h.scope.snapshot);
    h.scope.snapshot = state(); h.scope.snapshot.context.version = 2;
    h.api.renderTransfer(); h.requests[0].resolve(old); await flush();
    assert.equal(h.scope.transferAdvice, null);
    assert.equal(h.el('transfer-donor-button').hidden, true);
    h.requests[1].resolve(advice(h.scope.snapshot)); await flush();
    assert.equal(h.scope.transferAdvice.context_version, 2);
  });
  await check('changed target stop rejects the old recommendation', async () => {
    const h = harness(); const old = advice(h.scope.snapshot);
    h.scope.snapshot.vehicles[0].prediction.target.id = 'next-stop';
    assert.equal(h.api.transferMatches(old, h.scope.chosenVehicle(), h.scope.snapshot), false);
  });
  await check('another visit to the same stop is a different target', async () => {
    const h = harness(); const old = advice(h.scope.snapshot);
    h.scope.snapshot.vehicles[0].prediction.target.scheduled_at = '2026-09-27T13:12:00Z';
    assert.equal(h.api.transferMatches(old, h.scope.chosenVehicle(), h.scope.snapshot), false);
  });
  await check('stale target cannot activate a previously available action', async () => {
    const h = harness(); h.scope.transferAdvice = advice(h.scope.snapshot); h.api.paintTransfer();
    h.scope.snapshot.vehicles[0].status = 'stale'; h.api.showTransferDonor();
    assert.equal(h.centres.length, 0);
    h.api.renderTransfer(); assert.equal(h.el('detail-transfer-panel').hidden, true);
    assert.equal(h.scope.transferAdvice, null);
  });
  await check('API disconnection hides the scenario and blocks donor focus', async () => {
    const h = harness(); h.scope.transferAdvice = advice(h.scope.snapshot); h.api.paintTransfer();
    h.scope.disconnected = true; h.api.showTransferDonor(); h.api.renderTransfer();
    assert.equal(h.centres.length, 0); assert.equal(h.el('detail-transfer-panel').hidden, true);
  });
  await check('stale donor removes actionable scenario details', async () => {
    const h = harness(); h.scope.transferAdvice = advice(h.scope.snapshot); h.api.paintTransfer();
    h.scope.snapshot.vehicles[2].status = 'stale'; h.api.paintTransfer(); h.api.showTransferDonor();
    assert.equal(h.el('transfer-donor-button').hidden, true);
    assert.equal(h.el('transfer-effect').hidden, true); assert.equal(h.centres.length, 0);
  });
  await check('a donor that became late cannot be proposed or focused', async () => {
    const h = harness(); h.scope.transferAdvice = advice(h.scope.snapshot);
    h.scope.snapshot.vehicles[2].prediction.predicted_delay_s = 61;
    h.api.paintTransfer(); h.api.showTransferDonor();
    assert.equal(h.el('transfer-donor-button').hidden, true); assert.equal(h.centres.length, 0);
  });
  await check('removed donor makes the recommendation non-actionable', async () => {
    const h = harness(); h.scope.transferAdvice = advice(h.scope.snapshot);
    h.scope.snapshot.vehicles.pop(); h.api.paintTransfer(); h.api.showTransferDonor();
    assert.equal(h.el('transfer-donor-button').hidden, true); assert.equal(h.centres.length, 0);
  });
  await check('successful scenario and donor preview preserve forecasts and target selection', async () => {
    const h = harness(), before = JSON.stringify(h.scope.snapshot);
    h.el('risk-red').checked = true; h.el('vehicle-search').value = 'A';
    h.api.renderTransfer(); h.requests[0].resolve(advice(h.scope.snapshot)); await flush();
    assert.equal(h.el('transfer-donor-button').hidden, false);
    assert.match(h.el('transfer-effect').textContent, /раньше/);
    h.api.showTransferDonor(); assert.equal(h.centres.length, 1);
    assert.equal(h.scope.selectedId, 'A'); assert.equal(JSON.stringify(h.scope.snapshot), before);
    assert.equal(h.el('risk-red').checked, false); assert.equal(h.el('vehicle-search').value, '');
  });
  await check('not-needed response hides the scenario', async () => {
    const h = harness(); h.scope.transferAdvice = {...advice(h.scope.snapshot), status: 'not_needed'};
    h.api.paintTransfer(); assert.equal(h.el('detail-transfer-panel').hidden, true);
    assert.equal(h.el('transfer-donor-button').hidden, true);
  });
  await check('target below intervention threshold creates no request or action', async () => {
    const h = harness(); h.scope.transferAdvice = advice(h.scope.snapshot);
    h.scope.snapshot.vehicles[0].prediction.predicted_delay_s = 149;
    h.api.showTransferDonor(); h.api.renderTransfer();
    assert.equal(h.centres.length, 0); assert.equal(h.requests.length, 0);
    assert.equal(h.el('detail-transfer-panel').hidden, true);
  });
  await check('request failure never falls back to an actionable scenario', async () => {
    const h = harness(); h.api.renderTransfer(); h.requests[0].reject(new Error('offline')); await flush();
    assert.equal(h.scope.transferAdvice.status, 'unavailable');
    assert.equal(h.el('transfer-donor-button').hidden, true);
    assert.match(h.el('transfer-summary').textContent, /Повторяем/);
  });
  console.log(JSON.stringify({passed: true, checks, scope: 'transfer UI races, freshness and forecast isolation'}));
})().catch(error => { console.error(error); process.exitCode = 1; });
