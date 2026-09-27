// Run with Node: node tests/dashboard.test.cjs. No browser or packages required.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../dashboard/app.js'), 'utf8');
const riskPolicy = JSON.parse(fs.readFileSync(path.join(__dirname, '../common/risk_policy.json'), 'utf8'));
const bands = source.slice(source.indexOf('  const RISK ='), source.indexOf('  const $ ='));
const gate = source.slice(source.indexOf('  const effectiveRisk ='), source.indexOf('  const availabilityLabel ='));
let disconnected = false;
const scope = {apiStale: () => disconnected, snapshot: {risk_policy: riskPolicy}};
const { delayBand, delayColor, effectiveRisk, vehicleColor, displayedDelay, RISK } = vm.runInNewContext(
  bands + gate + '\n;({delayBand, delayColor, effectiveRisk, vehicleColor, displayedDelay, RISK});',
  scope,
);
let checks = 0;
for (const [value, expected] of [
  [null, 'unknown'], [undefined, 'unknown'], [NaN, 'unknown'], [Infinity, 'unknown'],
  [-100, 'blue'], [-30.001, 'blue'], [-30, 'green'], [0, 'green'],
  [59.999, 'green'], [60, 'amber'], [120, 'amber'], [130, 'amber'], [150, 'amber'], [150.001, 'red'],
]) {
  assert.equal(delayBand(value), expected, `delay=${value}`); checks++;
}
assert.equal(delayColor(null), delayColor(0)); checks++;
assert.notEqual(delayColor(151), delayColor(300)); checks++;
assert.notEqual(delayColor(300), delayColor(600)); checks++;
assert.equal(delayColor(600), '#671130'); checks++;
assert.equal(delayColor(900), delayColor(600)); checks++;
const vehicle = {status:'fresh', prediction_availability:{code:'ready'},
  prediction:{risk:'amber', predicted_delay_s:130, probability_late:null}};
assert.equal(effectiveRisk(vehicle), vehicle.prediction.risk, 'API and UI share the amber band'); checks++;
for (const [seconds, apiRisk] of [[0,'green'],[60,'amber'],[120,'amber'],[130,'amber'],[150,'amber'],[150.001,'red']]) {
  const input = {...vehicle,prediction:{...vehicle.prediction,predicted_delay_s:seconds,risk:apiRisk}};
  assert.equal(effectiveRisk(input), apiRisk, `UI matches API severity for delay=${seconds}`); checks++;
}
assert.equal(effectiveRisk({...vehicle,prediction:{...vehicle.prediction,predicted_delay_s:-31}}), 'blue'); checks++;
for (const changed of [
  {...vehicle,status:'stale'},
  {...vehicle,prediction_availability:{code:'deviation_expired'}},
  {...vehicle,prediction:{...vehicle.prediction,risk:'unknown'}},
  {...vehicle,prediction:null},
]) {
  assert.equal(displayedDelay(changed), null, 'never replace missing prediction with zero'); checks++;
  assert.equal(effectiveRisk(changed), 'unknown');
  assert.equal(vehicleColor(changed), delayColor(null)); checks+=2;
}
assert.equal(displayedDelay(vehicle), 130); checks++;
assert.equal(RISK.unknown.label, 'По плану'); checks++;
for (const state of [null, {}, {risk_policy: {}}, {risk_policy: {...riskPolicy, amber_from_s: NaN}}]) {
  scope.snapshot = state;
  assert.equal(delayBand(300), 'unknown', 'no local threshold fallback'); checks++;
  assert.equal(displayedDelay(vehicle), null); checks++;
  assert.equal(delayColor(300, 'red'), RISK.unknown.color); checks++;
}
scope.snapshot = {risk_policy: {...riskPolicy, amber_from_s: 80, red_above_s: 200, early_below_s: -50}};
for (const [seconds, band] of [[60,'green'],[80,'amber'],[150.001,'amber'],[200,'amber'],[200.001,'red'],[-31,'green'],[-51,'blue']]) {
  assert.equal(delayBand(seconds), band, 'bands must follow the API policy'); checks++;
}
assert.equal(delayColor(200, 'red'), '#d74e4b'); checks++;
assert.equal(delayColor(650), '#671130'); checks++;
scope.snapshot = {risk_policy: riskPolicy};
disconnected = true;
assert.equal(effectiveRisk(vehicle), 'unknown'); checks++;
const probabilitySource = source.slice(source.indexOf('  const probabilityLabel ='), source.indexOf('  const expectedArrival ='));
const probabilityLabel = vm.runInNewContext(probabilitySource + '\n;probabilityLabel;', {finite: Number.isFinite});
for (const [value, status, expected] of [
  [.28, 'validated', '28%'], [.28, 'transferred', '≈28% · приближённо'],
  [.28, 'unavailable', '—'], [null, 'transferred', '—'],
  [NaN, 'validated', '—'], [1.01, 'validated', '—'],
  [0, 'transferred', '≈0% · приближённо'], [1, 'validated', '100%'],
]) {
  assert.equal(probabilityLabel(value,status),expected); checks++;
}
const formatSource = source.slice(source.indexOf('  const duration ='), source.indexOf('  const probabilityLabel ='));
const format = vm.runInNewContext(formatSource + '\n;({delay});', {finite: Number.isFinite});
for (const [value, compact, expected] of [
  [-75,false,'Опережение на 1 мин 15 с'], [-75,true,'Опережение 1:15'],
  [-15,false,'Опережение на 15 с'], [75,true,'+1:15'], [null,false,'По расписанию'],
]) { assert.equal(format.delay(value,compact), expected); checks++; }
const filterSource = source.slice(source.indexOf('  const riskAllowed ='), source.indexOf('  function filteredVehicles'));
const riskAllowed = vm.runInNewContext(filterSource + '\n;riskAllowed;');
for (const [band, red, amber, expected] of [
  ['unknown',false,false,true], ['blue',false,false,true], ['red',true,false,true],
  ['amber',true,false,false], ['red',false,true,false], ['amber',false,true,true],
  ['red',true,true,true], ['amber',true,true,true], ['green',true,true,false],
]) { assert.equal(riskAllowed(band,red,amber),expected); checks++; }
console.log(JSON.stringify({passed:true,checks,scope:'display thresholds, missing/stale data, validated vs transferred probability'}));
