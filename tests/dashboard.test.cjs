// Run with Node: node tests/dashboard.test.cjs. No browser or packages required.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../dashboard/app.js'), 'utf8');
const bands = source.slice(source.indexOf('  const RISK ='), source.indexOf('  const $ ='));
const gate = source.slice(source.indexOf('  const effectiveRisk ='), source.indexOf('  const availabilityLabel ='));
let disconnected = false;
const { delayBand, delayColor, effectiveRisk, vehicleColor, displayedDelay, RISK } = vm.runInNewContext(
  bands + gate + '\n;({delayBand, delayColor, effectiveRisk, vehicleColor, displayedDelay, RISK});',
  {apiStale: () => disconnected},
);
let checks = 0;
for (const [value, expected] of [
  [null, 'unknown'], [undefined, 'unknown'], [NaN, 'unknown'], [Infinity, 'unknown'],
  [-100, 'blue'], [-30.001, 'blue'], [-30, 'green'], [0, 'green'],
  [59.999, 'green'], [60, 'amber'], [120, 'amber'], [150, 'amber'], [150.001, 'red'],
]) {
  assert.equal(delayBand(value), expected, `delay=${value}`); checks++;
}
assert.equal(delayColor(null), delayColor(0)); checks++;
assert.notEqual(delayColor(151), delayColor(300)); checks++;
assert.notEqual(delayColor(300), delayColor(600)); checks++;
assert.equal(delayColor(600), '#671130'); checks++;
assert.equal(delayColor(900), delayColor(600)); checks++;
const vehicle = {status:'fresh', prediction_availability:{code:'ready'},
  prediction:{risk:'red', predicted_delay_s:130, probability_late:null}};
assert.equal(effectiveRisk(vehicle), 'amber', 'UI threshold independent of old backend band'); checks++;
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
console.log(JSON.stringify({passed:true,checks,scope:'display thresholds, missing/stale data, validated vs transferred probability'}));
