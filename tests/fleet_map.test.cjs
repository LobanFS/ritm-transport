const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(require('node:path').join(__dirname, '../dashboard/app.js'), 'utf8');
const fragment = source.slice(source.indexOf('  const sameSelectedLine ='), source.indexOf('  function centreVehicle'));
const chosen = {tr_id:1, route_id:'a'};
const peer = {tr_id:2, route_id:'b'};
const other = {tr_id:3, route_id:'c'};
const state = {vehicles:[chosen, peer, other], line_memberships:{a:['a','b'], b:['a','b'], c:['c']}};
const filters = {'risk-red':{checked:false}, 'risk-amber':{checked:false}};
const context = {snapshot:state, chosenVehicle:()=>chosen, list:v=>Array.isArray(v)?v:[], idOf:v=>String(v??''), $:id=>filters[id]};
const {sameSelectedLine, vehiclesOnMap} = vm.runInNewContext(fragment+';({sameSelectedLine,vehiclesOnMap});',context);
assert.equal(sameSelectedLine(chosen,chosen,state),true);
assert.equal(sameSelectedLine(peer,chosen,state),true);
assert.equal(sameSelectedLine(other,chosen,state),false);
assert.equal(sameSelectedLine(peer,chosen,{}),false);
assert.equal(sameSelectedLine(chosen,chosen,{}),true);
assert.equal(sameSelectedLine(peer,null,state),false);
assert.deepEqual(Array.from(vehiclesOnMap([chosen]),v=>v.tr_id),[1,2]);
assert.deepEqual(Array.from(vehiclesOnMap([chosen,peer,other]),v=>v.tr_id),[1,2,3]);
assert.equal(sameSelectedLine(peer,chosen,{line_memberships:{a:['a']}}),false);
for (const [red,amber] of [[true,false],[false,true],[true,true]]) {
  filters['risk-red'].checked=red; filters['risk-amber'].checked=amber;
  assert.deepEqual(Array.from(vehiclesOnMap([chosen]),v=>v.tr_id),[1], 'Risk filter must not add peers of another color');
  assert.deepEqual(Array.from(vehiclesOnMap([])),[], 'Selection must not bypass an empty risk result');
}
console.log(JSON.stringify({passed:true,checks:15,scope:'line membership, archive replacement, search peers and strict risk filters'}));
