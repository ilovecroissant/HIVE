const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const ts=require('typescript');
const vm=require('node:vm');
const output=ts.transpileModule(fs.readFileSync('lib/event-presentation.ts','utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2020}}).outputText;
const scope={exports:{}};vm.runInNewContext(output,scope);
const {routingRecords,eventLabel}=scope.exports;
test('routing uses latest actual event, preserving reasons and skipped agents',()=>{
 const result=routingRecords([{tick:4,title:'Specialist Activation',message:'energy: ACTIVE — power low; engineering: SKIPPED — defer; fuel scarce; life_support: ACTIVE — oxygen low'}]);
 assert.equal(result.tick,4);assert.equal(result.records.length,3);
 assert.equal(result.records[1].status,'SKIPPED');assert.equal(result.records[1].reason,'defer; fuel scarce');
 assert.equal(routingRecords([]),null);
 assert.equal(routingRecords([{title:'Specialist Activation',message:'unknown format'}]),null);
 assert.equal(routingRecords([{tick:1,title:'Specialist Activation',message:'energy: ACTIVE — low'},{tick:2,title:'Specialist Activation',message:'energy: SKIPPED — stable'}]).records[0].status,'SKIPPED');
});
test('only real backend event titles receive special labels',()=>{
 for(const [title,label] of [['Conflict: x','CONFLICT'],['Commander Rejects: x','REJECTED'],['Executed: x','EXECUTED'],['Specialist Activation','ROUTING']]) assert.equal(eventLabel({title,type:'system'}),label);
 assert.equal(eventLabel({title:'Other',type:'error'}),'ERROR');
});
