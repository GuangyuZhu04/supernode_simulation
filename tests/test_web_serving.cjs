'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const {harness, fixture} = require('./web_frontend_harness.cjs');

function setup() {
  const h = harness();
  h.evaluate("$('#run-mode').value = 'serve'; attachServingEvents(); renderServingEditor();");
  return h;
}
function servingResult() {
  return {mode:'serve',engine_mode:'serving',status:'completed_with_failures',config:fixture(),
    hardware:{instance_count:2,npus_per_instance:4,total_npus:8}, clock_origin_s:1e20,
    time_window_s:{start:1e20,end:1e20,duration:4},time_window_relative_s:{start:0,end:4,duration:4},
    metrics:{request_count:500,completed_requests:499,failed_requests:1,
      distributions:{client_ttft_s:{sample_count:499,mean_s:.8,p50_s:.5,p90_s:1,p95_s:1.1,p99_s:2},
        tpot_s:{sample_count:498,mean_s:.02,p50_s:.02,p90_s:.03,p95_s:.04,p99_s:.05}}},
    throughput:{output_tokens_per_s:1000,completed_requests_per_s:124.75},
    goodput:{specified:true,passed_requests:480,joint_requests_per_s:120,passed_output_tokens_per_s:900},
    requests:[{id:'single',instance_id:0,batch_id:0,status:'completed',slo_passed:true,admission_wait_s:0,
      client_ttft_s:.1,server_ttft_s:.1,tpot_s:null,e2e_s:.1,output_tokens:1,requested_output_tokens:1},
      {id:'slow',instance_id:0,batch_id:1,status:'completed',slo_passed:false,admission_wait_s:1,
       client_ttft_s:2,server_ttft_s:1,tpot_s:.1,e2e_s:2.1,output_tokens:2,requested_output_tokens:2},
      {id:'oom',instance_id:1,batch_id:2,status:'failed',slo_passed:false,admission_wait_s:2,
       client_ttft_s:null,server_ttft_s:null,tpot_s:null,e2e_s:null,output_tokens:0,requested_output_tokens:2}],
    batches:[{id:0,instance_id:0,request_ids:['single'],status:'completed',duration_s:.1,
      admission_relative_s:0,model_start_relative_s:.01,end_relative_s:.11},
      {id:2,instance_id:1,request_ids:['oom'],status:'failed',duration_s:1,
       admission_relative_s:2,model_start_relative_s:2.01,end_relative_s:3.01,diagnostics:[{code:'memory_admission_deadlock',message:'显存不足'}]}],
    executions:[],display_limits:{request_count:500,batch_count:220},limitations:['不模拟连续 batching。']};
}

test('serving editor changes a separate model and preserves explicit zero thresholds', async () => {
  const h = setup(), before = h.json('state.config');
  h.node('serving-instances').value = '2'; await h.node('serving-instances').fire('input');
  h.node('serving-ttft').value = '0'; await h.node('serving-ttft').fire('input');
  const arrival = h.node('serving-request-rows').querySelector('[data-serving-field="arrival_s"]');
  arrival.value = '.25'; await arrival.fire('input');
  assert.equal(h.json('readServing()').instance_count,2);
  assert.equal(h.json('readServing()').slo.ttft_s,0);
  assert.equal(h.json('readServing()').requests[0].arrival_s,.25);
  assert.deepEqual(h.json('state.config'),before);
  assert.match(h.node('serving-resource-note').textContent,/合计 8 张卡/);
  assert.equal(h.node('batch').disabled,true);
});

test('training disables serve and retains the separate request draft', () => {
  const h = setup(), serving = h.json('servingDraft.value');
  h.evaluate("state.config.workload.scenario = 'training'; syncServingMode();");
  assert.equal(h.node('run-mode').value,'simulate');
  assert.equal(h.node('serve-mode-option').disabled,true);
  assert.equal(h.node('serving-editor').hidden,true);
  assert.deepEqual(h.json('servingDraft.value'),serving);
});

test('service validation and jobs submit config and serving separately', async () => {
  const h = setup(), sent=[];
  h.context.fetch = async (url,options) => {
    const body = JSON.parse(options.body); sent.push({url,body});
    return {ok:true,json:async () => url==='/api/validate' ? {valid:true,config:body.config,serving:body.serving,mode:'serve'} : {id:'served',status:'queued',mode:'serve'}};
  };
  await h.call('runSimulation',{preventDefault(){}});
  assert.deepEqual(sent.map(row=>row.url),['/api/validate','/api/jobs']);
  assert.equal(sent[0].body.mode,'serve');
  assert.deepEqual(sent[0].body.serving,sent[1].body.serving);
  assert.equal(Object.hasOwn(sent[1].body.config,'serving'),false);
  assert.equal(h.node('serving-instances').disabled,false);
});

test('invalid service JSON is preserved and blocks replacement or submission', async () => {
  const h = setup(), before = h.json('state.config'); let calls=0;
  h.context.fetch = async () => { calls++; throw new Error('unexpected'); };
  h.node('serving-json').value='{invalid'; await h.node('serving-json').fire('input');
  await h.call('loadServingExample');
  await h.call('runSimulation',{preventDefault(){}});
  assert.equal(calls,0); assert.equal(h.node('serving-json').value,'{invalid');
  assert.deepEqual(h.json('state.config'),before);
  assert.equal(h.node('serving-instances').disabled,true);
  await h.node('discard-serving-json').fire('click');
  assert.equal(h.json('servingDraft.jsonDirty'),false);
  assert.equal(h.node('serving-instances').disabled,false);
});

test('synthetic paired example does not overwrite unapplied model edits', async () => {
  const h = setup(); let calls=0;
  h.evaluate("state.jsonDirty = true; $('#config-json').value = '{pending';");
  h.context.fetch = async () => { calls++; throw new Error('unexpected'); };
  await h.call('loadServingExample');
  assert.equal(calls,0); assert.equal(h.node('config-json').value,'{pending');
});

test('paired example changes both inputs and preview source without reusing an old job', async () => {
  const h=setup(), saved=fixture(); saved.model.name='synthetic-paired';
  h.evaluate("simulationView.source='job'; state.currentJob={id:'previous',mode:'simulate',status:'completed'}");
  h.context.fetch=async()=>({ok:true,json:async()=>({config:saved,serving:{instance_count:2,max_batch_size:1,max_batch_wait_s:0,scheduler_overhead_s:0,requests:[{id:'new',arrival_s:0,input_tokens:8,output_tokens:2}]}})});
  await h.call('loadServingExample');
  assert.equal(h.json('state.config').model.name,'synthetic-paired');
  assert.equal(h.json('servingDraft.value').requests[0].id,'new');
  assert.equal(h.json('simulationView.source'),'draft');
  assert.equal(h.node('preset').value,'');
});

test('paired history restore deep copies both snapshots and stale responses keep edits', async () => {
  const h = setup(), saved=fixture(); saved.model.name='historical';
  const serving={instance_count:3,max_batch_size:1,max_batch_wait_s:0,scheduler_overhead_s:0,
    requests:[{id:'saved',arrival_s:1,input_tokens:8,output_tokens:1}]};
  h.evaluate("state.currentJob={id:'old',mode:'serve',status:'completed'}");
  h.context.fetch = async url => ({ok:true,json:async()=>url.endsWith('/serving') ? {serving} : {config:saved}});
  await h.call('restoreRunInputs');
  assert.equal(h.json('state.config').model.name,'historical');
  assert.deepEqual(h.json('servingDraft.value'),serving);
  h.evaluate("servingDraft.value.requests[0].id='changed'");
  assert.equal(serving.requests[0].id,'saved');
  const pending=[];
  h.context.fetch = url => new Promise(resolve=>pending.push({url,resolve}));
  const work=h.call('restoreRunInputs');
  h.evaluate("servingDraft.revision++; servingDraft.value.requests[0].id='fresh-edit'");
  pending.forEach(({url,resolve})=>resolve({ok:true,json:async()=>url.endsWith('/serving')?{serving}:{config:saved}}));
  await work;
  assert.equal(h.json('servingDraft.value').requests[0].id,'fresh-edit');
  assert.match(h.node('restore-run-status').textContent,/已保留当前输入/);
});

test('ordinary history restore clears stale serving state instead of borrowing a draft', async () => {
  const h=setup(); h.evaluate("state.currentJob={id:'ordinary',mode:'simulate',status:'completed'}");
  let count=0; h.context.fetch=async()=>{count++;return {ok:true,json:async()=>({config:fixture()})};};
  await h.call('restoreRunInputs');
  assert.equal(count,1); assert.equal(h.json('servingDraft.value'),null); assert.equal(h.node('run-mode').value,'simulate');
});

test('service result uses complete population metrics, relative clocks and distinct failure classes', () => {
  const h=setup(); h.evaluate('renderArtifactLinks=()=>{};');
  h.call('renderResults',servingResult(),{id:'served',mode:'serve'});
  assert.equal(h.node('ordinary-result-panels').hidden,true);
  assert.equal(h.node('serving-results').hidden,false);
  assert.match(h.node('serving-percentiles').textContent,/499/);
  assert.match(h.node('serving-request-result-note').textContent,/3 \/ 500/);
  const table=h.node('serving-request-results').textContent;
  assert.match(table,/— \/ 不适用/); assert.match(table,/完成 · SLO 未达标/); assert.match(table,/执行失败/);
  assert.match(h.node('serving-goodput').textContent,/480/);
  assert.equal(h.node('serving-timeline').querySelectorAll('.trace-row').length,2);
  assert.match(h.node('serving-timeline-note').textContent,/2 \/ 220/);
  assert.doesNotMatch(h.node('diagnostic-content').textContent,/不含请求到达/);
});

test('old results without relative clocks never synthesize timeline offsets', () => {
  const h=setup(), result=servingResult();
  result.batches.forEach(batch=>{delete batch.admission_relative_s;delete batch.model_start_relative_s;delete batch.end_relative_s;});
  h.call('renderServingTimeline',result);
  assert.equal(h.node('serving-timeline').querySelectorAll('.trace-row').length,0);
  assert.match(h.node('serving-timeline').textContent,/未提供相对批次时钟/);
});

test('completed execution with missed serving SLO is warning in result and history, not engine failure', () => {
  const h=setup(), result=servingResult();
  result.status='completed'; result.metrics.failed_requests=0; result.metrics.completed_requests=500;
  h.call('renderServingResults',result,{id:'slow-service',mode:'serve',status:'completed'});
  assert.match(h.node('result-status').textContent,/执行完成 · SLO 未达标/);
  assert.equal(h.node('result-status').classList.contains('warning'),true);
  assert.match(h.node('result-alert').textContent,/20 个成功完成的请求未满足服务 SLO/);
  assert.match(h.node('result-alert').textContent,/不是引擎执行失败/);
  assert.equal(h.node('serving-goodput').classList.contains('warning'),true);
  assert.deepEqual(h.json("servingJobOutcome({id:'slow-service',mode:'serve',status:'completed'})").tone,'warning');
  assert.equal(h.json("servingJobOutcome({id:'unread',mode:'serve',status:'completed'})").tone,'');
  assert.match(h.json("servingJobOutcome({id:'unread',mode:'serve',status:'completed'})").label,/查看 SLO/);
  result.goodput.passed_requests=500;
  h.call('renderServingResults',result,{id:'met',mode:'serve',status:'completed'});
  assert.equal(h.node('result-status').classList.contains('success'),true);
  assert.equal(h.node('result-alert').hidden,true);
  assert.equal(h.node('serving-goodput').classList.contains('warning'),false);
});

test('training roles, dtype peaks and separate CPU accounting preserve unavailable metrics', () => {
  const h=setup();
  h.call('renderTrainingAccounting',{metrics:{training_accounting:{status:'complete',model_mfu:.25,executed_compute_utilization:.4,
    npu_flops_by_phase:{forward:100,backward:200},npu_flops_by_role:{model_forward_backward:300,model_recompute:40,optimizer:20,auxiliary:5},
    npu_by_dtype:{fp16:{executed_flops:340,model_forward_backward_flops:300,model_recompute_flops:40,peak_flops_per_device_s:1000}},
    cpu:{device_count:1,executed_flops:10,optimizer_phase_flops:8,executed_compute_utilization:.1}}}});
  assert.equal(h.node('training-accounting-panel').hidden,false);
  assert.match(h.node('training-accounting-kpis').textContent,/25%/);
  assert.match(h.node('training-accounting-tables').textContent,/CPU（不混入 NPU MFU）/);
  h.call('renderTrainingAccounting',{metrics:{training_accounting:{status:'unavailable',model_mfu:null,executed_compute_utilization:null,reason:'incomplete'}}});
  assert.doesNotMatch(h.node('training-accounting-kpis').textContent,/0%/);
  h.call('renderTrainingAccounting',{metrics:{}});
  assert.equal(h.node('training-accounting-panel').hidden,true);
});

test('operator detail and calibration evidence are distinguished and missing legacy fields hide', () => {
  const h=setup(); const metadata={
    operator_trace:{matched_events:1,eligible_events:2,coverage_fraction:.5,generated_operations:3,additional_memory_bytes:64,skip_reasons:{metadata_mismatch:1}},
    kernel_calibration:{matched_events:1,eligible_events:3,coverage_fraction:1/3,scope_mismatch_reasons:{environment_mismatch:2}},
    collective_calibration:{matched_collectives:0,eligible_collectives:2,coverage_fraction:0,unmatched_reasons:{no_profile:2}}};
  h.call('renderExecutionEvidence',{mode:'simulate',model_metadata:metadata,trace:{events:[{id:'op1',metadata:{kernel:'gemm',original_kernel:'dense',trace_id:'test',phase:'forward',flops:64,matmul:{m:2,n:4,k:4}}}]}});
  const text=h.node('execution-evidence').textContent;
  assert.match(text,/算子展开/);assert.match(text,/environment_mismatch/);assert.match(text,/no_profile/);
  assert.match(h.node('operator-trace-events').textContent,/dense → gemm/);
  h.call('renderExecutionEvidence',{mode:'simulate'});
  assert.equal(h.node('execution-evidence-panel').hidden,true);
});

test('large request editing renders bounded initial rows and a newly added row', async () => {
  const h=setup();
  h.evaluate("servingDraft.value.requests=Array.from({length:1000},(_,i)=>({id:'r'+i,arrival_s:i,input_tokens:8,output_tokens:1}));renderServingRequestRows();");
  assert.equal(h.node('serving-request-rows').children.length,100);
  await h.node('serving-add-request').fire('click');
  assert.equal(h.json('servingDraft.value.requests.length'),1001);
  assert.equal(h.node('serving-request-rows').children.length,101);
});

test('empty length arrays use scalar batch defaults without expanding an oversized batch', () => {
  const h=setup();
  h.evaluate('servingDraft.value=null; state.config.workload.input_lengths=[]; state.config.workload.output_lengths=[]; ensureServingDraft();');
  const requests=h.json('readServing().requests');
  assert.equal(requests.length,2); assert.equal(requests[0].input_tokens,1024); assert.equal(requests[1].output_tokens,16);
  h.evaluate('servingDraft.value=null; state.config.workload.batch_size=100000; ensureServingDraft();');
  assert.equal(h.json('servingDraft.value.requests.length'),0);
  assert.match(h.node('serving-editor-status').textContent,/未自动展开或截断/);
});

test('microsecond plan times and zero eligible evidence remain explicit', () => {
  const h=setup();
  h.call('renderServingPlan',fixture(),{instance_count:1,max_batch_size:2,max_batch_wait_s:5e-6,scheduler_overhead_s:2e-6,requests:[{}]},null);
  assert.match(h.node('execution-facts').textContent,/5 μs/);
  assert.match(h.node('execution-facts').textContent,/2 μs/);
  h.call('renderExecutionEvidence',{mode:'simulate',model_metadata:{collective_calibration:{eligible_collectives:0,matched_collectives:0,coverage_fraction:0}}});
  assert.match(h.node('execution-evidence').textContent,/无适用事件 · —/);
  assert.doesNotMatch(h.node('execution-evidence').textContent,/0%/);
});
