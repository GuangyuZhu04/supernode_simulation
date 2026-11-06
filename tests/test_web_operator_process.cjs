'use strict';
const assert = require('node:assert/strict');
const {test} = require('node:test');
const {fixture, harness} = require('./web_frontend_harness.cjs');

function event(id, start, end, metadata = {}) {
  return {id, start_s: start, end_s: end, duration_s: end - start, startup_s: .001,
    dependency_ready_s: 0, slot_wait_s: 0, memory_wait_s: 0, slots: ['npu0.compute_stream'],
    metadata: {phase: 'prefill', component: 'Dense', layer: 0, token_index: 0, kernel: id, ...metadata}};
}
function result(events) {
  return {config: fixture(), mode: 'simulate', trace: {events, total_events: events.length, sampled: false},
    simulation: {duration_s: 1, completed_events: events.length}, execution_summary: {complete: true, completed_events: events.length}};
}
function setup(value) {
  const h = harness(); h.context.result = value; h.evaluate('state.result = result'); h.call('renderTrace'); return h;
}
function control(host, label) { return host.querySelector('[aria-label="' + label + '"]'); }
function button(host, text) { return host.querySelectorAll('button').find(node => node.textContent === text); }

test('process uses recorded times, preserves overlapping execution and classifies pack kernels as compute', async () => {
  const h = setup(result([event('late', .4, .9), event('communication_pack', 0, .6), event('done', .9, .9, {kernel: null, operator_trace_completion: true})]));
  const host = h.node('trace-chart'), rows = host.querySelectorAll('.operator-event-row');
  assert.equal(rows.length, 3);
  assert.match(rows[0].textContent, /communication_pack/);
  assert.ok(rows[0].querySelector('.operator-kind-compute'));
  const slider = control(host, '仿真时间回放位置'); slider.value = '500'; await slider.fire('input');
  assert.equal(rows.filter(row => row.classList.contains('is-running')).length, 2);
  slider.value = '1000'; await slider.fire('input');
  assert.equal(rows.filter(row => row.classList.contains('is-finished')).length, 3);
  assert.match(h.node('trace-note').textContent, /不代表当前任务实时进度/);
});

test('stage, device, layer zero and token zero filters use metadata without changing the source', async () => {
  const a = event('matmul', 0, .2), b = event('decode', .3, .4, {phase: 'decode', layer: 1, token_index: 1});
  b.slots = ['npu1.compute_stream'];
  const value = result([a, b]), before = structuredClone(value), h = setup(value), host = h.node('trace-chart');
  const layer = control(host, '过程层'); layer.value = '0'; await layer.fire('change');
  assert.equal(host.querySelectorAll('.operator-event-row').length, 1);
  const cycle = control(host, '过程Token / 训练步'); cycle.value = 'token 0'; await cycle.fire('change');
  assert.equal(host.querySelectorAll('.operator-event-row').length, 1);
  const device = control(host, '过程设备 / Rank'); device.value = 'npu1'; await device.fire('change');
  assert.equal(host.querySelectorAll('.operator-event-row').length, 0);
  assert.match(host.textContent, /采样未命中不表示/);
  assert.equal(button(host, '播放').disabled, true);
  await button(host, '重置筛选').fire('click');
  assert.equal(host.querySelectorAll('.operator-event-row').length, 2);
  assert.deepEqual(value, before);
});

test('independently retained parent chains expose suboperators absent from the uniform sample', async () => {
  const value = result([event('sample', 0, .1)]); value.trace.total_events = 1000; value.trace.sampled = true;
  const child = event('dense.up', .1, .2, {trace_id: 'synthetic', trace_parent_event: 'parent', original_kernel: 'dense', operation_index: 0,
    matmul: {m: 2, n: 3, k: 4}, flops: 48, trace_source: 'Synthetic; not measured'});
  value.operator_execution = {expanded_operations: 100, parent_count: 50, operations_per_group: 64,
    groups: [{parent_event: 'parent', original_kernel: 'dense', phase: 'prefill', event_count: 2, events: [child], truncated: true}]};
  const h = setup(value), host = h.node('trace-chart'), chain = control(host, '算子展开链');
  chain.value = '0'; await chain.fire('change');
  assert.equal(host.querySelectorAll('.operator-event-row').length, 1);
  assert.match(host.textContent, /前缀截取/);
  assert.match(host.querySelector('.operator-inspector').textContent, /dense.up/);
  assert.match(host.querySelector('.operator-inspector').textContent, /Synthetic; not measured/);
  assert.match(h.node('trace-note').textContent, /1 \/ 1000/);
});

test('step controls cross pages and playback stops on replacement results', async () => {
  const h = setup(result(Array.from({length: 43}, (_, i) => event('op' + i, i / 100, (i + 1) / 100))));
  const host = h.node('trace-chart');
  assert.equal(host.querySelectorAll('.operator-event-row').length, 40);
  for (let i = 0; i < 40; i++) await button(host, '下一事件').fire('click');
  assert.equal(host.querySelectorAll('.operator-event-row').length, 3);
  assert.match(host.querySelector('.operator-inspector').textContent, /op40/);
  let callback;
  h.context.setTimeout = cb => { callback = cb; return 77; };
  let cleared;
  h.context.clearTimeout = id => { cleared = id; };
  await button(host, '播放').fire('click');
  assert.ok(button(host, '暂停'));
  h.evaluate('state.result = null'); callback();
  assert.equal(cleared, 77); assert.ok(button(host, '播放'));
  assert.equal(h.evaluate('operatorPlayback.timer'), null);
});

test('estimates, empty interrupted runs and absent focused phases never invent execution', () => {
  const value = result([event('x', 0, .1)]), h = setup(value);
  h.call('renderOperatorTimeline', value, {phase: 'backward', component: 'Dense', operator: 'x'});
  assert.equal(h.node('trace-chart').querySelectorAll('.operator-event-row').length, 0);
  h.call('renderOperatorTimeline', {...value, mode: 'estimate'});
  assert.equal(h.node('trace-chart').querySelectorAll('.operator-event-row').length, 0);
  h.call('renderOperatorTimeline', {...value, trace: {events: []}, execution_summary: {complete: false}});
  assert.match(h.node('trace-chart').textContent, /本次执行未完成/);
  assert.match(h.node('trace-chart').textContent, /没有可展示/);
});

test('operator results paginate the complete ledger and filtering keeps the original denominator', async () => {
  const h = harness(), costs = {compute_s: 1, communication_s: 0, memory_access_s: 0, launch_s: 0, synchronization_wait_s: 0, scheduling_wait_s: 0,
    memory_wait_s: 0, resource_contention_s: 0, unattributed_service_s: 0, other_s: 0};
  const percentages = {...costs, compute_s: 100};
  const operators = Array.from({length: 60}, (_, i) => ({phase: i % 2 ? 'decode' : 'prefill', component: 'Dense', operator: 'op' + String(i).padStart(2, '0'),
    total_s: 1, share_pct: 100 / 60, components_s: costs, component_shares_pct: percentages}));
  h.context.result = {config: fixture(), metrics: {latency_breakdown: {ttft: {operator_breakdown: {total_s: 60, components_s: {...costs, compute_s: 60}, component_shares_pct: percentages, operators}}}}};
  h.evaluate('state.result = result'); h.call('renderLatencyBreakdown');
  const host = h.node('latency-breakdown'), table = () => host.querySelectorAll('.result-detail-table').at(-1);
  assert.equal(table().querySelector('tbody').children.length, 50);
  await button(host, '下一页').fire('click'); assert.equal(table().querySelector('tbody').children.length, 10);
  const phase = control(host, '算子结果阶段'); phase.value = 'decode'; await phase.fire('change');
  assert.equal(table().querySelector('tbody').children.length, 30);
  assert.match(host.textContent, /占原窗口 50%/);
  assert.equal(table().querySelector('tbody').children[0].children[4].textContent, '1.667%');
  const search = control(host, '搜索算子结果'); search.value = 'op59'; await search.fire('input');
  await button(host, 'op59').fire('click');
  assert.match(host.querySelector('.operator-inspector').textContent, /内存准入/);
  assert.match(host.querySelector('.operator-inspector').textContent, /调度等待/);
});
