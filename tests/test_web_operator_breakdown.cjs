// Run with: node --test tests/test_web_operator_breakdown.cjs
'use strict';
const assert = require('node:assert/strict');
const {test} = require('node:test');
const {fixture, harness} = require('./web_frontend_harness.cjs');

function record(operator, seconds) {
  const components = {compute_s: seconds * .4, communication_s: seconds * .2,
    memory_access_s: seconds * .3, launch_s: seconds * .1, synchronization_wait_s: 0,
    scheduling_wait_s: 0, memory_wait_s: 0, resource_contention_s: 0,
    unattributed_service_s: 0, other_s: 0};
  const percentages = Object.fromEntries(Object.entries(components).map(([key,value]) => [key, 100 * value / seconds]));
  return {total_s: seconds, components_s: {}, operator_breakdown: {total_s: seconds,
    components_s: components, component_shares_pct: percentages, operators: [{
      phase: 'test', component: 'Attention', operator, total_s: seconds, share_pct: 100,
      components_s: components, component_shares_pct: percentages,
    }]}};
}

test('operator window selector renders the actual values and switches inference metrics', async () => {
  const config = fixture(), h = harness(config);
  h.context.result = {config, metrics: {latency_breakdown: {ttft: record('prefill_kernel', .01), tpot: record('decode_kernel', .002)}}};
  h.evaluate('state.result = result');
  h.call('renderLatencyBreakdown');
  const host = h.node('latency-breakdown');
  const select = host.querySelector('select');
  const operatorRow = () => host.querySelectorAll('.result-detail-table').at(-1).querySelector('tbody').children[0].children.map(cell => cell.textContent);
  assert.deepEqual(operatorRow(), ['test', 'Attention', 'prefill_kernel', '10', '100%', '40%', '20%', '30%', '10%', '0%', '0%', '0%']);
  select.value = 'tpot'; await select.fire('change');
  assert.deepEqual(operatorRow().slice(2, 5), ['decode_kernel', '2', '100%']);
  assert.equal(host.textContent.includes('prefill_kernel'), false);
});

test('training uses step windows and missing steady-state is not displayed as zeros', async () => {
  const config = fixture(); config.workload.scenario = 'training';
  const h = harness(config);
  h.context.result = {config, metrics: {training_latency_breakdown: {step: record('train_kernel', .1),
    first_step: record('first_kernel', .12), steady_step: null, steady_step_unavailable_reason: 'At least two steps'}}};
  h.evaluate('state.result = result');
  h.call('renderLatencyBreakdown');
  const host = h.node('latency-breakdown'), select = host.querySelector('select');
  assert.deepEqual(select.options.map(option => option.value), ['step', 'first_step', 'steady_step']);
  assert.equal(host.textContent.includes('train_kernel'), true);
  select.value = 'first_step'; await select.fire('change');
  assert.equal(host.textContent.includes('first_kernel'), true);
  select.value = 'steady_step'; await select.fire('change');
  assert.equal(host.querySelectorAll('.result-detail-table').length, 0);
  assert.equal(host.textContent.includes('At least two steps'), true);
});

test('legacy reports without an operator ledger remain renderable and unavailable', () => {
  const h = harness();
  h.context.result = {config: fixture(), metrics: {latency_breakdown: {ttft: {total_s: 1, components_s: {compute_s: 1}}}}};
  h.evaluate('state.result = result');
  h.call('renderLatencyBreakdown');
  assert.equal(h.node('latency-breakdown').querySelectorAll('.result-detail-table').length, 0);
});
