'use strict';
const assert = require('node:assert/strict');
const {test} = require('node:test');
const {harness} = require('./web_frontend_harness.cjs');

test('streaming stages show observed chunks and do not invent retained operator events', () => {
  const h = harness();
  const result = {mode: 'simulate', execution_mode: 'structured_layer_streaming',
    execution_summary: {basis: 'completed_chunks', complete: true, completed_events: 1234,
      planned_events: null, phases: [{phase: 'prefill', start_s: 0, end_s: 2, event_count: 1234}]},
    trace: {events: [], total_events: 1234, retained: false, sampled: false}};
  h.call('renderExecutionSummary', result);
  assert.match(h.node('execution-observed').textContent, /1,234.*已完成事件/);
  assert.match(h.node('execution-observed').textContent, /streaming_chunks.csv/);
  assert.doesNotMatch(h.node('execution-observed').textContent, /计划事件/);
  h.call('renderOperatorTimeline', result);
  assert.match(h.node('trace-chart').textContent, /未保留逐算子回放轨迹/);
  assert.match(h.node('trace-note').textContent, /逐事件轨迹未保留/);
  assert.doesNotMatch(h.node('trace-note').textContent, /均匀采样/);
});

test('streaming communication states its completed-chunk basis even for partial runs', () => {
  const h = harness();
  h.context.streamResult = {communication_basis: 'completed_streaming_chunks',
    communication: {physical_bytes_by_domain: {SU: 1024}}};
  h.evaluate('state.result = streamResult');
  h.call('renderCommunication');
  assert.match(h.node('communication-note').textContent, /已完成分块.*中断分块/);
  assert.doesNotMatch(h.node('communication-note').textContent, /完整事件图/);
});
