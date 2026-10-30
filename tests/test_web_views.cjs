// Run with: node --test tests/test_web_views.cjs
'use strict';
const assert = require('node:assert/strict');
const {test} = require('node:test');
const {fixture, harness} = require('./web_frontend_harness.cjs');

const physicalDevices = model => Array.from(model.nodes).filter(node => node.kind === 'npu').map(node => node.physical).sort((a, b) => a - b);
const pairs = model => Array.from(model.edges).filter(edge => edge.domain === 'SU').map(edge => [edge.src, edge.dst].sort().join('|')).sort();

function topologyFixture(kind = 'full_mesh') {
  const config = fixture();
  Object.assign(config.hardware, {topology: kind, npu_count: 4, su_size: 4, mesh_width: 2, leaf_size: 2});
  return config;
}

test('fabric views follow full-mesh, mesh and leaf/spine connectivity', () => {
  const h = harness();
  const full = h.call('buildTopologyModel', topologyFixture(), {view: 'domain', domain: 0});
  assert.deepEqual(physicalDevices(full), [0, 1, 2, 3]);
  assert.deepEqual(pairs(full), ['npu0|npu1', 'npu0|npu2', 'npu0|npu3', 'npu1|npu2', 'npu1|npu3', 'npu2|npu3']);
  const mesh = h.call('buildTopologyModel', topologyFixture('mesh'), {view: 'domain', domain: 0});
  assert.deepEqual(pairs(mesh), ['npu0|npu1', 'npu0|npu2', 'npu1|npu3', 'npu2|npu3']);
  for (const kind of ['clos', 'fat_tree']) {
    const model = h.call('buildTopologyModel', topologyFixture(kind), {view: 'domain', domain: 0});
    assert.deepEqual(pairs(model), ['npu0|su0.leaf0', 'npu1|su0.leaf0', 'npu2|su0.leaf1', 'npu3|su0.leaf1', 'su0.leaf0|su0.spine', 'su0.leaf1|su0.spine']);
  }
});

test('a partial final SU contains only configured physical devices', () => {
  const config = topologyFixture(); config.hardware.npu_count = 10;
  const h = harness(config);
  const model = h.call('buildTopologyModel', config, {view: 'domain', domain: 2});
  assert.equal(model.groupCount, 3);
  assert.deepEqual(physicalDevices(model), [8, 9]);
  assert.deepEqual(pairs(model), ['npu8|npu9']);
});

test('explicit graphs retain isolated NPUs and the supplied directed link', () => {
  const config = topologyFixture('explicit');
  config.hardware.explicit_links = [{id: 'one-way', src: 'npu0', dst: 'switch-a', domain: 'SU', bandwidth_Bps: 2e11, latency_s: 3e-6}];
  const h = harness(config);
  const model = h.call('buildTopologyModel', config);
  assert.deepEqual(physicalDevices(model), [0, 1, 2, 3]);
  assert.equal(model.nodes.some(node => node.id === 'switch-a'), true);
  assert.equal(model.edges.length, 1);
  const edge = model.edges[0];
  assert.equal(edge.src, 'npu0'); assert.equal(edge.dst, 'switch-a');
  assert.equal(Boolean(edge.bidirectional), false);
  assert.equal(edge.bandwidth_Bps, 2e11); assert.equal(edge.latency_s, 3e-6);
});

test('rank placement follows the configured physical permutation and PD pools', () => {
  const config = topologyFixture();
  Object.assign(config.hardware, {npu_count: 8});
  Object.assign(config.execution, {tp: 2, ep: 2, dp: 2, rank_order: [7, 6, 5, 4, 3, 2, 1, 0]});
  const h = harness(config);
  const rank = h.call('topologyRank', config, 2);
  assert.equal(rank.rank, 5); assert.equal(rank.dp, 1); assert.equal(rank.pp, 0); assert.equal(rank.ep, 0); assert.equal(rank.tp, 1);
  Object.assign(config.execution, {dp: 1, rank_order: []});
  config.execution.features.prefill_decode_disaggregation = {enabled: true, prefill_ranks: [0, 2, 4, 6], decode_ranks: [1, 3, 5, 7]};
  const prefill = h.call('topologyRank', config, 4), decode = h.call('topologyRank', config, 5);
  assert.equal(prefill.rank, 2); assert.equal(decode.rank, 2);
  assert.notEqual(prefill.pool, decode.pool);
  assert.equal(prefill.ep, 1); assert.equal(decode.ep, 1);
});

test('large fabrics paginate without creating unbounded browser nodes or phantom devices', () => {
  const config = topologyFixture();
  Object.assign(config.hardware, {npu_count: 4097, su_size: 256});
  const h = harness(config);
  const overview = h.call('buildTopologyModel', config, {view: 'overview'});
  assert.equal(overview.groupCount, 17);
  assert.ok(overview.pageCount > 1);
  assert.ok(overview.nodes.length < 100);
  const first = h.call('buildTopologyModel', config, {view: 'domain', domain: 0, page: 0});
  const second = h.call('buildTopologyModel', config, {view: 'domain', domain: 0, page: 1});
  assert.ok(first.nodes.length < 100 && second.nodes.length < 100);
  assert.ok(first.pageCount > 1);
  assert.equal(physicalDevices(first).some(value => physicalDevices(second).includes(value)), false);
  const last = h.call('buildTopologyModel', config, {view: 'domain', domain: 16});
  assert.deepEqual(physicalDevices(last), [4096]);
});

const stepKeys = plan => Array.from(plan.steps, step => step.key);
const factValue = (plan, label) => plan.facts.find(row => row[0] === label)?.[1];

test('inference plans preserve variable request lengths and omit decode for one-token outputs', () => {
  const config = fixture(), h = harness(config);
  const plan = h.call('buildExecutionPlan', config, 'simulate');
  assert.deepEqual(stepKeys(plan), ['prefill', 'decode', 'output']);
  assert.match(plan.steps[0].detail, /192/);
  assert.match(plan.steps[1].detail, /3 轮/);
  assert.match(factValue(plan, '输入长度'), /64–128/);
  assert.match(factValue(plan, '输出长度'), /2–4/);
  config.workload.output_lengths = [1, 1];
  const single = h.call('buildExecutionPlan', config, 'simulate');
  assert.deepEqual(stepKeys(single), ['prefill', 'output']);
  assert.match(single.steps.at(-1).detail, /TPOT 未定义/);
  config.workload.output_lengths = [1];
  assert.equal(h.call('executionLengthStats', config.workload, 'output_lengths', 'output_length'), null);
});

test('training plans derive accumulation, global batch and enabled memory/optimizer operations', () => {
  const config = fixture();
  Object.assign(config.workload, {scenario: 'training', batch_size: 3, sequence_length: 512, gradient_accumulation: 4, training_steps: 2});
  Object.assign(config.execution, {dp: 4, fsdp: 2, pp: 2, sync_each_microbatch: false});
  Object.assign(config.execution.features, {activation_checkpointing: true, optimizer_offload: true});
  config.execution.optimizer.kind = 'muon';
  const h = harness(config), plan = h.call('buildExecutionPlan', config, 'simulate');
  assert.deepEqual(stepKeys(plan), ['weight-gather', 'forward', 'recompute', 'backward', 'gradient-sync', 'optimizer-offload']);
  assert.equal(factValue(plan, '每步全局样本'), '48');
  assert.match(factValue(plan, '训练循环'), /2 步 × 4 微批/);
  assert.match(plan.steps.at(-1).detail, /MUON/);
  Object.assign(config.execution, {dp: 1, fsdp: 1});
  Object.assign(config.execution.features, {activation_checkpointing: false, optimizer_offload: false});
  assert.deepEqual(stepKeys(h.call('buildExecutionPlan', config)), ['forward', 'backward', 'optimizer']);
});

test('PD, KV offload and speculative configurations produce their own dependency plans', () => {
  const config = fixture(), h = harness(config);
  config.execution.features.prefill_decode_disaggregation = {enabled: true};
  let plan = h.call('buildExecutionPlan', config);
  assert.deepEqual(stepKeys(plan), ['prefill', 'pd-handoff', 'decode', 'output']);
  assert.match(plan.steps[0].title, /P 池/); assert.match(plan.steps[2].title, /D 池/);
  config.execution.features.prefill_decode_disaggregation.enabled = false;
  Object.assign(config.execution.features, {kv_offload: 'host', kv_offload_fraction: .5});
  plan = h.call('buildExecutionPlan', config);
  assert.deepEqual(stepKeys(plan), ['prefill', 'kv-fetch', 'decode', 'output']);
  assert.match(plan.steps[1].detail, /50%/);
  Object.assign(config.execution.features, {kv_offload: 'none', speculative_decode: {enabled: true, method: 'dspark', draft_tokens: 7, accepted_length: 4}});
  plan = h.call('buildExecutionPlan', config);
  assert.deepEqual(stepKeys(plan), ['prefill', 'draft', 'verify', 'commit', 'output']);
  assert.match(plan.steps.find(step => step.key === 'draft').detail, /7 个候选/);
  assert.match(plan.steps.find(step => step.key === 'commit').detail, /4 tokens/);
});

test('execution preview uses pending feature overrides but never mutates the input', () => {
  const config = fixture();
  config.features = {parallelism: {tp: 4, ep: 1, pp: 1, dp: 2}, kv_offload: {location: 'remote', fraction: .25}, sequence_parallel: true};
  const before = structuredClone(config), h = harness(config);
  const plan = h.call('buildExecutionPlan', config, 'estimate');
  assert.equal(factValue(plan, '并行布局'), 'TP 4 / PP 1 / EP 1 / DP 2 / SP 4（随 TP）');
  assert.match(plan.steps.find(step => step.key === 'kv-fetch').detail, /远程内存.*25%/);
  assert.match(plan.notes.join(' '), /未实际执行/);
  assert.deepEqual(config, before);
});

test('observed stages render engine times and remain absent for estimates', () => {
  const h = harness(fixture(), {views: true});
  h.call('renderExecutionSummary', {mode: 'simulate', execution_summary: {basis: 'completed_trace', planned_events: 20, completed_events: 15,
    phases: [{phase: 'prefill', event_count: 10, start_s: 0, end_s: .2}, {phase: 'decode', event_count: 5, start_s: .1, end_s: .5}]}});
  const host = h.node('execution-observed');
  assert.equal(host.querySelectorAll('.execution-phase-row').length, 2);
  assert.ok(host.textContent.includes('15 已完成 / 20 计划事件'));
  const bars = host.querySelectorAll('.execution-phase-bar');
  assert.equal(bars[1].children[0].style.left, '20%');
  assert.equal(bars[1].children[0].style.width, '80%');
  h.call('renderExecutionSummary', {mode: 'estimate', execution_summary: {basis: 'estimate_only', planned_events: 20}});
  assert.equal(host.querySelectorAll('.execution-phase-row').length, 0);
  assert.match(host.textContent, /没有实际执行时间线.*20/);
});

test('topology SVG supports domain selection and device inspection from real configuration', async () => {
  const config = topologyFixture(); config.hardware.npu_count = 10;
  const h = harness(config, {views: true});
  h.call('renderTopology', config);
  const host = h.node('topology-visual');
  assert.ok(host.querySelector('svg'));
  const domain = host.querySelector('input[aria-label="SU 编号，从 0 开始"]');
  domain.value = '2'; await domain.fire('change');
  const suCanvas = host.querySelector('.topology-canvas');
  assert.equal(suCanvas.querySelectorAll('.topology-node').length, 2);
  await suCanvas.querySelectorAll('.topology-node')[1].fire('click');
  assert.match(host.querySelector('.topology-inspector').textContent, /NPU 9.*SU 2.*Host 4/);
  assert.equal(host.querySelectorAll('.topology-canvas').length, 1);
});

test('late simulation snapshots cannot replace newer selections or current parameter edits', async () => {
  const h = harness(fixture(), {views: true}), pending = new Map();
  h.context.fetch = url => new Promise(resolve => pending.set(url, resolve));
  h.evaluate("state.currentJob = {id: 'first', mode: 'simulate'}");
  const first = h.call('loadSimulationSnapshot', {id: 'first', mode: 'simulate'});
  h.evaluate("state.currentJob = {id: 'second', mode: 'simulate'}");
  const second = h.call('loadSimulationSnapshot', {id: 'second', mode: 'simulate'});
  const selected = topologyFixture('mesh'); selected.model.name = 'selected-snapshot';
  pending.get('/api/jobs/second/config')({ok: true, json: async () => ({config: selected})}); await second;
  assert.equal(h.evaluate('simulationView.snapshot.model.name'), 'selected-snapshot');
  assert.match(h.node('topology-type').textContent, /Mesh/);
  h.evaluate("state.config.model.name = 'edited-draft'; state.config.hardware.topology = 'clos'");
  h.call('markModified');
  pending.get('/api/jobs/first/config')({ok: true, json: async () => ({config: fixture()})}); await first;
  assert.equal(h.evaluate('simulationView.source'), 'draft');
  assert.equal(h.evaluate('simulationView.snapshot.model.name'), 'selected-snapshot');
  assert.match(h.node('topology-type').textContent, /Clos/);
  assert.match(h.node('execution-facts').textContent, /edited-draft/);
});

test('snapshot loading failures clear both displays rather than reuse unrelated draft data', async () => {
  const h = harness(fixture(), {views: true});
  h.call('renderSimulationViews');
  assert.ok(h.node('topology-visual').querySelector('svg'));
  h.evaluate("state.currentJob = {id: 'missing', mode: 'simulate'}");
  h.context.fetch = async () => ({ok: false, status: 404, json: async () => ({error: 'snapshot not found'})});
  await h.call('loadSimulationSnapshot', {id: 'missing', mode: 'simulate'});
  assert.equal(h.node('topology-visual').querySelector('svg'), null);
  assert.equal(h.node('execution-flow').querySelectorAll('.execution-step').length, 0);
  assert.equal(h.node('simulation-retry').hidden, false);
  assert.match(h.node('simulation-source-note').textContent, /snapshot not found/);
});

test('explicit star pages include adjacent fabric endpoints so configured links stay inspectable', () => {
  const config = topologyFixture('explicit');
  config.hardware.npu_count = 100;
  config.hardware.explicit_links = Array.from({length: 100}, (_, i) => ({id: 'link-' + i, src: 'npu' + i, dst: 'fabric', domain: 'SU', bandwidth_Bps: 1e11, latency_s: 1e-6}));
  const h = harness(config), first = h.call('buildTopologyModel', config, {page: 0});
  assert.ok(first.nodes.length <= 36);
  assert.ok(first.nodes.some(node => node.id === 'fabric'));
  assert.ok(first.edges.length > 0);
  const seen = new Set();
  for (let page = 0; page < first.pageCount; page++) {
    const model = h.call('buildTopologyModel', config, {page});
    for (const node of model.nodes) seen.add(node.id);
    assert.ok(model.nodes.length <= 36);
  }
  assert.equal(seen.size, 101);
});

test('empty request arrays resolve scalar lengths without expanding batch-sized arrays', () => {
  const h = harness();
  const stats = h.call('executionLengthStats', {batch_size: 100000, input_lengths: [], input_length: 1024}, 'input_lengths', 'input_length');
  assert.equal(stats.count, 100000); assert.equal(stats.sum, 102400000); assert.equal(stats.min, 1024); assert.equal(stats.max, 1024);
  assert.equal(h.call('executionLengthStats', {batch_size: 2, input_lengths: []}, 'input_lengths', 'input_length'), null);
});

test('incomplete engine results retain the interruption context even before the first event completes', () => {
  const h = harness(fixture(), {views: true});
  for (const phases of [[], [{phase: 'prefill', event_count: 2, start_s: 0, end_s: .1}]]) {
    h.call('renderExecutionSummary', {mode: 'simulate', execution_summary: {basis: 'completed_trace', complete: false, planned_events: 30,
      completed_events: phases.length ? 2 : 0, phases}});
    const host = h.node('execution-observed');
    assert.match(host.textContent, /本次执行未完成/);
    assert.ok(host.textContent.includes((phases.length ? '2' : '0') + ' 已完成 / 30 计划事件'));
    assert.equal(host.querySelectorAll('.execution-phase-row').length, phases.length);
  }
});

test('pending parallelism features drive the same rank mapping and execution layout', () => {
  const config = fixture();
  config.features = {parallelism: {tp: 1, ep: 4, pp: 1, dp: 1, rank_order: [3, 2, 1, 0]}};
  const before = structuredClone(config), h = harness(config, {views: true});
  const rank = h.call('topologyRank', config, 0);
  assert.equal(rank.rank, 3); assert.equal(rank.tp, 0); assert.equal(rank.ep, 3); assert.equal(rank.pp, 0); assert.equal(rank.dp, 0);
  const plan = h.call('buildExecutionPlan', config);
  assert.equal(factValue(plan, '并行布局'), 'TP 1 / PP 1 / EP 4 / DP 1 / SP 1（关闭）');
  h.call('renderSimulationViews');
  assert.match(h.node('execution-facts').textContent, /TP 1/);
  assert.deepEqual(config, before);
});

test('invalid JSON topology data clears both previews, preserves the draft and recovers after correction', () => {
  const h = harness(fixture(), {views: true});
  h.call('renderSimulationViews');
  assert.ok(h.node('topology-visual').querySelector('svg'));
  assert.ok(h.node('execution-flow').querySelectorAll('.execution-step').length > 0);
  const draft = topologyFixture('explicit'); draft.hardware.explicit_links = [null];
  const raw = JSON.stringify(draft), original = h.json('state.config');
  h.node('config-json').value = raw; h.evaluate('state.jsonDirty = true');
  assert.doesNotThrow(() => h.call('markModified'));
  assert.equal(h.node('topology-visual').querySelector('svg'), null);
  assert.equal(h.node('execution-flow').querySelectorAll('.execution-step').length, 0);
  assert.equal(h.node('config-json').value, raw);
  assert.equal(h.evaluate('state.jsonDirty'), true);
  assert.deepEqual(h.json('state.config'), original);
  assert.match(h.node('simulation-source-note').textContent, /请完成校验/);
  draft.hardware.explicit_links = [];
  h.node('config-json').value = JSON.stringify(draft);
  h.call('markModified');
  assert.ok(h.node('topology-visual').querySelector('svg'));
  assert.ok(h.node('execution-flow').querySelectorAll('.execution-step').length > 0);
  assert.equal(h.evaluate('state.jsonDirty'), true);
});


test('Scale-out model exposes configured SU gateways and the shared spine independently of the SU view', () => {
  const config = topologyFixture(); config.hardware.npu_count = 10;
  const h = harness(config), model = h.call('buildScaleOutModel', config);
  const ids = new Set(Array.from(model.nodes, node => node.id));
  for (const id of ['su0.devices', 'su1.devices', 'su2.devices', 'su0.gateway', 'su1.gateway', 'su2.gateway', 'so.spine']) assert.ok(ids.has(id), id);
  const attachments = Array.from(model.edges).filter(edge => edge.aggregated);
  assert.equal(attachments.length, 3);
  assert.deepEqual(attachments.map(edge => edge.device_count), [4, 4, 2]);
  for (const edge of attachments) { assert.equal(edge.domain, 'SO'); assert.equal(edge.bandwidth_Bps, 50e9); }
  for (const domain of [0, 1, 2]) {
    const uplink = model.edges.find(edge => edge.src === 'su' + domain + '.gateway' && edge.dst === 'so.spine');
    assert.ok(uplink); assert.equal(uplink.domain, 'SO');
    assert.equal(uplink.bandwidth_Bps, domain === 2 ? 50e9 : 100e9);
  }
  assert.equal(model.nodes.some(node => node.kind === 'host'), false);
});

test('Scale-out uplinks respect partial host reachability and explicit uplink capacity', () => {
  const config = topologyFixture();
  Object.assign(config.hardware, {npu_count: 10, npus_per_host: 3});
  config.hardware.scale_out.host_injection_Bps = 80e9;
  const h = harness(config);
  const edgeFor = model => model.edges.find(edge => edge.src === 'su1.gateway' && edge.dst === 'so.spine');
  assert.equal(edgeFor(h.call('buildScaleOutModel', config)).bandwidth_Bps, 160e9);
  config.hardware.scale_out.uplink_Bps = 120e9;
  assert.equal(edgeFor(h.call('buildScaleOutModel', config)).bandwidth_Bps, 120e9);
});

test('explicit Scale-out view isolates SO links from large amounts of SU connectivity', () => {
  const config = topologyFixture('explicit'); config.hardware.npu_count = 100;
  const suLinks = Array.from({length: 1000}, (_, i) => ({id: 'su-' + i, src: 'npu' + i % 100, dst: 'su-switch-' + Math.floor(i / 100), domain: 'SU', bandwidth_Bps: 1e11, latency_s: 1e-6}));
  const soLinks = [{id: 'so-tx', src: 'npu99', dst: 'so-fabric', domain: 'SO', bandwidth_Bps: 25e9, latency_s: 4e-6},
    {id: 'so-rx', src: 'so-fabric', dst: 'npu99', domain: 'SO', bandwidth_Bps: 30e9, latency_s: 5e-6}];
  config.hardware.explicit_links = [...suLinks, ...soLinks];
  const h = harness(config), model = h.call('buildScaleOutModel', config);
  assert.equal(model.edges.length, 2);
  assert.ok(model.nodes.length <= 36);
  assert.ok(model.nodes.some(node => node.id === 'npu99'));
  assert.ok(model.nodes.some(node => node.id === 'so-fabric'));
  assert.equal(model.nodes.some(node => node.id === 'so.spine'), false);
  assert.deepEqual(Array.from(model.edges, edge => edge.id).sort(), ['so-rx', 'so-tx']);
  for (const expected of soLinks) {
    const actual = model.edges.find(edge => edge.id === expected.id);
    assert.equal(actual.src, expected.src); assert.equal(actual.dst, expected.dst);
    assert.equal(actual.domain, 'SO'); assert.equal(actual.bandwidth_Bps, expected.bandwidth_Bps);
    assert.equal(actual.latency_s, expected.latency_s); assert.equal(Boolean(actual.bidirectional), false);
  }
  config.hardware.explicit_links = suLinks;
  const absent = h.call('buildScaleOutModel', config);
  assert.equal(absent.nodes.length, 0); assert.equal(absent.edges.length, 0);
});

test('Scale-out remains visible beside default SU detail and after domain changes', async () => {
  const config = topologyFixture(); config.hardware.npu_count = 10;
  const h = harness(config, {views: true});
  h.call('renderTopology', config);
  const host = h.node('topology-visual');
  assert.ok(host.querySelector('.so-topology-canvas'));
  assert.ok(host.querySelector('.topology-canvas'));
  const domain = host.querySelector('input[aria-label="SU 编号，从 0 开始"]');
  domain.value = '2'; await domain.fire('change');
  assert.equal(host.querySelectorAll('.so-topology-canvas').length, 1);
  assert.equal(host.querySelectorAll('.topology-canvas').length, 1);
  assert.match(host.querySelector('.so-topology-canvas').textContent, /SO/);
  const suCanvas = host.querySelector('.topology-canvas');
  assert.match(suCanvas.textContent, /NPU 8/); assert.match(suCanvas.textContent, /NPU 9/);
  h.call('renderTopology', config);
  assert.equal(host.querySelectorAll('.so-topology-canvas').length, 1);
});


test('SP degree shares TP ranks, preserves explicit false and follows pending feature selections', () => {
  const config = fixture(), h = harness(config);
  assert.equal(h.call('sequenceParallelInfo', config).label, 'SP 1（关闭）');
  config.execution.sequence_parallel = true;
  assert.equal(h.call('sequenceParallelInfo', config).degree, 2);
  config.execution.tp = 8;
  assert.equal(h.call('sequenceParallelInfo', config).label, 'SP 8（随 TP）');
  config.features = {sequence_parallel: false, parallelism: {tp: 4}};
  const disabled = h.call('sequenceParallelInfo', config);
  assert.equal(disabled.enabled, false); assert.equal(disabled.degree, 1); assert.equal(disabled.label, 'SP 1（关闭）');
  config.features.sequence_parallel = true;
  assert.equal(h.call('sequenceParallelInfo', config).label, 'SP 4（随 TP）');
  config.features.sequence_parallel = 'false';
  assert.equal(h.call('sequenceParallelInfo', config).degree, null);
  assert.match(h.call('sequenceParallelInfo', config).label, /待校验/);
  config.features.sequence_parallel = true; config.features.parallelism.tp = 0;
  assert.equal(h.call('sequenceParallelInfo', config).degree, null);
});

test('SP edits update derived degree, execution summary and topology without adding devices', async () => {
  const config = fixture();
  config.model.architecture = {vocab_size: 256, attention_by_layer: ['GQA', 'GQA', 'GQA', 'GQA'], ffn_by_layer: ['Dense', 'Dense', 'Dense', 'Dense']};
  const h = harness(config, {views: true});
  h.evaluate('renderInputNotes = realRenderInputNotes');
  await h.call('beginInputEditing');
  await h.call('commitInputValue', ['execution', 'sequence_parallel'], true);
  assert.equal(h.node('sp-degree').textContent, '2');
  assert.equal(h.node('sequence-parallel').checked, true);
  await h.call('commitInputValue', ['execution', 'tp'], 4);
  await h.call('commitInputValue', ['execution', 'ep'], 1);
  assert.equal(h.node('sp-degree').textContent, '4');
  const computed = h.node('input-parallel-body').querySelector('[data-input-computed-label="SP 并行度"]');
  assert.equal(computed.querySelector('.input-display-value').textContent, 'SP 4（随 TP）');
  assert.match(h.node('execution-facts').textContent, /SP 4（随 TP）/);
  assert.ok(h.node('parallel-note').textContent.includes('并行乘积 4 / 4 NPU'));
  assert.equal(h.node('parallel-note').classList.contains('error'), false);
  assert.equal(h.evaluate('state.config.hardware.npu_count'), 4);
  const canvas = h.node('topology-visual').querySelector('.topology-canvas');
  await canvas.querySelectorAll('.topology-node')[1].fire('click');
  assert.match(h.node('topology-visual').querySelector('.topology-inspector').textContent, /SP 4（随 TP）.*SP 分片索引 1/);
  await h.call('commitInputValue', ['execution', 'sequence_parallel'], false);
  assert.equal(h.node('sp-degree').textContent, '1');
  assert.equal(computed.querySelector('.input-display-value').textContent, 'SP 1（关闭）');
  assert.match(h.node('execution-facts').textContent, /SP 1（关闭）/);
  assert.ok(h.node('parallel-note').textContent.includes('并行乘积 4 / 4 NPU'));
});
