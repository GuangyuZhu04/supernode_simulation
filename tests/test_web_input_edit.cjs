// Run with: node --test tests/test_web_input_edit.cjs
// Exercise production browser handlers in isolation; real browser QA covers layout.
'use strict';

const assert = require('node:assert/strict');
const {test} = require('node:test');
const {fixture, harness} = require('./web_frontend_harness.cjs');

test('PD pool counts stay consistent in the form, overview and live edits', async () => {
  const config = fixture();
  config.execution.features.prefill_decode_disaggregation = {enabled: true};
  config.hardware.npu_count = 2 * config.execution.tp * config.execution.ep;
  const h = harness(config);
  await h.call('beginInputEditing');
  h.call('realRenderInputNotes');
  assert.equal(h.node('parallel-note').classList.contains('error'), false);
  assert.match(h.node('parallel-note').textContent, /PD 双池总卡数 8 \/ 8/);
  const note = () => h.node('input-parallel-body').querySelector('[data-input-live-note="parallel"]').textContent;
  assert.match(note(), /PD 双池：2 × TP × EP = 8.*两者匹配/);
  await h.call('commitInputValue', ['execution', 'tp'], 1);
  assert.match(note(), /两者不匹配/);
  await h.call('commitInputValue', ['hardware', 'npu_count'], 4);
  assert.match(note(), /PD 双池：2 × TP × EP = 4.*两者匹配/);
  await h.call('commitInputValue', ['execution', 'pp'], 2);
  assert.match(note(), /PD 双池要求 PP = DP = 1/);
  h.call('realRenderInputNotes');
  assert.equal(h.node('parallel-note').classList.contains('error'), true);
});

test('edits across the five input sections reach validation and the submitted simulation', async () => {
  const h = harness();
  await h.call('beginInputEditing');
  const edits = [
    [['model', 'name'], 'edited-input-model'],
    [['hardware', 'topology'], 'mesh'],
    [['hardware', 'mesh_width'], 2],
    [['hardware', 'npu', 'hbm_bytes'], 96e9],
    [['hardware', 'scale_up', 'bandwidth_Bps'], 200e9],
    [['execution', 'tp'], 1],
    [['execution', 'ep'], 4],
    [['execution', 'precision', 'weight'], 'fp16'],
    [['workload', 'output_lengths'], [5, 2]],
  ];
  for (const [tokens, value] of edits) await h.call('commitInputValue', tokens, value);
  const edited = h.json('state.config');
  assert.equal(edited.model.name, 'edited-input-model');
  assert.equal(edited.hardware.topology, 'mesh');
  assert.equal(edited.hardware.npu.hbm_bytes, 96e9);
  assert.equal(edited.hardware.scale_up.bandwidth_Bps, 200e9);
  assert.equal(edited.execution.tp * edited.execution.ep, 4);
  assert.deepEqual(edited.workload.output_lengths, [5, 2]);
  assert.deepEqual(edited.workload.input_lengths, [128, 64]);
  assert.equal(edited.execution.sequence_parallel, false);
  assert.equal(edited.hardware.npu.runtime_bytes, 0);
  assert.equal(edited.execution.precision.routed_expert_weight, null);
  assert.equal(edited.sources['hardware.calibration'].description, 'Original provenance.');
  assert.deepEqual(edited.assumptions, ['Keep this assumption.']);

  const requests = [];
  h.context.fetch = async (url, options) => {
    const body = JSON.parse(options.body);
    requests.push({url, body});
    if (url === '/api/validate') return {ok: true, status: 200, json: async () => ({valid: true, config: body.config})};
    if (url === '/api/jobs') return {ok: true, status: 202, json: async () => ({id: 'a'.repeat(32), status: 'queued', mode: body.mode})};
    throw new Error('Unexpected request: ' + url);
  };
  await h.call('runSimulation', {preventDefault() {}});
  assert.deepEqual(requests.map(request => request.url), ['/api/validate', '/api/jobs']);
  assert.deepEqual(requests[0].body.config, edited);
  assert.deepEqual(requests[1].body.config, edited);
  assert.equal(requests[1].body.mode, 'simulate');
});

test('copying a historical snapshot creates an independent editable configuration', async () => {
  const h = harness();
  h.context.archived = fixture();
  h.evaluate(`inputView.source = 'job'; inputView.jobId = 'saved-job'; inputView.snapshot = archived;`);
  const before = h.json('inputView.snapshot');
  await h.call('copySnapshotForEditing');
  assert.equal(h.evaluate('inputView.source'), 'draft');
  assert.equal(h.evaluate('state.config === inputView.snapshot'), false);
  await h.call('commitInputValue', ['workload', 'output_lengths'], [7, 3]);
  await h.call('commitInputValue', ['model', 'name'], 'copy-edited');
  assert.deepEqual(h.json('inputView.snapshot'), before);
  assert.deepEqual(h.json('state.config.workload.output_lengths'), [7, 3]);
  assert.equal(h.json('state.config.model.name'), 'copy-edited');
});

test('display units round-trip to engine bytes, seconds, efficiency, and FLOP/s', async () => {
  const h = harness();
  await h.call('beginInputEditing');
  for (const [tokens, entered, expected] of [
    [['hardware', 'npu', 'hbm_bytes'], '96', 96e9],
    [['hardware', 'scale_up', 'bandwidth_Bps'], '200', 200e9],
    [['hardware', 'scale_up', 'latency_s'], '2.5', 2.5e-6],
    [['hardware', 'npu', 'efficiency'], '50', .5],
    [['hardware', 'npu', 'flops', 'fp16'], '800', 800e12],
    [['targets', 'ttft_s'], '750', .75],
  ]) {
    const editor = h.call('inputFieldEditor', tokens, 'test field');
    const control = editor.querySelector('input');
    assert.ok(control, tokens.join('.'));
    control.value = entered;
    await control.fire('input'); // No blur/change event is needed before submitting.
    const actual = h.call('inputOwnValue', h.evaluate('state.config'), tokens);
    assert.ok(Math.abs(actual - expected) <= Math.abs(expected) * 1e-12, `${tokens.join('.')}: ${actual}`);
  }
});

test('false, zero and nullable selection edits preserve their JSON types', async () => {
  const h = harness();
  await h.call('beginInputEditing');
  const falseEditor = h.call('inputFieldEditor', ['execution', 'sequence_parallel'], 'SP');
  const selection = falseEditor.querySelector('select');
  selection.value = 'true'; await selection.fire('change');
  selection.value = 'false'; await selection.fire('change');
  const zeroEditor = h.call('inputFieldEditor', ['hardware', 'npu', 'runtime_bytes'], 'Runtime');
  zeroEditor.querySelector('input').value = '0'; await zeroEditor.querySelector('input').fire('input');
  const nullable = h.call('inputFieldEditor', ['targets', 'ttft_s'], 'TTFT').querySelector('input');
  nullable.value = ''; await nullable.fire('input');
  assert.equal(h.evaluate('state.config.execution.sequence_parallel'), false);
  assert.equal(h.evaluate('state.config.hardware.npu.runtime_bytes'), 0);
  assert.equal(h.evaluate('state.config.targets.ttft_s'), null);
  assert.equal(h.call('hasUnappliedInputEdits'), false);
});

test('incomplete numeric text stays visible, blocks run, and recovers without losing prior edits', async () => {
  const h = harness();
  await h.call('beginInputEditing');
  await h.call('commitInputValue', ['model', 'name'], 'keep-valid-edit');
  const control = h.call('inputFieldEditor', ['hardware', 'npu', 'hbm_bytes'], 'HBM').querySelector('input');
  control.value = '1e'; await control.fire('input');
  assert.equal(control.value, '1e');
  assert.equal(control.getAttribute('aria-invalid'), 'true');
  assert.equal(h.evaluate('state.config.hardware.npu.hbm_bytes'), 80e9);
  assert.equal(h.call('hasUnappliedInputEdits'), true);
  assert.throws(() => h.call('readConfig'), /未应用/);
  let calls = 0;
  h.context.fetch = async () => { calls++; throw new Error('Invalid edit must block all requests'); };
  await h.call('runSimulation', {preventDefault() {}});
  assert.equal(calls, 0);
  assert.equal(h.evaluate('state.config.model.name'), 'keep-valid-edit');
  control.value = '100'; await control.fire('input');
  assert.equal(h.call('hasUnappliedInputEdits'), false);
  assert.equal(h.call('readConfig').hardware.npu.hbm_bytes, 100e9);
  assert.equal(h.evaluate('state.config.model.name'), 'keep-valid-edit');
});

test('field JSON retains an invalid draft and applies ragged arrays only explicitly', async () => {
  const h = harness();
  h.call('attachInputEditors');
  await h.call('beginInputEditing');
  await h.call('openInputJsonEditor', ['workload', 'output_lengths']);
  const editor = h.node('input-json-editor');
  editor.value = '[5,'; await editor.fire('input');
  await h.call('applyInputJsonEdit');
  assert.equal(editor.value, '[5,');
  assert.equal(h.call('hasUnappliedInputEdits'), true);
  assert.deepEqual(h.json('state.config.workload.output_lengths'), [4, 2]);
  assert.throws(() => h.call('readConfig'), /未应用/);
  editor.value = '[5, 2]'; await editor.fire('input');
  assert.deepEqual(h.json('state.config.workload.output_lengths'), [4, 2]);
  await h.call('applyInputJsonEdit');
  assert.equal(h.call('hasUnappliedInputEdits'), false);
  assert.deepEqual(h.json('state.config.workload.output_lengths'), [5, 2]);
  assert.deepEqual(h.json('state.config.workload.input_lengths'), [128, 64]);
});

test('structured shape edits clear only derived counts and prevent direct derived-field edits', async () => {
  const config = fixture();
  config.model.architecture = {vocab_size: 256, attention_by_layer: ['GQA', 'GQA', 'GQA', 'GQA'], ffn_by_layer: ['Dense', 'Dense', 'Dense', 'Dense']};
  config.model.reported_total_params = 2e9;
  config.model.reported_active_params = 8e8;
  const h = harness(config);
  await h.call('beginInputEditing');
  assert.throws(() => h.call('commitInputValue', ['model', 'total_params'], 123), /推导/);
  const derived = h.call('inputFieldEditor', ['model', 'total_params'], 'Total');
  assert.equal(derived.querySelector('input'), null);
  await h.call('commitInputValue', ['model', 'hidden_dim'], 2048);
  assert.equal(h.evaluate('state.config.model.total_params == null'), true);
  assert.equal(h.evaluate('state.config.model.active_params == null'), true);
  assert.equal(h.evaluate('state.config.model.reported_total_params'), 2e9);
  assert.equal(h.evaluate('state.config.model.reported_active_params'), 8e8);
  assert.deepEqual(h.json('state.config.model.architecture.attention_by_layer'), config.model.architecture.attention_by_layer);
});

test('literal dotted keys and prototype-like path segments remain own data properties', async () => {
  const h = harness();
  await h.call('beginInputEditing');
  await h.call('commitInputValue', ['sources', 'hardware.calibration', 'description'], 'Updated source');
  assert.equal(h.evaluate('state.config.sources["hardware.calibration"].description'), 'Updated source');
  assert.equal(h.evaluate('Object.hasOwn(state.config.sources, "hardware")'), false);
  await h.call('commitInputValue', ['__proto__', 'inputEditPolluted'], 'data-only');
  await h.call('commitInputValue', ['constructor', 'prototype', 'inputEditPolluted'], 'also-data-only');
  assert.equal(h.evaluate('({}).inputEditPolluted'), undefined);
  assert.equal({}.inputEditPolluted, undefined);
  assert.equal(h.evaluate('Object.hasOwn(state.config, "__proto__")'), true);
  assert.equal(h.evaluate('state.config["__proto__"].inputEditPolluted'), 'data-only');
  assert.equal(h.evaluate('Object.hasOwn(state.config, "constructor")'), true);
});

test('backend validation failure retains the edited configuration for correction', async () => {
  const h = harness();
  await h.call('beginInputEditing');
  await h.call('commitInputValue', ['execution', 'tp'], 1);
  await h.call('commitInputValue', ['model', 'name'], 'keep-after-validation-error');
  const edited = h.json('state.config');
  h.context.fetch = async () => ({ok: false, status: 400,
    json: async () => ({error: '配置校验失败', details: 'tp * pp * ep * dp must equal hardware.npu_count'})});
  await h.call('handleValidation');
  assert.deepEqual(h.json('state.config'), edited);
  assert.equal(h.evaluate('state.busy'), false);
  assert.match(h.node('input-edit-message').textContent, /配置校验失败/);
  await h.call('commitInputValue', ['execution', 'ep'], 4);
  assert.equal(h.evaluate('state.config.execution.tp * state.config.execution.ep'), 4);
});

test('validation locks active input controls and accepts new edits after the response', async () => {
  const h = harness();
  await h.call('beginInputEditing');
  h.evaluate('updateRunButton = realUpdateRunButton');
  const findName = () => h.document.querySelectorAll('[data-input-edit-path]').find(node => node.dataset.inputEditPath === '["model","name"]');
  let control = findName();
  assert.ok(control);
  control.value = 'before-request'; await control.fire('input');
  let release, submitted;
  h.context.fetch = async (_url, options) => {
    submitted = JSON.parse(options.body).config;
    return new Promise(resolve => { release = resolve; });
  };
  const pending = h.call('handleValidation');
  assert.equal(h.evaluate('state.busy'), true);
  assert.equal(control.disabled, true);
  control.value = 'cannot-commit-while-request-active'; await control.fire('input');
  assert.equal(h.evaluate('state.config.model.name'), 'before-request');
  release({ok: true, status: 200, json: async () => ({valid: true, config: submitted})});
  await pending;
  control = findName();
  assert.equal(control.disabled, false);
  assert.equal(control.value, 'before-request');
  control.value = 'after-response'; await control.fire('input');
  assert.equal(h.evaluate('state.config.model.name'), 'after-response');
});

test('a late history response cannot replace the selected snapshot or its edited copy', async () => {
  const h = harness();
  const pending = new Map();
  h.context.fetch = url => new Promise(resolve => pending.set(url, resolve));
  h.evaluate("inputView.source = 'job'");
  const first = h.call('loadInputSnapshot', 'first');
  const second = h.call('loadInputSnapshot', 'second');
  const selected = fixture(); selected.model.name = 'second-snapshot';
  pending.get('/api/jobs/second/config')({ok: true, status: 200, json: async () => ({config: selected})});
  await second;
  await h.call('copySnapshotForEditing');
  await h.call('commitInputValue', ['model', 'name'], 'copied-and-edited');
  pending.get('/api/jobs/first/config')({ok: true, status: 200, json: async () => ({config: fixture()})});
  await first;
  assert.equal(h.evaluate('inputView.source'), 'draft');
  assert.equal(h.evaluate('state.config.model.name'), 'copied-and-edited');
  assert.equal(h.evaluate('inputView.snapshot.model.name'), 'second-snapshot');
});

test('invalid experiment JSON stays intact when entering the input editor', async () => {
  for (const raw of ['{"model":', '{"model":{"total_params":1e999}}']) {
    const h = harness();
    const before = h.json('state.config');
    h.node('config-json').value = raw;
    h.evaluate('state.jsonDirty = true');
    await h.call('beginInputEditing');
    assert.equal(h.node('config-json').value, raw);
    assert.equal(h.evaluate('state.jsonDirty'), true);
    assert.equal(h.evaluate('inputEdit.editing'), false);
    assert.deepEqual(h.json('state.config'), before);
  }
});

test('invalid sources in an imported draft are retained instead of silently replaced', async () => {
  for (const sources of ['do not drop this invalid source', [], null]) {
    const h = harness();
    const before = h.json('state.config');
    const draft = fixture(); draft.sources = sources;
    const raw = JSON.stringify(draft);
    h.node('config-json').value = raw;
    h.evaluate('state.jsonDirty = true');
    await h.call('beginInputEditing');
    assert.equal(h.node('config-json').value, raw);
    assert.equal(h.evaluate('state.jsonDirty'), true);
    assert.equal(h.evaluate('inputEdit.editing'), false);
    assert.deepEqual(h.json('state.config'), before);
    assert.match(h.node('input-edit-message').textContent, /sources/);

    const fields = harness();
    fields.call('attachInputEditors');
    await fields.call('beginInputEditing');
    await fields.call('openInputJsonEditor', ['sources']);
    const sourceEditor = fields.node('input-json-editor');
    sourceEditor.value = JSON.stringify(sources); await sourceEditor.fire('input');
    await fields.call('applyInputJsonEdit');
    assert.equal(sourceEditor.value, JSON.stringify(sources));
    assert.equal(fields.call('hasUnappliedInputEdits'), true);
    assert.deepEqual(fields.json('state.config.sources'), fixture().sources);
    assert.match(fields.node('input-json-error').textContent, /sources/);
  }
});

test('expanding the full field tree after an edit displays the current configuration', async () => {
  const h = harness();
  await h.call('beginInputEditing');
  await h.call('commitInputValue', ['hardware', 'npu', 'hbm_bytes'], 96e9);
  const allHardware = h.node('input-hardware-body').querySelectorAll('details').find(node => node.children[0].textContent.startsWith('全部硬件字段'));
  assert.ok(allHardware);
  allHardware.open = true; await allHardware.fire('toggle');
  const npu = allHardware.querySelectorAll('details').find(node => node.children[0].textContent.startsWith('hardware.npu ·'));
  assert.ok(npu);
  npu.open = true; await npu.fire('toggle');
  const row = npu.querySelectorAll('tr').find(node => node.children[0]?.textContent === 'hbm_bytes');
  assert.ok(row);
  assert.match(row.children[1].textContent, /96000000000/);
});



