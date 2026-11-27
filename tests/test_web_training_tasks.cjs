'use strict';
const assert = require('node:assert/strict');
const {test} = require('node:test');
const {fixture, harness} = require('./web_frontend_harness.cjs');

test('pretraining results show censored completion quantiles and clear the comparison for a healthy run', () => {
  const h = harness(fixture());
  const projection = {reliability: {completed: 5, trials: 6, wall_s_p50: 42, wall_s_p95: null,
    net_tokens_s_p50: 500, rollback_tokens_p50: 128, lost_card_hours_p50: 2.5}};
  const result = {metrics: {training_accounting: {status: 'complete'},
    training_task: {kind: 'pretrain', budget_projection: projection}}};
  h.call('renderTrainingAccounting', result);
  const table = h.node('training-accounting-tables');
  assert.match(table.textContent, /5 \/ 6/);
  assert.match(table.textContent, /超出观测窗口/);
  delete projection.reliability;
  h.call('renderTrainingAccounting', result);
  assert.doesNotMatch(table.textContent, /超出观测窗口/);
});

test('training report shows the job chart and exact record and clears both on job change', () => {
  const h = harness(fixture());
  Object.assign(h.context.location, {origin:'http://localhost:8090', href:'http://localhost:8090/'});
  const files = ['training_breakdown.svg', 'training_summary.json'];
  h.call('renderTrainingSummary', {artifacts:files.map(name => ({name, url:'/api/jobs/1/artifacts/'+name}))});
  const panel = h.node('training-summary-panel'), chart = h.node('training-summary-chart');
  assert.equal(panel.hidden, false);
  assert.equal(chart.children[0].src, 'http://localhost:8090/api/jobs/1/artifacts/training_breakdown.svg');
  assert.equal(chart.children[1].download, files[1]);
  h.call('renderTrainingSummary', {artifacts:[]});
  assert.equal(panel.hidden, true);
  assert.equal(chart.children.length, 0);
});

test('RL preview separates generation, scoring and update roles including pending feature input', () => {
  const config = fixture();
  Object.assign(config.workload, {scenario:'training', training_steps:2, gradient_accumulation:2});
  config.features = {training_task:{kind:'rl', rl:{algorithm:'ppo', prompts_per_step:2, samples_per_prompt:4,
    rollout_batch_size:4, update_epochs:3, old_log_probs:'recompute', reference_policy:true, reward:'model'}}};
  const h = harness(config), plan = h.call('buildExecutionPlan', config);
  assert.deepEqual(Array.from(plan.steps, row => row.key), ['rollout','old-log-probs','reference','reward','value','advantage','actor-update','critic-update']);
  assert.equal(plan.facts.find(row => row[0] === '训练循环')[1], '2 次 RL 迭代');
  config.features.training_task.rl.algorithm = 'grpo';
  config.features.training_task.rl.reference_policy = false;
  config.features.training_task.rl.old_log_probs = 'rollout';
  const simplified = h.call('buildExecutionPlan', config);
  assert.deepEqual(Array.from(simplified.steps, row => row.key), ['rollout','reward','advantage','actor-update']);
});

test('DPO online reference disappears when precomputed while paired actor loss stays', () => {
  const config = fixture();
  Object.assign(config.workload, {scenario:'training', training_steps:1, gradient_accumulation:1,
    training:{kind:'dpo', data:{batching:'packed'}, dpo:{reference_log_probs:'online'}}});
  const h = harness(config), first = h.call('buildExecutionPlan', config);
  assert.ok(first.steps.some(row => row.key === 'reference'));
  assert.ok(first.steps.some(row => row.key === 'loss'));
  config.workload.training.dpo.reference_log_probs = 'precomputed';
  const cached = h.call('buildExecutionPlan', config);
  assert.ok(!cached.steps.some(row => row.key === 'reference'));
  assert.match(cached.steps.find(row => row.key === 'forward').detail, /chosen\/rejected/);
});
