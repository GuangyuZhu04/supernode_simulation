'use strict';
const assert = require('node:assert/strict');
const {test} = require('node:test');
const {fixture, harness} = require('./web_frontend_harness.cjs');

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
