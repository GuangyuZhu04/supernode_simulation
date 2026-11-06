'use strict';

// Plans describe configured dependencies; only execution_summary describes completed work.
function executionLengthStats(workload, key, fallback) {
  const raw = workload[key];
  const values = Array.isArray(raw) && raw.length ? raw : null;
  const batch = workload.batch_size;
  if (!Number.isSafeInteger(batch) || batch < 1) return null;
  if (!values) {
    const value = workload[fallback];
    return Number.isSafeInteger(value) && value > 0 ? {count: batch, min: value, max: value, sum: batch * value} : null;
  }
  if (values.length !== batch) return null;
  let min = Infinity, max = 0, sum = 0;
  for (const value of values) {
    if (!Number.isSafeInteger(value) || value < 1) return null;
    min = Math.min(min, value); max = Math.max(max, value); sum += value;
  }
  return {count: batch, min, max, sum};
}
function buildExecutionPlan(config, mode = 'simulate') {
  const w = config.workload || {}, e = config.execution || {}, m = config.model || {};
  const selected = config.features || {}, f = {...e.features};
  for (const key of ['activation_checkpointing', 'optimizer_offload', 'engram', 'speculative_decode', 'megakernel', 'prefill_decode_disaggregation']) {
    if (selected[key] != null) f[key] = selected[key];
  }
  if (selected.kv_offload) { f.kv_offload = selected.kv_offload.location; f.kv_offload_fraction = selected.kv_offload.fraction; }
  const parallel = {...e, ...selected.parallelism};
  const fsdp = selected.fsdp?.degree ?? e.fsdp ?? 1;
  const sp = selected.sequence_parallel ?? e.sequence_parallel;
  const optimizer = selected.optimizer || e.optimizer || {};
  const pd = f.prefill_decode_disaggregation?.enabled === true;
  const speculative = f.speculative_decode?.enabled === true;
  const plan = {scenario: w.scenario, mode, facts: [], steps: [], notes: []};
  const add = (key, title, detail) => plan.steps.push({key, title, detail});
  const count = value => number(value, 0);
  const range = stat => !stat ? '待校验' : (stat.min === stat.max ? count(stat.min) : count(stat.min) + '–' + count(stat.max)) + ' tokens / 请求';
  plan.facts.push(['场景', w.scenario === 'training' ? '训练' : '推理'], ['模型', (m.name || '自定义模型') + ' · ' + count(m.layers) + ' 层'], ['并行布局', ['tp', 'pp', 'ep', 'dp'].map(key => key.toUpperCase() + ' ' + count(parallel[key] ?? 1)).join(' / ') + ' / ' + sequenceParallelInfo(config).label]);
  if (selected && Object.keys(selected).length) plan.notes.push('含待解析的顶层 features；此处为配置预览，完整校验后以 resolved 配置为准。');
  if (mode === 'estimate') plan.notes.push('解析估算只计算计划事件与必要耗时下界；下列流程未实际执行，不产生完成进度或实际阶段耗时。');
  if (w.scenario === 'training') {
    const task = selected.training_task || w.training;
    if (task) plan.facts.push(['训练任务', {pretrain:'预训练', sft:'SFT', dpo:'DPO', rl:'RL · ' + (task.rl?.algorithm || 'grpo').toUpperCase()}[task.kind] || task.kind]);
    if (task?.kind === 'rl') {
      const rl = task.rl || {}, samples = (rl.prompts_per_step ?? 1) * (rl.samples_per_prompt ?? 2);
      plan.facts.push(['训练循环', count(w.training_steps) + ' 次 RL 迭代'], ['每迭代样本', count(samples) + ' / DP 副本'], ['更新轮数', count(rl.update_epochs ?? 1)], ['更新微批', count(w.batch_size) + ' 序列 × ' + count(w.gradient_accumulation) + ' 累积']);
      add('rollout', 'Rollout · Prefill / Decode / 采样', '每批 ' + count(rl.rollout_batch_size ?? 1) + ' 序列；按实际 prompt/response 长度生成，计 KV 与采样算子。');
      if (rl.old_log_probs === 'recompute') add('old-log-probs', '行为策略 log-prob', '第一次更新前执行额外冻结 actor 前向。');
      if (rl.reference_policy !== false) add('reference', 'Reference · 冻结参考', '完整上下文前向及 log-prob；独立权重，没有梯度或优化器。');
      add('reward', rl.reward === 'model' ? 'Reward · 奖励模型' : 'Reward · 外部规则/环境', rl.reward === 'model' ? '显式标量输出头模型的前向成本。' : '每回复 ' + number(rl.rule_reward_latency_s ?? 0, 6) + ' s；' + count(rl.reward_workers ?? 1) + ' 个外部 worker。');
      if (rl.algorithm === 'ppo') add('value', 'Critic · 价值前向', '收集旧价值，独立 critic 权重与状态驻留。');
      add('advantage', rl.algorithm === 'ppo' ? 'GAE 优势计算' : 'GRPO 组相对优势', '保留经验缓冲至本轮全部更新完成。');
      add('actor-update', 'Actor · 前向 / loss / 反向 / 更新', count(rl.update_epochs ?? 1) + ' 轮复用经验；' + (f.activation_checkpointing ? '含激活重计算；' : '') + (f.optimizer_offload ? '主机 optimizer 与 D2H/H2D。' : 'NPU optimizer。'));
      if (rl.algorithm === 'ppo') add('critic-update', 'Critic · Value loss / 更新', '每个 actor 更新批之后执行 critic 更新，优化步单列。');
      plan.notes.push('当前各角色共用物理池、PP=FSDP=1，阶段串行；此图概括一次迭代，评分按微批交错，actor/critic 按更新批交替执行。');
    } else {
    plan.facts.push(['训练循环', count(w.training_steps) + ' 步 × ' + count(w.gradient_accumulation) + ' 微批'], ['每微批', 'Batch ' + count(w.batch_size) + ' × ' + count(w.sequence_length) + ' tokens'], ['每步全局样本', [w.batch_size,w.gradient_accumulation,parallel.dp ?? 1].every(v => Number.isSafeInteger(v) && v > 0) ? count(w.batch_size * w.gradient_accumulation * (parallel.dp ?? 1)) : '待校验']);
    if (task?.kind === 'dpo' && task.dpo?.reference_log_probs !== 'precomputed') add('reference', '冻结参考 log-prob', 'chosen/rejected 双序列先执行参考前向；batch 计偏好对。');
    if (fsdp > 1) add('weight-gather', 'FSDP 权重聚合', '按 ' + count(fsdp) + ' 个 DP rank 分片；前向后 ' + ((selected.fsdp?.reshard_after_forward ?? e.reshard_after_forward ?? true) ? '重新分片' : '保留完整权重') + '。');
    add('forward', '前向计算', count(m.layers) + ' 层 · 每微批 ' + count(w.batch_size * w.sequence_length) + ' tokens；PP ' + count(parallel.pp ?? 1) + ' 个流水阶段。');
    if (task) {
      plan.steps[plan.steps.length - 1].detail = '按任务内的长度及 ' + (task.data?.batching || 'padded') + ' 方式计算；' + (task.kind === 'dpo' ? '每对包含 chosen/rejected 两条序列。' : 'batch 计序列，监督掩码不裁剪主干计算。');
      add('loss', task.kind === 'dpo' ? '成对偏好 loss' : '交叉熵 / 标签掩码', '全位置 logits 与 log-softmax，监督 token 或偏好对单独计数。');
    }
    if (f.activation_checkpointing) add('recompute', '激活重计算', '反向阶段按配置重新计算激活，减少保存的激活容量。');
    add('backward', '反向与梯度累积', '每步累积 ' + count(w.gradient_accumulation) + ' 个微批；同步策略：' + ((selected.fsdp?.sync_each_microbatch ?? e.sync_each_microbatch) ? '每个微批' : '累积后') + '。');
    if ((parallel.dp ?? 1) > 1 || fsdp > 1) add('gradient-sync', '梯度同步', fsdp > 1 ? 'FSDP reduce-scatter 与剩余 DP 副本同步；具体依赖以事件图为准。' : 'DP ' + count(parallel.dp) + ' 个副本执行梯度 all-reduce。');
    if (f.optimizer_offload) add('optimizer-offload', '梯度卸载 → 主机更新 → 权重回传', 'Host 执行 ' + (optimizer.kind || 'adam').toUpperCase() + '；回传精度 ' + (e.precision?.weight_transfer || '待校验') + '。');
    else add('optimizer', (optimizer.kind || 'adam').toUpperCase() + ' 更新', '在 NPU 更新权重，随后进入下一训练步。');
    plan.notes.push('流程展示每个训练步的依赖骨架；组件更新和通信可能重叠，不能按卡片数量推算进度。');
    }
  } else if (w.scenario === 'inference') {
    const input = executionLengthStats(w, 'input_lengths', 'input_length');
    const output = executionLengthStats(w, 'output_lengths', 'output_length');
    plan.facts.push(['请求规模', 'Batch ' + count(w.batch_size) + ' / DP 副本'], ['输入长度', range(input)], ['输出长度', range(output)]);
    if (!input || !output) plan.notes.push('请求长度必须是正整数，且显式列表长度应等于 Batch；完成校验后才能确认执行规模。');
    add('prefill', pd ? 'P 池 · Prefill' : 'Prefill · 首 token', (input ? count(input.sum) + ' 输入 tokens / 副本' : '输入长度待校验') + '，经过 ' + count(m.layers) + ' 层并建立 KV / recurrent 状态。');
    if (pd) add('pd-handoff', 'P → D 状态交接', '两个独立物理池，每池 TP × EP = ' + count((parallel.tp ?? 1) * (parallel.ep ?? 1)) + ' 卡；传输 KV / recurrent 状态与首 token ID。');
    const rounds = output ? Math.max(0, output.max - 1) : null;
    if (rounds !== 0) {
      if (f.kv_offload && f.kv_offload !== 'none' && (f.kv_offload_fraction ?? 1) > 0) add('kv-fetch', 'Decode · KV 取回', (f.kv_offload === 'host' ? '主机 DRAM' : '远程内存') + ' ↔ NPU，卸载比例 ' + percentage(f.kv_offload_fraction ?? 1) + '；随解码重复。');
      if (speculative) {
        const spec = f.speculative_decode;
        add('draft', 'Draft · 候选生成', (spec.method || 'speculative').toUpperCase() + ' 每轮最多 ' + count(spec.draft_tokens) + ' 个候选；draft 成本与驻留由配置指定。');
        add('verify', 'Verify · 目标模型验证', '按候选块执行目标模型，保留状态快照与回滚成本。');
        add('commit', 'Commit · 提交与回滚', Array.isArray(spec.accepted_lengths) && spec.accepted_lengths.length ? '使用 ' + count(spec.accepted_lengths.length) + ' 项逐轮提交长度，直到请求目标完成。' : '每轮配置提交 ' + count(spec.accepted_length) + ' tokens（含 bonus），末轮按剩余输出截断。');
      } else add('decode', pd ? 'D 池 · 自回归 Decode' : '自回归 Decode', (rounds == null ? '轮数待校验' : '最多 ' + count(rounds) + ' 轮 / 副本') + '；已完成的短请求退出，活动 Batch 随请求长度变化。');
    }
    add('output', '请求完成', output ? '输出合计 ' + count(output.sum * (parallel.dp ?? 1)) + ' tokens（含首 token）；' + (rounds === 0 ? '全部只生成 1 token，无 Decode 循环，TPOT 未定义。' : 'TTFT / TPOT 与吞吐由实际结果计算。') : '输出规模待校验。');
    if (speculative && rounds !== 0) plan.notes.push('投机轮数由逐轮接受长度决定；不将输出 token 数直接当作 draft/verify 轮数。');
  } else plan.notes.push('缺少有效 workload.scenario，请先载入训练或推理配置。');
  if ((parallel.tp ?? 1) > 1) plan.notes.push('层内通信：TP ' + count(parallel.tp) + (sp ? '，SP 复用 TP rank 执行序列分片通信。' : '，按算子依赖执行张量并行 collective。'));
  if ((m.num_experts ?? 0) > 0 && (m.moe_layer_fraction ?? 0) > 0) plan.notes.push('MoE 层：' + count(m.num_experts) + ' 专家 / Top-' + count(m.top_k) + '；EP ' + count(parallel.ep ?? 1) + ((parallel.ep ?? 1) > 1 ? ' 执行 dispatch → 专家计算 → combine。' : '，专家计算在本地完成。'));
  if (f.engram?.enabled) plan.notes.push('Engram：在配置层访问 ' + (f.engram.location || 'host') + ' 表，命中率 ' + percentage(f.engram.hit_rate) + '。');
  if (f.megakernel?.enabled) plan.notes.push('Megakernel：选择 ' + count(f.megakernel.groups?.length || 0) + ' 个融合规则，实际匹配数以引擎报告为准。');
  return plan;
}
function renderExecutionPlan(config, mode, result = null) {
  const facts = $('#execution-facts'), flow = $('#execution-flow'), notes = $('#execution-notes');
  facts.replaceChildren(); flow.replaceChildren(); notes.replaceChildren();
  if (!config) { text('#execution-plan-label', '配置执行计划 · 等待配置'); empty(flow, '载入配置后生成对应的执行流程。'); renderExecutionSummary(null); return; }
  const plan = buildExecutionPlan(config, mode);
  text('#execution-plan-label', (mode === 'estimate' ? '估算对象' : '配置执行计划') + ' · ' + (plan.scenario === 'training' ? '训练' : '推理'));
  plan.facts.forEach(([label, value]) => { const item = el('div', 'execution-fact'); item.append(el('span', '', label), el('strong', '', value)); facts.append(item); });
  plan.steps.forEach((step, index) => {
    const item = el('li', 'execution-step'); item.dataset.step = step.key;
    item.append(el('span', 'execution-step-index', String(index + 1).padStart(2, '0')));
    const body = el('div'); body.append(el('h4', '', step.title), el('p', '', step.detail)); item.append(body); flow.append(item);
  });
  plan.notes.forEach(note => notes.append(el('p', '', note)));
  renderExecutionSummary(result);
}
function renderExecutionSummary(result) {
  const host = $('#execution-observed'); host.replaceChildren();
  if (!result) { empty(host, '运行结束后，按引擎记录展示实际阶段与时间范围。配置流程不代表完成进度。'); return; }
  const summary = result.execution_summary;
  if (result.mode === 'estimate' || summary?.basis === 'estimate_only') { empty(host, '本次为解析估算，没有实际执行时间线。计划事件数：' + number(summary?.planned_events ?? result.estimate?.event_count, 0) + '。'); return; }
  if (!summary || !['completed_trace', 'completed_chunks'].includes(summary.basis)) { empty(host, '该运行未保留阶段汇总；可在下方事件时间线查看已保存记录，完整明细以导出文件为准。'); return; }
  const streaming = summary.basis === 'completed_chunks';
  const phases = summary.phases || [];
  if (summary.complete === false) host.append(el('p', 'execution-partial-note', '本次执行未完成：以下仅包含中断前已完成的事件，不能代表完整任务。'));
  host.append(el('p', 'execution-observed-title', '实际执行记录 · ' + number(summary.completed_events, 0) + (streaming ? ' 已完成事件（含分块边界）' : ' 已完成 / ' + number(summary.planned_events, 0) + ' 计划事件')));
  if (!phases.length) { host.append(el('p', 'muted-empty', '未记录已完成的事件阶段。')); return; }
  const maxEnd = Math.max(...phases.map(p => Number(p.end_s) || 0), 1e-12);
  phases.forEach(phase => {
    const row = el('div', 'execution-phase-row');
    const name = el('strong', '', phase.phase); name.title = phase.phase;
    const track = el('div', 'execution-phase-bar');
    const bar = el('span'); bar.style.left = Math.max(0, Number(phase.start_s) / maxEnd * 100) + '%'; bar.style.width = Math.max(.25, (Number(phase.end_s) - Number(phase.start_s)) / maxEnd * 100) + '%';
    track.append(bar);
    row.append(name, track, el('span', 'mono', duration(phase.start_s) + ' → ' + duration(phase.end_s)), el('span', '', number(phase.event_count, 0) + ' 事件')); host.append(row);
  });
  host.append(el('p', 'chart-caption', streaming ? '时间从本次仿真开始计，按实际执行分块汇总；中断阶段只包含已观测部分。完整分块见 streaming_chunks.csv。' : '时间从本次仿真开始计。条形为该阶段首个开始至最后完成的跨度，可与其他阶段重叠，不是连续占用时长。' + (summary.grouped_phases ? '超出展示上限的阶段合并为 other_phases。' : '')));
}
