'use strict';

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));
const state = {config: null, presets: [], scenario: 'inference', presetId: '', jsonDirty: false, modified: false, busy: false, currentJob: null, result: null, pollTimer: null, pollFailures: 0, pollVersion: 0, commView: 'physical_bytes_by_domain', history: [], loadingPreset: 0};
const terminal = new Set(['completed', 'failed', 'cancelled']);
const statusNames = {queued: '排队中', running: '运行中', completed: '已完成', failed: '运行失败', cancelled: '已取消', feasible: '满足模型约束', infeasible: '资源不可行', slo_failed: '未满足 SLO', error: '引擎执行错误', estimated: '解析估算'};
const phaseNames = {queued: '等待本地引擎', validating: '校验输入配置', simulating: '执行离散事件仿真', estimating: '计算解析必要下界', exporting: '导出报告与原始统计', summarizing: '整理可视化结果', completed: '计算与导出完成', failed: '执行失败', cancelled: '运行已取消'};
const sourceLabel = {assumed: '假设', public: '公开资料', user: '用户输入', measured: '实测'};
const clone = value => JSON.parse(JSON.stringify(value));
const getPath = (obj, path) => path.split('.').reduce((value, key) => value == null ? undefined : value[key], obj);
const number = (value, digits = 3) => value == null || !Number.isFinite(Number(value)) ? '—' : Number(value).toLocaleString('zh-CN', {maximumFractionDigits: digits});
const compact = value => !Number.isFinite(Number(value)) || value == null ? '—' : Intl.NumberFormat('en', {notation: 'compact', maximumFractionDigits: 2}).format(value);
const percentage = value => Number.isFinite(Number(value)) && value != null ? number(Number(value) * 100, 1) + '%' : '—';
const duration = value => value == null || !Number.isFinite(Number(value)) ? '—' : Number(value) < 1 ? number(Number(value) * 1000, 2) + ' ms' : number(value, 2) + ' s';
const errorText = error => error && error.message ? error.message : String(error);

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
function empty(node, message) {
  node.replaceChildren(el('p', 'muted-empty', message));
}
function text(selector, value) { $(selector).textContent = value; }
function sourceFor(path) {
  state.config.sources = state.config.sources || {};
  state.config.sources[path] = {status: 'user', description: '通过本地仿真网页明确修改此字段。'};
}
function setPath(path, value) {
  const parts = path.split('.');
  let current = state.config;
  for (let i = 0; i < parts.length - 1; i++) {
    if (!current[parts[i]] || typeof current[parts[i]] !== 'object') current[parts[i]] = {};
    current = current[parts[i]];
  }
  current[parts[parts.length - 1]] = value;
  sourceFor(path);
}
function markJsonChanges(previous, next, prefix = '') {
  if (!previous || !next || typeof previous !== 'object' || typeof next !== 'object') return;
  for (const key of new Set([...Object.keys(previous), ...Object.keys(next)])) {
    if (!prefix && key === 'sources') continue;
    const path = prefix ? prefix + '.' + key : key;
    const a = previous[key], b = next[key];
    if (JSON.stringify(a) === JSON.stringify(b)) continue;
    if (a && b && typeof a === 'object' && typeof b === 'object' && !Array.isArray(a) && !Array.isArray(b)) markJsonChanges(a, b, path);
    else {
      state.config.sources[path] = {status: 'user', description: '通过网页完整 JSON 配置明确修改此字段。'};
    }
  }
}
function markModified() {
  state.modified = true;
  simulationView.source = 'draft';
  renderSimulationViews();
  text('#config-state', '已修改');
  $('#config-state').className = 'mini-badge warning';
  $('#validation-message').textContent = '';
  scheduleInputOverview();
}
function toast(message) {
  const node = $('#toast'); node.textContent = message; node.hidden = false;
  clearTimeout(toast.timer); toast.timer = setTimeout(() => { node.hidden = true; }, 4500);
}
function feedback(message, isError = false) {
  const node = $('#validation-message'); node.textContent = message;
  node.className = 'validation-message' + (isError ? ' error' : '');
}
async function api(url, options = {}) {
  if (options.body && new Blob([options.body]).size > 2 * 1024 * 1024) throw new Error('提交配置（含字段来源）超过服务的 2 MiB 请求上限，请精简配置。');
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 30000);
  try {
    const response = await fetch(url, {...options, signal: controller.signal, headers: {'Content-Type': 'application/json', ...options.headers}});
    let data;
    try { data = await response.json(); } catch { throw new Error('服务返回了无法解析的响应（HTTP ' + response.status + '）。'); }
    if (!response.ok) {
      const details = data.details ? '\n' + (typeof data.details === 'string' ? data.details : JSON.stringify(data.details, null, 2)) : '';
      throw new Error((typeof data.error === 'string' ? data.error : '请求失败（HTTP ' + response.status + '）') + details);
    }
    setConnected(true);
    return data;
  } catch (error) {
    if (error.name === 'AbortError') throw new Error('本地服务响应超时。请确认服务仍在运行后重试。');
    if (error instanceof TypeError) { setConnected(false); throw new Error('无法连接本地仿真服务。请检查启动服务的终端。'); }
    throw error;
  } finally { clearTimeout(timeout); }
}
function setConnected(online) {
  $('#connection').className = 'connection ' + (online ? 'online' : 'offline');
  $('#connection span:last-child').textContent = online ? '本地引擎已连接' : '本地引擎连接中断';
}
function syncJson() { if (state.config && !state.jsonDirty) $('#config-json').value = JSON.stringify(state.config, null, 2); }
function readConfig() {
  requireAppliedInputEdits();
  if (state.jsonDirty) {
    let parsed;
    try { parsed = JSON.parse($('#config-json').value); }
    catch (error) { throw new Error('完整配置不是有效 JSON：' + error.message); }
    if (!parsed || Array.isArray(parsed) || typeof parsed !== 'object') throw new Error('完整配置必须为 JSON 对象。');
    assertInputFinite(parsed);
    if (Object.hasOwn(parsed, 'sources') && (!parsed.sources || typeof parsed.sources !== 'object' || Array.isArray(parsed.sources))) throw new Error('sources 必须为对象。原始 JSON 已保留，请修正后重试。');
    const old = state.config;
    state.config = parsed;
    if (!state.config.sources || typeof state.config.sources !== 'object' || Array.isArray(state.config.sources)) state.config.sources = {};
    if (old) markJsonChanges(old, state.config);
    else state.config.sources['web.import'] = {status: 'user', description: '通过网页导入的完整配置。'};
    state.jsonDirty = false;
    state.modified = true;
    renderForm();
  }
  if (!state.config) throw new Error('请先选择一个预设，或导入完整 JSON 配置。');
  return clone(state.config);
}
function renderForm() {
  if (!state.config) return;
  const config = state.config;
  state.scenario = config.workload && config.workload.scenario || 'inference';
  $('#config-fields').disabled = false;
  $$('[data-path]').forEach(input => {
    const value = getPath(config, input.dataset.path);
    if (input.type === 'checkbox') input.checked = Boolean(value);
    else input.value = Array.isArray(value) ? value.join(', ') : value == null ? '' : String(input.dataset.scale ? Number(value) / Number(input.dataset.scale) : value);
  });
  $$('[data-scenario]').forEach(button => {
    const selected = button.dataset.scenario === state.scenario;
    button.classList.toggle('selected', selected); button.setAttribute('aria-pressed', selected);
  });
  $$('[data-for-scenario]').forEach(node => {
    node.hidden = node.dataset.forScenario !== state.scenario;
    $$('input, select', node).forEach(input => { input.disabled = node.hidden; });
  });
  $('#fsdp').disabled = state.scenario === 'inference';
  const hasExplicit = config.hardware && (config.hardware.topology === 'explicit' || (config.hardware.explicit_links || []).length > 0);
  $('#topology-select').disabled = Boolean(hasExplicit);
  $('#topology-select option[value="explicit"]').disabled = !hasExplicit;
  $('#explicit-topology-note').hidden = !hasExplicit;
  $('#su-bandwidth').disabled = Boolean(hasExplicit);
  $('#so-bandwidth').disabled = Boolean(hasExplicit);
  text('#model-name', config.model && config.model.name || '自定义模型');
  text('#model-params', compact(config.model && config.model.total_params) + ' 参数');
  text('#model-layers', number(config.model && config.model.layers, 0) + ' 层');
  $('#validate-button').disabled = state.busy;
  updateRunButton();
  renderSequenceParallelControl();
  renderInputNotes();
  renderSimulationViews();
  syncJson();
  scheduleInputOverview();
}
function sequenceParallelInfo(config) {
  const setting = config.features?.sequence_parallel ?? config.execution?.sequence_parallel ?? false;
  const tp = config.features?.parallelism?.tp ?? config.execution?.tp ?? 1;
  const enabled = typeof setting === 'boolean' ? setting : null;
  const degree = enabled === false ? 1 : enabled && Number.isSafeInteger(tp) && tp > 0 ? tp : null;
  return {enabled, degree, label: degree == null ? 'SP 待校验' : 'SP ' + number(degree, 0) + (enabled ? '（随 TP）' : '（关闭）')};
}
function renderSequenceParallelControl() {
  const sp = sequenceParallelInfo(state.config);
  $('#sequence-parallel').checked = sp.enabled === true;
  text('#sp-degree', sp.degree == null ? '—' : number(sp.degree, 0));
  $('#sp-degree').setAttribute('aria-label', sp.label);
}

function renderInputNotes() {
  if (!state.config) return;
  const e = state.config.execution || {}, h = state.config.hardware || {}, w = state.config.workload || {};
  renderSequenceParallelControl();
  const pd = pdEnabled(state.config);
  const product = (pd ? 2 : 1) * ['tp', 'pp', 'ep', 'dp'].reduce((a, key) => a * (e[key] || 1), 1);
  text('#parallel-note', (pd ? 'PD 双池总卡数 ' : '并行乘积 ') + number(product, 0) + ' / ' + number(h.npu_count, 0) + ' NPU · ' + sequenceParallelInfo(state.config).label + ' · SP 不增加卡数，FSDP 为 DP 子组');
  $('#parallel-note').classList.toggle('error', product !== h.npu_count || (e.dp || 1) % (e.fsdp || 1) !== 0 || pd && (e.dp !== 1 || e.pp !== 1));
  const inCount = Array.isArray(w.input_lengths) ? w.input_lengths.length : 0;
  const outCount = Array.isArray(w.output_lengths) ? w.output_lengths.length : 0;
  text('#length-note', '已填 ' + inCount + ' / ' + outCount + ' 条输入 / 输出；需要 ' + (w.batch_size || '—') + ' 条。');
  $('#length-note').classList.toggle('error', inCount !== w.batch_size || outCount !== w.batch_size);
}
function updateRunButton() {
  syncInputEditorControls();
  $('#config-fields').disabled = state.busy;
  $('#run-mode').disabled = state.busy;
  const running = state.currentJob && !terminal.has(state.currentJob.status);
  $('#run-button').disabled = state.busy || running || hasUnappliedInputEdits() || (!state.config && !state.jsonDirty);
  $('#validate-button').disabled = state.busy || (!state.config && !state.jsonDirty);
  $('#run-button').lastChild.textContent = state.busy ? ' 处理中…' : running ? ' 正在运行' : $('#run-mode').value === 'estimate' ? ' 开始估算' : ' 开始模拟';
}
function populatePresets() {
  const select = $('#preset'); select.replaceChildren();
  const matching = state.presets.filter(preset => preset.scenario === state.scenario);
  if (!matching.length) { const option = el('option', '', '暂无预设 · 请导入 JSON'); option.value = ''; select.append(option); }
  matching.forEach(preset => { const option = el('option', '', preset.name); option.value = preset.id; select.append(option); });
  if (matching.some(preset => preset.id === state.presetId)) select.value = state.presetId;
}
async function loadPreset(id) {
  if (!id) return;
  const token = ++state.loadingPreset;
  state.busy = true; updateRunButton();
  text('#config-state', '载入中');
  try {
    const data = await api('/api/presets/' + encodeURIComponent(id));
    if (token !== state.loadingPreset) return;
    simulationView.source = 'draft';
    state.config = data.config; state.presetId = id; state.modified = false; state.jsonDirty = false;
    const preset = state.presets.find(item => item.id === id);
    text('#preset-description', preset && preset.description || '保留预设中的全部模型、硬件、精度与来源信息。');
    text('#config-state', '预设已载入'); $('#config-state').className = 'mini-badge';
    feedback(''); renderForm(); populatePresets();
  } catch (error) {
    if (token !== state.loadingPreset) return;
    feedback(errorText(error), true);
    $('#config-fields').disabled = false;
    $('#advanced').open = true;
    text('#config-state', '载入失败');
  } finally { if (token === state.loadingPreset) { state.busy = false; updateRunButton(); } }
}
async function validateConfig(showSuccess = true) {
  const candidate = readConfig();
  const response = await api('/api/validate', {method: 'POST', body: JSON.stringify({config: candidate})});
  state.config = response.config;
  state.jsonDirty = false;
  renderForm();
  text('#config-state', '校验通过'); $('#config-state').className = 'mini-badge success';
  if (showSuccess) feedback('配置校验通过。可以提交至本地引擎运行。');
  return clone(response.config);
}
async function handleValidation() {
  if (state.busy) return;
  state.busy = true; updateRunButton(); feedback('正在校验完整配置…');
  try { await validateConfig(); inputEditMessage('当前实际 Config 已通过完整校验；可以从实验页开始运行。'); renderInputOverview(); } catch (error) { feedback(errorText(error), true); inputEditMessage(errorText(error), true); text('#config-state', '校验失败'); $('#config-state').className = 'mini-badge error'; renderInputOverview(); }
  finally { state.busy = false; updateRunButton(); }
}
async function runSimulation(event) {
  event.preventDefault();
  if (state.busy || state.currentJob && !terminal.has(state.currentJob.status)) return;
  state.busy = true; updateRunButton(); feedback('正在校验完整配置…');
  try {
    const config = await validateConfig(false);
    const mode = $('#run-mode').value;
    const job = await api('/api/jobs', {method: 'POST', body: JSON.stringify({config, mode})});
    state.result = null;
    $('#process-artifact-links').replaceChildren();
    $('#result-content').hidden = true; $('#empty-results').hidden = false;
    text('#empty-results h3', '引擎正在计算这一次实验');
    text('#empty-results p', '结果生成后自动展示。当前输入修改不会改变已提交的运行。');
    $('#result-status').hidden = true;
    text('#result-mode', '本次运行：' + (mode === 'estimate' ? '解析估算' : '离散事件仿真'));
    feedback('配置已提交。正在执行的任务使用提交时的独立快照。');
    selectJob(job);
    refreshHistory();
  } catch (error) { feedback(errorText(error), true); }
  finally { state.busy = false; updateRunButton(); }
}
function selectJob(job) {
  $('#process-artifact-links').replaceChildren();
  clearTimeout(state.pollTimer);
  state.pollVersion++;
  state.currentJob = job; state.pollFailures = 0; state.result = null;
  loadSimulationSnapshot(job);
  renderJob(job);
  renderHistory();
  pollJob(job.id, state.pollVersion);
}
async function pollJob(id, version) {
  if (version !== state.pollVersion) return;
  try {
    const job = await api('/api/jobs/' + encodeURIComponent(id));
    if (version !== state.pollVersion) return;
    state.currentJob = job; state.pollFailures = 0; renderJob(job);
    if (terminal.has(job.status)) {
      await refreshHistory();
      if (version !== state.pollVersion) return;
      if (job.status === 'completed') {
        try {
          const result = await api('/api/jobs/' + encodeURIComponent(id) + '/result');
          if (version !== state.pollVersion) return;
          state.result = result; renderResults(result, job); renderSimulationViews();
        } catch (error) {
          if (version !== state.pollVersion) return;
          text('#process-message', '运行已完成，但结果读取失败：' + errorText(error) + '。可点击历史记录重试。');
          renderArtifactLinks(job, $('#process-artifact-links'));
        }
      } else {
        renderArtifactLinks(job, $('#process-artifact-links'));
        state.result = null; $('#result-content').hidden = true; $('#empty-results').hidden = false;
        text('#empty-results h3', job.status === 'cancelled' ? '本次运行已取消' : '本次运行未完成');
        text('#empty-results p', job.error || '未生成完整结果。修改输入后可以重新运行。');
        text('#result-mode', '本次任务没有完整仿真结果');
        $('#result-status').hidden = true;
      }
      updateRunButton(); return;
    }
    state.pollTimer = setTimeout(() => pollJob(id, version), 1200);
  } catch (error) {
    if (version !== state.pollVersion) return;
    state.pollFailures++;
    text('#process-message', '暂时无法获取任务状态，正在重试。' + errorText(error));
    state.pollTimer = setTimeout(() => pollJob(id, version), Math.min(10000, 1500 * state.pollFailures));
  }
}
function renderJob(job) {
  const status = $('#job-status');
  status.textContent = statusNames[job.status] || job.status;
  status.className = 'mini-badge ' + (job.status === 'completed' ? 'success' : job.status === 'failed' ? 'error' : job.status === 'cancelled' ? 'warning' : 'running');
  text('#elapsed', '墙钟 ' + duration(job.elapsed_s));
  text('#engine-phase-label', job.mode === 'estimate' ? '解析估算' : '引擎计算');
  text('#runtime-context', '任务 ' + job.id + ' · ' + (job.mode === 'estimate' ? '解析估算' : '事件仿真') + ' · 实际服务状态');
  const active = !terminal.has(job.status);
  $('#cancel-button').hidden = !active;
  $('#cancel-button').disabled = false;
  text('#process-message', job.error || (phaseNames[job.phase] || job.phase || phaseNames[job.status]) + (active ? ' · 仅展示实际阶段，不估算完成百分比。' : ''));
  const mapping = {queued: -1, validating: 0, simulating: 1, estimating: 1, exporting: 2, summarizing: 2, completed: 4};
  const index = mapping[job.phase] == null ? -1 : mapping[job.phase];
  $$('#phase-track > div').forEach((node, i) => {
    node.className = i < index ? 'complete' : active && i === index ? 'current' : job.status === 'failed' && i === index ? 'failed' : '';
    node.firstElementChild.textContent = i < index ? '✓' : String(i + 1);
  });
  const logs = Array.isArray(job.logs) ? job.logs : [];
  text('#log-count', logs.length + ' 条');
  const logText = logs.map(item => (item.time ? '[' + formatTime(item.time, true) + '] ' : '') + (item.message || '')).join('\n') || '等待运行日志…';
  if ($('#job-logs').textContent !== logText) { $('#job-logs').textContent = logText; $('#job-logs').scrollTop = $('#job-logs').scrollHeight; }
  if (job.status === 'failed') $('#logs-details').open = true;
  updateRunButton();
}
function formatTime(value, timeOnly = false) {
  if (!value) return '—';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return new Intl.DateTimeFormat('zh-CN', {timeZone: 'Asia/Shanghai', ...(timeOnly ? {} : {month: '2-digit', day: '2-digit'}), hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false}).format(date);
}
async function refreshHistory() {
  try {
    const data = await api('/api/jobs'); state.history = data.jobs || []; renderHistory();
  } catch (error) { empty($('#history-list'), errorText(error)); }
}
function renderHistory() {
  inputJobOptions();
  const host = $('#history-list'); host.replaceChildren();
  if (!state.history.length) { empty(host, '还没有运行记录。第一次实验将从这里开始。'); return; }
  const active = state.currentJob && !terminal.has(state.currentJob.status);
  state.history.slice(0, 12).forEach(job => {
    const row = el('button', 'history-row' + (state.currentJob && job.id === state.currentJob.id ? ' selected' : ''));
    row.type = 'button'; row.disabled = Boolean(active && job.id !== state.currentJob.id);
    row.title = '查看运行 ' + job.id + '；输入面板保留当前编辑内容';
    row.append(el('span', 'history-id', job.id), el('span', 'history-mode', job.mode === 'estimate' ? '解析估算' : '仿真'));
    const badge = el('span', 'mini-badge ' + (job.status === 'completed' ? 'success' : job.status === 'failed' ? 'error' : ''), statusNames[job.result_status] || statusNames[job.status] || job.status);
    row.append(badge, el('span', 'history-time', formatTime(job.created_at)));
    row.addEventListener('click', () => {
      state.result = null; $('#result-content').hidden = true; $('#empty-results').hidden = false;
      text('#empty-results h3', '正在读取运行结果'); text('#empty-results p', '历史结果绑定各自的配置快照。');
      selectJob(job);
    });
    host.append(row);
  });
}
function svgEl(name, attrs = {}, label) {
  const node = document.createElementNS('http://www.w3.org/2000/svg', name);
  Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, String(value)));
  if (label !== undefined) node.textContent = label;
  return node;
}
function metricCard(label, value, unit, note, stateClass = '') {
  const card = el('div', 'kpi-card');
  const val = el('div', 'kpi-value', value); val.append(el('span', '', unit));
  card.append(el('div', 'kpi-label', label), val, el('div', 'kpi-note ' + stateClass, note));
  return card;
}
function renderResults(result, job) {
  $('#empty-results').hidden = true; $('#result-content').hidden = false;
  const isEstimate = result.mode === 'estimate', config = result.config || {}, metrics = result.metrics || {};
  const status = $('#result-status'); status.hidden = false;
  status.textContent = statusNames[result.status] || result.status || '结果已生成';
  status.className = 'mini-badge ' + (['infeasible', 'error'].includes(result.status) ? 'error' : result.status === 'slo_failed' ? 'warning' : isEstimate ? '' : 'success');
  text('#result-mode', isEstimate ? '解析估算 · 必要下界' : '离散事件仿真');
  text('#result-context', '运行 ' + job.id + ' · ' + (config.model && config.model.name || '模型') + ' · ' + (config.workload && config.workload.scenario === 'training' ? '训练' : '推理') + ' · ' + number(config.hardware && config.hardware.npu_count, 0) + ' NPU · 提交时配置快照');
  const notices = [];
  if (isEstimate) notices.push('解析估算提供必要下界，不能证明资源争用下的 SLO 可行性。此模式不产生仿真时间线或资源利用率。');
  if (result.status === 'infeasible') notices.push('资源约束不可行。已观测内存峰值可能仅覆盖中断前执行，不能代表完整任务可运行。请查看下方诊断。');
  if (result.status === 'error') notices.push('引擎未能完成本次执行。以下统计可能不完整，请查看执行诊断与原始报告。');
  if (result.status === 'slo_failed') notices.push('执行完成，但至少一项 SLO 目标未满足。请检查性能指标与约束诊断。');
  if (notices.length) { $('#result-alert').hidden = false; text('#result-alert', notices.join('\n')); $('#result-alert').className = 'notice' + (['infeasible', 'error'].includes(result.status) ? ' error' : ''); }
  else $('#result-alert').hidden = true;
  const kpis = $('#kpi-grid'); kpis.replaceChildren();
  const checks = result.slo && result.slo.checks || [];
  const sloNote = key => {
    const check = checks.find(check => check.metric === key);
    const target = key === 'ttft_s' || key === 'tpot_s' ? number(check && check.target * 1000, 3) + ' ms' : number(check && check.target, 3) + (key === 'throughput_tokens_s' ? ' tok/s' : ' s');
    return check ? [(check.passed ? '✓ 达标' : '未达标') + ' · 目标 ' + target, check.passed ? 'pass' : 'fail'] : ['未设置此项 SLO', ''];
  };
  if (isEstimate) {
    kpis.append(metricCard('执行时间必要下界', number(result.estimate && result.estimate.duration_lower_bound_s, 5), 's', '理想资源下界 · 非仿真耗时'));
    kpis.append(metricCard('计划事件数', compact(result.estimate && result.estimate.event_count), '', '完整事件图'));
    const traffic = Object.values((result.communication || {}).physical_bytes_by_domain || {}).reduce((sum, v) => sum + (Number(v) || 0), 0);
    kpis.append(metricCard('逐跳物理传输总量', number(traffic / 1e9, 2), 'GB', '完整计划 · 不含重复语义视图'));
    kpis.append(metricCard('内存池数量', number(result.memory && result.memory.pool_count, 0), '', '固定驻留 / 解析峰值'));
  } else {
    kpis.append(metricCard('总执行时间', number(metrics.duration_s, 5), 's', '仿真时间 · 非墙钟耗时'));
    if (config.workload && config.workload.scenario === 'training') {
      kpis.append(metricCard('平均训练步时间', number(metrics.step_time_s, 5), 's', ...sloNote('step_time_s')));
      kpis.append(metricCard('训练步数', number(config.workload.training_steps, 0), '', '提交时训练负载'));
      kpis.append(metricCard('已完成事件', compact(result.simulation && result.simulation.completed_events), '', '引擎执行计数'));
    } else {
      kpis.append(metricCard('执行 TTFT', number(metrics.ttft_s == null ? null : metrics.ttft_s * 1000, 3), 'ms', ...sloNote('ttft_s')));
      kpis.append(metricCard('批内最大平均 TPOT（TPOP）', number(metrics.tpot_s == null ? null : metrics.tpot_s * 1000, 3), 'ms', ...sloNote('tpot_s')));
      kpis.append(metricCard('聚合输出吞吐', number(metrics.throughput_tokens_s, 2), 'tok/s', ...sloNote('throughput_tokens_s')));
    }
  }
  renderLatencyBreakdown(); renderCommunication(); renderMemoryFilter(); renderMemory(); renderResources(); renderTrace(); renderDiagnostics(); renderArtifactLinks(job);
}
function renderLatencyBreakdown() {
  const breakdown = (state.result.metrics || {}).latency_breakdown || {};
  const host = $('#latency-breakdown'); host.replaceChildren();
  const labels = [
    ['total_s', '总时延'], ['compute_s', '计算 / kernel 访存'],
    ['communication_node_s', '通信 · node 内'], ['communication_scaleup_s', '通信 · scale-up'],
    ['communication_scaleout_s', '通信 · scale-out'], ['communication_other_s', '通信 · 其他路径 / 内存限制'],
    ['synchronization_wait_s', '显式同步等待'], ['scheduling_wait_s', '调度槽位等待'],
    ['memory_wait_s', '内存准入等待'], ['resource_contention_s', '资源争用'], ['other_s', '其他'],
  ];
  const table = el('table', 'memory-matrix'), head = el('thead'), header = el('tr');
  ['分项', 'TTFT / ms', 'TPOT（TPOP）/ ms'].forEach(label => header.append(el('th', '', label)));
  head.append(header); table.append(head);
  const body = el('tbody');
  labels.forEach(([key, label]) => {
    const row = el('tr'); row.append(el('td', '', label));
    ['ttft', 'tpot'].forEach(metric => {
      const record = breakdown[metric];
      const value = record && (key === 'total_s' ? record.total_s : (record.components_s || {})[key]);
      const cell = el('td', '', number(value == null ? null : value * 1000, 5));
      if (record) cell.title = 'DP 副本 ' + (record.replica ?? '—') + ' · 序列 ' + (record.sequence ?? '—');
      row.append(cell);
    });
    body.append(row);
  });
  table.append(body); host.append(table);
  const note = '各列分项相加得到该请求的总时延，不累加并发 rank。TTFT 与最大平均 TPOT 可能来自不同请求。依赖同步的上游时间已归因，不重复算等待；通信服务按路径限制资源分类（同主机 / 跨主机 SU / SO），启动延迟按各跳域分类。跨 token 边界事件按窗口覆盖比例归因。';
  const scenario = (state.result.config && state.result.config.workload || {}).scenario;
  const explanation = scenario === 'training' ? '训练不定义 TTFT / TPOT。' : state.result.mode === 'estimate' ? '解析估算没有实际调度轨迹，不提供执行时延拆解。' : (breakdown.unavailable_reason || '');
  const unavailable = (!breakdown.ttft || !breakdown.tpot) ? ' “—”为未定义或不可用，不能视为零。' + explanation : '';
  text('#latency-note', note + unavailable + ' 完整秒值与候选拆解见 latency_breakdown.csv。');
}
function renderCommunication() {
  const result = state.result;
  if (!result) return;
  const comm = result.communication || result.estimate && result.estimate.communication || {};
  let values = comm[state.commView] || {};
  // Older report schemas use this spelling for the collective view.
  if (state.commView === 'payload_by_collective_bytes' && !Object.keys(values).length) values = comm.payload_by_collective || {};
  const entries = Object.entries(values).filter(([, value]) => typeof value === 'number' && Number.isFinite(value)).sort((a, b) => b[1] - a[1]);
  const host = $('#communication-chart'); host.replaceChildren();
  if (!entries.length) empty(host, '此视图没有通信统计数据。');
  const max = Math.max(...entries.map(([, v]) => v), 1);
  entries.forEach(([key, value]) => {
    const row = el('div', 'bar-row'), label = el('div', 'bar-label', key), track = el('div', 'bar-track'), fill = el('div', 'bar-fill');
    label.title = key; fill.style.width = (value / max * 100) + '%'; track.append(fill);
    row.append(label, track, el('div', 'bar-value', number(value / 1e9, 3))); row.title = key + ': ' + number(value, 0) + ' bytes'; host.append(row);
  });
  text('#communication-note', state.commView === 'physical_bytes_by_domain' ? '计划通信流量 / 完整事件图。物理字节按路径逐跳累计，SU / SO 为链路类别，不等同于具体域 ID。' : '计划 payload / 完整事件图。应用语义与 Collective 是同一批 payload 的不同视图，不可相加；与逐跳物理字节分开解读。');
}
function renderMemoryFilter() {
  const filter = $('#memory-filter'); filter.replaceChildren();
  const all = el('option', '', '所有内存池'); all.value = 'all'; filter.append(all);
  const kinds = [...new Set((state.result.memory && state.result.memory.pools || []).map(pool => pool.pool_kind || '其他'))];
  kinds.forEach(kind => { const option = el('option', '', kind); option.value = kind; filter.append(option); });
  filter.value = 'all';
}
function renderMemory() {
  const memory = state.result.memory || {}, host = $('#memory-chart'); host.replaceChildren();
  const filter = $('#memory-filter').value;
  const pools = (memory.pools || []).filter(pool => filter === 'all' || (pool.pool_kind || '其他') === filter);
  pools.sort((a, b) => (b.peak_bytes || b.resident_bytes || 0) / (b.capacity_bytes || 1) - (a.peak_bytes || a.resident_bytes || 0) / (a.capacity_bytes || 1));
  if (!pools.length) empty(host, '没有可展示的内存池。');
  pools.forEach(pool => {
    const peak = pool.peak_bytes == null ? pool.resident_bytes : pool.peak_bytes;
    const capacity = pool.capacity_bytes, ratio = capacity > 0 && peak != null ? peak / capacity : null;
    const row = el('div', 'memory-row');
    const top = el('div', 'memory-row-top'); top.append(el('span', '', pool.pool), el('span', '', number(peak == null ? null : peak / 1e9, 2) + ' / ' + number(capacity == null ? null : capacity / 1e9, 1) + ' GB'));
    const track = el('div', 'memory-track'), fill = el('div', 'memory-fill' + (ratio > 1 ? ' over' : ''));
    fill.style.width = (ratio == null ? 0 : Math.min(100, Math.max(0, ratio * 100))) + '%'; track.append(fill);
    const basis = pool.peak_basis || memory.mode;
    const note = {simulated: '已模拟并发峰值', observed_partial: '中断前已观测峰值', resident_only: '固定驻留 · 非执行峰值'}[basis] || ({simulated: '已模拟并发峰值', observed_partial: '中断前已观测峰值', resident_only: '固定驻留 · 非执行峰值'}[memory.mode]) || basis || '引擎内存统计';
    row.append(top, track, el('div', 'memory-caption', note + ' · 容量占用 ' + percentage(ratio) + (pool.optimistic_peak_bytes == null ? '' : ' · 乐观峰值 ' + number(pool.optimistic_peak_bytes / 1e9, 2) + ' GB')));
    if (pool.composition_at_peak && pool.composition_at_peak.length) row.title = pool.composition_at_peak.map(item => [item.component, item.state, number(item.bytes / 1e9, 3) + ' GB'].join(' · ')).join('\n');
    host.append(row);
  });
  text('#memory-note', '按容量压力排序，展示 ' + pools.length + ' 个池' + (memory.sampled ? '（总计 ' + memory.pool_count + '，网页仅展示部分）' : '') + '。不同内存池峰值时刻可能不同，不能直接相加。全部逐池结果见 CSV。');
}
function renderResources() {
  const result = state.result, host = $('#resource-chart'); host.replaceChildren();
  const all = result.resources || [], resources = all.filter(resource => resource.utilization != null && Number.isFinite(Number(resource.utilization))).sort((a, b) => b.utilization - a.utilization);
  text('#resource-meta', '展示 ' + resources.length + ' / ' + (result.resource_count || all.length) + ' 个资源');
  if (!resources.length) { empty(host, result.mode === 'estimate' ? '解析估算无实际执行利用率，请查看 resources.csv 中的必要下界。' : '没有可用的资源利用率统计。'); return; }
  resources.forEach(resource => {
    const row = el('div', 'bar-row'), label = el('span', 'bar-label', resource.resource), track = el('div', 'bar-track'), fill = el('div', 'bar-fill');
    label.title = resource.resource; fill.style.width = Math.min(100, Math.max(0, resource.utilization * 100)) + '%';
    track.append(fill); row.append(label, track, el('span', 'bar-value', percentage(resource.utilization)));
    row.title = resource.resource + ' · busy ' + duration(resource.busy_s) + ' · 容量 ' + number(resource.capacity) + ' ' + (resource.unit || '');
    host.append(row);
  });
}
function renderTrace() {
  const trace = state.result.trace || {}, events = trace.events || [], host = $('#trace-chart'); host.replaceChildren();
  text('#trace-note', '展示 ' + events.length + ' / ' + (trace.total_events || 0) + ' 个事件' + (trace.sampled ? ' · 均匀抽样，仅用于浏览；完整导出以下载文件清单为准。' : ' · 悬停查看事件细节。') + ' 横轴为仿真时间。');
  if (!events.length) { empty(host, state.result.mode === 'estimate' ? '解析估算不执行事件调度，无事件时间线。' : '没有可展示的已执行事件。'); return; }
  const max = Math.max(...events.map(event => Number(event.end_s) || 0), 1e-9);
  const axis = el('div', 'trace-axis'); [0, .25, .5, .75, 1].forEach(fraction => axis.append(el('span', '', duration(max * fraction)))); host.append(axis);
  events.forEach(event => {
    const row = el('div', 'trace-row'), label = el('span', 'trace-name', event.id), lane = el('div', 'trace-lane');
    const meta = event.metadata || {}, name = (String(event.id) + ' ' + JSON.stringify(meta)).toLowerCase();
    const kind = /collective|all_to|allreduce|all_reduce|broadcast|transfer|dispatch|combine|send|recv/.test(name) ? 'comm-color' : /compute|kernel|attention|matmul|forward|backward|ffn/.test(name) ? 'compute-color' : 'other-color';
    const bar = el('div', 'trace-event ' + kind);
    const start = Math.max(0, Number(event.start_s) || 0), end = Math.max(start, Number(event.end_s) || 0);
    bar.style.left = Math.min(100, start / max * 100) + '%'; bar.style.width = Math.max(0, Math.min(100, (end - start) / max * 100)) + '%';
    const detail = event.id + '\n开始 ' + number(start, 8) + ' s · 结束 ' + number(end, 8) + ' s\n持续 ' + duration(event.duration_s == null ? end - start : event.duration_s);
    row.title = detail; label.title = detail; bar.title = detail; lane.append(bar); row.append(label, lane); host.append(row);
  });
}
function addList(parent, title, entries) {
  if (!entries.length) return;
  parent.append(el('h4', '', title));
  const list = el('ul'); entries.forEach(entry => list.append(el('li', '', entry))); parent.append(list);
}
function renderDiagnostics() {
  const result = state.result, host = $('#diagnostic-content'); host.replaceChildren();
  const checks = result.slo && result.slo.checks || [], diagnostics = result.diagnostics || [], sources = result.sources || {}, assumptions = result.assumptions || [];
  text('#diagnostic-count', diagnostics.length + ' 项诊断 · ' + Object.keys(sources).length + ' 项来源');
  addList(host, 'SLO 检查', checks.map(check => (check.passed ? '满足' : '不满足') + ' · ' + check.metric + ' · 目标 ' + (check.metric.endsWith('_s') && check.metric !== 'throughput_tokens_s' ? duration(check.target) : number(check.target, 6) + ' tok/s') + (check.actual == null ? '' : ' · 实际 ' + (check.metric.endsWith('_s') && check.metric !== 'throughput_tokens_s' ? duration(check.actual) : number(check.actual, 6) + ' tok/s'))));
  if (diagnostics.length) { host.append(el('h4', '', '执行诊断')); host.append(el('pre', '', JSON.stringify(diagnostics, null, 2))); $('#diagnostic-details').open = true; }
  const logical = result.memory && result.memory.logical_component_state_bytes || {};
  if (Object.keys(logical).length) {
    host.append(el('h4', '', '逻辑组件 × 状态容量（GB，分片与复制之前）'));
    const states = [...new Set(Object.values(logical).flatMap(row => Object.keys(row)))];
    const wrap = el('div', 'matrix-wrap'), table = el('table', 'memory-matrix'), head = el('thead'), header = el('tr');
    ['组件', ...states].forEach(value => header.append(el('th', '', value))); head.append(header); table.append(head);
    const body = el('tbody');
    Object.entries(logical).forEach(([component, values]) => {
      const row = el('tr'); row.append(el('td', '', component));
      states.forEach(stateName => row.append(el('td', '', number((values[stateName] || 0) / 1e9, 3)))); body.append(row);
    });
    table.append(body); wrap.append(table); host.append(wrap);
  }
  addList(host, '建模假设', [...assumptions, '固定负载模型，不含请求到达排队或生产 P99 尾延迟。', '可行状态只代表当前模型假设下的结果；未校准配置用于架构比较。']);
  addList(host, '参数来源', Object.entries(sources).map(([key, source]) => key + ' · ' + (sourceLabel[source.status] || source.status || '未标注') + ' · ' + (source.description || '') + (source.uri ? ' · ' + source.uri : '')));
  if (result.config_sha256) addList(host, '复现标识', ['配置 SHA256: ' + result.config_sha256]);
}
function renderArtifactLinks(job, host = $('#artifact-links')) {
  host.replaceChildren();
  (job.artifacts || []).forEach(artifact => {
    if (typeof artifact.url !== 'string') return;
    const url = new URL(artifact.url, location.href);
    if (url.origin !== location.origin) return;
    const link = el('a', '', '↓ ' + artifact.name); link.href = url.href; link.download = artifact.name; host.append(link);
  });
  if (!host.childNodes.length) host.append(el('span', 'subtle', '暂无导出文件'));
}
function downloadJson(config, name) {
  const blob = new Blob([JSON.stringify(config, null, 2) + '\n'], {type: 'application/json;charset=utf-8'});
  const url = URL.createObjectURL(blob), link = el('a');
  link.href = url; link.download = name; document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function attachEvents() {
  $('#config-form').addEventListener('submit', runSimulation);
  $('#validate-button').addEventListener('click', handleValidation);
  $('#apply-json').addEventListener('click', handleValidation);
  $('#preset').addEventListener('change', event => loadPreset(event.target.value));
  $$('[data-scenario]').forEach(button => button.addEventListener('click', () => {
    if (button.dataset.scenario === state.scenario) return;
    state.scenario = button.dataset.scenario; populatePresets();
    const next = state.presets.find(preset => preset.scenario === state.scenario);
    if (next) loadPreset(next.id);
    else { toast('没有此场景的预设。请导入对应的完整配置。'); renderForm(); }
  }));
  $$('[data-path]').forEach(input => input.addEventListener('change', () => {
    if (!state.config) return;
    const editedValue = input.value, editedChecked = input.checked;
    if (state.jsonDirty) {
      try { readConfig(); } catch (error) { feedback(errorText(error), true); return; }
    }
    let value;
    if (input.type === 'checkbox') value = editedChecked;
    else if (input.dataset.type === 'array') value = editedValue.trim() ? editedValue.trim().split(/[\s,，;；]+/).map(Number) : [];
    else if (input.type === 'number') value = editedValue === '' ? null : Number(editedValue) * (Number(input.dataset.scale) || 1);
    else value = editedValue;
    setPath(input.dataset.path, value);
    if (input.dataset.path === 'execution.sequence_parallel' && state.config.features?.sequence_parallel != null) setPath('features.sequence_parallel', value);
    if (input.dataset.path === 'workload.batch_size' && state.scenario === 'training') {
      setPath('workload.input_lengths', []); setPath('workload.output_lengths', []);
    }
    markModified(); renderForm();
  }));
  $('#config-json').addEventListener('input', () => { state.jsonDirty = true; markModified(); updateRunButton(); });
  $('#uniform-lengths').addEventListener('click', () => {
    try {
      readConfig();
      const workload = state.config.workload || {}, batch = Number(workload.batch_size);
      if (!Number.isInteger(batch) || batch < 1 || batch > 100000) throw new Error('请先输入 1 至 100000 的整数 Batch size。');
      const input = (workload.input_lengths || [])[0] || workload.input_length, output = (workload.output_lengths || [])[0] || workload.output_length;
      if (!Number.isInteger(input) || input < 1 || !Number.isInteger(output) || output < 1) throw new Error('请先提供有效的输入、输出首项长度。');
      setPath('workload.input_lengths', Array(batch).fill(input)); setPath('workload.output_lengths', Array(batch).fill(output));
      markModified(); renderForm(); toast('已按首项长度显式生成 ' + batch + ' 条等长请求。');
    } catch (error) { feedback(errorText(error), true); }
  });
  $('#run-mode').addEventListener('change', () => {
    text('#mode-note', $('#run-mode').value === 'estimate' ? '仅计算资源与时间必要下界，不能证明 SLO 可行。' : '模拟依赖、资源争用与内存生命周期。');
    updateRunButton(); renderSimulationViews();
  });
  $('#import-json').addEventListener('click', () => $('#config-file').click());
  $('#config-file').addEventListener('change', async event => {
    const file = event.target.files[0]; if (!file) return;
    try {
      if (file.size > 2 * 1024 * 1024) throw new Error('配置文件超过 2 MiB。服务的请求上限为 2 MiB（含字段来源），请使用更小的 JSON 配置。');
      const contents = (await file.text()).replace(/^\uFEFF/, ''); JSON.parse(contents);
      $('#config-json').value = contents; state.jsonDirty = true; markModified(); readConfig();
      text('#preset-description', '已导入 ' + file.name + '。完整配置以导入内容为准。'); $('#preset').value = '';
      await handleValidation();
    } catch (error) { feedback('导入失败：' + errorText(error), true); }
    finally { event.target.value = ''; }
  });
  $('#export-config').addEventListener('click', () => {
    try { downloadJson(readConfig(), 'supernode-config.json'); } catch (error) { feedback(errorText(error), true); }
  });
  $('#cancel-button').addEventListener('click', async () => {
    if (!state.currentJob || terminal.has(state.currentJob.status)) return;
    $('#cancel-button').disabled = true;
    const cancellingId = state.currentJob.id, cancellingVersion = state.pollVersion;
    try {
      const job = await api('/api/jobs/' + encodeURIComponent(cancellingId) + '/cancel', {method: 'POST', body: '{}'});
      if (state.pollVersion !== cancellingVersion || !state.currentJob || state.currentJob.id !== cancellingId) return;
      state.currentJob = job; renderJob(job);
      text('#process-message', terminal.has(job.status) ? (phaseNames[job.status] || job.status) : '已发送取消请求，等待引擎停止。');
    } catch (error) { if (state.pollVersion === cancellingVersion) { text('#process-message', errorText(error)); $('#cancel-button').disabled = false; } }
  });
  $('#refresh-history').addEventListener('click', refreshHistory);
  $$('[data-comm-view]').forEach(button => button.addEventListener('click', () => {
    state.commView = button.dataset.commView;
    $$('[data-comm-view]').forEach(item => { item.classList.toggle('selected', item === button); item.setAttribute('aria-pressed', item === button); });
    renderCommunication();
  }));
  $('#memory-filter').addEventListener('change', renderMemory);
}
async function initialize() {
  attachEvents();
  attachSimulationViews();
  attachInputOverview();
  attachInputEditors();
  try {
    const data = await api('/api/presets'); state.presets = data.presets || [];
    const preferred = state.presets.find(preset => /demo.*inference|inference.*demo/.test(preset.id)) || state.presets.find(preset => preset.scenario === 'inference') || state.presets[0];
    if (preferred) { state.scenario = preferred.scenario; populatePresets(); await loadPreset(preferred.id); }
    else {
      populatePresets(); $('#config-fields').disabled = false; $('#advanced').open = true;
      text('#config-state', '等待导入');
      text('#preset-description', '当前安装未包含示例预设。请通过下方完整配置导入 JSON。');
      feedback('请导入完整 JSON 配置，然后校验并运行。');
    }
    await refreshHistory();
    const active = state.history.find(job => !terminal.has(job.status));
    if (active) selectJob(active);
  } catch (error) {
    setConnected(false); feedback(errorText(error), true);
    $('#config-fields').disabled = false; $('#advanced').open = true;
    text('#config-state', '服务未连接'); empty($('#history-list'), '服务恢复后可刷新运行记录。');
  }
}
// Browsing is read-only; explicit input editors below synchronize the shared Config.
const inputView = {source: 'draft', jobId: '', snapshot: null, loading: false, error: '', version: 0, viewedConfig: null, refreshTimer: null};
const inputObject = value => value && typeof value === 'object' && !Array.isArray(value) ? value : {};
const inputPositiveInt = value => typeof value === 'number' && Number.isInteger(value) && value > 0;
function inputScalar(value) {
  if (value === undefined) return '未提供';
  if (value === null) return '未设置（null）';
  if (value === true) return '是（true）';
  if (value === false) return '否（false）';
  if (value === '') return '空字符串（""）';
  if (Array.isArray(value)) return '数组 · ' + value.length + ' 项';
  if (typeof value === 'object') return '对象 · ' + Object.keys(value).length + ' 个字段';
  return String(value);
}
function inputUnit(value, unit, scale = 1, digits = 3) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return inputScalar(value);
  const scaled = value / scale;
  let formatted = number(scaled, digits);
  if (scaled !== 0 && Number(scaled.toFixed(digits)) === 0) formatted = scaled.toPrecision(3);
  return formatted + (unit ? ' ' + unit : '');
}
function inputNote(host, message, liveKey) { const note = el('p', 'input-section-note', message); if (liveKey) note.dataset.inputLiveNote = liveKey; host.append(note); }
function inputFacts(host, rows, className = '') {
  const list = el('dl', 'input-facts ' + className);
  rows.forEach(([label, value, path]) => {
    const item = el('div', 'input-fact'), term = el('dt', '', label), definition = el('dd');
    if (path) { term.title = path; definition.title = path; definition.dataset.inputDisplayPath = path; }
    definition.dataset.inputComputedLabel = label;
    const tokens = path && !path.includes(' / ') ? path.split('.') : null;
    const editor = tokens ? inputFieldEditor(tokens, label) : null;
    if (editor) definition.append(editor);
    else {
      definition.append(el('span', 'input-display-value', inputScalar(value)));
      if (tokens) { const button = inputJsonButton(tokens, '编辑字段'); if (button) definition.append(button); }
    }
    item.append(term, definition); list.append(item);
  });
  host.append(list); return list;
}
function inputTable(host, headers, rows, caption) {
  const wrap = el('div', 'input-table-wrap'), table = el('table', 'input-data-table');
  if (caption) table.append(el('caption', '', caption));
  const head = el('thead'), header = el('tr');
  headers.forEach(label => { const cell = el('th', '', label); cell.setAttribute('scope', 'col'); header.append(cell); });
  head.append(header); table.append(head);
  const body = el('tbody');
  rows.forEach(values => {
    const row = el('tr');
    values.forEach((value, index) => {
      const cell = el('td');
      if (value && typeof value === 'object' && Array.isArray(value.inputTokens)) {
        const editor = inputFieldEditor(value.inputTokens, headers[index] || inputTokenPath(value.inputTokens));
        cell.append(editor || el('span', '', inputScalar(value.display)));
      } else cell.textContent = inputScalar(value);
      row.append(cell);
    });
    body.append(row);
  });
  table.append(body); wrap.append(table); host.append(wrap); return body;
}
function inputTree(value, path, label = path, tokens = path === 'config' ? [] : path.split('.')) {
  // Filtered section trees retain their section keys; full trees follow every key.
  const original = value;
  const atPath = inputEditorActive() ? inputOwnValue(state.config, tokens) : value;
  const projection = value && typeof value === 'object' && !Array.isArray(value) && atPath !== value ? Object.keys(value) : null;
  function liveValue() {
    if (!inputEditorActive()) return original;
    const current = inputOwnValue(state.config, tokens);
    if (!projection || !current || typeof current !== 'object') return current;
    const selected = {};
    projection.forEach(key => { if (Object.hasOwn(current, key)) Object.defineProperty(selected, key, {value: current[key], enumerable: true}); });
    return selected;
  }
  let source = liveValue();
  if (source === null || typeof source !== 'object') {
    const scalar = el('span', 'input-raw-value', inputScalar(source)); scalar.title = path;
    scalar.dataset.inputRawTokens = JSON.stringify(tokens);
    const button = inputJsonButton(tokens, '编辑');
    if (!button) return scalar;
    const wrap = el('div', 'input-tree-edit-value'); wrap.append(scalar, button); return wrap;
  }
  const detail = el('details', 'input-config-tree');
  detail.dataset.inputTreeTokens = JSON.stringify(tokens);
  let keys = Object.keys(source), isArray = Array.isArray(source), built = false, renderedKeys = '';
  const summary = el('summary');
  function updateSummary() { summary.textContent = label + ' · ' + keys.length + (isArray ? ' 项' : ' 个字段'); }
  updateSummary(); detail.append(summary);
  const editButton = inputJsonButton(tokens, '编辑这组完整字段'); if (editButton) detail.append(editButton);
  let content = null, more = null;
  function build() {
    source = liveValue(); keys = source && typeof source === 'object' ? Object.keys(source) : []; isArray = Array.isArray(source);
    updateSummary(); built = true; renderedKeys = JSON.stringify(keys);
    if (!keys.length) { content = el('p', 'input-section-note', Array.isArray(source) ? '空数组 []' : inputScalar(source)); detail.append(content); return; }
    content = el('div', 'input-table-wrap'); const table = el('table', 'input-data-table input-raw-table'), body = el('tbody');
    const head = el('thead'), row = el('tr');
    ['字段 / 索引', '原始值'].forEach(value => { const cell = el('th', '', value); cell.setAttribute('scope', 'col'); row.append(cell); }); head.append(row); table.append(head, body); content.append(table); detail.append(content);
    let shown = 0; more = el('button', 'text-button input-show-more'); more.type = 'button';
    function appendBatch() {
      source = liveValue();
      const end = Math.min(keys.length, shown + 40);
      for (; shown < end; shown++) {
        const key = keys[shown], itemPath = isArray ? path + '[' + key + ']' : path + '.' + key;
        const row = el('tr'), term = el('th'), cell = el('td'); term.setAttribute('scope', 'row'); term.append(el('code', '', key));
        cell.append(inputTree(source && source[key], itemPath, itemPath, tokens.concat(isArray ? Number(key) : key))); row.append(term, cell); body.append(row);
      }
      more.textContent = '继续显示 · 已显示 ' + shown + ' / ' + keys.length + ' 项'; more.hidden = shown >= keys.length;
    }
    more.addEventListener('click', appendBatch); detail.append(more); appendBatch();
  }
  detail.refreshInputTree = () => {
    source = liveValue(); keys = source && typeof source === 'object' ? Object.keys(source) : []; isArray = Array.isArray(source); updateSummary();
    if (built && renderedKeys !== JSON.stringify(keys)) {
      if (content) content.remove(); if (more) more.remove(); content = null; more = null; built = false;
      if (detail.open) build();
    }
  };
  detail.addEventListener('toggle', () => { if (detail.open && !built) build(); });
  return detail;
}
function inputFields(host, value, path, label) {
  const container = el('div', 'input-full-fields');
  if (value === undefined || value === null || typeof value !== 'object') {
    container.append(el('p', 'input-section-note', label + '：' + inputScalar(value)));
  } else container.append(inputTree(value, path, label));
  host.append(container);
}
function assertInputFinite(value) {
  if (typeof value === 'number' && !Number.isFinite(value)) throw new Error('草稿包含超出有限范围的数字（例如 1e999），无法准确展示或导出。');
  if (value && typeof value === 'object') Object.values(value).forEach(assertInputFinite);
}
function currentInputDraft() {
  const config = state.jsonDirty ? JSON.parse($('#config-json').value) : state.config;
  if (!config) throw new Error('尚无当前配置。请返回实验页选择预设或导入 JSON。');
  if (typeof config !== 'object' || Array.isArray(config)) throw new Error('配置根节点必须为 JSON 对象。');
  assertInputFinite(config);
  return config;
}
function scheduleInputOverview() {
  if ($('#inputs').hidden || inputEdit.editing || inputEdit.jsonDirty) return;
  clearTimeout(inputView.refreshTimer);
  inputView.refreshTimer = setTimeout(() => { if (!inputEdit.editing && !inputEdit.jsonDirty) renderInputOverview(); }, 120);
}
function inputJobOptions() {
  const select = $('#input-job-select'), known = new Map(state.history.map(job => [job.id, job]));
  if (state.currentJob) known.set(state.currentJob.id, state.currentJob);
  if (inputView.jobId && !known.has(inputView.jobId)) known.set(inputView.jobId, {id: inputView.jobId});
  select.replaceChildren();
  if (!known.size) { const option = el('option', '', '暂无运行记录'); option.value = ''; select.append(option); }
  for (const job of known.values()) {
    const option = el('option', '', job.id + (job.status ? ' · ' + (statusNames[job.status] || job.status) : '') + (job.created_at ? ' · ' + formatTime(job.created_at) : ''));
    option.value = job.id; select.append(option);
  }
  select.value = inputView.jobId;
  select.disabled = !known.size;
}
async function loadInputSnapshot(id) {
  const version = ++inputView.version;
  inputView.jobId = id; inputView.snapshot = null; inputView.error = '';
  inputView.loading = Boolean(id); inputJobOptions(); renderInputOverview();
  if (!id) return;
  try {
    const response = await api('/api/jobs/' + encodeURIComponent(id) + '/config');
    if (version !== inputView.version || inputView.source !== 'job' || id !== inputView.jobId) return;
    if (!response.config || typeof response.config !== 'object' || Array.isArray(response.config)) throw new Error('该任务未返回有效配置对象。');
    assertInputFinite(response.config);
    inputView.snapshot = response.config;
  } catch (error) {
    if (version !== inputView.version || inputView.source !== 'job') return;
    inputView.error = errorText(error);
  } finally {
    if (version === inputView.version && inputView.source === 'job') { inputView.loading = false; renderInputOverview(); }
  }
}
function renderInputModel(config) {
  const host = $('#input-model-body'); host.replaceChildren();
  const model = inputObject(config.model), architecture = inputObject(model.architecture);
  inputFacts(host, [
    ['模型名称', model.name, 'model.name'],
    ['总参数量', inputUnit(model.total_params, '参数', 1, 0), 'model.total_params'],
    ['每 token 活跃参数', inputUnit(model.active_params, '参数', 1, 0), 'model.active_params'],
    ['层数', model.layers, 'model.layers'], ['隐藏维度', model.hidden_dim, 'model.hidden_dim'],
    ['通用中间维度', model.intermediate_dim, 'model.intermediate_dim'],
    ['Attention 头数 / KV 头数', inputScalar(model.num_heads) + ' / ' + inputScalar(model.kv_heads), 'model.num_heads / model.kv_heads'],
    ['Head 维度', model.head_dim, 'model.head_dim'],
    ['通用 Attention 类型', model.attention_kind, 'model.attention_kind'],
    ['专家数 / 每 token Top-k', inputScalar(model.num_experts) + ' / ' + inputScalar(model.top_k), 'model.num_experts / model.top_k'],
    ['MoE 层占比', inputUnit(model.moe_layer_fraction, '%', .01), 'model.moe_layer_fraction'],
    ['专家参数占比', inputUnit(model.expert_parameter_fraction, '%', .01), 'model.expert_parameter_fraction']
  ]);
  inputExtraFacts(host, [['Attention 头数', 'model.num_heads'], ['KV 头数', 'model.kv_heads'], ['专家数', 'model.num_experts'], ['Top-k', 'model.top_k']]);
  inputNote(host, '总参数、每 token 活跃参数与激活内存是不同量。具有 architecture 时，张量形状与逐层类型给出更具体的结构；原始草稿缺失值保留为未提供。');
  if (model.architecture != null) {
    // Use Map for arbitrary user-supplied layer labels, with no object-key mutation.
    const layerLabels = values => {
      if (!Array.isArray(values)) return inputScalar(values);
      const counts = new Map(); values.forEach(kind => { const key = inputScalar(kind); counts.set(key, (counts.get(key) || 0) + 1); });
      return [...counts].slice(0, 12).map(([kind, count]) => kind + ' × ' + count).join(' · ') + (counts.size > 12 ? ' · 其余 ' + (counts.size - 12) + ' 种类型见完整字段' : '') || '空数组 []';
    };
    inputFacts(host, [['逐层 Attention', layerLabels(architecture.attention_by_layer), 'model.architecture.attention_by_layer'], ['逐层 FFN', layerLabels(architecture.ffn_by_layer), 'model.architecture.ffn_by_layer'], ['词表大小', architecture.vocab_size, 'model.architecture.vocab_size'], ['共享输入 / 输出权重', architecture.tie_embeddings, 'model.architecture.tie_embeddings']]);
  }
  inputFields(host, config.model, 'model', '全部模型字段（含 MLA / KDA / MoE 结构）');
}
function renderInputTopology(config) {
  const host = $('#input-topology-body'); host.replaceChildren();
  const h = inputObject(config.hardware), e = inputObject(config.execution), npu = h.npu_count, su = h.su_size, perHost = h.npus_per_host;
  const suCount = inputPositiveInt(npu) && inputPositiveInt(su) && npu % su === 0 ? number(npu / su, 0) + '（由 NPU / SU 卡数推导）' : '无法推导：需要可整除的正整数 NPU / SU 卡数';
  const hosts = inputPositiveInt(npu) && inputPositiveInt(perHost) ? number(Math.ceil(npu / perHost), 0) + '（向上取整）' : '无法推导：NPU / 每主机卡数未就绪';
  inputFacts(host, [
    ['拓扑类型', h.topology, 'hardware.topology'], ['路由策略', h.routing_policy, 'hardware.routing_policy'],
    ['NPU 总数 / 每 SU 卡数', inputScalar(npu) + ' / ' + inputScalar(su), 'hardware.npu_count / hardware.su_size'],
    ['Scale-up 域数', suCount], ['主机数', hosts],
    ['Scale-out 域数', '原生配置未单独声明，不从 SU 数推断'],
    ['Mesh 宽度', h.mesh_width, 'hardware.mesh_width'], ['Leaf 分组大小', h.leaf_size, 'hardware.leaf_size'],
    ['显式有向链路', Array.isArray(h.explicit_links) ? h.explicit_links.length + ' 条' : inputScalar(h.explicit_links), 'hardware.explicit_links'],
    ['指定路由', h.routes && typeof h.routes === 'object' && !Array.isArray(h.routes) ? Object.keys(h.routes).length + ' 条' : inputScalar(h.routes), 'hardware.routes']
  ]);
  inputExtraFacts(host, [['NPU 总数', 'hardware.npu_count'], ['每 SU 卡数', 'hardware.su_size']]);
  inputNote(host, 'SU / SO 是链路类别。显式图的边、路由与共享组保留在下方；域成员不能由型号名称、链路类别或域数量猜测。');
  const topology = {};
  ['topology', 'routing_policy', 'npu_count', 'su_size', 'npus_per_host', 'mesh_width', 'leaf_size', 'explicit_links', 'routes'].forEach(key => { if (Object.hasOwn(h, key)) topology[key] = h[key]; });
  inputFields(host, topology, 'hardware', '完整拓扑字段与逐条显式链路');
  inputFields(host, e.rank_order, 'execution.rank_order', 'Rank → 物理 NPU 放置');
}
function renderInputHardware(config) {
  const host = $('#input-hardware-body'); host.replaceChildren();
  const h = inputObject(config.hardware), npu = inputObject(h.npu), cpu = inputObject(h.cpu);
  inputNote(host, '硬件名称：' + inputScalar(h.name) + '。容量使用十进制 GB；带宽是 GB/s（byte/s），算力为配置峰值，效率单独展示。');
  const grid = el('div', 'input-hardware-grid'), npuCard = el('div'), cpuCard = el('div');
  npuCard.append(el('h3', '', '每颗 NPU'));
  inputFacts(npuCard, [
    ['HBM 容量', inputUnit(npu.hbm_bytes, 'GB', 1e9), 'hardware.npu.hbm_bytes'],
    ['HBM 带宽', inputUnit(npu.hbm_Bps, 'GB/s', 1e9), 'hardware.npu.hbm_Bps'],
    ['计算效率', inputUnit(npu.efficiency, '%', .01), 'hardware.npu.efficiency'],
    ['运行时预留', inputUnit(npu.runtime_bytes, 'GB', 1e9), 'hardware.npu.runtime_bytes'],
    ['Workspace', inputUnit(npu.workspace_bytes, 'GB', 1e9), 'hardware.npu.workspace_bytes']
  ], 'input-facts-single');
  inputTable(npuCard, ['计算精度', '配置峰值 / TFLOP/s'], Object.entries(inputObject(npu.flops)).map(([dtype, rate]) => [dtype, inputEditableValue(['hardware', 'npu', 'flops', dtype], inputUnit(rate, '', 1e12))]), '各精度独立算力');
  if (!Object.keys(inputObject(npu.flops)).length) inputNote(npuCard, '算力映射：' + inputScalar(npu.flops));
  cpuCard.append(el('h3', '', '每台主机 CPU'));
  inputFacts(cpuCard, [
    ['每主机 NPU 数', h.npus_per_host, 'hardware.npus_per_host'],
    ['DRAM 容量', inputUnit(cpu.dram_bytes, 'GB', 1e9), 'hardware.cpu.dram_bytes'],
    ['DRAM 带宽', inputUnit(cpu.dram_Bps, 'GB/s', 1e9), 'hardware.cpu.dram_Bps'],
    ['CPU 峰值算力', inputUnit(cpu.flops, 'TFLOP/s', 1e12), 'hardware.cpu.flops'],
    ['计算效率', inputUnit(cpu.efficiency, '%', .01), 'hardware.cpu.efficiency'],
    ['IOPS', inputUnit(cpu.iops, '次/s', 1, 0), 'hardware.cpu.iops']
  ], 'input-facts-single');
  grid.append(npuCard, cpuCard); host.append(grid);
  host.append(el('h3', 'input-subheading', '互联链路'));
  inputTable(host, ['链路', '带宽 GB/s', '延迟 μs', '效率', '端点注入 GB/s', '共享上联 GB/s'],
    [['Scale-up', 'scale_up'], ['Scale-out', 'scale_out'], ['Host ↔ NPU', 'host_link']].map(([label, key]) => {
      const link = inputObject(h[key]);
      return [label, ...[['bandwidth_Bps', 1e9, ''], ['latency_s', 1e-6, ''], ['efficiency', .01, '%'], ['injection_Bps', 1e9, ''], ['uplink_Bps', 1e9, '']].map(([field, scale, unit]) => inputEditableValue(['hardware', key, field], inputUnit(link[field], unit, scale)))];
    }));
  inputNote(host, h.topology === 'explicit' ? '显式边的带宽、延迟、效率及共享容量以 explicit_links 为准；本表的端点注入等资源约束仍参与计算。' : '链路带宽、端点注入能力与共享上联能力是独立约束；未设置的值不会在此自动补齐。');
  host.append(el('h3', 'input-subheading', '外部存储'));
  inputTable(host, ['类型', '启用', '容量 GB', '带宽 GB/s', '延迟 μs', 'IOPS'],
    [['远程内存', 'remote'], ['SSD', 'ssd']].map(([label, key]) => {
      const storage = inputObject(h[key]); return [label, inputEditableValue(['hardware', key, 'enabled'], storage.enabled), ...[['capacity_bytes', 1e9], ['bandwidth_Bps', 1e9], ['latency_s', 1e-6], ['iops', 1]].map(([field, scale]) => inputEditableValue(['hardware', key, field], inputUnit(storage[field], '', scale)))];
    }));
  inputFields(host, config.hardware, 'hardware', '全部硬件字段（含显式图、kernel profiles 与共享资源）');
}
function pdEnabled(config) {
  return (config.features?.prefill_decode_disaggregation?.enabled ?? config.execution?.features?.prefill_decode_disaggregation?.enabled) === true;
}
function parallelInputNote(config) {
  const e = inputObject(config.execution), h = inputObject(config.hardware), pd = pdEnabled(config);
  const factors = [e.tp, e.pp, e.ep, e.dp];
  if (!factors.every(inputPositiveInt)) return '并行乘积尚不可推导。TP / PP / EP / DP 必须都是正整数；SP 共用 TP rank，FSDP 是 DP 子组。';
  const product = factors.reduce((a, b) => a * b, pd ? 2 : 1);
  if (pd && (e.dp !== 1 || e.pp !== 1)) return 'PD 双池要求 PP = DP = 1；请校验并修正并行配置。';
  return (pd ? 'PD 双池：2 × TP × EP = ' : 'TP × PP × EP × DP = ') + number(product, 0) + '；配置 NPU = ' + inputScalar(h.npu_count) + (product === h.npu_count ? '，两者匹配。' : '，两者不匹配或 NPU 尚未提供。') + ' SP / FSDP 不额外乘入卡数。';
}
function renderInputParallel(config) {
  const host = $('#input-parallel-body'); host.replaceChildren();
  const e = inputObject(config.execution), h = inputObject(config.hardware), precision = inputObject(e.precision), features = inputObject(e.features), optimizer = inputObject(e.optimizer);
  inputFacts(host, [['TP · 张量并行', e.tp, 'execution.tp'], ['PP · 流水线并行', e.pp, 'execution.pp'], ['EP · 专家并行', e.ep, 'execution.ep'], ['DP · 数据并行', e.dp, 'execution.dp'], ['FSDP · DP 子组', e.fsdp, 'execution.fsdp'], ['SP · 序列并行开关', e.sequence_parallel, 'execution.sequence_parallel'], ['SP 并行度', sequenceParallelInfo(config).label]]);
  inputNote(host, parallelInputNote(config), 'parallel');
  const precisionLabels = {weight: '权重驻留', compute: '计算', gradient: '梯度', master: '主权重', optimizer: '优化器状态', kv: 'KV', weight_transfer: '权重回传', activation: '激活', communication: '通信', recurrent: '循环状态', routed_expert_weight: '路由专家权重覆盖'};
  inputTable(host, ['独立精度项', '数据类型'], Object.entries(precision).map(([key, value]) => [(precisionLabels[key] || key) + ' · ' + key, inputEditableValue(['execution', 'precision', key], value)]), '各精度独立配置；未设置的覆盖项保留 null');
  if (!Object.keys(precision).length) inputNote(host, '精度映射：' + inputScalar(e.precision));
  inputFacts(host, [
    ['KV offload', features.kv_offload, 'execution.features.kv_offload'],
    ['KV offload 比例', inputUnit(features.kv_offload_fraction, '%', .01), 'execution.features.kv_offload_fraction'],
    ['优化器 offload', features.optimizer_offload, 'execution.features.optimizer_offload'],
    ['激活重计算', features.activation_checkpointing, 'execution.features.activation_checkpointing'],
    ['优化器类型', optimizer.kind, 'execution.optimizer.kind'],
    ['Engram', inputObject(features.engram).enabled, 'execution.features.engram.enabled'],
    ['FSDP 前向后 reshard', e.reshard_after_forward, 'execution.reshard_after_forward'],
    ['每微批同步 / 预取', inputScalar(e.sync_each_microbatch) + ' / ' + inputScalar(e.prefetch), 'execution.sync_each_microbatch / execution.prefetch']
  ]);
  inputExtraFacts(host, [['每微批同步', 'execution.sync_each_microbatch'], ['预取', 'execution.prefetch']]);
  inputNote(host, Array.isArray(e.rank_order) && !e.rank_order.length ? 'rank_order = []：引擎使用默认恒等放置，逻辑 rank 与物理 NPU 索引一致。' : 'rank_order 每一项的数组索引是逻辑 rank，值是物理 NPU 索引。未提供时不在此生成放置。');
  inputFields(host, config.execution, 'execution', '全部执行字段（含 rank 放置、卸载、Engram 与优化器细节）');
}
function inputLengthSummary(values) {
  if (!Array.isArray(values)) return inputScalar(values);
  if (!values.length) return '空数组 [] · 本页不自动展开标量默认长度';
  if (!values.every(inputPositiveInt)) return values.length + ' 项 · 含非正整数，无法计算长度统计';
  let min = Infinity, max = -Infinity, sum = 0; const counts = new Map();
  values.forEach(value => { min = Math.min(min, value); max = Math.max(max, value); sum += value; counts.set(value, (counts.get(value) || 0) + 1); });
  return values.length + ' 条 · ' + min + '–' + max + ' tokens · 均值 ' + number(sum / values.length, 2) + ' · ' + counts.size + ' 种长度';
}
function inputLengthDistribution(host, values, label) {
  if (!Array.isArray(values) || !values.length || !values.every(inputPositiveInt)) return;
  const counts = new Map(); values.forEach(value => counts.set(value, (counts.get(value) || 0) + 1));
  const detail = el('details', 'input-config-tree');
  detail.append(el('summary', '', label + '长度分布 · ' + counts.size + ' 种长度'));
  let built = false;
  detail.addEventListener('toggle', () => {
    if (!detail.open || built) return; built = true;
    const entries = [...counts].sort((a, b) => a[0] - b[0]), body = inputTable(detail, ['长度 / tokens', '请求数量', '占比'], []);
    let shown = 0; const more = el('button', 'text-button input-show-more'); more.type = 'button';
    function appendBatch() {
      const end = Math.min(entries.length, shown + 40);
      for (; shown < end; shown++) {
        const [length, count] = entries[shown], row = el('tr');
        [length, count, inputUnit(count / values.length, '%', .01)].forEach(value => row.append(el('td', '', inputScalar(value)))); body.append(row);
      }
      more.textContent = '继续显示 · 已显示 ' + shown + ' / ' + entries.length + ' 种长度'; more.hidden = shown >= entries.length;
    }
    more.addEventListener('click', appendBatch); detail.append(more); appendBatch();
  });
  host.append(detail);
}
function inputRequestLengths(host, workload) {
  const inputs = Array.isArray(workload.input_lengths) ? workload.input_lengths : [], outputs = Array.isArray(workload.output_lengths) ? workload.output_lengths : [];
  const count = Math.max(inputs.length, outputs.length);
  const detail = el('details', 'input-config-tree'); detail.append(el('summary', '', '每个请求的精确长度 · ' + count + ' 项 / DP 副本'));
  let built = false;
  detail.addEventListener('toggle', () => {
    if (!detail.open || built) return; built = true;
    if (!count) { inputNote(detail, '没有显式请求长度；标量回退值保留在 workload 完整字段中。'); return; }
    const body = inputTable(detail, ['请求索引（从 0 开始）', '输入 tokens', '输出 tokens'], []);
    let shown = 0; const more = el('button', 'text-button input-show-more'); more.type = 'button';
    function appendBatch() {
      const end = Math.min(count, shown + 40);
      for (; shown < end; shown++) { const row = el('tr'); [shown, inputs[shown], outputs[shown]].forEach(value => row.append(el('td', '', inputScalar(value)))); body.append(row); }
      more.textContent = '继续显示 · 已显示 ' + shown + ' / ' + count + ' 个请求'; more.hidden = shown >= count;
    }
    more.addEventListener('click', appendBatch); detail.append(more); appendBatch();
  });
  host.append(detail);
}
function renderInputWorkload(config) {
  const host = $('#input-workload-body'); host.replaceChildren();
  const w = inputObject(config.workload), e = inputObject(config.execution), targets = inputObject(config.targets);
  inputFacts(host, [['场景', w.scenario === 'inference' ? '推理 · inference' : w.scenario === 'training' ? '训练 · training' : w.scenario, 'workload.scenario'], ['每 DP 副本 Batch', w.batch_size, 'workload.batch_size']]);
  if (w.scenario === 'training') {
    inputFacts(host, [['序列长度', inputUnit(w.sequence_length, 'tokens', 1, 0), 'workload.sequence_length'], ['梯度累积次数', w.gradient_accumulation, 'workload.gradient_accumulation'], ['训练步数', w.training_steps, 'workload.training_steps']]);
    const factors = [w.batch_size, w.gradient_accumulation, e.dp];
    inputNote(host, factors.every(inputPositiveInt) ? '每优化器步全局样本数 = Batch × 梯度累积 × DP = ' + number(factors.reduce((a, b) => a * b, 1), 0) + '。' : '全局样本数需要有效的 Batch、梯度累积和 DP 才能推导。', 'training-batch');
    inputNote(host, '训练采用 sequence_length；完整字段中的 input_length(s) / output_length(s) 是归一化配置保留的推理字段，此训练场景不据此生成负载。');
  } else if (w.scenario === 'inference') {
    inputFacts(host, [['输入长度', inputLengthSummary(w.input_lengths), 'workload.input_lengths'], ['输出长度', inputLengthSummary(w.output_lengths), 'workload.output_lengths']]);
    if (Array.isArray(w.input_lengths) && Array.isArray(w.output_lengths)) {
      inputNote(host, w.input_lengths.length === w.batch_size && w.output_lengths.length === w.batch_size ? '显式长度列表与每副本 Batch 数量一致；保留每条请求的差异。' : '显式列表数量与 Batch 不一致或仍未展开；此页面不补齐、不覆盖请求长度。', 'length-count');
    }
    inputLengthDistribution(host, w.input_lengths, '输入'); inputLengthDistribution(host, w.output_lengths, '输出'); inputRequestLengths(host, w);
    inputNote(host, inputPositiveInt(w.batch_size) && inputPositiveInt(e.dp) ? '配置请求总数 = 每副本 Batch × DP = ' + number(w.batch_size * e.dp, 0) + '。这是固定负载，不包含请求到达排队。' : '请求总数需有效 Batch 与 DP 才能推导。', 'inference-batch');
  } else inputNote(host, '场景尚未明确；以下完整字段保留当前输入，不选择默认推理或训练场景。');
  host.append(el('h3', 'input-subheading', '性能目标（输入约束）'));
  inputFacts(host, [['TTFT 上限', inputUnit(targets.ttft_s, 'ms', .001), 'targets.ttft_s'], ['TPOT 上限', inputUnit(targets.tpot_s, 'ms', .001), 'targets.tpot_s'], ['训练步时间上限', inputUnit(targets.step_time_s, 's'), 'targets.step_time_s'], ['聚合吞吐下限', inputUnit(targets.throughput_tokens_s, 'tokens/s'), 'targets.throughput_tokens_s']]);
  inputNote(host, '目标值不是仿真输出；不适用当前场景的目标仍如实保留，是否允许由实验页的完整校验决定。');
  inputFields(host, config.workload, 'workload', '全部负载字段与精确数组');
  inputFields(host, config.targets, 'targets', '全部目标字段与搜索候选');
}
function renderInputMetadata(config) {
  const host = $('#input-meta-body'); host.replaceChildren();
  inputFacts(host, [['Schema 版本', config.schema_version, 'schema_version'], ['随机种子', config.seed, 'seed'], ['校准引用', config.calibration, 'calibration'], ['假设条目', Array.isArray(config.assumptions) ? config.assumptions.length : inputScalar(config.assumptions), 'assumptions']]);
  const metadata = {};
  Object.entries(config).forEach(([key, value]) => { if (!['model', 'hardware', 'execution', 'workload', 'targets'].includes(key)) Object.defineProperty(metadata, key, {value, enumerable: true}); });
  inputFields(host, metadata, 'config', '来源、假设与所有其他根字段');
  inputFields(host, config, 'config', '完整配置字段树（原始值，按需展开）');
}
function renderInputOverview() {
  if ($('#inputs').hidden) return;
  syncInputEditorControls();
  inputJobOptions();
  $('#input-source').value = inputView.source;
  $('#input-job-control').hidden = inputView.source !== 'job';
  $('#retry-input-snapshot').hidden = inputView.source !== 'job' || !inputView.jobId;
  $('#retry-input-snapshot').disabled = inputView.loading;
  inputView.viewedConfig = null; $('#download-viewed-input').disabled = true;
  const badge = $('#input-state-badge'), notice = $('#input-view-notice');
  notice.hidden = true; notice.className = 'notice';
  let config;
  if (inputView.source === 'job') {
    badge.textContent = inputView.loading ? '快照读取中' : inputView.snapshot ? '只读快照' : '等待快照';
    badge.className = 'mini-badge' + (inputView.error ? ' error' : '');
    text('#input-source-description', '运行 ' + (inputView.jobId || '尚未选择') + ' · 以下是提交时的配置，不随当前编辑内容改变。');
    if (inputView.loading || inputView.error || !inputView.snapshot) {
      $('#input-view-content').hidden = true; $('#input-view-empty').hidden = false;
      text('#input-view-empty h2', inputView.loading ? '正在读取运行配置' : inputView.error ? '配置快照读取失败' : '请选择一个运行');
      text('#input-view-empty p', inputView.error || (inputView.loading ? '从该任务的独立配置文件读取，尚不展示上一份配置。' : '尚无运行时可先查看当前编辑配置。'));
      return;
    }
    config = inputView.snapshot;
  } else {
    badge.textContent = state.jsonDirty ? 'JSON 草稿 · 未应用' : $('#config-state').textContent;
    badge.className = 'mini-badge' + (state.jsonDirty || state.modified ? ' warning' : '');
    text('#input-source-description', state.jsonDirty ? '来源：当前 JSON 编辑器。只读解析尚未应用的草稿；查看不会保存、校验或替换实验配置。' : '来源：当前实际 Config。输入页与实验页共享参数；编辑后的值用于下一次仿真，未提供字段不会自动补默认值。');
    try { config = currentInputDraft(); }
    catch (error) {
      $('#input-view-content').hidden = true; $('#input-view-empty').hidden = false;
      text('#input-view-empty h2', state.jsonDirty ? '当前 JSON 草稿无法展示' : '尚无可展示的输入');
      text('#input-view-empty p', errorText(error));
      badge.textContent = state.jsonDirty ? '草稿无效' : '等待配置'; badge.className = 'mini-badge error';
      return;
    }
    if (state.jsonDirty) { notice.hidden = false; notice.textContent = '该 JSON 草稿尚未应用或执行 schema 校验。此处仅展示原始输入；类型、约束与来源请在实验页应用并校验后确认。'; }
  }
  $('#input-view-empty').hidden = true; $('#input-view-content').hidden = false;
  renderInputModel(config); renderInputTopology(config); renderInputHardware(config); renderInputParallel(config); renderInputWorkload(config); renderInputMetadata(config);
  inputView.viewedConfig = config; $('#download-viewed-input').disabled = false;
  syncInputEditorControls();
}
function routeDashboard(focus = false) {
  const hash = location.hash || '#workspace', showInputs = hash === '#inputs' || hash.startsWith('#inputs-');
  const entering = showInputs && $('#inputs').hidden;
  $('#workspace').hidden = showInputs; $('#inputs').hidden = !showInputs;
  const active = showInputs ? '#inputs' : hash === '#results' ? '#results' : hash === '#history-panel' ? '#history-panel' : '#workspace';
  $$('.rail-link').forEach(link => { const selected = link.getAttribute('href') === active; link.classList.toggle('active', selected); if (selected) link.setAttribute('aria-current', 'page'); else link.removeAttribute('aria-current'); });
  document.title = showInputs ? '输入总览 · 超节点流量仿真实验室' : '超节点 · 流量仿真实验室';
  if (showInputs && (entering || !inputView.viewedConfig)) renderInputOverview();
  if (focus) {
    const target = document.getElementById(hash.slice(1)) || (showInputs ? $('#inputs-heading') : $('#workspace'));
    requestAnimationFrame(() => { target.scrollIntoView({block: 'start'}); if (showInputs) (hash === '#inputs' ? $('#inputs-heading') : target).focus({preventScroll: true}); });
  }
}
function attachInputOverview() {
  $('#input-source').addEventListener('change', () => {
    if (state.busy) { $('#input-source').value = inputView.source; return; }
    inputView.source = $('#input-source').value; inputView.version++; inputView.loading = false; inputView.error = '';
    if (inputView.source === 'job') {
      const id = inputView.jobId || state.currentJob && state.currentJob.id || state.history[0] && state.history[0].id || '';
      loadInputSnapshot(id);
    } else renderInputOverview();
  });
  $('#input-job-select').addEventListener('change', () => loadInputSnapshot($('#input-job-select').value));
  $('#retry-input-snapshot').addEventListener('click', () => loadInputSnapshot(inputView.jobId));
  $('#download-viewed-input').addEventListener('click', () => {
    renderInputOverview();
    if (!inputView.viewedConfig) return;
    downloadJson(inputView.viewedConfig, inputView.source === 'job' ? 'supernode-run-' + inputView.jobId + '-config.json' : 'supernode-current-input.json');
  });
  window.addEventListener('hashchange', () => routeDashboard(true));
  routeDashboard();
}

// Editors share state.config with the experiment form and /api/jobs.
// Invalid text stays in independent buffers until explicitly corrected or discarded.
const inputEdit = {editing: false, invalidFields: new Map(), jsonTokens: null, jsonRaw: '', jsonDirty: false, jsonOpen: false, message: '', messageError: false};
const inputTokenKey = tokens => JSON.stringify(tokens);
function inputTokenPath(tokens) {
  if (!tokens.length) return 'config';
  return tokens.map((token, index) => typeof token === 'number' ? '[' + token + ']' : /^[A-Za-z_]\w*$/.test(token) ? (index ? '.' : '') + token : '[' + JSON.stringify(token) + ']').join('');
}
function inputOwnValue(object, tokens) {
  return tokens.reduce((value, key) => value && typeof value === 'object' && Object.hasOwn(value, key) ? value[key] : undefined, object);
}
function inputSetOwn(object, tokens, value) {
  let parent = object;
  tokens.slice(0, -1).forEach((key, index) => {
    if (!Object.hasOwn(parent, key) || !parent[key] || typeof parent[key] !== 'object') Object.defineProperty(parent, key, {value: typeof tokens[index + 1] === 'number' ? [] : {}, writable: true, enumerable: true, configurable: true});
    parent = parent[key];
  });
  Object.defineProperty(parent, tokens[tokens.length - 1], {value, writable: true, enumerable: true, configurable: true});
}
function hasUnappliedInputEdits() { return inputEdit.invalidFields.size > 0 || inputEdit.jsonDirty; }
function inputEditMessage(message, error = false) {
  inputEdit.message = message; inputEdit.messageError = error;
  text('#input-edit-message', message);
  $('#input-edit-message').className = error ? 'input-edit-error' : '';
}
function requireAppliedInputEdits() {
  if (hasUnappliedInputEdits()) throw new Error('输入页仍有未应用内容：' + (inputEdit.invalidFields.size ? inputEdit.invalidFields.size + ' 个无效输入' : '') + (inputEdit.jsonDirty ? ' 字段 JSON 草稿' : '') + '。请回到输入页修正、应用或明确放弃后再校验 / 运行。');
}
function inputDerivedField(tokens, config = state.config) {
  if (tokens.length !== 2 || tokens[0] !== 'model' || !inputObject(config && config.model).architecture) return false;
  return ['total_params', 'active_params', 'expert_parameter_fraction', 'moe_layer_fraction', 'num_experts', 'top_k'].includes(tokens[1]);
}
function inputMarkSource(config, tokens, description) {
  if (!config.sources) config.sources = {};
  if (typeof config.sources !== 'object' || Array.isArray(config.sources)) throw new Error('sources 必须是对象，不能在记录编辑来源时丢弃原有内容。请先修正该字段。');
  Object.defineProperty(config.sources, inputTokenPath(tokens), {value: {status: 'user', description}, enumerable: true, configurable: true, writable: true});
}
function ensureInputEditorReady() {
  if (state.busy) throw new Error('正在校验或提交配置，请等待当前操作完成。');
  if (inputView.source !== 'draft') throw new Error('运行快照不可直接修改。请先复制为当前配置。');
  if (state.jsonDirty) {
    requireAppliedInputEdits();
    currentInputDraft(); // Parse and check finite numbers before readConfig can adopt anything.
    readConfig();
  }
  if (!state.config) throw new Error('尚无当前配置，请先选择预设或导入 JSON。');
}
function commitInputValue(tokens, value) {
  ensureInputEditorReady();
  assertInputFinite(value);
  if (inputDerivedField(tokens)) throw new Error('此字段由结构化 architecture 推导，请编辑张量形状 / 架构后校验。');
  const previous = state.config;
  const next = tokens.length ? clone(previous) : clone(value);
  if (!next || typeof next !== 'object' || Array.isArray(next)) throw new Error('完整配置必须是 JSON 对象。');
  if (tokens.length) inputSetOwn(next, tokens, value);
  if (tokens.join('.') === 'execution.sequence_parallel' && next.features?.sequence_parallel != null) {
    next.features.sequence_parallel = value;
    inputMarkSource(next, ['features', 'sequence_parallel'], 'SP 开关修改同步至已有的顶层同名配置。');
  }
  if (tokens.join('.') === 'workload.batch_size' && inputObject(next.workload).scenario === 'training') {
    next.workload.input_lengths = []; next.workload.output_lengths = [];
    inputMarkSource(next, ['workload', 'input_lengths'], '训练 Batch 修改后清空不参与训练的推理长度列表，完整校验时归一化。');
    inputMarkSource(next, ['workload', 'output_lengths'], '训练 Batch 修改后清空不参与训练的推理长度列表，完整校验时归一化。');
  }
  if (Object.hasOwn(next, 'sources') && (!next.sources || typeof next.sources !== 'object' || Array.isArray(next.sources))) throw new Error('sources 必须保留为对象；该字段草稿尚未应用。');
  const model = inputObject(next.model);
  const changedShape = model.architecture && (
    (!tokens.length && JSON.stringify(previous.model) !== JSON.stringify(next.model)) ||
    tokens[0] === 'model' && (tokens.length === 1 || !['name', 'reported_total_params', 'reported_active_params'].includes(tokens[1]))
  );
  if (changedShape) {
    delete model.total_params; delete model.active_params;
    inputMarkSource(next, ['model', 'total_params'], '结构输入已修改，清除旧派生总参数量；完整校验时按张量形状重新推导。');
    inputMarkSource(next, ['model', 'active_params'], '结构输入已修改，清除旧派生活跃参数量；完整校验时按张量形状重新推导。');
  }
  inputMarkSource(next, tokens, '通过输入总览编辑' + (tokens.length ? '字段 ' + inputTokenPath(tokens) : '完整配置') + '；作为下一次仿真的实际 Config 输入。');
  state.config = next; state.jsonDirty = false;
  inputEdit.invalidFields.delete(inputTokenKey(tokens));
  markModified(); renderForm();
  $$('[data-input-derived]').forEach(node => {
    const key = node.dataset.inputDerived;
    node.textContent = inputScalar(inputOwnValue(state.config, ['model', key])) + ' · 由 architecture 推导';
  });
  refreshInputLiveViews();
  syncInputEditorControls();
  inputEditMessage(changedShape ? '结构修改已同步。旧 total_params / active_params 已清除，请校验以重新推导；reported_* 保持不变。' : '修改已同步到当前 Config、实验表单与完整 JSON。下一次运行将使用这些值。');
  return state.config;
}
const inputEditSpecs = new Map();
function addInputSpecs(paths, spec) { paths.split(' ').filter(Boolean).forEach(path => inputEditSpecs.set(path, {...spec})); }
addInputSpecs('model.name hardware.name', {type: 'text'});
addInputSpecs('model.total_params model.active_params model.reported_total_params model.reported_active_params', {type: 'number'});
addInputSpecs('model.layers model.hidden_dim model.intermediate_dim model.num_heads model.kv_heads model.head_dim model.num_experts model.top_k model.kv_latent_dim model.kv_rope_dim model.architecture.vocab_size hardware.npu_count hardware.su_size hardware.npus_per_host hardware.mesh_width hardware.leaf_size execution.tp execution.pp execution.ep execution.dp execution.fsdp workload.batch_size workload.sequence_length workload.gradient_accumulation workload.training_steps seed', {type: 'number', integer: true});
addInputSpecs('hardware.mesh_width hardware.leaf_size', {type: 'number', integer: true, nullable: true});
addInputSpecs('model.moe_layer_fraction model.expert_parameter_fraction hardware.npu.efficiency hardware.cpu.efficiency execution.features.kv_offload_fraction', {type: 'number', scale: .01, unit: '%'});
addInputSpecs('hardware.npu.hbm_bytes hardware.npu.runtime_bytes hardware.npu.workspace_bytes hardware.cpu.dram_bytes hardware.remote.capacity_bytes hardware.ssd.capacity_bytes', {type: 'number', scale: 1e9, unit: 'GB'});
addInputSpecs('hardware.npu.hbm_Bps hardware.cpu.dram_Bps hardware.remote.bandwidth_Bps hardware.ssd.bandwidth_Bps', {type: 'number', scale: 1e9, unit: 'GB/s'});
addInputSpecs('hardware.cpu.flops', {type: 'number', scale: 1e12, unit: 'TFLOP/s'});
addInputSpecs('hardware.cpu.iops hardware.remote.iops hardware.ssd.iops', {type: 'number', unit: 'IOPS'});
addInputSpecs('hardware.remote.latency_s hardware.ssd.latency_s', {type: 'number', scale: 1e-6, unit: 'μs'});
['scale_up', 'scale_out', 'host_link'].forEach(link => {
  addInputSpecs('hardware.' + link + '.bandwidth_Bps', {type: 'number', scale: 1e9, unit: 'GB/s'});
  addInputSpecs('hardware.' + link + '.injection_Bps hardware.' + link + '.uplink_Bps', {type: 'number', scale: 1e9, unit: 'GB/s', nullable: true});
  addInputSpecs('hardware.' + link + '.latency_s', {type: 'number', scale: 1e-6, unit: 'μs'});
  addInputSpecs('hardware.' + link + '.efficiency', {type: 'number', scale: .01, unit: '%'});
});
addInputSpecs('targets.ttft_s targets.tpot_s', {type: 'number', scale: .001, unit: 'ms', nullable: true});
addInputSpecs('targets.step_time_s', {type: 'number', unit: 's', nullable: true});
addInputSpecs('targets.throughput_tokens_s', {type: 'number', unit: 'tokens/s', nullable: true});
addInputSpecs('model.architecture.tie_embeddings execution.sequence_parallel execution.reshard_after_forward execution.sync_each_microbatch execution.prefetch execution.features.optimizer_offload execution.features.activation_checkpointing execution.features.engram.enabled hardware.remote.enabled hardware.ssd.enabled', {type: 'boolean'});
inputEditSpecs.set('hardware.topology', {type: 'select', choices: ['full_mesh', 'mesh', 'clos', 'fat_tree', 'explicit']});
inputEditSpecs.set('hardware.routing_policy', {type: 'select', choices: ['shortest_hop', 'su_so']});
inputEditSpecs.set('model.attention_kind', {type: 'select', choices: ['mha', 'gqa', 'mqa', 'latent']});
inputEditSpecs.set('workload.scenario', {type: 'select', choices: ['inference', 'training']});
inputEditSpecs.set('execution.features.kv_offload', {type: 'select', choices: ['none', 'host', 'remote']});
inputEditSpecs.set('execution.optimizer.kind', {type: 'select', choices: ['adam', 'muon']});
function inputEditSpec(tokens) {
  const path = tokens.join('.');
  if (inputEditSpecs.has(path)) return inputEditSpecs.get(path);
  if (tokens[0] === 'hardware' && tokens[1] === 'npu' && tokens[2] === 'flops' && tokens.length === 4) return {type: 'number', scale: 1e12, unit: 'TFLOP/s'};
  if (tokens[0] === 'execution' && tokens[1] === 'precision' && tokens.length === 3) return {type: 'select', choices: ['fp4', 'fp8', 'fp16', 'bf16', 'fp32', 'int8'], nullable: tokens[2] === 'routed_expert_weight'};
  return null;
}
function inputEditorActive() { return inputEdit.editing && !state.jsonDirty && inputView.source === 'draft' && Boolean(state.config); }
function inputNumericText(value, scale) {
  const scaled = value / (scale || 1);
  const rounded = Number(scaled.toPrecision(15));
  return String(Math.abs(scaled - rounded) <= Number.EPSILON * Math.max(1, Math.abs(scaled)) * 2 ? rounded : scaled);
}
function inputFieldEditor(tokens, label) {
  if (!inputEditorActive()) return null;
  const spec = inputEditSpec(tokens), value = inputOwnValue(state.config, tokens);
  if (!spec) return null;
  if (inputDerivedField(tokens)) {
    const derived = el('span', 'input-derived-value', inputScalar(value) + ' · 由 architecture 推导'); derived.dataset.inputDerived = tokens[1]; return derived;
  }
  const box = el('div', 'input-field-editor'), key = inputTokenKey(tokens), pending = inputEdit.invalidFields.get(key);
  let control;
  if (spec.type === 'select' || spec.type === 'boolean') {
    control = el('select');
    const choices = spec.type === 'boolean' ? [['true', '是（true）'], ['false', '否（false）']] : spec.choices.map(choice => [choice, choice]);
    const emptyOption = el('option', '', value === undefined ? '未提供' : spec.nullable ? '未设置（null）' : '请选择');
    emptyOption.value = ''; emptyOption.disabled = !spec.nullable; control.append(emptyOption);
    choices.forEach(([raw, name]) => { const option = el('option', '', name); option.value = raw; control.append(option); });
    if (value != null && !choices.some(([raw]) => raw === String(value))) { const unknown = el('option', '', String(value)); unknown.value = String(value); control.append(unknown); }
    control.value = value == null ? '' : String(value);
  } else {
    control = el('input'); control.type = 'text';
    if (spec.type === 'number') { control.inputMode = spec.integer ? 'numeric' : 'decimal'; control.spellcheck = false; }
    control.value = pending ? pending.raw : value == null ? '' : spec.type === 'number' && typeof value === 'number' ? inputNumericText(value, spec.scale) : String(value);
    control.placeholder = spec.nullable ? '留空设为 null' : value === undefined ? '未提供' : '';
  }
  control.dataset.inputEditPath = key;
  control.setAttribute('aria-label', label + (spec.unit ? '，单位 ' + spec.unit : '') + '，' + inputTokenPath(tokens));
  control.disabled = state.busy;
  const message = el('span', 'input-field-error', pending ? pending.error : '');
  if (pending) control.setAttribute('aria-invalid', 'true');
  box.append(control); if (spec.unit) box.append(el('span', 'input-editor-unit', spec.unit)); box.append(message);
  function changeValue() {
    if (state.busy) return;
    const raw = control.value;
    try {
      let next;
      if (spec.type === 'boolean') {
        if (raw !== 'true' && raw !== 'false') throw new Error('请选择 true 或 false。');
        next = raw === 'true';
      } else if (spec.type === 'select') next = raw === '' && spec.nullable ? null : raw;
      else if (spec.type === 'number') {
        if (!raw.trim()) {
          if (!spec.nullable) throw new Error('请输入数字；空白不会自动变为 0。');
          next = null;
        } else {
          if (!/^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/.test(raw.trim())) throw new Error('数字尚未完整，请继续输入。');
          const parsed = Number(raw);
          if (!Number.isFinite(parsed) || !Number.isFinite(parsed * (spec.scale || 1))) throw new Error('数值必须为有限数字。');
          if (spec.integer && !Number.isSafeInteger(parsed)) throw new Error('请输入可精确表示的整数。');
          next = parsed * (spec.scale || 1);
        }
      } else next = raw;
      // Other invalid fields are retained while valid fields continue to synchronize.
      commitInputValue(tokens, next);
      control.removeAttribute('aria-invalid'); message.textContent = '';
      if (tokens.join('.') === 'workload.scenario') renderInputOverview();
    } catch (error) {
      inputEdit.invalidFields.set(key, {tokens: [...tokens], raw, error: errorText(error)});
      control.setAttribute('aria-invalid', 'true'); message.textContent = errorText(error);
      inputEditMessage('此字段尚未应用：' + inputTokenPath(tokens) + '。原始输入已保留；修正或放弃后才能运行。', true);
      updateRunButton(); syncInputEditorControls();
    }
  }
  control.addEventListener(spec.type === 'boolean' || spec.type === 'select' ? 'change' : 'input', changeValue);
  return box;
}
function inputJsonButton(tokens, label = '编辑 JSON') {
  if (!inputEditorActive()) return null;
  const button = el('button', 'text-button input-json-field-button', label); button.type = 'button'; button.dataset.inputJsonTokens = JSON.stringify(tokens);
  button.disabled = state.busy || inputDerivedField(tokens);
  button.addEventListener('click', () => openInputJsonEditor(tokens));
  return button;
}
function inputEditableValue(tokens, value) { return {inputTokens: tokens, display: value}; }
function inputExtraFacts(host, paths) {
  if (!inputEditorActive()) return;
  inputFacts(host, paths.map(([label, path]) => [label, inputOwnValue(state.config, path.split('.')), path]), 'input-inline-extra');
}
function syncInputEditorControls() {
  const editable = inputView.source === 'draft' && Boolean(state.config || state.jsonDirty);
  $('#edit-inputs-toggle').hidden = inputView.source !== 'draft';
  $('#edit-inputs-toggle').textContent = state.jsonDirty ? '应用现有 JSON 并编辑' : inputEdit.editing ? '完成编辑' : '编辑参数';
  $('#edit-inputs-toggle').disabled = state.busy || !editable;
  $('#validate-inputs').hidden = inputView.source !== 'draft';
  $('#validate-inputs').disabled = state.busy || !editable;
  $('#copy-input-snapshot').hidden = inputView.source !== 'job';
  $('#copy-input-snapshot').disabled = state.busy || inputView.loading || !inputView.snapshot || hasUnappliedInputEdits();
  $('#discard-input-pending').hidden = !hasUnappliedInputEdits();
  $('#discard-input-pending').disabled = state.busy;
  $('#input-source').disabled = state.busy;
  $('#input-job-select').disabled = state.busy || !$('#input-job-select').options.length;
  $('#retry-input-snapshot').disabled = state.busy || inputView.loading;
  $$('[data-input-edit-path]').forEach(control => { control.disabled = state.busy; });
  $$('[data-input-json-tokens]').forEach(button => { button.disabled = state.busy || inputDerivedField(JSON.parse(button.dataset.inputJsonTokens)); });
  $('#input-json-editor').disabled = state.busy;
  ['#apply-input-json', '#discard-input-json', '#close-input-json'].forEach(selector => { $(selector).disabled = state.busy; });
  $('#input-json-edit-panel').hidden = inputView.source !== 'draft' || !inputEdit.jsonOpen;
  if (inputView.source === 'draft' && hasUnappliedInputEdits()) $('#download-viewed-input').disabled = true;
  if (inputEdit.message) { text('#input-edit-message', inputEdit.message); $('#input-edit-message').className = inputEdit.messageError ? 'input-edit-error' : ''; }
}
function beginInputEditing() {
  clearTimeout(inputView.refreshTimer);
  try {
    if (inputEdit.editing && !state.jsonDirty) {
      requireAppliedInputEdits(); inputEdit.editing = false;
      inputEditMessage('编辑已完成。已应用的修改保留在当前 Config，可校验后运行。');
    } else {
      ensureInputEditorReady(); inputEdit.editing = true;
      inputEditMessage('编辑模式：有效字段即时同步；可以依次修改并行布局，完成后统一校验。高级 JSON 需要点击应用。');
    }
    renderInputOverview();
  } catch (error) { inputEditMessage(errorText(error) + ' 请修正实验页的完整 JSON 或未应用输入后重试。', true); }
}
function openInputJsonEditor(tokens) {
  try {
    ensureInputEditorReady();
    if (inputDerivedField(tokens)) throw new Error('此字段由 architecture 推导，请编辑结构字段。');
    if (inputEdit.jsonDirty && inputTokenKey(tokens) !== inputTokenKey(inputEdit.jsonTokens || [])) throw new Error('已有未应用的字段 JSON。请先应用或放弃该草稿，再编辑其他字段。');
    if (!inputEdit.jsonDirty) {
      inputEdit.jsonTokens = [...tokens];
      const value = inputOwnValue(state.config, tokens);
      inputEdit.jsonRaw = value === undefined ? 'null' : JSON.stringify(value, null, 2);
      $('#input-json-editor').value = inputEdit.jsonRaw;
    }
    inputEdit.jsonOpen = true;
    text('#input-json-edit-path', inputTokenPath(tokens)); text('#input-json-error', '');
    syncInputEditorControls();
    $('#input-json-edit-panel').scrollIntoView({block: 'start'});
    $('#input-json-editor').focus();
  } catch (error) { inputEditMessage(errorText(error), true); }
}
function applyInputJsonEdit() {
  try {
    ensureInputEditorReady();
    if (!inputEdit.jsonTokens) throw new Error('请先选择要编辑的字段。');
    const value = JSON.parse($('#input-json-editor').value);
    assertInputFinite(value);
    const tokens = [...inputEdit.jsonTokens];
    commitInputValue(tokens, value);
    inputEdit.jsonDirty = false; inputEdit.jsonRaw = $('#input-json-editor').value;
    // Applying a parent JSON replacement also resolves pending controls inside it.
    for (const [key, pending] of inputEdit.invalidFields) if (tokens.every((token, index) => pending.tokens[index] === token)) inputEdit.invalidFields.delete(key);
    inputEdit.jsonOpen = false; text('#input-json-error', '');
    renderInputOverview(); updateRunButton();
    inputEditMessage('字段 ' + inputTokenPath(tokens) + ' 已应用。实验表单与下一次仿真配置已同步。');
  } catch (error) { text('#input-json-error', errorText(error)); inputEditMessage('字段 JSON 尚未应用，草稿已保留。请修正后重试。', true); }
}
function discardInputPending() {
  if (state.busy) return;
  inputEdit.invalidFields.clear(); inputEdit.jsonDirty = false; inputEdit.jsonOpen = false; inputEdit.jsonTokens = null; inputEdit.jsonRaw = '';
  text('#input-json-error', '');
  inputEditMessage('已放弃未应用的无效文本和字段 JSON；此前已成功应用的配置修改继续保留。');
  renderInputOverview(); updateRunButton();
}
function copySnapshotForEditing() {
  try {
    if (state.busy || inputView.loading || !inputView.snapshot) throw new Error('请等待有效运行快照读取完成。');
    requireAppliedInputEdits();
    if (state.jsonDirty) throw new Error('实验页尚有未应用的完整 JSON，请先应用或明确放弃，避免复制快照时丢失草稿。');
    const config = clone(inputView.snapshot); assertInputFinite(config);
    inputView.source = 'draft'; inputView.version++; inputView.loading = false;
    state.config = config; state.jsonDirty = false; state.modified = true; state.presetId = '';
    inputEdit.editing = true;
    markModified(); renderForm(); $('#preset').value = '';
    text('#preset-description', '从运行 ' + inputView.jobId + ' 复制的独立当前配置；原始运行保持不变。');
    renderInputOverview();
    inputEditMessage('已深拷贝运行 ' + inputView.jobId + ' 的配置。后续修改只影响当前 Config 与下一次运行。');
  } catch (error) { inputEditMessage(errorText(error), true); }
}
function attachInputEditors() {
  $('#edit-inputs-toggle').addEventListener('click', beginInputEditing);
  $('#validate-inputs').addEventListener('click', handleValidation);
  $('#copy-input-snapshot').addEventListener('click', copySnapshotForEditing);
  $('#discard-input-pending').addEventListener('click', discardInputPending);
  $('#input-json-editor').addEventListener('input', () => {
    inputEdit.jsonRaw = $('#input-json-editor').value; inputEdit.jsonDirty = true;
    inputEditMessage('字段 JSON 草稿尚未应用；应用或放弃后才能运行。', true); updateRunButton();
  });
  $('#apply-input-json').addEventListener('click', applyInputJsonEdit);
  $('#discard-input-json').addEventListener('click', () => {
    if (state.busy) return;
    inputEdit.jsonDirty = false; inputEdit.jsonOpen = false; inputEdit.jsonTokens = null; inputEdit.jsonRaw = '';
    text('#input-json-error', ''); updateRunButton(); syncInputEditorControls();
    inputEditMessage('已放弃本字段的未应用 JSON，当前 Config 未受影响。');
  });
  $('#close-input-json').addEventListener('click', () => {
    inputEdit.jsonOpen = false; syncInputEditorControls();
    inputEditMessage(inputEdit.jsonDirty ? '字段 JSON 已收起但仍保留。再次点击该字段编辑可继续；当前禁止提交仿真。' : '字段编辑器已收起。', inputEdit.jsonDirty);
  });
}

function refreshInputLiveViews() {
  if (!inputEditorActive()) return;
  $$('[data-input-tree-tokens]').forEach(detail => { if (detail.refreshInputTree) detail.refreshInputTree(); });
  $$('[data-input-raw-tokens]').forEach(node => { node.textContent = inputScalar(inputOwnValue(state.config, JSON.parse(node.dataset.inputRawTokens))); });
  $$('[data-input-display-path]').forEach(node => {
    const path = node.dataset.inputDisplayPath, target = $('.input-display-value', node);
    if (!target) return;
    if (path.includes(' / ')) target.textContent = path.split(' / ').map(part => inputScalar(getPath(state.config, part))).join(' / ');
    else if (path === 'workload.input_lengths' || path === 'workload.output_lengths') target.textContent = inputLengthSummary(getPath(state.config, path));
  });
  const h = inputObject(state.config.hardware), e = inputObject(state.config.execution), w = inputObject(state.config.workload);
  $$('[data-input-computed-label]').forEach(node => {
    const target = $('.input-display-value', node); if (!target) return;
    if (node.dataset.inputComputedLabel === 'Scale-up 域数') target.textContent = inputPositiveInt(h.npu_count) && inputPositiveInt(h.su_size) && h.npu_count % h.su_size === 0 ? number(h.npu_count / h.su_size, 0) + '（由 NPU / SU 卡数推导）' : '无法推导：需要可整除的正整数 NPU / SU 卡数';
    if (node.dataset.inputComputedLabel === 'SP 并行度') target.textContent = sequenceParallelInfo(state.config).label;
    if (node.dataset.inputComputedLabel === '主机数') target.textContent = inputPositiveInt(h.npu_count) && inputPositiveInt(h.npus_per_host) ? number(Math.ceil(h.npu_count / h.npus_per_host), 0) + '（向上取整）' : '无法推导：NPU / 每主机卡数未就绪';
  });
  $$('[data-input-live-note]').forEach(node => {
    if (node.dataset.inputLiveNote === 'parallel') node.textContent = parallelInputNote(state.config);
    if (node.dataset.inputLiveNote === 'training-batch') node.textContent = [w.batch_size, w.gradient_accumulation, e.dp].every(inputPositiveInt) ? '每优化器步全局样本数 = Batch × 梯度累积 × DP = ' + number(w.batch_size * w.gradient_accumulation * e.dp, 0) + '。' : '全局样本数需要有效的 Batch、梯度累积和 DP 才能推导。';
    if (node.dataset.inputLiveNote === 'inference-batch') node.textContent = inputPositiveInt(w.batch_size) && inputPositiveInt(e.dp) ? '配置请求总数 = 每副本 Batch × DP = ' + number(w.batch_size * e.dp, 0) + '。这是固定负载，不包含请求到达排队。' : '请求总数需有效 Batch 与 DP 才能推导。';
    if (node.dataset.inputLiveNote === 'length-count') node.textContent = Array.isArray(w.input_lengths) && Array.isArray(w.output_lengths) && w.input_lengths.length === w.batch_size && w.output_lengths.length === w.batch_size ? '显式长度列表与每副本 Batch 数量一致；保留每条请求的差异。' : '显式列表数量与 Batch 不一致或仍未展开；此页面不补齐、不覆盖请求长度。';
  });
}


// Both diagrams use one explicit source; job snapshots never borrow the editable draft.
const simulationView = {source: 'draft', snapshot: null, jobId: '', loading: false, error: '', version: 0};
async function loadSimulationSnapshot(job) {
  const version = ++simulationView.version;
  simulationView.source = 'job'; simulationView.jobId = job.id;
  simulationView.snapshot = null; simulationView.loading = true; simulationView.error = '';
  renderSimulationViews();
  try {
    const response = await api('/api/jobs/' + encodeURIComponent(job.id) + '/config');
    if (version !== simulationView.version || state.currentJob?.id !== job.id) return;
    if (!response.config || typeof response.config !== 'object' || Array.isArray(response.config)) throw new Error('该运行没有有效配置快照。');
    assertInputFinite(response.config);
    simulationView.snapshot = response.config;
  } catch (error) {
    if (version === simulationView.version) simulationView.error = errorText(error);
  } finally {
    if (version === simulationView.version) { simulationView.loading = false; renderSimulationViews(); }
  }
}
function renderSimulationViews() {
  const jobView = simulationView.source === 'job';
  const select = $('#simulation-source'); if (!select) return;
  select.value = simulationView.source;
  $('#simulation-job-option').disabled = !state.currentJob;
  $('#simulation-retry').hidden = !(jobView && simulationView.error);
  let config, note, mode = $('#run-mode').value || 'simulate';
  if (jobView) {
    config = simulationView.jobId === state.currentJob?.id ? simulationView.snapshot : null;
    mode = state.currentJob?.mode || mode;
    note = simulationView.loading ? '正在读取运行快照…' : simulationView.error ? '快照读取失败：' + simulationView.error : '运行 ' + simulationView.jobId + ' 的提交配置 · 拓扑与执行计划共用此快照';
    if (simulationView.loading || simulationView.error) config = null;
  } else {
    try { config = currentInputDraft(); note = '当前编辑配置 · 修改参数后同步生成拓扑与执行计划'; }
    catch (error) { config = null; note = '当前配置暂不可预览：' + errorText(error); }
    if (state.jsonDirty || hasUnappliedInputEdits()) note += ' · 含未校验或未应用内容，提交前请完成校验';
  }
  text('#simulation-source-note', note);
  text('#topology-source', jobView ? '运行快照' : '当前草稿');
  if (config) {
    try {
      renderTopology(config);
      renderExecutionPlan(config, mode, jobView && !simulationView.loading && !simulationView.error && simulationView.jobId === state.currentJob?.id ? state.result : null);
    } catch (error) {
      config = null; note = '配置字段暂不可预览，请完成校验：' + errorText(error);
      text('#simulation-source-note', note);
    }
  }
  if (!config) {
    empty($('#topology-visual'), note);
    for (const id of ['topology-npu','topology-su','topology-hbm','topology-bandwidth']) text('#' + id, '—');
    text('#topology-type', '等待配置'); text('#topology-note', '配置可用后生成拓扑。');
    renderExecutionPlan(null, mode);
  }
}
function attachSimulationViews() {
  $('#simulation-source').addEventListener('change', event => {
    simulationView.source = event.target.value === 'job' && state.currentJob ? 'job' : 'draft';
    if (simulationView.source === 'job' && simulationView.jobId !== state.currentJob.id) loadSimulationSnapshot(state.currentJob);
    else renderSimulationViews();
  });
  $('#simulation-retry').addEventListener('click', () => { if (state.currentJob) loadSimulationSnapshot(state.currentJob); });
}

initialize();
