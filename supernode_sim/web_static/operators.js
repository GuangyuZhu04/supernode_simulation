'use strict';

// Exact ledgers and observed event samples have deliberately separate views.
const operatorCosts = [
  ['compute_s', '计算', 'compute'], ['communication_s', '通信', 'communication'],
  ['memory_access_s', '访存', 'memory'], ['launch_s', '启动', 'launch'],
  ['synchronization_wait_s', '同步等待', 'sync'], ['scheduling_wait_s', '调度等待', 'schedule'],
  ['memory_wait_s', '内存准入', 'admission'], ['resource_contention_s', '资源争用', 'contention'],
  ['unattributed_service_s', '校准服务（未细分）', 'unattributed'], ['other_s', '其他', 'other'],
];
const operatorMs = value => number(value == null ? null : value * 1000, 5);
const operatorPct = value => value == null || !Number.isFinite(value) ? '—' : number(value, 3) + '%';
const operatorPlayback = {timer: null, onStop: null};

function operatorSelect(label, choices) {
  const select = el('select'); select.setAttribute('aria-label', label);
  choices.forEach(([value, title]) => { const option = el('option', '', title); option.value = String(value); select.append(option); });
  select.value = String(choices[0]?.[0] ?? '');
  return select;
}
function operatorField(label, control) {
  const field = el('label', 'operator-field'); field.append(el('span', '', label), control); return field;
}
function operatorButton(title, action, className = 'text-button') {
  const button = el('button', className, title); button.type = 'button'; button.addEventListener('click', action); return button;
}
function operatorFacts(host, facts) {
  const list = el('dl', 'operator-facts');
  facts.forEach(([label, value]) => { const item = el('div'); item.append(el('dt', '', label), el('dd', '', value ?? '—')); list.append(item); });
  host.append(list);
}
function operatorStrip(record) {
  const bar = el('div', 'operator-cost-strip');
  operatorCosts.forEach(([key, label, color]) => {
    const share = record.component_shares_pct?.[key];
    if (!(share > 0)) return;
    const segment = el('span', 'operator-cost-' + color); segment.style.width = Math.min(100, share) + '%';
    segment.title = label + ' · ' + operatorMs(record.components_s?.[key]) + ' ms · ' + operatorPct(share);
    bar.append(segment);
  });
  bar.setAttribute('aria-label', operatorCosts.map(([key, label]) => label + ' ' + operatorPct(record.component_shares_pct?.[key])).join('；'));
  return bar;
}
function operatorLegend(host) {
  const legend = el('div', 'operator-cost-legend');
  operatorCosts.forEach(([, label, color]) => { const item = el('span'); item.append(el('i', 'operator-cost-' + color), el('span', '', label)); legend.append(item); });
  host.append(legend);
}

function renderOperatorBreakdown(host, scenario) {
  const result = state.result, metrics = result.metrics || {};
  const training = scenario === 'training' || !!metrics.training_latency_breakdown;
  const source = (training ? metrics.training_latency_breakdown : metrics.latency_breakdown) || {};
  const choices = training ? [['step', '平均训练步'], ['first_step', '首个训练步'], ['steady_step', '稳态平均训练步（第 2 步起）']] : [['ttft', 'TTFT'], ['tpot', 'TPOT（TPOP）']];
  const panel = el('section', 'operator-analysis'); host.append(panel);
  const heading = el('div', 'operator-section-heading');
  const select = operatorSelect('算子占比统计窗口', choices);
  heading.append(el('h3', '', '算子级结果 · 耗时与内部开销'), operatorField('统计窗口', select)); panel.append(heading);
  panel.append(el('p', 'chart-caption', '占窗口比例以窗口总时延为分母；算子内比例以该算子归因耗时为分母。计算与访存重叠时归到限制资源。仅列对关键路径有时间贡献的算子；未出现不表示未执行。'));
  const content = el('div'); panel.append(content);
  const drawWindow = () => {
    content.replaceChildren();
    const record = source[select.value], breakdown = record?.operator_breakdown;
    if (result.mode === 'estimate' || !breakdown) {
      empty(content, '没有可用算子归因，不能视为零。' + (source[select.value + '_unavailable_reason'] || source.unavailable_reason || '仅估算、旧报告或该窗口未定义。')); return;
    }
    const operators = breakdown.operators || [];
    const stats = el('div', 'operator-summary');
    operatorFacts(stats, [['窗口时延', operatorMs(breakdown.total_s) + ' ms'], ['归因算子', number(operators.length, 0)],
      ['统计对象', record.replica != null ? 'DP ' + record.replica + ' / sequence ' + record.sequence : choices.find(([key]) => key === select.value)[1]],
      ['归因口径', '关键依赖路径 · 不累加并发 rank']]);
    stats.append(operatorStrip(breakdown)); operatorLegend(stats); content.append(stats);
    const costs = el('details', 'operator-window-costs'); costs.append(el('summary', '', '窗口全部开销分项 · ms / %'));
    resultTable(costs, ['窗口分项', '耗时 / ms', '占窗口'], [
      ['总时延', operatorMs(breakdown.total_s), breakdown.total_s > 0 ? '100%' : '—'],
      ...operatorCosts.map(([key, title]) => [title, operatorMs(breakdown.components_s?.[key]), operatorPct(breakdown.component_shares_pct?.[key])]),
    ]); content.append(costs);
    const controls = el('div', 'operator-toolbar');
    const values = key => [['', '全部'], ...[...new Set(operators.map(row => row[key] ?? 'Unassigned'))].sort().map(value => [value, value])];
    const phase = operatorSelect('算子结果阶段', values('phase')), component = operatorSelect('算子结果组件', values('component'));
    const search = el('input'); search.type = 'search'; search.placeholder = '搜索算子名'; search.setAttribute('aria-label', '搜索算子结果');
    const sort = operatorSelect('算子结果排序', [['duration', '耗时从高到低'], ['ascending', '耗时从低到高'], ['name', '算子名称']]);
    controls.append(operatorField('阶段', phase), operatorField('组件', component), operatorField('算子', search), operatorField('排序', sort)); content.append(controls);
    const chart = el('div', 'operator-ranking'), summary = el('p', 'chart-caption'), table = el('div', 'matrix-wrap'), pager = el('div', 'operator-pager'), detail = el('div', 'operator-inspector');
    detail.hidden = true; detail.setAttribute('aria-live', 'polite');
    content.append(summary, chart, table, pager, detail);
    let page = 0;
    const inspect = row => {
      detail.hidden = false; detail.replaceChildren(el('h4', '', row.operator));
      operatorFacts(detail, [['阶段 / 组件', row.phase + ' / ' + row.component], ['归因耗时', operatorMs(row.total_s) + ' ms'], ['占窗口', operatorPct(row.share_pct)]]);
      operatorFacts(detail, operatorCosts.map(([key, title]) => [title, operatorMs(row.components_s?.[key]) + ' ms · ' + operatorPct(row.component_shares_pct?.[key])]));
      detail.append(operatorButton('在执行记录中查找此算子', () => {
        renderOperatorTimeline(result, {phase: row.phase, component: row.component, operator: row.operator});
        $('#operator-process-panel')?.scrollIntoView({behavior: 'smooth', block: 'start'});
      }));
    };
    const drawRows = () => {
      const query = search.value.trim().toLowerCase();
      const rows = operators.filter(row => (!phase.value || row.phase === phase.value) && (!component.value || row.component === component.value)
        && (!query || String(row.operator).toLowerCase().includes(query))).slice();
      rows.sort(sort.value === 'name' ? (a, b) => String(a.operator).localeCompare(String(b.operator)) : (a, b) => (sort.value === 'ascending' ? 1 : -1) * (a.total_s - b.total_s));
      page = Math.min(page, Math.max(0, Math.ceil(rows.length / 50) - 1));
      chart.replaceChildren(); table.replaceChildren(); pager.replaceChildren(); detail.hidden = true;
      const share = breakdown.total_s > 0 ? rows.reduce((sum, row) => sum + row.total_s, 0) / breakdown.total_s * 100 : null;
      summary.textContent = '筛选 ' + rows.length + ' / ' + operators.length + ' 个算子 · 占原窗口 ' + operatorPct(share) + '。筛选不改变比例分母；图示耗时最高的 12 项，完整排行可翻页查看。';
      const top = rows.slice().sort((a, b) => b.total_s - a.total_s).slice(0, 12);
      const max = Math.max(...top.map(row => row.total_s), 1e-15);
      top.forEach(row => {
        const item = operatorButton('', () => inspect(row), 'operator-rank-row'); item.title = row.phase + ' / ' + row.component + ' / ' + row.operator;
        const track = el('div', 'operator-rank-track'), strip = operatorStrip(row); strip.style.width = Math.max(0, row.total_s / max * 100) + '%'; track.append(strip);
        item.append(el('span', 'operator-rank-name', row.operator), track, el('span', 'mono', operatorMs(row.total_s) + ' ms'), el('span', 'mono', operatorPct(row.share_pct))); chart.append(item);
      });
      if (!rows.length) { empty(table, '没有符合筛选条件的算子。'); return; }
      const groups = [['compute_s'], ['communication_s'], ['memory_access_s'], ['launch_s'],
        ['synchronization_wait_s', 'scheduling_wait_s', 'memory_wait_s'], ['resource_contention_s'], ['unattributed_service_s', 'other_s']];
      const visible = rows.slice(page * 50, (page + 1) * 50);
      resultTable(table, ['阶段', '组件', '算子', '耗时 / ms', '占窗口', '计算', '通信', '访存', '启动', '等待', '争用', '未细分及其他'],
        visible.map(row => [row.phase, row.component, operatorButton(row.operator, () => inspect(row)), operatorMs(row.total_s), operatorPct(row.share_pct),
          ...groups.map(keys => operatorPct(row.total_s > 0 && keys.every(key => row.component_shares_pct?.[key] != null)
            ? keys.reduce((sum, key) => sum + row.component_shares_pct[key], 0) : null))]));
      const previous = operatorButton('上一页', () => { page--; drawRows(); }), next = operatorButton('下一页', () => { page++; drawRows(); });
      previous.disabled = page === 0; next.disabled = (page + 1) * 50 >= rows.length;
      pager.append(previous, el('span', '', (page * 50 + 1) + '–' + Math.min(rows.length, (page + 1) * 50) + ' / ' + rows.length), next);
    };
    [phase, component, sort].forEach(control => control.addEventListener('change', () => { page = 0; drawRows(); }));
    search.addEventListener('input', () => { page = 0; drawRows(); }); drawRows();
    content.append(el('p', 'chart-caption', '点击算子查看全部开销项。未筛选的算子占窗口比例合计为 100%；每行内部比例合计为 100%（舍入可能有尾差）。零耗时比例为“—”。完整数值见 operator_breakdown.csv。'));
  };
  select.addEventListener('change', drawWindow); drawWindow();
}

function stopOperatorPlayback() {
  if (operatorPlayback.timer != null) clearTimeout(operatorPlayback.timer);
  operatorPlayback.timer = null;
  const stop = operatorPlayback.onStop; operatorPlayback.onStop = null; if (stop) stop();
}
function operatorEventKind(event) {
  const meta = event.metadata || {};
  if (meta.operator_trace_completion) return 'completion';
  if (meta.kernel || meta.flops) return 'compute';
  if (meta.type === 'transfer' || meta.collective || meta.semantic) return 'communication';
  return 'other';
}
function operatorEventName(event) {
  const meta = event.metadata || {};
  return meta.kernel || (meta.operator_trace_completion ? '父算子完成点' : meta.semantic || meta.operation || meta.collective || meta.category || meta.type || meta.kind || event.id);
}
function operatorEventDevices(event) {
  const meta = event.metadata || {};
  const devices = (event.slots || []).map(slot => String(slot).split('.')[0]);
  [meta.src, meta.dst].forEach(value => { if (value != null) devices.push(String(value)); });
  if (meta.rank != null) devices.push('rank ' + meta.rank);
  return [...new Set(devices)];
}
function operatorEventCycle(event) {
  const meta = event.metadata || {};
  return [['step', 'step'], ['token_index', 'token'], ['token', 'token'], ['training_iteration', 'iteration'], ['round_index', 'round']]
    .filter(([key]) => meta[key] != null).map(([key, label]) => label + ' ' + meta[key]);
}
function renderOperatorEventDetail(host, event, onParent) {
  const meta = event.metadata || {};
  host.replaceChildren(el('h4', '', operatorEventName(event)), el('p', 'operator-event-id', event.id));
  operatorFacts(host, [['阶段 / 组件', (meta.phase ?? '未标注') + ' / ' + (meta.component ?? 'Unassigned')],
    ['设备 / 执行槽', operatorEventDevices(event).join('、') || '未标注'], ['层 / 迭代', (meta.layer ?? '—') + ' / ' + (operatorEventCycle(event).join(' · ') || '—')],
    ['开始 → 完成', operatorMs(event.start_s) + ' → ' + operatorMs(event.end_s) + ' ms'], ['事件跨度', operatorMs(event.duration_s) + ' ms'],
    ['启动开销', operatorMs(event.startup_s) + ' ms'], ['依赖就绪时刻', operatorMs(event.dependency_ready_s) + ' ms'],
    ['调度等待', operatorMs(event.slot_wait_s) + ' ms'], ['内存准入等待', operatorMs(event.memory_wait_s) + ' ms'],
    ['FLOPs', number(meta.flops, 0)], ['Matmul', meta.matmul ? JSON.stringify(meta.matmul) : '—'], ['局部 workspace', meta.operator_workspace_bytes == null ? '—' : number(meta.operator_workspace_bytes, 0) + ' B']]);
  if (meta.trace_parent_event) {
    host.append(el('p', 'chart-caption', '父 kernel ' + (meta.original_kernel ?? '—') + ' → 子算子 #' + meta.operation_index + ' · trace ' + meta.trace_id));
    host.append(operatorButton('查看父算子的展开链', () => onParent(meta.trace_parent_event)));
  }
  if (meta.training_role) host.append(el('p', 'chart-caption', '训练任务 ' + (meta.training_task ?? '—') + ' / ' + meta.training_role + ' / ' + (meta.training_stage ?? '—')));
  [['资源工作量（引擎保留项）', event.work], ['工作负载形状', meta.workload_shape], ['Kernel 实现', meta.implementation], ['计时校准', meta.calibration], ['展开来源', meta.trace_source]].forEach(([title, value]) => {
    if (value == null) return;
    const details = el('details'); details.append(el('summary', '', title), el('pre', '', typeof value === 'string' ? value : JSON.stringify(value, null, 2))); host.append(details);
  });
  host.append(el('p', 'chart-caption', '事件跨度含启动与执行期间争用，调度 / 内存准入等待发生在开始之前。此处不是关键路径归因，不能累加并发事件推算总时延。' + (event.details_truncated ? ' 本条元数据已截取。' : '')));
}

function renderOperatorTimeline(result, focus = null) {
  stopOperatorPlayback();
  const host = $('#trace-chart'); host.replaceChildren();
  const trace = result?.trace || {}, events = trace.events || [], execution = result?.operator_execution, groups = execution?.groups || [];
  if (trace.retained === false) {
    text('#trace-note', '流式执行保留完整算子耗时归因；逐事件轨迹未保留。');
    empty(host, '本次使用分块执行以控制内存，未保留逐算子回放轨迹。阶段时间见执行摘要，完整分块见 streaming_chunks.csv。'); return;
  }
  text('#trace-note', '实际记录回放 · 横轴为仿真时间，播放位置不代表当前任务实时进度。全局时间线保留 ' + events.length + ' / ' + (trace.total_events ?? events.length) + ' 个事件' + (trace.sampled ? '（均匀采样）' : '') + '；算子归因使用完整统计，不从时间线采样重算。');
  if (!result || result.mode === 'estimate' || isServingResult(result)) { empty(host, '此模式没有可回放的算子执行轨迹。解析估算仅提供必要下界。'); return; }
  if (result.execution_summary?.complete === false) host.append(el('p', 'execution-partial-note', '本次执行未完成；这里只回放中断前已完成的事件，不代表完整任务。'));
  if (!events.length && !groups.length) { host.append(el('p', 'muted-empty', '没有可展示的执行记录。旧结果或精简导出可能未保留逐算子轨迹。')); return; }
  if (execution) {
    const pipeline = el('div', 'operator-pipeline');
    [['构建事件图', number(result.execution_summary?.planned_events ?? result.estimate?.event_count, 0) + ' 计划事件'],
      ['展开子算子', number(execution.expanded_operations, 0) + ' 个已执行子算子'],
      ['资源调度', number(result.execution_summary?.completed_events ?? result.simulation?.completed_events, 0) + ' 已完成事件'],
      ['关键路径归因', '窗口耗时见上方算子结果']].forEach(([title, detail], index) => {
        const step = el('div'); step.append(el('span', 'operator-pipeline-index', String(index + 1).padStart(2, '0')), el('strong', '', title), el('small', '', detail)); pipeline.append(step);
      }); host.append(pipeline);
  }
  const chain = operatorSelect('算子展开链', [['', '全局事件时间线'], ...groups.map((group, index) => [String(index), (group.phase || '未标注') + ' · ' + group.original_kernel + ' · ' + group.parent_event])]);
  host.append(operatorField('过程视图', chain));
  const scopeNote = el('p', 'chart-caption'); host.append(scopeNote);
  const chainFlow = el('div', 'operator-chain-flow'); chainFlow.hidden = true; host.append(chainFlow);
  const controls = el('div', 'operator-toolbar'); host.append(controls);
  const filters = {};
  [['phase', '阶段'], ['component', '组件'], ['device', '设备 / Rank'], ['layer', '层'], ['cycle', 'Token / 训练步'], ['kind', '事件类型']].forEach(([key, title]) => {
    filters[key] = operatorSelect('过程' + title, [['', '全部']]); controls.append(operatorField(title, filters[key]));
  });
  const search = el('input'); search.type = 'search'; search.placeholder = '算子名或事件 ID'; search.setAttribute('aria-label', '搜索执行事件');
  controls.append(operatorField('搜索', search), operatorButton('重置筛选', () => { Object.values(filters).forEach(control => { control.value = ''; }); search.value = ''; redraw(); }));
  const transport = el('div', 'operator-transport'), play = operatorButton('播放', () => togglePlayback()), previousEvent = operatorButton('上一事件', () => step(-1)), nextEvent = operatorButton('下一事件', () => step(1));
  const slider = el('input'); slider.type = 'range'; slider.min = '0'; slider.max = '1000'; slider.step = '1'; slider.value = '0'; slider.setAttribute('aria-label', '仿真时间回放位置');
  const speed = operatorSelect('回放速度', [['1', '1×'], ['0.5', '0.5×'], ['2', '2×'], ['4', '4×']]);
  const zoom = operatorSelect('时间轴范围', [['full', '完整执行时间'], ['filtered', '筛选事件范围']]);
  const clock = el('span', 'mono'); transport.append(play, previousEvent, nextEvent, slider, clock, speed, zoom); host.append(transport);
  const note = el('p', 'chart-caption'), lanes = el('div', 'operator-lanes'), pager = el('div', 'operator-pager'), detail = el('aside', 'operator-inspector');
  detail.setAttribute('aria-label', '执行事件详情'); host.append(note, lanes, pager, detail);
  let base = [], filtered = [], page = 0, selected = null, rows = [], start = 0, end = 0, time = 0;
  const getValues = (event, key) => {
    if (key === 'device') return operatorEventDevices(event);
    if (key === 'cycle') return operatorEventCycle(event);
    if (key === 'kind') return [operatorEventKind(event)];
    if (key === 'phase') return [String(event.metadata?.phase || 'unspecified')];
    if (key === 'component') return [String(event.metadata?.component || 'Unassigned')];
    const value = event.metadata?.[key]; return value == null ? [] : [String(value)];
  };
  const selectEvent = event => {
    selected = event;
    renderOperatorEventDetail(detail, event, parent => {
      const index = groups.findIndex(group => group.parent_event === parent);
      if (index < 0) { toast('此父算子的完整展开链未保留在网页摘要中；请查看该运行的导出记录。'); return; }
      chain.value = String(index); setBase();
    });
    rows.forEach(row => { row.node.classList.toggle('selected', row.event === selected); row.button.setAttribute('aria-pressed', row.event === selected ? 'true' : 'false'); });
  };
  const updateTime = value => {
    time = Math.min(end, Math.max(start, value));
    slider.value = String(end > start ? Math.round((time - start) / (end - start) * 1000) : 0);
    clock.textContent = operatorMs(time) + ' / ' + operatorMs(end) + ' ms';
    slider.setAttribute('aria-valuetext', operatorMs(time) + ' 毫秒');
    rows.forEach(({event, node, status, lane}) => {
      const finished = time >= event.end_s, running = !finished && time >= event.start_s;
      node.classList.toggle('is-running', running); node.classList.toggle('is-finished', finished);
      status.textContent = finished ? '已完成' : running ? '执行中' : '待开始';
      lane.style.setProperty ? lane.style.setProperty('--operator-cursor', (end > start ? (time - start) / (end - start) * 100 : 0) + '%') : null;
    });
  };
  const paint = () => {
    lanes.replaceChildren(); pager.replaceChildren(); rows = [];
    if (!filtered.length) {
      empty(lanes, '当前网页记录中没有匹配事件。采样未命中不表示算子没有执行；可尝试选择展开链。'); detail.replaceChildren();
    } else {
      const axis = el('div', 'operator-axis'); [0, .25, .5, .75, 1].forEach(f => axis.append(el('span', '', operatorMs(start + (end - start) * f) + ' ms'))); lanes.append(axis);
      filtered.slice(page * 40, (page + 1) * 40).forEach(event => {
        const row = el('div', 'operator-event-row'), meta = event.metadata || {}, label = operatorButton(operatorEventName(event), () => { selectEvent(event); updateTime(event.start_s); }, 'operator-event-label');
        label.title = event.id; label.append(el('small', '', [meta.phase, meta.component, meta.layer != null ? 'L' + meta.layer : '', ...operatorEventCycle(event)].filter(v => v !== '' && v != null).join(' · ')));
        const lane = el('div', 'operator-event-lane'), bar = operatorButton('', () => { selectEvent(event); updateTime(event.start_s); }, 'operator-event-bar operator-kind-' + operatorEventKind(event));
        bar.setAttribute('aria-label', operatorEventName(event) + '，' + operatorMs(event.start_s) + ' 至 ' + operatorMs(event.end_s) + ' 毫秒');
        bar.style.left = (end > start ? (event.start_s - start) / (end - start) * 100 : 0) + '%';
        bar.style.width = (end > start ? (event.end_s - event.start_s) / (end - start) * 100 : 0) + '%';
        bar.title = event.id + '\n' + operatorMs(event.start_s) + ' → ' + operatorMs(event.end_s) + ' ms';
        lane.append(bar, el('i', 'operator-cursor')); const status = el('span', 'operator-event-status');
        row.append(label, lane, status); lanes.append(row); rows.push({event, node: row, status, lane, button: label});
      });
      const previous = operatorButton('上一页', () => { page--; paint(); }), next = operatorButton('下一页', () => { page++; paint(); });
      previous.disabled = page === 0; next.disabled = (page + 1) * 40 >= filtered.length;
      pager.append(previous, el('span', '', (page * 40 + 1) + '–' + Math.min(filtered.length, (page + 1) * 40) + ' / ' + filtered.length), next);
      selectEvent(filtered.includes(selected) ? selected : filtered[page * 40]);
    }
    play.disabled = !filtered.length || end <= start; previousEvent.disabled = nextEvent.disabled = !filtered.length;
    slider.disabled = play.disabled;
    note.textContent = '符合筛选的记录 ' + filtered.length + ' / ' + base.length + ' · 每页 40 条，按开始时间排序。颜色：青色计算 / 蓝色通信 / 灰色完成点与其他；同一时刻可有多个事件执行。';
    updateTime(time);
  };
  const redraw = () => {
    stopOperatorPlayback(); page = 0;
    const query = search.value.trim().toLowerCase();
    filtered = base.filter(event => Object.entries(filters).every(([key, control]) => !control.value || getValues(event, key).includes(control.value))
      && (!query || (String(operatorEventName(event)) + ' ' + event.id).toLowerCase().includes(query)))
      .filter(event => Number.isFinite(event.start_s) && Number.isFinite(event.end_s) && event.end_s >= event.start_s)
      .sort((a, b) => a.start_s - b.start_s || a.end_s - b.end_s || String(a.id).localeCompare(String(b.id)));
    const bounds = zoom.value === 'filtered' ? filtered : base;
    start = zoom.value === 'filtered' && bounds.length ? Math.min(...bounds.map(event => event.start_s)) : 0;
    end = Math.max(start, ...bounds.map(event => Number(event.end_s) || 0), zoom.value === 'full' ? result.simulation?.duration_s || 0 : 0);
    time = start; paint();
  };
  const setBase = () => {
    const group = chain.value === '' ? null : groups[Number(chain.value)];
    base = group ? group.events || [] : events;
    chainFlow.replaceChildren(); chainFlow.hidden = !group;
    if (group) {
      chainFlow.append(el('span', 'operator-chain-parent', group.original_kernel || group.parent_event));
      let previous = -1;
      base.slice().sort((a, b) => a.metadata.operation_index - b.metadata.operation_index).forEach(event => {
        const index = event.metadata.operation_index;
        chainFlow.append(el('span', 'operator-chain-arrow', index === previous + 1 ? '→' : '…'));
        const chip = operatorButton('#' + index + ' ' + operatorEventName(event), () => {
          Object.values(filters).forEach(control => { control.value = ''; }); search.value = ''; redraw();
          selected = event; page = Math.floor(filtered.indexOf(event) / 40); time = event.start_s; paint();
        }, 'operator-chain-step');
        chip.title = event.id; chainFlow.append(chip); previous = index;
      });
      if (group.truncated) chainFlow.append(el('span', 'operator-chain-arrow', '… 未全部保留'));
    }
    scopeNote.textContent = group ? '父事件 ' + group.parent_event + ' · ' + group.original_kernel + ' → ' + group.event_count + ' 个已执行子算子；当前保留 ' + base.length + ' 个' + (group.truncated ? '（前缀截取）' : '') + '。按真实开始时间显示，不补造缺失算子。'
      : (execution ? '独立保留最先观测到的 ' + groups.length + ' / ' + execution.parent_count + ' 个父算子展开链，每链最多 ' + execution.operations_per_group + ' 项。' : '该历史结果未保留独立展开链，可浏览已保存的全局事件。') + ' 事件总览与展开链为不同浏览样本。';
    Object.entries(filters).forEach(([key, control]) => {
      control.replaceChildren(); const values = [...new Set(base.flatMap(event => getValues(event, key)))].sort((a, b) => a.localeCompare(b, 'zh-CN', {numeric: true}));
      [['', '全部'], ...values.map(value => [value, ({compute: '计算', communication: '通信', completion: '父算子完成点', other: '其他'})[value] || value])].forEach(([value, title]) => { const option = el('option', '', title); option.value = value; control.append(option); });
      control.value = '';
    });
    search.value = ''; selected = null; redraw();
  };
  const step = direction => {
    stopOperatorPlayback(); if (!filtered.length) return;
    let index = filtered.indexOf(selected); index = Math.max(0, Math.min(filtered.length - 1, index + direction));
    selected = filtered[index]; page = Math.floor(index / 40); time = selected.start_s; paint();
  };
  const togglePlayback = () => {
    if (operatorPlayback.onStop) { stopOperatorPlayback(); return; }
    if (time >= end) updateTime(start);
    play.textContent = '暂停'; operatorPlayback.onStop = () => { play.textContent = '播放'; };
    let last = Date.now();
    const tick = () => {
      if (state.result !== result || document.hidden) { stopOperatorPlayback(); return; }
      const now = Date.now(); updateTime(time + (now - last) / 10000 * (end - start) * Number(speed.value)); last = now;
      if (time >= end) { stopOperatorPlayback(); return; }
      operatorPlayback.timer = setTimeout(tick, 50);
    };
    operatorPlayback.timer = setTimeout(tick, 50);
  };
  slider.addEventListener('input', () => { stopOperatorPlayback(); updateTime(start + Number(slider.value) / 1000 * (end - start)); });
  Object.values(filters).forEach(control => control.addEventListener('change', redraw)); search.addEventListener('input', redraw);
  chain.addEventListener('change', setBase); zoom.addEventListener('change', redraw); setBase();
  if (focus) {
    ['phase', 'component'].forEach(key => {
      if (focus[key] == null) return;
      if (!Array.from(filters[key].options).some(option => option.value === focus[key])) {
        const option = el('option', '', focus[key]); option.value = focus[key]; filters[key].append(option);
      }
      filters[key].value = focus[key];
    });
    search.value = focus.operator || ''; zoom.value = 'filtered'; redraw();
  }
}
