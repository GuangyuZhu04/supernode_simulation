'use strict';

// Bounded views of the same templates implemented by topology.Topology.
const topologyView = {view: 'domain', domain: 0, page: 0, selected: null};
const topologyInteger = (value, fallback = 0) => Number.isSafeInteger(Number(value)) && Number(value) > 0 ? Number(value) : fallback;
const topologyKinds = {full_mesh: 'Full mesh · 全连接', mesh: 'Mesh · 网格', clos: 'Clos · 叶脊', fat_tree: 'Fat tree · 叶脊模板', explicit: 'Explicit · 显式有向图'};

function topologyRank(config, physical) {
  const e = {...config.execution, ...config.features?.parallelism}, tp = topologyInteger(e.tp, 1), ep = topologyInteger(e.ep, 1), pp = topologyInteger(e.pp, 1), dp = topologyInteger(e.dp, 1);
  const pd = config.features?.prefill_decode_disaggregation ?? e.features?.prefill_decode_disaggregation;
  const independent = (e.cp ?? 1) !== 1 || e.expert_rank_order?.length || [e.expert_tp, e.attention_dp, e.expert_dp].some(v => v != null);
  const cp = e.cp ?? 1, etp = independent ? (e.expert_tp ?? 1) : tp;
  const sharedMesh = independent || (config.features?.fsdp?.domain ?? e.fsdp_domain);
  const width = sharedMesh ? config.hardware.npu_count / ((pd?.enabled ? 2 : 1) * pp * dp) : tp * ep;
  if (sharedMesh && (!Number.isInteger(width/(tp*cp)) || !Number.isInteger(width/(etp*ep)))) return null;
  let rank, pool = '';
  if (pd?.enabled) {
    const count = width;
    if (pd.prefill_ranks?.length || pd.decode_ranks?.length) {
      rank = (pd.prefill_ranks || []).indexOf(physical); pool = 'Prefill';
      if (rank < 0) { rank = (pd.decode_ranks || []).indexOf(physical); pool = 'Decode'; }
    } else { pool = physical < count ? 'Prefill' : 'Decode'; rank = physical % count; if (physical >= 2 * count) return null; }
  } else rank = e.rank_order?.length ? e.rank_order.indexOf(physical) : physical;
  if (rank < 0 || rank >= width * pp * dp) return null;
  const slot = rank % width;
  const expert = e.expert_rank_order?.length ? (Array.isArray(e.expert_rank_order) ? e.expert_rank_order.indexOf(slot) : -1) : slot;
  if (expert < 0) return null;
  const result = {rank, pool, tp: slot % tp, ep: Math.floor(expert / etp) % ep, pp: Math.floor(rank / width) % pp, dp: Math.floor(rank / (width * pp))};
  if (sharedMesh) Object.assign(result, {cp: Math.floor(slot / tp) % cp, attention_dp: Math.floor(slot / (tp*cp)), expert_tp: expert % etp, expert_dp: Math.floor(expert / (ep*etp))});
  return result;
}

function buildTopologyModel(config, options = {}) {
  const h = config?.hardware || {}, npuCount = topologyInteger(h.npu_count), suSize = topologyInteger(h.su_size), perHost = topologyInteger(h.npus_per_host);
  const kind = h.topology || 'full_mesh', groupCount = suSize ? Math.ceil(npuCount / suSize) : 0;
  const model = {kind, npuCount, suSize, groupCount, hostCount: perHost ? Math.ceil(npuCount / perHost) : null, nodes: [], edges: [], page: 0, pageCount: 1, shownNodes: 0, totalNodes: 0, width: 820, height: 350, note: '', view: options.view || 'overview'};
  if (!npuCount || !suSize || !perHost) { model.note = '请填写有效的 NPU 总数、每 SU 卡数和每主机卡数。'; return model; }
  const su = {bandwidth_Bps: 4e11, latency_s: 1e-6, efficiency: .85, ...h.scale_up};
  const so = {bandwidth_Bps: 5e10, latency_s: 5e-6, efficiency: .8, ...h.scale_out};
  const pageFor = (total, limit) => { model.totalNodes = total; model.pageCount = Math.max(1, Math.ceil(total / limit)); model.page = Math.max(0, Math.min(model.pageCount - 1, Math.floor(Number(options.page) || 0))); return model.page * limit; };
  const addNpu = (physical, x, y) => {
    const rank = topologyRank(config, physical);
    model.nodes.push({id: 'npu' + physical, label: 'NPU ' + physical, kind: 'npu', physical, rank, pool: rank?.pool || '', x, y, host: Math.floor(physical / perHost), su: Math.floor(physical / suSize)});
  };
  const edge = (src, dst, domain, params, extra = {}) => model.edges.push({src, dst, domain, bandwidth_Bps: params.bandwidth_Bps, latency_s: params.latency_s, efficiency: params.efficiency, bidirectional: true, ...extra});
  if (kind === 'explicit') {
    const links = Array.isArray(h.explicit_links) ? h.explicit_links : [];
    const isDevice = name => /^npu\d+$/.test(String(name)) && Number(String(name).slice(3)) < npuCount && String(name) === 'npu' + Number(String(name).slice(3));
    const extraNames = [...new Set(links.flatMap(link => [link.src, link.dst]).filter(name => name != null && !isDevice(name)))];
    const start = pageFor(npuCount + extraNames.length, 24), stop = Math.min(start + 24, model.totalNodes);
    const names = [];
    for (let i = start; i < stop; i++) names.push(i < npuCount ? 'npu' + i : extraNames[i - npuCount]);
    const primary = new Set(names), included = new Set(names);
    for (const link of links) {
      const peer = primary.has(link.src) ? link.dst : primary.has(link.dst) ? link.src : null;
      if (peer != null && !included.has(peer) && names.length < 36) { names.push(peer); included.add(peer); }
    }
    const columns = Math.min(6, names.length);
    model.width = Math.max(280, columns * 137);
    names.forEach((id, offset) => {
      const rowCount = Math.min(columns, names.length - Math.floor(offset / columns) * columns);
      const x = model.width / 2 + ((offset % columns) - (rowCount - 1) / 2) * 137, y = 40 + Math.floor(offset / columns) * 61;
      if (isDevice(id)) addNpu(Number(String(id).slice(3)), x, y);
      else model.nodes.push({id, label: String(id), kind: 'switch', x, y});
      model.nodes[model.nodes.length - 1].context = !primary.has(id);
    });
    const visible = new Set(model.nodes.map(node => node.id));
    const candidates = links.filter(link => visible.has(link.src) && visible.has(link.dst));
    for (const link of candidates.slice(0, 240)) edge(link.src, link.dst, link.domain || 'SU', link, {id: link.id, bidirectional: false, shared_group: link.shared_group, shared_Bps: link.shared_Bps});
    model.height = Math.max(100, Math.ceil(names.length / columns) * 61 + 24);
    model.shownNodes = model.nodes.length;
    model.note = '显式有向图：本页 ' + (stop - start) + ' 个主端点 + ' + (names.length - (stop - start)) + ' 个邻接端点 / 共 ' + model.totalNodes + ' 个端点，' + model.edges.length + ' / ' + links.length + ' 条配置边。未展示端点相关边及超过 240 条的页内边省略；无链路的 NPU 仍展示。位置不代表物理距离。';
    return model;
  }
  if (model.view === 'overview') {
    const start = pageFor(groupCount, 8), stop = Math.min(groupCount, start + 8);
    const columns = Math.min(4, stop - start);
    model.width = Math.max(340, columns * 204 + 4);
    model.nodes.push({id: 'so.spine', label: 'SO spine', kind: 'switch', x: model.width / 2, y: 27});
    for (let domain = start; domain < stop; domain++) {
      const offset = domain - start, first = domain * suSize, count = Math.min(suSize, npuCount - first);
      const firstHost = Math.floor(first / perHost), lastHost = Math.floor((first + count - 1) / perHost);
      const rowCount = Math.min(columns, stop - start - Math.floor(offset / columns) * columns);
      const node = {id: 'su' + domain, label: 'SU ' + domain, kind: 'su', domain, count, first, last: first + count - 1, hosts: lastHost - firstHost + 1, x: model.width / 2 + ((offset % columns) - (rowCount - 1) / 2) * 204, y: 117 + Math.floor(offset / columns) * 133};
      model.nodes.push(node);
      let defaultUplink = so.bandwidth_Bps * node.hosts;
      if (so.host_injection_Bps != null) {
        const perDevice = Math.min(so.bandwidth_Bps, so.injection_Bps || so.bandwidth_Bps);
        const firstCount = Math.min(firstHost * perHost + perHost, first + count) - first;
        const lastCount = firstHost === lastHost ? 0 : first + count - lastHost * perHost;
        defaultUplink = Math.min(so.host_injection_Bps, firstCount * perDevice) + Math.max(0, node.hosts - 2) * Math.min(so.host_injection_Bps, perHost * perDevice) + Math.min(so.host_injection_Bps, lastCount * perDevice);
      }
      edge(node.id, 'so.spine', 'SO', {...so, bandwidth_Bps: so.uplink_Bps || defaultUplink});
    }
    model.height = stop - start > 4 ? 320 : 190;
    model.shownNodes = stop - start;
    model.note = '集群概览：展示 ' + model.shownNodes + ' / ' + groupCount + ' 个 SU 域。连线为每个 SU gateway 到共享 SO spine 的聚合上联；点击 SU 查看域内链路与物理卡。';
    return model;
  }
  const domain = Math.max(0, Math.min(groupCount - 1, Math.floor(Number(options.domain) || 0)));
  model.domain = domain;
  const first = domain * suSize, count = Math.min(suSize, npuCount - first), limit = kind === 'full_mesh' ? 16 : 32;
  const start = pageFor(count, limit), stop = Math.min(count, start + limit), shown = stop - start;
  let width = kind === 'mesh' ? topologyInteger(h.mesh_width, Math.ceil(Math.sqrt(count))) : 8;
  const leafSize = topologyInteger(h.leaf_size, Math.max(1, Math.min(8, suSize)));
  const columns = kind === 'full_mesh' && shown <= 4 ? Math.min(shown, 3) : kind === 'mesh' ? Math.min(width, 8, shown) : Math.min(8, shown);
  if (kind === 'full_mesh' && shown <= 4) { model.width = Math.max(320, columns * 120 + 40); model.height = shown <= 3 ? 110 : 180; }
  else if (kind !== 'full_mesh') model.width = Math.max(320, columns * 101 + 12);
  for (let offset = start; offset < stop; offset++) {
    const local = offset - start;
    let x, y;
    if (kind === 'full_mesh') {
      if (shown <= 4) {
        const cols = shown === 4 ? 2 : shown, rowCount = Math.min(cols, shown - Math.floor(local / cols) * cols);
        x = model.width / 2 + ((local % cols) - (rowCount - 1) / 2) * 135; y = 55 + Math.floor(local / cols) * 70;
      } else { const angle = local * Math.PI * 2 / shown - Math.PI / 2; x = 410 + Math.cos(angle) * 280; y = 163 + Math.sin(angle) * 115; }
    } else {
      const rowCount = Math.min(columns, shown - Math.floor(local / columns) * columns);
      x = model.width / 2 + ((local % columns) - (rowCount - 1) / 2) * 101; y = 147 + Math.floor(local / columns) * 59;
    }
    if (kind === 'mesh') y = 53 + Math.floor(local / columns) * 62;
    addNpu(first + offset, x, y);
  }
  const displayed = new Set(model.nodes.map(node => node.id));
  if (kind === 'full_mesh') {
    for (let a = start; a < stop; a++) for (let b = a + 1; b < stop; b++) edge('npu' + (first + a), 'npu' + (first + b), 'SU', su);
  } else if (kind === 'mesh') {
    for (let offset = start; offset < stop; offset++) {
      for (const before of [offset % width ? offset - 1 : -1, offset >= width ? offset - width : -1]) {
        if (before >= start && displayed.has('npu' + (first + before))) edge('npu' + (first + before), 'npu' + (first + offset), 'SU', su);
      }
    }
    model.height = Math.max(120, Math.ceil(shown / columns) * 62 + 25);
  } else if (kind === 'clos' || kind === 'fat_tree') {
    const leaves = [...new Set(model.nodes.map(node => Math.floor((node.physical - first) / leafSize)))];
    model.nodes.push({id: 'su' + domain + '.spine', label: 'SU spine', kind: 'switch', x: model.width / 2, y: 24});
    leaves.forEach((leaf, i) => {
      const id = 'su' + domain + '.leaf' + leaf;
      model.nodes.push({id, label: 'Leaf ' + leaf, kind: 'switch', x: 50 + (i + .5) * (model.width - 100) / leaves.length, y: 82});
      edge(id, 'su' + domain + '.spine', 'SU', {...su, bandwidth_Bps: su.uplink_Bps || su.bandwidth_Bps * leafSize});
    });
    for (let offset = start; offset < stop; offset++) edge('npu' + (first + offset), 'su' + domain + '.leaf' + Math.floor(offset / leafSize), 'SU', su);
    model.height = 147 + Math.ceil(shown / columns) * 59;
  }
  model.shownNodes = shown;
  model.note = 'SU ' + domain + '：展示 ' + shown + ' / ' + count + ' 张物理卡（NPU ' + (first + start) + '–' + (first + stop - 1) + '），仅绘制本页域内双向链路。' + (kind === 'mesh' ? '网格宽度 ' + width + '；布局折行，连接以边为准。' : kind === 'clos' || kind === 'fat_tree' ? '每 leaf ' + leafSize + ' 卡；本引擎的 Clos / Fat tree 均使用单 spine 模板。' : '全连接每对卡都有独立方向链路。') + ' SO 接入、主机与 I/O 约束见设备详情。';
  return model;
}

function renderTopology(config = state.config) {
  if (!config) return;
  const h = config.hardware || {}, model = buildTopologyModel(config, topologyView);
  topologyView.page = model.page;
  if (model.domain != null) topologyView.domain = model.domain;
  const host = $('#topology-visual'); host.replaceChildren();
  text('#topology-type', topologyKinds[model.kind] || model.kind);
  text('#topology-npu', number(model.npuCount, 0) + ' NPU');
  text('#topology-su', number(model.groupCount, 0) + ' 域 · ≤ ' + number(model.suSize, 0) + ' 卡');
  text('#topology-hbm', number(h.npu?.hbm_bytes / 1e9, 1) + ' GB');
  text('#topology-bandwidth', model.kind === 'explicit' ? '逐条显式链路' : number(h.scale_up?.bandwidth_Bps / 1e9, 1) + ' / ' + number(h.scale_out?.bandwidth_Bps / 1e9, 1) + ' GB/s');
  text('#topology-note', model.note);
  host.setAttribute('aria-label', '由当前所看配置生成的集群拓扑；' + model.note);
  if (!model.nodes.length) { empty(host, model.note); return; }
  renderScaleOutTopology(config, host);
  host.append(el('h3', 'su-topology-heading', model.kind === 'explicit' ? '完整显式拓扑 · SU / SO' : 'SU · 域内拓扑'));
  const controls = el('div', 'topology-toolbar');
  const button = (label, selected, action) => { const b = el('button', 'text-button' + (selected ? ' selected' : ''), label); b.type = 'button'; b.setAttribute('aria-pressed', String(selected)); b.addEventListener('click', action); return b; };
  if (model.kind !== 'explicit') {
    controls.append(button('集群概览', model.view === 'overview', () => { topologyView.view = 'overview'; topologyView.page = 0; renderTopology(config); }), button('SU 域内链路', model.view === 'domain', () => { topologyView.view = 'domain'; topologyView.page = 0; renderTopology(config); }));
    if (model.view === 'domain') {
      const label = el('label', '', 'SU 编号 '), input = el('input'); input.type = 'number'; input.min = '0'; input.max = String(model.groupCount - 1); input.step = '1'; input.value = String(model.domain); input.setAttribute('aria-label', 'SU 编号，从 0 开始');
      input.addEventListener('change', () => { topologyView.domain = Math.max(0, Math.min(model.groupCount - 1, Math.floor(Number(input.value) || 0))); topologyView.page = 0; renderTopology(config); }); label.append(input); controls.append(label);
    }
  }
  if (model.pageCount > 1) {
    const prev = button('上一页', false, () => { topologyView.page--; renderTopology(config); }); prev.disabled = model.page === 0;
    const next = button('下一页', false, () => { topologyView.page++; renderTopology(config); }); next.disabled = model.page + 1 >= model.pageCount;
    const label = el('label', '', '页 '), input = el('input'); input.type = 'number'; input.min = '1'; input.max = String(model.pageCount); input.step = '1'; input.value = String(model.page + 1); input.setAttribute('aria-label', '拓扑页码'); input.addEventListener('change', () => { topologyView.page = Math.max(0, Math.floor(Number(input.value) || 1) - 1); renderTopology(config); }); label.append(input, el('span', '', ' / ' + number(model.pageCount, 0))); controls.append(prev, label, next);
  }
  host.append(controls);
  const viewport = el('div', 'topology-viewport'); viewport.setAttribute('tabindex', '0'); viewport.setAttribute('aria-label', '拓扑图，窄屏可横向滚动查看');
  const svg = svgEl('svg', {viewBox: '0 0 ' + model.width + ' ' + model.height, width: model.width, height: model.height, role: 'img', 'aria-label': model.note, class: 'topology-canvas'});
  svg.style.maxWidth = model.width + 'px';
  svg.style.minWidth = Math.min(720, model.width) + 'px';
  svg.append(svgEl('title', {}, topologyKinds[model.kind] + ' · ' + model.note));
  const defs = svgEl('defs'), marker = svgEl('marker', {id: 'topology-arrow', markerWidth: 6, markerHeight: 6, refX: 5, refY: 3, orient: 'auto'});
  marker.append(svgEl('path', {d: 'M0 0 L6 3 L0 6 Z', fill: '#7e91ab'})); defs.append(marker); svg.append(defs);
  const positions = new Map(model.nodes.map(node => [node.id, node]));
  const colors = {SU: '#76b5a6', SO: '#83a2d1', H2D: '#bc9a69', D2H: '#bc9a69', remote: '#a38eb8'};
  model.edges.forEach(link => {
    const a = positions.get(link.src), b = positions.get(link.dst); if (!a || !b) return;
    const dx = b.x - a.x, dy = b.y - a.y, distance = Math.hypot(dx, dy) || 1;
    const halfWidth = b.kind === 'su' ? 87 : b.kind === 'npu' ? 37 : 44, halfHeight = b.kind === 'su' ? 44 : b.kind === 'npu' ? 18.5 : 13.5;
    const clearance = Math.min(dx ? (halfWidth + 5) * distance / Math.abs(dx) : Infinity, dy ? (halfHeight + 5) * distance / Math.abs(dy) : Infinity);
    const end = Number.isFinite(clearance) ? clearance : 0;
    const attrs = {x1: a.x, y1: a.y, x2: b.x - dx / distance * end, y2: b.y - dy / distance * end, stroke: colors[link.domain] || '#99a6b9', 'stroke-width': model.kind === 'full_mesh' && model.view === 'domain' ? 1 : 1.5, opacity: model.edges.length > 80 ? .35 : .7};
    if (!link.bidirectional) attrs['marker-end'] = 'url(#topology-arrow)';
    const line = link.src === link.dst
      ? svgEl('path', {...attrs, d: 'M ' + (a.x + halfWidth) + ' ' + (a.y - 7) + ' C ' + (a.x + halfWidth + 38) + ' ' + (a.y - 40) + ', ' + (a.x + halfWidth + 38) + ' ' + (a.y + 40) + ', ' + (a.x + halfWidth + 5) + ' ' + (a.y + 7), fill: 'none'})
      : svgEl('line', attrs);
    line.append(svgEl('title', {}, link.src + (link.bidirectional ? ' ↔ ' : ' → ') + link.dst + ' · ' + link.domain + ' · ' + number(link.bandwidth_Bps / 1e9) + ' GB/s / 方向 · ' + number(link.latency_s * 1e6) + ' µs · 效率 ' + number((link.efficiency ?? 1) * 100, 1) + '%' + (link.shared_group ? ' · 共享组 ' + link.shared_group : '')));
    svg.append(line);
  });
  const inspector = el('div', 'topology-inspector'); inspector.setAttribute('aria-live', 'polite');
  const inspect = node => {
    topologyView.selected = node.id; inspector.replaceChildren();
    if (node.kind === 'su') { topologyView.view = 'domain'; topologyView.domain = node.domain; topologyView.page = 0; renderTopology(config); return; }
    inspector.append(el('strong', '', node.label));
    if (node.kind === 'npu') {
      const r = node.rank, sp = sequenceParallelInfo(config);
      inspector.append(el('span', '', '物理 NPU ' + node.physical + ' · SU ' + node.su + ' · Host ' + node.host));
      inspector.append(el('span', '', r ? (r.pool ? r.pool + ' 池 · ' : '') + 'Rank ' + r.rank + ' · TP ' + r.tp + ' / EP ' + r.ep + ' / PP ' + r.pp + ' / DP ' + r.dp : '未找到合法的 rank 映射，请校验并行配置。'));
      if (r?.cp != null) inspector.append(el('span', '', 'Attention 副本 ' + r.attention_dp + ' / CP ' + r.cp + ' · 专家副本 ' + r.expert_dp + ' / 专家 TP ' + r.expert_tp));
      inspector.append(el('span', '', sp.label + (sp.enabled && r ? ' · SP 分片索引 ' + r.tp + '（与 TP 索引共用，从 0 开始）' : '')));
      const groups = (h.io_groups || []).filter(group => (group.members || []).includes(node.physical));
      if (groups.length) inspector.append(el('span', '', 'I/O 共享约束：' + groups.map(group => group.id + ' (' + number(group.bandwidth_Bps / 1e9) + ' GB/s)').join('、')));
      if (model.kind !== 'explicit') inspector.append(el('span', '', 'SO 接入 su' + node.su + '.gateway · H2D / D2H 接入 cpu' + node.host));
      else inspector.append(el('span', '', '显式图仅使用配置中的边；不会自动补充 SO 或主机连接。'));
    } else inspector.append(el('span', '', '端点 ' + node.id + ' · 相关链路参数可悬停查看。'));
  };
  model.nodes.forEach(node => {
    const group = svgEl('g', {class: 'topology-node', role: 'button', tabindex: 0, 'aria-label': node.kind === 'su' ? node.label + '，' + node.count + ' 张卡，点击查看域内链路' : node.label + '，点击查看设备详情'});
    const aggregate = node.kind === 'su', width = aggregate ? 174 : node.kind === 'npu' ? 74 : 88, height = aggregate ? 88 : node.kind === 'npu' ? 37 : 27;
    const fill = node.pool === 'Decode' ? '#edf1fc' : node.kind === 'switch' ? '#f0f4fa' : '#eef8f4';
    group.append(svgEl('rect', {x: node.x - width / 2, y: node.y - height / 2, width, height, rx: aggregate ? 9 : 6, fill, stroke: node.pool === 'Decode' ? '#a6b6dd' : '#accdc1', 'stroke-width': 1.2, ...(node.context ? {'stroke-dasharray': '4 2'} : {})}));
    group.append(svgEl('text', {x: node.x, y: node.y + (aggregate ? -18 : node.kind === 'npu' ? -2 : 4), 'text-anchor': 'middle', 'font-size': aggregate ? 13 : 10, fill: '#284d54'}, String(node.label).length > 17 ? String(node.label).slice(0, 16) + '…' : node.label));
    if (aggregate) {
      group.append(svgEl('text', {x: node.x, y: node.y + 3, 'text-anchor': 'middle', 'font-size': 11, fill: '#52736d'}, number(node.count, 0) + ' NPU · ' + number(node.hosts, 0) + ' Host'));
      group.append(svgEl('text', {x: node.x, y: node.y + 22, 'text-anchor': 'middle', 'font-size': 10, fill: '#6c8983'}, '物理卡 ' + node.first + '–' + node.last + '  ↗'));
    } else if (node.kind === 'npu') group.append(svgEl('text', {x: node.x, y: node.y + 12, 'text-anchor': 'middle', 'font-size': 8, fill: '#6c8983'}, node.rank ? (node.pool ? node.pool[0] + ' · ' : '') + 'rank ' + node.rank.rank : 'rank ?'));
    group.append(svgEl('title', {}, node.id + (node.rank ? ' · ' + node.pool + ' rank ' + node.rank.rank + ' · TP ' + node.rank.tp + ' / EP ' + node.rank.ep + ' / PP ' + node.rank.pp + ' / DP ' + node.rank.dp : '')));
    group.addEventListener('click', () => inspect(node)); group.addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); inspect(node); } }); svg.append(group);
  });
  viewport.append(svg); host.append(viewport);
  const legend = el('div', 'topology-legend'); legend.append(el('span', 'topology-su-legend', '● SU 域内'), el('span', 'topology-so-legend', '● SO 互联'), el('span', '', model.kind === 'explicit' ? '箭头为单向边 · 虚线框为邻接端点' : '实线表示双向链路 · 带宽按方向计'), el('span', '', '点击节点查看详情'));
  host.append(legend, inspector);
  const selected = model.nodes.find(node => node.id === topologyView.selected && node.kind !== 'su');
  if (selected) inspect(selected); else inspector.append(el('span', '', number(model.hostCount, 0) + ' 台主机 · 每主机 ' + number(h.npus_per_host, 0) + ' 卡 · 路由 ' + (h.routing_algorithm || 'shortest_hop') + ' / ' + (h.routing_policy || 'shortest_hop') + (' · ' + sequenceParallelInfo(config).label)));
}


// Scale-out is always visible alongside the selected SU view.
const scaleOutView = {page: 0};
function topologyScaleOutUplink(h, domain, so) {
  const first = domain * h.su_size, count = Math.min(h.su_size, h.npu_count - first), perHost = h.npus_per_host;
  const firstHost = Math.floor(first / perHost), lastHost = Math.floor((first + count - 1) / perHost), hosts = lastHost - firstHost + 1;
  if (so.uplink_Bps != null) return so.uplink_Bps;
  if (so.host_injection_Bps == null) return so.bandwidth_Bps * hosts;
  const perDevice = Math.min(so.bandwidth_Bps, so.injection_Bps ?? so.bandwidth_Bps);
  const firstCount = Math.min((firstHost + 1) * perHost, first + count) - first;
  const lastCount = firstHost === lastHost ? 0 : first + count - lastHost * perHost;
  return Math.min(so.host_injection_Bps, firstCount * perDevice) + Math.max(0, hosts - 2) * Math.min(so.host_injection_Bps, perHost * perDevice) + Math.min(so.host_injection_Bps, lastCount * perDevice);
}
function buildScaleOutModel(config, options = {}) {
  const h = config.hardware || {}, explicit = h.topology === 'explicit';
  const model = {kind: explicit ? 'explicit' : 'template', nodes: [], edges: [], page: 0, pageCount: 1, width: 820, height: 310, totalLinks: 0, totalNodes: 0, groupCount: 0, note: ''};
  const paginate = (count, limit) => { model.pageCount = Math.max(1, Math.ceil(count / limit)); model.page = Math.max(0, Math.min(model.pageCount - 1, Math.floor(Number(options.page) || 0))); return model.page * limit; };
  if (explicit) {
    const links = (h.explicit_links || []).filter(link => link.domain === 'SO');
    const names = [...new Set(links.flatMap(link => [link.src, link.dst]))];
    model.totalNodes = names.length; model.totalLinks = links.length;
    if (!links.length) { model.note = '当前显式图未配置 SO 链路；不会自动补画网关或交换节点。'; return model; }
    const start = paginate(names.length, 24), primary = names.slice(start, start + 24), visible = new Set(primary);
    for (const link of links) {
      const peer = visible.has(link.src) ? link.dst : visible.has(link.dst) ? link.src : null;
      if (peer != null && visible.size < 36) visible.add(peer);
    }
    const displayed = [...visible], columns = Math.min(6, displayed.length);
    model.width = Math.max(400, columns * 150);
    model.height = Math.max(150, Math.ceil(displayed.length / columns) * 85 + 35);
    displayed.forEach((id, i) => {
      const rowCount = Math.min(columns, displayed.length - Math.floor(i / columns) * columns);
      model.nodes.push({id, label: String(id), kind: 'endpoint', x: model.width / 2 + ((i % columns) - (rowCount - 1) / 2) * 150, y: 55 + Math.floor(i / columns) * 85, context: !primary.includes(id)});
    });
    model.edges = links.filter(link => visible.has(link.src) && visible.has(link.dst)).slice(0, 240).map(link => ({...link, bidirectional: false}));
    model.note = '按 domain=SO 提取真实有向链路：本页 ' + model.nodes.length + ' / ' + names.length + ' 个端点、' + model.edges.length + ' / ' + links.length + ' 条 SO 边。SU 边不占用此图名额；未推测 SO 域划分。虚线框为补充邻接端点，完整链路可查 JSON。';
    return model;
  }
  const count = topologyInteger(h.npu_count), size = topologyInteger(h.su_size), perHost = topologyInteger(h.npus_per_host);
  if (!count || !size || !perHost) { model.note = '填写有效 NPU、SU 与主机卡数后生成 SO 网络。'; return model; }
  const so = {bandwidth_Bps: 5e10, latency_s: 5e-6, efficiency: .8, ...h.scale_out};
  model.groupCount = Math.ceil(count / size); model.totalNodes = 2 * model.groupCount + 1; model.totalLinks = 2 * model.groupCount;
  const start = paginate(model.groupCount, 4), stop = Math.min(model.groupCount, start + 4), shown = stop - start;
  model.width = Math.max(460, shown * 215);
  model.nodes.push({id: 'so.spine', label: 'SO spine', kind: 'spine', x: model.width / 2, y: 40});
  for (let domain = start; domain < stop; domain++) {
    const first = domain * size, devices = Math.min(size, count - first), x = model.width / 2 + ((domain - start) - (shown - 1) / 2) * 215;
    const hosts = Math.floor((first + devices - 1) / perHost) - Math.floor(first / perHost) + 1;
    model.nodes.push({id: 'su' + domain + '.gateway', label: 'su' + domain + '.gateway', kind: 'gateway', domain, x, y: 155});
    model.nodes.push({id: 'su' + domain + '.devices', label: 'SU ' + domain, kind: 'su', domain, first, last: first + devices - 1, count: devices, hosts, selected: domain === options.selectedDomain, x, y: 260});
    model.edges.push({src: 'su' + domain + '.devices', dst: 'su' + domain + '.gateway', domain: 'SO', bandwidth_Bps: so.bandwidth_Bps, latency_s: so.latency_s, efficiency: so.efficiency, bidirectional: true, aggregated: true, device_count: devices});
    model.edges.push({src: 'su' + domain + '.gateway', dst: 'so.spine', domain: 'SO', bandwidth_Bps: topologyScaleOutUplink(h, domain, so), latency_s: so.latency_s, efficiency: so.efficiency, bidirectional: true});
  }
  model.note = 'SO 跨 SU 网络：本页 ' + shown + ' / ' + model.groupCount + ' 个 SU。每个 SU 卡组聚合表示其 NPU→gateway 接入，带宽标为每卡；gateway→spine 为共享上联，每个方向独立计。' + (model.groupCount === 1 ? '当前只有 1 个 SU，无跨 SU 通信需求。' : '跨 SU 路径：源 NPU→源 gateway→SO spine→目标 gateway→目标 NPU。') + ' 点击 SU 可查看下方域内拓扑。';
  return model;
}
function renderScaleOutTopology(config, host) {
  const model = buildScaleOutModel(config, {...scaleOutView, selectedDomain: topologyView.domain});
  scaleOutView.page = model.page;
  const section = el('section', 'so-topology-section'); section.setAttribute('aria-label', 'SO 跨 SU 拓扑');
  const heading = el('div', 'so-topology-heading'); heading.append(el('h3', '', 'SO · 跨 SU 拓扑'), el('span', 'mini-badge', model.kind === 'explicit' ? '显式 SO 链路' : 'Gateway / Spine'));
  section.append(heading); host.append(section);
  if (!model.nodes.length) { section.append(el('p', 'so-topology-note', model.note)); return; }
  if (model.pageCount > 1) {
    const controls = el('div', 'so-topology-controls');
    const prev = el('button', 'text-button', '上一页 SO'), next = el('button', 'text-button', '下一页 SO'); prev.type = next.type = 'button';
    prev.disabled = model.page === 0; next.disabled = model.page + 1 >= model.pageCount;
    prev.addEventListener('click', () => { scaleOutView.page--; renderTopology(config); }); next.addEventListener('click', () => { scaleOutView.page++; renderTopology(config); });
    const label = el('label', '', 'SO 页码 '), input = el('input'); input.type = 'number'; input.min = '1'; input.max = String(model.pageCount); input.value = String(model.page + 1); input.setAttribute('aria-label', 'SO 页码');
    input.addEventListener('change', () => { scaleOutView.page = Math.floor(Number(input.value) || 1) - 1; renderTopology(config); }); label.append(input, el('span', '', ' / ' + number(model.pageCount, 0)));
    controls.append(prev, label, next); section.append(controls);
  }
  const viewport = el('div', 'topology-viewport'); viewport.setAttribute('tabindex', '0'); viewport.setAttribute('aria-label', 'SO 网络图，窄屏可横向滚动');
  const svg = svgEl('svg', {class: 'so-topology-canvas', viewBox: '0 0 ' + model.width + ' ' + model.height, role: 'img', 'aria-label': model.note});
  svg.style.maxWidth = model.width + 'px'; svg.style.minWidth = Math.min(720, model.width) + 'px';
  const defs = svgEl('defs'), marker = svgEl('marker', {id: 'so-arrow', markerWidth: 6, markerHeight: 6, refX: 5, refY: 3, orient: 'auto'});
  marker.append(svgEl('path', {d: 'M0 0 L6 3 L0 6 Z', fill: '#4f77b7'})); defs.append(marker); svg.append(defs);
  const positions = new Map(model.nodes.map(node => [node.id, node]));
  const linkDescription = link => link.src + (link.bidirectional ? ' ↔ ' : ' → ') + link.dst + ' · ' + number(link.bandwidth_Bps / 1e9) + ' GB/s / 方向' + (link.aggregated ? ' / 卡（' + number(link.device_count, 0) + ' 卡接入聚合显示）' : '') + ' · ' + number(link.latency_s * 1e6) + ' µs / 跳 · 效率 ' + percentage(link.efficiency ?? 1) + (link.shared_group ? ' · 共享组 ' + link.shared_group + ' / ' + number(link.shared_Bps / 1e9) + ' GB/s' : '');
  const details = el('div', 'so-topology-details'); details.setAttribute('aria-live', 'polite');
  const activate = (node, callback) => { node.setAttribute('tabindex', '0'); node.setAttribute('role', 'button'); node.addEventListener('click', callback); node.addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); callback(); } }); };
  model.edges.forEach(link => {
    const a = positions.get(link.src), b = positions.get(link.dst), dx = b.x - a.x, dy = b.y - a.y, distance = Math.hypot(dx, dy) || 1;
    const pad = Math.min(dx ? 69 * distance / Math.abs(dx) : Infinity, dy ? 22 * distance / Math.abs(dy) : Infinity);
    const group = svgEl('g', {class: 'so-topology-link', 'data-domain': 'SO', 'aria-label': linkDescription(link)});
    const attrs = {x1: a.x, y1: a.y, x2: b.x - dx / distance * pad, y2: b.y - dy / distance * pad, stroke: '#7399cb', 'stroke-width': 2, ...(link.aggregated ? {'stroke-dasharray': '5 3'} : {}), ...(!link.bidirectional ? {'marker-end': 'url(#so-arrow)'} : {})};
    group.append(link.src === link.dst ? svgEl('path', {...attrs, d: 'M ' + (a.x + 55) + ' ' + a.y + ' C ' + (a.x + 90) + ' ' + (a.y - 40) + ' ' + (a.x + 90) + ' ' + (a.y + 40) + ' ' + (a.x + 60) + ' ' + (a.y + 7), fill: 'none'}) : svgEl('line', attrs));
    group.append(svgEl('title', {}, linkDescription(link)));
    if (model.kind === 'template') {
      const t = link.aggregated ? .48 : .35, x = a.x + dx * t, y = a.y + dy * t;
      group.append(svgEl('rect', {x: x - 62, y: y - 10, width: 124, height: 20, rx: 4, fill: '#f4f8ff'}));
      group.append(svgEl('text', {x, y: y + 4, 'text-anchor': 'middle', 'font-size': 11, fill: '#426698'}, number(link.bandwidth_Bps / 1e9) + ' GB/s' + (link.aggregated ? ' / 卡' : ' 上联')));
    }
    activate(group, () => { details.textContent = linkDescription(link); }); svg.append(group);
  });
  model.nodes.forEach(node => {
    const width = node.kind === 'su' ? 174 : 134, height = node.kind === 'su' ? 66 : 34;
    const group = svgEl('g', {class: 'so-topology-node', 'data-node-id': node.id, 'aria-label': node.label + (node.kind === 'su' ? '，查看 SU 域内拓扑' : '，查看 SO 连接')});
    group.append(svgEl('rect', {x: node.x - width / 2, y: node.y - height / 2, width, height, rx: 7, fill: node.kind === 'su' ? '#edf7f3' : '#eaf1fc', stroke: node.selected ? '#168b79' : '#a9bfdf', 'stroke-width': node.selected ? 2 : 1, ...(node.context ? {'stroke-dasharray': '4 2'} : {})}));
    group.append(svgEl('text', {x: node.x, y: node.y + (node.kind === 'su' ? -13 : 4), 'text-anchor': 'middle', 'font-size': 12, fill: '#365779'}, node.label.length > 19 ? node.label.slice(0, 18) + '…' : node.label));
    if (node.kind === 'su') {
      group.append(svgEl('text', {x: node.x, y: node.y + 6, 'text-anchor': 'middle', 'font-size': 11, fill: '#4d776f'}, number(node.count, 0) + ' NPU · ' + number(node.hosts, 0) + ' Host'));
      group.append(svgEl('text', {x: node.x, y: node.y + 22, 'text-anchor': 'middle', 'font-size': 10, fill: '#617e76'}, '物理卡 ' + node.first + '–' + node.last + '  ↘'));
    }
    group.append(svgEl('title', {}, node.id));
    activate(group, () => {
      if (node.kind === 'su') { topologyView.view = 'domain'; topologyView.domain = node.domain; topologyView.page = 0; renderTopology(config); }
      else details.textContent = model.edges.filter(link => link.src === node.id || link.dst === node.id).map(linkDescription).join('\n');
    }); svg.append(group);
  });
  viewport.append(svg); section.append(viewport);
  section.append(el('p', 'so-topology-note', model.note));
  if (model.kind === 'template') {
    const so = {bandwidth_Bps: 5e10, latency_s: 5e-6, efficiency: .8, ...config.hardware.scale_out};
    details.textContent = '链路效率 ' + percentage(so.efficiency) + ' · 每跳延迟 ' + number(so.latency_s * 1e6) + ' µs · 每卡注入 ' + number((so.injection_Bps ?? so.bandwidth_Bps) / 1e9) + ' GB/s · 每主机 NIC 注入 ' + number((so.host_injection_Bps ?? so.injection_Bps ?? so.bandwidth_Bps) / 1e9) + ' GB/s（原始容量；各约束另乘效率）\n接入、主机 NIC 与共享上联是独立约束；点击链路或节点查看详情。';
  } else details.textContent = '点击节点或链路查看 SO 方向、带宽、延迟、效率与共享容量。';
  section.append(details);
}
