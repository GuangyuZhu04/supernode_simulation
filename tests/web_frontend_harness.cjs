// Shared lightweight DOM for production frontend behavior tests.
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const root = path.resolve(__dirname, '..');
const source = fs.readFileSync(path.join(root, 'supernode_sim/web_static/app.js'), 'utf8');
const html = fs.readFileSync(path.join(root, 'supernode_sim/web_static/index.html'), 'utf8');

function dataName(name) {
  return name.slice(5).replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
}

class Element {
  constructor(tag = 'div') {
    this.tagName = tag.toUpperCase();
    this.children = [];
    this.parentElement = null;
    this.attributes = new Map();
    this.dataset = {};
    this.listeners = new Map();
    this.value = '';
    this.checked = this.hidden = this.disabled = this.open = false;
    this.className = '';
    this.style = {};
    this._text = '';
    this.classList = {
      contains: name => this.className.split(/\s+/).includes(name),
      toggle: (name, force) => {
        const present = this.classList.contains(name);
        const add = force === undefined ? !present : force;
        this.className = this.className.split(/\s+/).filter(value => value && value !== name).concat(add ? [name] : []).join(' ');
        return add;
      },
      add: name => this.classList.toggle(name, true),
      remove: name => this.classList.toggle(name, false),
    };
  }
  get textContent() { return this._text + this.children.map(child => child.textContent || '').join(''); }
  set textContent(value) { this._text = String(value); this.children = []; }
  get firstElementChild() { return this.children[0] || null; }
  get options() { return this.children.filter(child => child.tagName === 'OPTION'); }
  get lastChild() { return this.children[this.children.length - 1] || {textContent: ''}; }
  append(...nodes) {
    for (const node of nodes) {
      if (typeof node === 'string') { const text = new Element('span'); text.textContent = node; this.append(text); }
      else if (node) { node.parentElement = this; this.children.push(node); }
    }
  }
  appendChild(node) { this.append(node); return node; }
  replaceChildren(...nodes) { this.children = []; this._text = ''; this.append(...nodes); }
  setAttribute(name, value) {
    this.attributes.set(name, String(value));
    if (name.startsWith('data-')) this.dataset[dataName(name)] = String(value);
    else if (name === 'class') this.className = String(value);
    else if (name === 'id') this.id = String(value);
    else if (name === 'type') this.type = String(value);
  }
  getAttribute(name) { return this.attributes.get(name) ?? null; }
  removeAttribute(name) { this.attributes.delete(name); }
  addEventListener(name, callback) {
    if (!this.listeners.has(name)) this.listeners.set(name, []);
    this.listeners.get(name).push(callback);
  }
  async fire(name) {
    for (const callback of this.listeners.get(name) || []) await callback({target: this, currentTarget: this, preventDefault() {}});
  }
  matches(selector) {
    selector = selector.trim();
    const id = selector.match(/#([\w-]+)/);
    if (id && this.id !== id[1]) return false;
    const tag = selector.match(/^[a-zA-Z]+/);
    if (tag && this.tagName !== tag[0].toUpperCase()) return false;
    for (const match of selector.matchAll(/\.([\w-]+)/g)) if (!this.classList.contains(match[1])) return false;
    for (const match of selector.matchAll(/\[([\w-]+)(?:=["']([^"']*)["'])?\]/g)) {
      const value = match[1].startsWith('data-') ? this.dataset[dataName(match[1])] : this.getAttribute(match[1]);
      if (value == null || match[2] !== undefined && value !== match[2]) return false;
    }
    return true;
  }
  querySelectorAll(selector) {
    const selected = [];
    const selectors = selector.split(',');
    function visit(node) {
      for (const child of node.children) {
        if (selectors.some(part => child.matches(part))) selected.push(child);
        visit(child);
      }
    }
    visit(this);
    return selected;
  }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
  closest(selector) { return this.matches(selector) ? this : this.parentElement?.closest(selector) || null; }
  contains(node) { return node === this || this.children.some(child => child.contains(node)); }
  focus() {}
  scrollIntoView() {}
  setCustomValidity(message) { this.validationMessage = message; }
  reportValidity() { return !this.validationMessage; }
  remove() { if (this.parentElement) this.parentElement.children = this.parentElement.children.filter(child => child !== this); }
}

function fixture() {
  return {
    schema_version: '1.0',
    model: {name: 'synthetic-edit', total_params: 2000000000, active_params: 800000000,
      layers: 4, hidden_dim: 1024, intermediate_dim: 4096, num_heads: 8, kv_heads: 2,
      head_dim: 128, num_experts: 8, top_k: 2, moe_layer_fraction: .5, expert_parameter_fraction: .8},
    hardware: {npu_count: 4, su_size: 2, npus_per_host: 2, topology: 'full_mesh', routing_policy: 'shortest_hop',
      npu: {hbm_bytes: 80e9, hbm_Bps: 3e12, runtime_bytes: 0, workspace_bytes: 64e6, efficiency: .6, flops: {fp16: 1e15}},
      cpu: {dram_bytes: 2e12}, scale_up: {bandwidth_Bps: 9e11, latency_s: 1e-6, efficiency: .85},
      scale_out: {bandwidth_Bps: 50e9}, explicit_links: [], routes: {}},
    execution: {tp: 2, pp: 1, ep: 2, dp: 1, fsdp: 1, sequence_parallel: false, rank_order: [],
      precision: {weight: 'fp8', compute: 'fp16', communication: 'fp16', routed_expert_weight: null},
      features: {kv_offload: 'none', kv_offload_fraction: 1, optimizer_offload: false, activation_checkpointing: false,
        engram: {enabled: false, hit_rate: 0}}, optimizer: {kind: 'adam'}},
    workload: {scenario: 'inference', batch_size: 2, input_lengths: [128, 64], output_lengths: [4, 2], input_length: 1024, output_length: 16},
    targets: {ttft_s: 1, tpot_s: 1}, seed: 0,
    sources: {'*': {status: 'assumed', description: 'Synthetic test case.'}, 'hardware.calibration': {status: 'user', description: 'Original provenance.'}},
    assumptions: ['Keep this assumption.'], calibration: null,
  };
}

function harness(config = fixture(), {views = false} = {}) {
  const document = new Element('document');
  const ids = new Map();
  for (const match of html.matchAll(/<([\w-]+)\b([^>]*\bid="([^"]+)"[^>]*)>/g)) {
    const node = new Element(match[1]); node.id = match[3]; node.setAttribute('id', node.id);
    for (const attribute of match[2].matchAll(/([\w-]+)="([^"]*)"/g)) node.setAttribute(attribute[1], attribute[2]);
    node.hidden = /\bhidden\b/.test(match[2]);
    ids.set(node.id, node); document.append(node);
  }
  document.createElement = tag => new Element(tag);
  document.createElementNS = (_namespace, tag) => new Element(tag);
  document.querySelector = selector => {
    const direct = selector.match(/^#([\w-]+)$/);
    if (direct) return ids.get(direct[1]) || null;
    const child = selector.match(/^#([\w-]+) (.+)$/);
    if (child && ids.has(child[1])) {
      const parent = ids.get(child[1]);
      let node = parent.querySelector(child[2]);
      if (!node) { node = new Element(child[2].split(':')[0]); parent.append(node); }
      return node;
    }
    return Element.prototype.querySelector.call(document, selector);
  };
  const context = vm.createContext({document, console, Blob, AbortController, URL, TextEncoder,
    location: {hash: '#inputs'}, window: {addEventListener() {}},
    setTimeout: () => 1, clearTimeout() {}, requestAnimationFrame: callback => callback(),
    fetch: async () => { throw new Error('Unexpected network request'); },
  });
  for (const filename of ['topology.js', 'execution.js']) {
    const file = path.join(root, 'supernode_sim/web_static', filename);
    if (fs.existsSync(file)) vm.runInContext(fs.readFileSync(file, 'utf8'), context, {filename});
  }
  assert.match(source, /initialize\(\);\s*$/);
  vm.runInContext(source.replace(/initialize\(\);\s*$/, ''), context, {filename: 'app.js'});
  const evaluate = expression => vm.runInContext(expression, context);
  context.initialConfig = structuredClone(config);
  evaluate(`state.config = initialConfig; state.modified = false; state.jsonDirty = false;
    inputView.source = 'draft'; $('#inputs').hidden = false; $('#run-mode').value = 'simulate';
    globalThis.realRenderTopology = renderTopology; globalThis.realRenderJob = renderJob; globalThis.realRenderForm = renderForm;
    renderTopology = () => {}; globalThis.realRenderInputNotes = renderInputNotes; renderInputNotes = () => {}; renderHistory = () => {};
    renderJob = () => {}; globalThis.realUpdateRunButton = updateRunButton; updateRunButton = () => {}; setConnected = () => {};
    selectJob = job => { state.currentJob = job; }; refreshHistory = async () => {};
    renderForm = () => { syncJson(); }; routeDashboard = () => {};
    syncJson();`);
  if (views) evaluate('renderTopology = realRenderTopology; renderJob = realRenderJob; renderForm = realRenderForm');
  return {
    context, document, ids, evaluate,
    json: expression => JSON.parse(evaluate(`JSON.stringify(${expression})`)),
    call: (name, ...args) => { context.callArgs = args; return evaluate(`${name}(...callArgs)`); },
    node: id => ids.get(id),
  };
}


module.exports = {Element, fixture, harness};
