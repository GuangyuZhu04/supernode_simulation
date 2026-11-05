'use strict';
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {Element} = require('./web_frontend_harness.cjs');
const root = path.resolve(__dirname, '..');
const html = fs.readFileSync(path.join(root, 'supernode_sim/web_static/fp4-case.html'), 'utf8');
const source = fs.readFileSync(path.join(root, 'supernode_sim/web_static/fp4-case.js'), 'utf8');
const config = JSON.parse(fs.readFileSync(path.join(root, 'examples/deepseek_v41_fp4_512_ep128_950dt.json')));
const catalog = {id:config.case_id, presets:[{config}], contexts:{'128K':131072,'4M':4194304}, notes:['条件仿真']};
const copy = value => JSON.parse(JSON.stringify(value));

async function setup() {
  const ids=new Map();
  for(const match of html.matchAll(/<([\w-]+)\b([^>]*\bid="([^"]+)"[^>]*)>/g)) {
    const n=new Element(match[1]);n.id=match[3];ids.set(n.id,n);n.hidden=/\bhidden\b/.test(match[2]);
  }
  const create=tag=>{const e=new Element(tag);e.style.setProperty=(key,value)=>{e.style[key]=value;};return e;};
  const calls=[],jobs=new Map();
  const context=vm.createContext({document:{querySelector:selector=>ids.get(selector.slice(1)),createElement:create},
    Option:function(text,value){const e=create('option');e.textContent=text;e.value=String(value);return e;},
    URL,URLSearchParams,Blob,location:{search:''},history:{replaceState(){}},
    setTimeout:()=>1,clearTimeout(){},console,
    fetch:async(url,options)=>{
      const body=options?.body?JSON.parse(options.body):null;calls.push({url,body});let data;
      if(url==='/api/case-presets')data=catalog;
      else if(url==='/api/validate')data={valid:true,config:body.config,summary:{dp:512/(128*body.config.tp)}};
      else if(url==='/api/jobs'&&body){const job={id:'a'.repeat(32),mode:'pod_dflash',status:'running',elapsed_s:0,created_at:'2026-11-02T00:00:00Z',logs:[]};jobs.set(job.id,{job,config:copy(body.config)});data=job;}
      else if(url==='/api/jobs')data={jobs:[...jobs.values()].map(x=>x.job)};
      else {const parts=url.split('/'),saved=jobs.get(parts[3]);data=parts[4]==='config'?{config:saved.config}:saved.job;}
      return {ok:true,json:async()=>copy(data)};
    }});
  await vm.runInContext(source,context,{filename:'fp4-case.js'});
  return {ids,calls,jobs,context,evaluate:s=>vm.runInContext(s,context),json:s=>JSON.parse(vm.runInContext('JSON.stringify('+s+')',context))};
}

test('FP4 case defaults and layout changes preserve 512 cards and global decode batch',async()=>{
  const h=await setup();assert.deepEqual(h.json('draft()'),config);
  h.ids.get('chip').value='950pr';h.ids.get('tp').value='2';await h.ids.get('tp').fire('change');
  h.ids.get('sp').value='2';await h.ids.get('case-form').fire('change');
  const draft=h.json('draft()');assert.equal(draft.chip,'950pr');assert.equal(draft.tp,2);assert.equal(draft.sp,2);assert.equal(draft.decode_batch,32);
  assert.match(h.ids.get('layout-note').textContent,/DP2 = 512/);assert.match(h.ids.get('layout-note').textContent,/每副本 batch 16/);
  assert.match(h.ids.get('hardware-note').textContent,/128 GB/);
});

test('run submits a validated independent case snapshot once and restores it explicitly',async()=>{
  const h=await setup();await h.ids.get('case-form').fire('submit');
  assert.equal(h.calls.filter(x=>x.url==='/api/jobs'&&x.body).length,1);
  assert.deepEqual(h.calls.find(x=>x.url==='/api/jobs'&&x.body).body,{mode:'pod_dflash',config});
  h.ids.get('chip').value='950pr';await h.ids.get('case-form').fire('change');
  assert.equal(h.json('currentSnapshot.chip'),'950dt');assert.equal(h.json('draft().chip'),'950pr');
  await h.ids.get('case-form').fire('submit');assert.equal(h.calls.filter(x=>x.url==='/api/jobs'&&x.body).length,1);
  await h.ids.get('restore').fire('click');assert.equal(h.json('draft().chip'),'950dt');
});

test('capacity failure never renders missing TTFT as zero and preserves decode result',async()=>{
  const h=await setup();h.context.result={config,status:'infeasible',metrics:{ttft_s:null,tpot_s:.02,round_s:.12,decode_tokens_s:1600,latency_breakdown:{ttft:null,tpot:{hbm_bound_s:.02}}},phases:{prefill:{status:'infeasible',duration_s:null,native_completed_events:0,memory:{hbm_peak_bytes:200e9,host_peak_bytes:60e9}}},decode_physical_work:{hbm_bytes:12e9}};
  h.evaluate('renderResult(result,{artifacts:[]})');
  assert.match(h.ids.get('metrics').children[0].textContent,/不可用/);assert.match(h.ids.get('metrics').children[1].textContent,/20/);
  assert.match(h.ids.get('phases').textContent,/容量不可行/);assert.match(h.ids.get('breakdown').textContent,/无完整时延拆解/);
  assert.equal(h.ids.get('result-panel').hidden,false);
});

test('late job input responses do not overwrite a newer selection',async()=>{
  const h=await setup();let resolveOld;
  const first='a'.repeat(32),second='b'.repeat(32);
  h.context.fetch=async url=>{
    if(url==='/api/jobs')return {ok:true,json:async()=>({jobs:[]})};
    const id=url.split('/')[3];
    if(url.endsWith('/config')) {
      if(id===first)await new Promise(resolve=>{resolveOld=resolve;});
      return {ok:true,json:async()=>({config:{...config,chip:id===first?'950dt':'950pr'}})};
    }
    return {ok:true,json:async()=>({id,mode:'pod_dflash',status:'cancelled',elapsed_s:1,logs:[]})};
  };
  const old=h.evaluate(`selectJob('${first}')`);
  while(!resolveOld)await new Promise(resolve=>setImmediate(resolve));
  await h.evaluate(`selectJob('${second}')`);resolveOld();await old;
  assert.equal(h.json('currentSnapshot.chip'),'950pr');assert.equal(h.json('currentJob.id'),second);
});
