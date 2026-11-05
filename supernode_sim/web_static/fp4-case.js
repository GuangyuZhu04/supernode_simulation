'use strict';
const $ = selector => document.querySelector(selector);
const MODE = 'pod_dflash';
let catalog, currentJob = null, currentSnapshot = null, pollTimer, selection = 0, busy = false;
const active = job => job && ['queued', 'running'].includes(job.status);
const labels = {queued:'等待启动', running:'运行中', completed:'已完成', failed:'执行失败', cancelled:'已取消', feasible:'容量可行', infeasible:'容量不可行'};
const fmt = (value, digits=3) => Number.isFinite(value) ? value.toLocaleString('zh-CN', {maximumFractionDigits:digits}) : '不可用';
function node(tag, text, className) { const n=document.createElement(tag); if(text!==undefined)n.textContent=text; if(className)n.className=className; return n; }
async function api(path, payload) {
  const response=await fetch(path, payload===undefined ? {} : {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
  const data=await response.json();
  if(!response.ok) throw new Error(data.error + (data.details ? '：'+data.details.map(x=>x.msg).join('；') : ''));
  return data;
}
function draft() {
  const [accepted,block]=$('#dflash').value.split('/').map(Number);
  return {case_id:catalog.id, chip:$('#chip').value, cards:512, ep:128, tp:Number($('#tp').value), sp:Number($('#sp').value),
          context_tokens:Number($('#context').value), prefill_batch:1, decode_batch:Number($('#decode-batch').value), accepted, block};
}
function spOptions(value=Number($('#sp').value)) {
  const tp=Number($('#tp').value); $('#sp').replaceChildren(...[...new Set([1,tp])].map(x=>new Option(String(x),String(x))));
  $('#sp').value=String([1,tp].includes(value)?value:1);
}
function describe(c) { return `${c.chip.toUpperCase()} · 512 卡 · TP${c.tp} / EP128 / DP${512/(128*c.tp)} / SP${c.sp} · ${fmt(c.context_tokens/1024,0)}K · P1 / D${c.decode_batch} · 接受 ${c.accepted}/${c.block}`; }
function changed() {
  const c=draft(), dp=512/(128*c.tp), replicas=Math.min(dp,c.decode_batch);
  $('#hardware-note').textContent=c.chip==='950dt'?'HBM 96 GB / 卡，4 TB/s；FP8 峰值 1 PF/s × 50%。':'HBM 128 GB / 卡，1.6 TB/s；FP8 峰值 1 PF/s × 50%。';
  $('#layout-note').textContent=`TP${c.tp} × EP128 × DP${dp} = 512；活跃 Decode 副本 ${replicas}，每副本 batch ${c.decode_batch/replicas}。SU 1 TB/s/方向 × 80%，启动 2.1 μs。`;
  $('#config-json').textContent=JSON.stringify(c,null,2); message('当前编辑配置尚未校验。');
}
function fill(c) {
  $('#chip').value=c.chip;$('#tp').value=c.tp;spOptions(c.sp);$('#context').value=c.context_tokens;
  $('#decode-batch').value=c.decode_batch;$('#dflash').value=c.accepted+'/'+c.block;changed();
}
function message(text, error=false) { $('#input-status').textContent=text;$('#input-status').classList.toggle('case-error',error); }
function controls() { $('#case-fields').disabled=busy||!catalog;$('#run').disabled=busy||active(currentJob);$('#validate').disabled=busy;$('#cancel').disabled=!active(currentJob);$('#restore').disabled=!currentSnapshot||busy; }
async function validate() {
  const result=await api('/api/validate',{mode:MODE,config:draft()});
  message(`校验通过 · 512 卡 / EP128 / DP${result.summary.dp}`);return result.config;
}
function renderJob(job) {
  $('#run-status').textContent=labels[job.status]||job.status;
  $('#run-caption').textContent=`任务 ${job.id.slice(0,8)} · 运行耗时 ${fmt(job.elapsed_s,1)} 秒${currentSnapshot?' · '+describe(currentSnapshot):''}`;
  $('#logs').textContent=(job.logs||[]).map(x=>x.message).join('\n')+(job.error?'\n'+job.error:'');controls();
}
function table(host, headers, rows) {
  const t=node('table'),head=node('thead'),tr=node('tr'); headers.forEach(x=>tr.append(node('th',x)));head.append(tr);t.append(head);
  const body=node('tbody');rows.forEach(values=>{const row=node('tr');values.forEach(v=>row.append(node('td',String(v))));body.append(row);});t.append(body);host.replaceChildren(t);
}
function renderResult(r, job) {
  const m=r.metrics;$('#result-panel').hidden=false;$('#result-status').textContent=labels[r.status]||r.status;
  $('#result-caption').textContent='结果来自运行快照：'+describe(r.config)+'。修改左侧输入后，需重新运行才会生成新结果。';
  $('#metrics').replaceChildren(...[
    ['TTFT',fmt(m.ttft_s*1000),'ms · Prefill batch 1',m.ttft_s],
    ['TPOT',fmt(m.tpot_s*1000),'ms · 固定上下文',m.tpot_s],
    ['Decode 吞吐',fmt(m.decode_tokens_s,1),'token/s · 条件预测',m.decode_tokens_s],
    ['每轮耗时',fmt(m.round_s*1000),'ms · draft + verify + commit',m.round_s]
  ].map(([label,value,unit,raw])=>{const card=node('div',undefined,'case-metric');card.append(node('span',label),node('strong',raw===null?'不可用':value),node('small',unit));return card;}));
  const categories=[['计算',['compute_bound_s','cpu_compute_s'],'#168d7e'],['HBM',['hbm_bound_s'],'#607fc2'],['主机内存',['host_memory_s'],'#8b70ae'],['SU 传输',['su_transfer_s'],'#e09d4e'],['Host 传输',['host_transfer_s'],'#cc785a'],['通信启动',['communication_startup_s'],'#8c9ca9']];
  $('#breakdown').replaceChildren();
  for(const metric of ['ttft','tpot']) {
    const values=m.latency_breakdown[metric],box=node('div',undefined,'case-breakdown');box.append(node('strong',metric.toUpperCase()));
    if(!values){box.append(node('p','阶段不可行，无完整时延拆解。'));$('#breakdown').append(box);continue;}
    const total=Object.values(values).reduce((a,b)=>a+b,0),bar=node('div',undefined,'case-bar'),legend=node('div',undefined,'case-legend');
    categories.forEach(([label,keys,color])=>{const value=keys.reduce((sum,key)=>sum+(values[key]||0),0);const part=node('span');part.style.width=(total?value/total*100:0)+'%';part.style.background=color;part.title=label+' '+fmt(value*1000)+' ms';bar.append(part);const text=node('span',label+' '+fmt(value*1000)+' ms');text.style.setProperty('--color',color);legend.append(text);});
    box.append(bar,legend);$('#breakdown').append(box);
  }
  const names={prefill:'Prefill + KV seed',draft:'DFlash draft',verify:'Target verify',commit:'Commit + KV catchup'};
  table($('#phases'),['阶段','状态','时延 ms','HBM GB/卡','主机 GB','容量依据','完成事件'],Object.entries(r.phases).map(([name,p])=>[names[name],labels[p.status]||p.status,p.duration_s===null?'不可用':fmt(p.duration_s*1000),fmt(p.memory.hbm_peak_bytes/1e9),fmt(p.memory.host_peak_bytes/1e9),p.capacity_check==='persistent_lower_bound'?'驻留下界':'完整阶段预留',p.native_completed_events]));
  $('#traffic').replaceChildren(...[['HBM','hbm_bytes'],['SU 逐跳','su_physical_bytes'],['Host','host_physical_bytes']].map(([label,key])=>{const box=node('div');box.append(node('span',label),node('strong',r.decode_physical_work?fmt((r.decode_physical_work[key]||0)/1e9)+' GB':'不可用'));return box;}));
  $('#artifacts').replaceChildren(...(job.artifacts||[]).filter(x=>['result.json','report.md','config.resolved.json','latency_breakdown.csv','phases.csv','provenance.json'].includes(x.name)).map(x=>{const a=node('a',x.name);a.href=x.url;a.download=x.name;return a;}));
}
async function refreshHistory() {
  const data=await api('/api/jobs'),jobs=(data.jobs||[]).filter(j=>j.mode===MODE);
  $('#history').replaceChildren(...jobs.slice(0,30).map(job=>{const button=node('button',`${labels[job.status]||job.status} · ${new Date(job.created_at).toLocaleString('zh-CN')} · ${job.id.slice(0,8)}`,'case-history');button.type='button';button.addEventListener('click',()=>selectJob(job.id).catch(error=>message(error.message,true)));return button;}));
  if(!jobs.length)$('#history').textContent='尚无案例运行。';return jobs;
}
async function pollJob(id, token) {
  const job=await api('/api/jobs/'+id);if(token!==selection)return;
  currentJob=job;renderJob(job);
  if(job.status==='completed') {const result=await api('/api/jobs/'+id+'/result');if(token===selection)renderResult(result,job);}
  if(active(job))pollTimer=setTimeout(()=>pollJob(id,token).catch(error=>message('状态读取失败，请从历史选择原任务重试：'+error.message,true)),1000);
  else await refreshHistory();
}
async function selectJob(id) {
  clearTimeout(pollTimer);const token=++selection;currentSnapshot=null;$('#result-panel').hidden=true;$('#restore').disabled=true;
  const job=await api('/api/jobs/'+id);if(token!==selection)return;
  if(job.mode!==MODE)throw new Error('所选任务不属于 FP4 POD / DFlash 案例');
  currentJob=job;const saved=await api('/api/jobs/'+id+'/config');if(token!==selection)return;
  currentSnapshot=saved.config;history.replaceState(null,'','/fp4-case.html?job='+id);await pollJob(id,token);
}
$('#tp').addEventListener('change',()=>{spOptions();changed();});
$('#case-form').addEventListener('change',changed);
$('#validate').addEventListener('click',()=>validate().catch(error=>message(error.message,true)));
$('#case-form').addEventListener('submit',async event=>{
  event.preventDefault();if(busy||active(currentJob))return;busy=true;controls();let posting=false,created=false;
  try {const config=await validate();posting=true;const job=await api('/api/jobs',{mode:MODE,config});created=true;await refreshHistory();await selectJob(job.id);message('任务已创建；当前结果与本次输入快照绑定。');}
  catch(error){message((posting&&!created?'提交结果未确认，请先刷新历史检查是否已创建，勿重复提交。':'')+error.message,true);await refreshHistory().catch(()=>{});}
  finally{busy=false;controls();}
});
$('#cancel').addEventListener('click',async()=>{if(!active(currentJob))return;try{const id=currentJob.id;await api('/api/jobs/'+id+'/cancel',{});await selectJob(id);}catch(error){message(error.message,true);}});
$('#restore').addEventListener('click',()=>{if(currentSnapshot){fill(currentSnapshot);message('已复制所选运行的独立输入；可编辑后重新运行。');}});
$('#refresh').addEventListener('click',()=>refreshHistory().catch(error=>message(error.message,true)));
$('#export').addEventListener('click',()=>{const blob=new Blob([JSON.stringify(draft(),null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=node('a');a.href=url;a.download='deepseek-v41-fp4-512-ep128-'+draft().chip+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
(async()=>{
  try {
    catalog=await api('/api/case-presets');$('#context').replaceChildren(...Object.entries(catalog.contexts).map(([label,value])=>new Option(label+' · '+fmt(value,0)+' token',value)));
    $('#notes').replaceChildren(...catalog.notes.map(text=>node('li',text)));fill(catalog.presets[0].config);$('#case-fields').disabled=false;
    $('#connection').textContent='本地引擎已连接';const jobs=await refreshHistory(),id=new URLSearchParams(location.search).get('job');
    if(id&&/^[0-9a-f]{32}$/.test(id))await selectJob(id);else if(jobs.length)await selectJob(jobs[0].id);
  }catch(error){$('#connection').textContent='加载失败';message(error.message,true);}
})();
