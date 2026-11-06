"""Reproducible full POD/DFlash matrix, with bounded workers and SQLite resume."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import csv
import hashlib
import itertools
import json
from pathlib import Path
import sqlite3
import sys
import time
import zlib

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from supernode_sim.case_studies.v41_fp4 import provenance
from supernode_sim.case_studies.v41_fp4 import model
from supernode_sim.case_studies.v41_fp4.fine import run_phase, PROFILE

DEFAULT_OUTPUT=ROOT/'outputs/deepseek-v41-fp4-operator-matrix-20261105'


def save(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2),encoding='utf-8')


def key(task): return '|'.join(map(str,task))


def tasks_for(chip,cards,context,batch,ep,tp,sp,accepted,block):
    dp=cards//(tp*ep)
    local=batch//min(batch,dp)
    return [(chip,tp,ep,sp,1,context,block,'prefill',1)]+[
        (chip,tp,ep,sp,local,context,block,phase,count)
        for phase,count in [('draft',1),('verify',1),('commit',accepted)]]


def task_set():
    result=set()
    for chip,cards,ctx,batch,ep in itertools.product(['950pr','950dt'],model.CARDS,model.CONTEXTS.values(),model.BATCHES,model.EPS):
        for tp in (1,2,4,8):
            if tp*ep>cards: continue
            for sp in sorted({1,tp}):
                for a,k in model.PAIRS:
                    result.update(tasks_for(chip,cards,ctx,batch,ep,tp,sp,a,k))
    return sorted(result)


def worker(tasks):
    output=[]
    for task in tasks:
        r=run_phase(*task)
        r.pop('stages',None)
        r.pop('operator_counts',None)
        # Static memory per lane and operator component detail remain in the
        # compressed phase checkpoint; only the summary crosses into tables.
        output.append((key(task),r))
    return output


def get_parts(result,divisor=1):
    if result['breakdown_s'] is None:
        return {k:None for k in ('compute','compute_service','memory','communication','operator_launch','hbm','host_memory','su','host_link','comm_startup')}
    b=result['breakdown_s']; scale=1000/divisor
    compute=b.get('compute_bound_s',0)+b.get('cpu_compute_s',0)
    launch=b.get('operator_launch_s',0)
    return dict(compute=(compute+launch)*scale,compute_service=compute*scale,
                memory=(b.get('hbm_bound_s',0)+b.get('host_memory_s',0))*scale,
                communication=sum(b.get(x,0) for x in ('su_transfer_s','host_transfer_s','communication_startup_s'))*scale,
                operator_launch=launch*scale,hbm=b.get('hbm_bound_s',0)*scale,
                host_memory=b.get('host_memory_s',0)*scale,su=b.get('su_transfer_s',0)*scale,
                host_link=b.get('host_transfer_s',0)*scale,comm_startup=b.get('communication_startup_s',0)*scale)


def result_row(chip,cards,label,batch,ep,tp,sp,a,k,results):
    context=model.CONTEXTS[label];dp=cards//(tp*ep)
    tasks=tasks_for(chip,cards,context,batch,ep,tp,sp,a,k)
    pre,draft,verify,commit=[results[key(t)] for t in tasks]
    decode=model.aggregate([draft,verify,commit])
    pok,dok=pre['status']=='feasible',decode['status']=='feasible'
    row=dict(chip=chip,cards=cards,context=label,context_tokens=context,prefill_batch=1,decode_batch=batch,
             ep=ep,accepted=a,block=k,tp=tp,sp=sp,dp=dp,active_decode_replicas=min(batch,dp),
             local_decode_batch=batch//min(batch,dp),
             prefill_status=pre['status'],decode_status=decode['status'],joint_status='feasible' if pok and dok else 'infeasible',
             reason=';'.join(f'{phase}_{"HBM" if r["memory"]["hbm_peak_ratio"]>1 else "host"}_capacity'
                             for phase,r in [('prefill',pre),('decode',decode)] if r['status']!='feasible'),
             ttft_ms=pre['duration_s']*1000 if pok else None,
             tpot_ms=decode['duration_s']*1000/a if dok else None,
             round_ms=decode['duration_s']*1000 if dok else None,
             decode_tokens_s=batch*a/decode['duration_s'] if dok else None,
             prefill_peak_GB=pre['memory']['hbm_peak_bytes']/1e9,
             decode_peak_GB=decode['memory']['hbm_peak_bytes']/1e9,
             static_GB_per_card=pre['memory']['static_bytes_per_card']/1e9,
             host_peak_GB=max(pre['memory']['host_peak_bytes'],decode['memory']['host_peak_bytes'])/1e9,
             prefill_capacity_check=pre['capacity_check'],
             ttft_target_ms=(pre['duration_s']-pre['seed_s'])*1000 if pok else None,
             ttft_draft_seed_ms=pre['seed_s']*1000 if pok else None,
             draft_round_ms=draft['duration_s']*1000 if draft['duration_s'] is not None else None,
             verify_round_ms=verify['duration_s']*1000 if verify['duration_s'] is not None else None,
             commit_round_ms=commit['duration_s']*1000 if commit['duration_s'] is not None else None,
             prefill_phase_key=key(tasks[0]),draft_phase_key=key(tasks[1]),verify_phase_key=key(tasks[2]),commit_phase_key=key(tasks[3]))
    for prefix,r,divisor in [('ttft',pre,1),('tpot',decode,a)]:
        parts=get_parts(r,divisor)
        row.update({prefix+'_'+name+'_ms':value for name,value in parts.items()})
        total=row[prefix+'_ms'];launch=parts['operator_launch']
        row[prefix+'_launch_half_ms']=total-launch*.5 if total is not None else None
        row[prefix+'_launch_double_ms']=total+launch if total is not None else None
        if total is not None:
            assert abs(total-sum(parts[k] for k in ('compute','memory','communication'))) < max(1e-6,total*1e-9)
    row['prefill_operator_events']=pre['operator_trace'].get('generated_operations',0) if pok else None
    row['decode_operator_events']=sum(r['operator_trace'].get('generated_operations',0) for r in (draft,verify,commit)) if dok else None
    return row


def csv_write(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def export(results,output):
    detail=[];matrix=[];bestpre=[];bestdecode=[]
    for chip,cards,label,batch,ep,(a,k) in itertools.product(['950pr','950dt'],model.CARDS,model.CONTEXTS,model.BATCHES,model.EPS,model.PAIRS):
        candidates=[]
        for tp in (1,2,4,8):
            if tp*ep>cards:continue
            for sp in sorted({1,tp}):
                row=result_row(chip,cards,label,batch,ep,tp,sp,a,k,results)
                candidates.append(row);detail.append(row)
        if not candidates:
            row=dict(chip=chip,cards=cards,context=label,context_tokens=model.CONTEXTS[label],prefill_batch=1,
                     decode_batch=batch,ep=ep,accepted=a,block=k,prefill_status='incompatible',
                     decode_status='incompatible',joint_status='incompatible',reason='EP_exceeds_total_cards')
            detail.append(row);matrix.append(row);bestpre.append(row);bestdecode.append(row)
            continue
        viable=[r for r in candidates if r['joint_status']=='feasible']
        dec=[r for r in candidates if r['decode_status']=='feasible']
        pre=[r for r in candidates if r['prefill_status']=='feasible']
        fallback=min(candidates,key=lambda r:r['static_GB_per_card'])
        chosen=min(viable or dec,key=lambda r:r['tpot_ms']) if viable or dec else min(pre,key=lambda r:r['ttft_ms']) if pre else fallback
        matrix.append(chosen)
        bestpre.append(min(pre,key=lambda r:r['ttft_ms']) if pre else fallback)
        bestdecode.append(min(dec,key=lambda r:r['tpot_ms']) if dec else fallback)
    assert len(matrix)==23040
    for filename,rows in [('matrix',matrix),('layout_detail',detail),('best_prefill_by_EP',bestpre),('best_decode_by_EP',bestdecode)]:
        csv_write(output/(filename+'.csv'),rows)
    save(output/'matrix.json',matrix)
    # Interactive report reads compact arrays rather than duplicating keys.
    fields=list(dict.fromkeys(k for row in matrix for k in row))
    save(output/'matrix.compact.json',dict(fields=fields,rows=[[r.get(k) for k in fields] for r in matrix]))
    stats=dict(matrix_rows=len(matrix),layout_rows=len(detail),phase_cases=len(results),
               joint_status=dict(Counter(r['joint_status'] for r in matrix)),
               prefill_status=dict(Counter(r['prefill_status'] for r in matrix)),
               decode_status=dict(Counter(r['decode_status'] for r in matrix)),
               trace_eligible=sum(r['operator_trace'].get('eligible_events',0) for r in results.values()),
               trace_matched=sum(r['operator_trace'].get('matched_events',0) for r in results.values()),
               generated_suboperators=sum(r['operator_trace'].get('generated_operations',0) for r in results.values()),
               executed_events=sum(r['native_completed_events'] for r in results.values()))
    save(output/'execution_summary.json',stats)
    return stats


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-o','--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--export-only',action='store_true')
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args();output=args.output;output.mkdir(parents=True,exist_ok=True)
    tasks=task_set();prov=provenance()
    script_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    fingerprint=hashlib.sha256((prov['numeric_fingerprint']+script_hash).encode()).hexdigest()
    manifest=dict(fingerprint=fingerprint,provenance=prov,runner_sha256=script_hash,
                  contexts=model.CONTEXTS,batches=model.BATCHES,ep=model.EPS,cards=model.CARDS,dflash=model.PAIRS,
                  tp=[1,2,4,8],sp='1 or TP',dp='cards/(TP*EP)',prefill_batch=1,decode_batch_scope='all cards, across active DP replicas',
                  precision=model.PRECISION_NOTE,model=model.MODEL,draft=model.DRAFT,
                  hardware={chip:model.hardware(chip,512) for chip in ('950pr','950dt')},
                  operator_launch=PROFILE,phase_tasks=len(tasks),
                  topology='One logical full-mesh SU POD. 1 TB/s/direction at 80%; 2.1 us/round. No SO. Disjoint replicas, no shared uplink contention.',
                  timing='TTFT=prefill+draft KV seed, batch 1. TPOT=(draft+verify+commit)/accepted at fixed context. No bonus token. No client queuing.',
                  decomposition='compute=compute service+operator launch; memory=HBM/DRAM/IOPS bottleneck service; communication=SU/host transfer+network startup. Per-op roofline max is attributed once.',
                  selection='Main matrix minimizes TPOT among jointly feasible layouts, otherwise independent decode, prefill, or smallest static footprint. Best-prefill and best-decode tables are independent.',
                  limitations=['Synthetic design, not official checkpoint or measured platform results.',
                               'No chunked prefill or context parallelism; 4M is a cost/capacity extrapolation.',
                               'Rank resource envelopes and serialized barriers inherited from prior study.',
                               'Fused attention/mHC/pointwise kernel boundaries are assumptions; no binary-level runtime trace.',
                               'Extra unfused HBM is reported per critical rank envelope, not invented aggregate physical traffic.'])
    save(output/'experiment.json',manifest)
    save(output/'operator_launch_assumptions.json',PROFILE)
    print(json.dumps(dict(tasks=len(tasks),fingerprint=fingerprint,workers=args.workers)),flush=True)
    if args.prepare_only:return
    db=sqlite3.connect(output/'phase_checkpoints.sqlite3')
    db.execute('CREATE TABLE IF NOT EXISTS phases (key TEXT PRIMARY KEY, fingerprint TEXT, payload BLOB)')
    results={}
    for label,payload in db.execute('SELECT key,payload FROM phases WHERE fingerprint=?',(fingerprint,)):
        results[label]=json.loads(zlib.decompress(payload))
    pending=[task for task in tasks if key(task) not in results]
    if args.export_only and pending:raise RuntimeError(f'{len(pending)} missing matching phase checkpoints')
    started=time.monotonic();last=started
    if pending:
        batches=iter([pending[i:i+32] for i in range(0,len(pending),32)])
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures={pool.submit(worker,chunk) for chunk in itertools.islice(batches,args.workers*2)}
            while futures:
                done,futures=wait(futures,return_when=FIRST_COMPLETED)
                for future in done:
                    for label,r in future.result():
                        results[label]=r
                        payload=zlib.compress(json.dumps(r,separators=(',',':'),allow_nan=False).encode(),1)
                        db.execute('INSERT OR REPLACE INTO phases VALUES (?,?,?)',(label,fingerprint,payload))
                    db.commit()
                    chunk=next(batches,None)
                    if chunk:futures.add(pool.submit(worker,chunk))
                if time.monotonic()-last>=15:
                    progress=dict(done=len(results),total=len(tasks),elapsed_s=round(time.monotonic()-started,1))
                    save(output/'progress.json',progress);print(json.dumps(progress),flush=True);last=time.monotonic()
    stats=export(results,output);stats['wall_seconds']=time.monotonic()-started
    stats['fingerprint']=fingerprint
    save(output/'execution_summary.json',stats);save(output/'progress.json',dict(status='complete',**stats))
    print(json.dumps(stats),flush=True)


if __name__=='__main__':main()
