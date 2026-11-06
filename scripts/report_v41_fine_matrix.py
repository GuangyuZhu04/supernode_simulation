"""Validate and present the full operator-level V4.1 sweep without rerunning it."""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import csv
import gzip
import itertools
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from supernode_sim.case_studies.v41_fp4 import model
from supernode_sim.case_studies.v41_fp4.architecture import summarize
from supernode_sim.case_studies.v41_fp4.fine import run_phase, PROFILE, startup_kind
from run_v41_fine_matrix import DEFAULT_OUTPUT,save,csv_write


def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as handle:
        rows=list(csv.DictReader(handle))
    strings={'chip','context','prefill_status','decode_status','joint_status','reason','prefill_capacity_check',
             'prefill_phase_key','draft_phase_key','verify_phase_key','commit_phase_key'}
    for row in rows:
        for k,v in row.items():
            if v=='':row[k]=None
            elif k not in strings:row[k]=float(v) if any(c in v for c in '.eE') else int(v)
    return rows


def fmt(value,digits=3): return '不可用' if value is None else f'{value:,.{digits}f}'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-o','--output',type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args();out=args.output
    matrix=json.loads((out/'matrix.json').read_text(encoding='utf-8'))
    detail=read_csv(out/'layout_detail.csv')
    stats=json.loads((out/'execution_summary.json').read_text(encoding='utf-8'))
    manifest=json.loads((out/'experiment.json').read_text(encoding='utf-8'))
    expected=set(itertools.product(['950pr','950dt'],model.CARDS,model.CONTEXTS,model.BATCHES,model.EPS,model.PAIRS))
    observed={(r['chip'],r['cards'],r['context'],r['decode_batch'],r['ep'],(r['accepted'],r['block'])) for r in matrix}
    assert observed==expected and len(matrix)==len(expected)==23040
    checked=0;max_error=0
    for row in detail:
        if row['joint_status']=='incompatible':
            assert row['ep']>row['cards']
            continue
        assert row['tp']*row['ep']*row['dp']==row['cards']
        assert row['local_decode_batch']*row['active_decode_replicas']==row['decode_batch']
        for metric,status in [('ttft','prefill_status'),('tpot','decode_status')]:
            total=row[metric+'_ms']
            if row[status]!='feasible':
                assert total is None
                continue
            parts=sum(row[metric+'_'+p+'_ms'] for p in ('compute','memory','communication'))
            error=abs(total-parts);max_error=max(max_error,error)
            assert error<max(1e-6,total*1e-9)
            assert row[metric+'_launch_half_ms']<=total<=row[metric+'_launch_double_ms']
            checked+=1
        if row['tpot_ms'] is not None:
            assert math.isclose(row['tpot_ms']*row['accepted'],sum(row[p+'_round_ms'] for p in ('draft','verify','commit')),rel_tol=1e-9)
    assert stats['trace_matched']==stats['trace_eligible']
    validation=dict(all_23040_requested_combinations_present=True,layout_rows=len(detail),
                    latency_additivity_checks=checked,max_additivity_error_ms=max_error,
                    all_trace_selectors_matched=True,trace_eligible=stats['trace_eligible'],
                    generated_suboperators=stats['generated_suboperators'],
                    failed_latencies_are_null=True,parallel_and_batch_products_correct=True,
                    sensitivity_is_analytical_exact_for_serial_graph=True,
                    sensitivity_test='Native reruns at 1x and 2x for every phase verified by tests/test_v41_fine.py')
    save(out/'validation.json',validation)
    fixed=[r for r in detail if r.get('cards')==512 and r.get('ep')==128 and r.get('tp')==4 and r.get('sp')==1 and r.get('decode_batch')==32 and r.get('accepted')==6]
    assert len(fixed)==20
    csv_write(out/'reference_512_EP128_TP4_B32_DFlash6of8.csv',fixed)
    grouped=defaultdict(list)
    for row in matrix:
        if row['decode_batch']==32 and row['accepted']==6:
            grouped[(row['chip'],row['cards'],row['context'])].append(row)
    optimal=[]
    for chip,cards,context in itertools.product(['950pr','950dt'],model.CARDS,model.CONTEXTS):
        rows=grouped[(chip,cards,context)]
        feasible=[r for r in rows if r['joint_status']=='feasible']
        if feasible:chosen=min(feasible,key=lambda r:r['tpot_ms'])
        else:chosen=dict(chip=chip,cards=cards,context=context,joint_status='infeasible',reason='no_jointly_feasible_layout')
        optimal.append(chosen)
    csv_write(out/'best_joint_B32_DFlash6of8.csv',optimal)
    long=[]
    identity=['chip','cards','context','context_tokens','decode_batch','ep','accepted','block','tp','sp','dp']
    for row in matrix:
        for metric,status in [('ttft','prefill_status'),('tpot','decode_status')]:
            long.append({**{k:row.get(k) for k in identity},'metric':metric,'status':row[status],
                         'total_ms':row.get(metric+'_ms'),
                         **{p+'_ms':row.get(metric+'_'+p+'_ms') for p in ('compute','compute_service','operator_launch','memory','hbm','host_memory','communication','su','host_link','comm_startup')}})
    csv_write(out/'latency_breakdown.csv',long)
    audit=[];launches={}
    for chip in ('950pr','950dt'):
        phases={}
        for phase,batch,count in [('prefill',1,1),('draft',32,1),('verify',32,1),('commit',32,6)]:
            result=run_phase(chip,4,128,1,batch,131072,8,phase,count,keep_stages=True,keep_graph=True)
            phases[phase]=result
            assert result['status']=='feasible'
            for event in result['stages']:
                if 'SU.tx' not in event['work'] and 'HOST' not in event['work']:
                    name=event['name'];kind=startup_kind(name)
                    launches[name]=dict(operator=name,kind=kind,startup_us=PROFILE['startup_us'][kind],evidence='assumed_not_measured')
            audit.append(dict(chip=chip,phase=phase,events=result['native_completed_events'],duration_ms=result['duration_s']*1000,
                              trace_parents=result['operator_trace']['matched_events'],suboperators=result['operator_trace']['generated_operations']))
        with gzip.open(out/f'audit_{chip}_512_EP128_TP4_B32_128K_DFlash6of8.json.gz','wt',encoding='utf-8') as handle:
            json.dump(phases,handle,ensure_ascii=False,allow_nan=False,separators=(',',':'))
        save(out/f'audit_{chip}_summary.json',{p:{k:v for k,v in r.items() if k not in ('trace_config','stages','operator_counts')} for p,r in phases.items()})
        row=next(r for r in fixed if r['chip']==chip and r['context']=='128k')
        assert math.isclose(row['ttft_ms'],phases['prefill']['duration_s']*1000,rel_tol=1e-12)
        assert math.isclose(row['tpot_ms'],sum(phases[p]['duration_s'] for p in ('draft','verify','commit'))*1000/6,rel_tol=1e-12)
    csv_write(out/'operator_startups.csv',list(launches.values()))
    save(out/'audit_execution.json',audit)
    original_summary=summarize(model.MODEL)
    model_summary={k:original_summary[k] for k in ('backbone_params','ngram_params','prefill_bulk_active_params','decode_active_params','active_count_definition')}
    draft_params=sum(r['weight_bytes_per_card']/(4 if r['tensor']=='DFlash.norms' else 1) for r in model.draft_weights(1))
    model_summary.update(dflash_additional_params=draft_params,
                         resident_logical_params_with_dflash=model_summary['backbone_params']+model_summary['ngram_params']+draft_params,
                         scope='Target backbone + host Engram table + current 5-layer DFlash; old 3-layer DSpark is excluded.')
    save(out/'model_parameter_audit.json',model_summary)
    counts=[]
    for chip,cards in itertools.product(['950pr','950dt'],model.CARDS):
        rows=[r for r in matrix if r['chip']==chip and r['cards']==cards]
        counts.append(dict(chip=chip,cards=cards,joint_feasible=sum(r['joint_status']=='feasible' for r in rows),
                           prefill_feasible=sum(r['prefill_status']=='feasible' for r in rows),
                           decode_feasible=sum(r['decode_status']=='feasible' for r in rows),
                           incompatible=sum(r['joint_status']=='incompatible' for r in rows),total=len(rows)))
    csv_write(out/'feasibility_by_cards.csv',counts)
    assert not any(r['context']=='4M' and r['prefill_status']=='feasible' for r in matrix)
    report=['# DeepSeek V4.1 Pro 10T FP4：算子级 POD / DFlash 全矩阵','',
            '日期：2026-11-05（Asia/Shanghai）。条件预测；模型结构和启动开销未做同条件实机校准。','',
            f'完整覆盖 **23,040** 组请求组合，保留 **{len(detail):,}** 行合法布局及不兼容记录，执行 **{stats["phase_cases"]:,}** 个去重阶段任务。',
            f'同一布局下 Prefill / Decode 均可行：**{stats["joint_status"].get("feasible",0):,}** 组；Decode 单独可行：**{stats["decode_status"].get("feasible",0):,}** 组。','',
            '在本次候选、权重驻留及非分块 Prefill 策略下，950PR 从64卡开始有可行组合，950DT从128卡开始。4M输入的 Prefill 全部容量不可行，独立 Decode 仍保留其实际结果。','',
            '## 数值口径','',
            '- 输入：1K、4K、16K、32K、64K、128K、256K、512K、1M、4M；K=1024，M=1048576。',
            '- Prefill batch=1；Decode batch=1/4/8/16/32/64，为全部物理卡上的总请求数。',
            '- DFlash 净接受/验证块长=3/4、6/8、9/12、12/16；分母使用净接受数，不额外赠送 bonus token。',
            '- 卡数=16/32/64/128/256/512；EP=2/4/8/16/32/64/128/256。扫描合法 TP=1/2/4/8、SP=1 或 TP，DP=卡数/(TP×EP)。SP 不额外乘入总卡数。',
            '- TTFT=目标 Prefill + DFlash 上下文 KV 初始化。TPOT=(draft+verify+commit)/净接受数，为固定上下文轮次均摊；两个阶段 batch 不同，不拼接为单请求端到端时间线。',
            '- 三项加和：计算（含算子启动）+访存+通信=总时延。计算服务和算子启动另列；单算子 FLOPs/算力与 HBM/带宽取 max，仅归因到限制项，避免把重叠工作重复累加。',
            '- 通信包含 SU 服务、host 搬运及原网络启动费用。全 POD 无 SO；保留原 rank 包络及阶段 barrier 假设，未假装实现逐物理链路拥塞、连续成批或 P99。','',
            '## 硬件与结构','',
            '| 参数 | 950PR | 950DT |','|---|---:|---:|',
            '| HBM 容量（十进制 GB/卡） | 128 | 96 |',
            '| HBM 带宽（沿用原假设） | 1.6 TB/s | 4 TB/s |',
            '| FP8 峰值 × 效率 | 1 PF/s × 50% | 1 PF/s × 50% |',
            '| FP32 峰值 × 效率 | 50 TF/s × 50% | 50 TF/s × 50% |',
            '| SU 单方向带宽 × 效率 | 1 TB/s × 80% | 1 TB/s × 80% |',
            '| SU 每轮启动 | 2.1 μs | 2.1 μs |','',
            '每主机 8 卡、4 TB DDR、512 GB/s DDR、1e8 lookup IOPS、host link 64 GB/s×85%、2 μs 启动。所有部署均是单个逻辑 Full mesh POD，DP 副本按整主机隔离。',
            f'沿用 synthetic 64 层、hidden7168、1792 路由专家、top24、CED/CSA2/mHC/Engram 设计。主干参数 {model_summary["backbone_params"]/1e12:.6f} T；主机 Engram 表额外 {model_summary["ngram_params"]/1e12:.4f} T。',
            f'路由专家 gate/up/down 采用 FP4 驻留，每 32 个末轴元素计 1 byte scale；仍使用 FP8 GEMM，显式计入 FP4→FP8 转换、暂存、读写与启动。5 层 DFlash 草稿额外参数 {draft_params/1e9:.6f} B，保持原设计，不宣称真实接受率或模型质量。旧3层DSpark不计入本次驻留。','',
            '## 算子细分和启动假设','',
            f'最新 `apply_kernel_traces` 已接入四阶段。构建的 {stats["trace_eligible"]:,} 个可展开计算事件全部匹配，生成 {stats["generated_suboperators"]:,} 个子算子；实际完成原生事件数 {stats["executed_events"]:,}。匹配覆盖率不是实测精度。',
            '矩阵投影按模型张量拆分；专家 gate/up/down 独立启动，SwiGLU 位于 down 前。DFlash Q/K/V、gate/up 分离并补计非融合输入读取。注意力 tile、mHC Sinkhorn 和命名逐点组保留明确的融合边界；未获得真实二进制 kernel trace。',
            '启动参数为工程假设，两芯片使用同一组参数：','',
            '| 算子类别 | 启动 μs/次 |','|---|---:|']
    report += [f'| {kind} | {value:g} |' for kind,value in PROFILE['startup_us'].items()]
    report += ['', '所有结果附 0.5×/1×/2× 算子启动敏感性列；只改变算子启动，通信启动保持不变。串行图下敏感性为精确代数重算，不是置信区间；四阶段分别用原生重跑验证 1×→2× 增量。','',
               '## 固定参考布局','',
               '512 卡 / EP128 / TP4 / SP1 / DP1，Decode batch32，DFlash6/8。下表每行保留同一布局的两个阶段；TTFT 单位 s，TPOT 单位 ms。','',
               '| 芯片 | 输入 | TTFT s | TPOT ms | P峰值 GB/卡 | D峰值 GB/卡 |','|---|---|---:|---:|---:|---:|']
    report += [f'| {r["chip"].upper()} | {r["context"]} | {fmt(r["ttft_ms"]/1000 if r["ttft_ms"] is not None else None)} | {fmt(r["tpot_ms"])} | {fmt(r["prefill_peak_GB"],1)} | {fmt(r["decode_peak_GB"],1)} |' for r in fixed]
    report += ['', '## 各卡数可行数量','',
               '| 芯片 | 卡数 | P与D同时可行 | D单独可行 | EP不兼容 | 请求组合数 |','|---|---:|---:|---:|---:|---:|']
    report += [f'| {r["chip"].upper()} | {r["cards"]} | {r["joint_feasible"]} | {r["decode_feasible"]} | {r["incompatible"]} | {r["total"]} |' for r in counts]
    report += ['', '容量不足的时延保持空值。`persistent_lower_bound` 表示仅权重/保留状态已超容量，该数值是必要下界；`full_reserved_phase` 表示包含完整阶段的保留 arena。未运行图的配置没有假造算子覆盖或时延。4M 仅为成本和容量外推，不证明模型质量。','',
               '## 结果文件','',
               '- `matrix.csv`：23,040 行主矩阵；优先在两阶段均可行布局中选最小 TPOT，若不存在则清楚保留独立阶段结果和失败状态。',
               '- `layout_detail.csv`：全部 TP/SP/DP 布局，含计算、访存、通信、算子启动、阶段耗时、容量和敏感性。',
               '- `best_prefill_by_EP.csv`、`best_decode_by_EP.csv`：分别按 TTFT / TPOT 优选，不能拼成一个未实际执行的布局。',
               '- `latency_breakdown.csv`：主矩阵 TTFT / TPOT 的长表拆解。',
               '- `phase_checkpoints.sqlite3`：每个去重阶段的原生数值检查点，按源码/配置指纹校验恢复。',
               '- `audit_*.json.gz`：两个参考案例的完整四阶段逐算子记录、精确 trace 配置和原生执行时钟。',
               '- `operator_startups.csv`、`operator_launch_assumptions.json`：逐算子启动参数、分类和来源。',
               '- `experiment.json`、`validation.json`：全部场景参数、代码哈希与验证结果。','',
               '复跑：','.venv/Scripts/python.exe scripts/run_v41_fine_matrix.py --workers 4',
               '.venv/Scripts/python.exe scripts/report_v41_fine_matrix.py','',
               '核心版本：仓库基础提交 85d3c10，加本次 `fine.py` 适配；精确源码 SHA256 见 experiment.json。原有实验文件保持可追溯。']
    (out/'report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    save(out/'presentation_data.json',dict(stats=stats,fixed=fixed,counts=counts,optimal=optimal,audit=audit,validation=validation))
    print(json.dumps(dict(validation=validation,reference_rows=len(fixed),audit_cases=2),ensure_ascii=False))


if __name__=='__main__':main()
