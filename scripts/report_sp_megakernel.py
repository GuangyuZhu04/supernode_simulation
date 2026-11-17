"""Report context parallelism and startup interventions from saved evidence."""
from __future__ import annotations
import csv
import html
import json
from math import gcd
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.report_scaling_study import plt, np, save, fmt, table, decorate

BASE=ROOT/'outputs/scaling-10case-20261113'
OUT=BASE/'sp-megakernel'


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def csv_save(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def normalized_ratio(row):
    a,b=row['tp_a'],row['tp_f'];d=gcd(a,b)
    return f'{a//d}:{b//d}'


def main():
    main_data=load(BASE/'comparison.json')
    reductions=load(OUT/'startup-reduction/summary.json')
    cp=load(OUT/'cp-owner-star/summary.json')
    sp=load(OUT/'sp-analysis/summary.json')
    assert reductions['complete'] and cp['complete']
    transformed={(r['case_id'],r['startup_remaining_fraction']):r for r in reductions['rows'] if r['scope']=='main'}
    cp_executed={(r['case_id'],r['startup_factor']):r for r in cp['rows']}
    bounds={r['case_id']:r for r in sp['pure_cp_bounds']}
    rows=[]
    for i,r in enumerate(main_data):
        a,b=transformed[r['id'],.5],transformed[r['id'],.3]
        c=cp_executed.get((r['id'],1.0)); bound=bounds[r['id']]
        rows.append({'case_id':r['id'],'scenario':f'C{i%10+1:02d}','architecture':r['architecture'],
            'model_T':r['model_T'],'hardware':r['hardware'],'target_ms':r['target_ms'],
            'baseline_tp_ms':r['tpot_ms'],'tp_startup_50_ms':a['tpot_ms'],'tp_startup_50_passed':a['target_passed'],
            'tp_startup_30_ms':b['tpot_ms'],'tp_startup_30_passed':b['target_passed'],
            'pure_cp_ms':c['tpot_ms'] if c else None,'pure_cp_passed':c['target_satisfied'] if c else False,
            'pure_cp_evidence':'executed_owner_star' if c else 'excluded_by_necessary_bound',
            'cp_shared_weight_floor_ms':bound['shared_weight_read_floor_ms'],
            'shared_dense_B':bound['shared_dense_params_B'],
            'cp_degree':bound['assumed_cp']})
    csv_save(OUT/'comparison.csv',rows)
    (OUT/'comparison.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    figures=OUT/'figures';figures.mkdir(exist_ok=True)
    startup_plot(rows,figures)
    cp_plot(cp,figures)
    ratio_plot(reductions,figures)
    lines=report_lines(rows,cp,reductions)
    if (OUT/'component_breakdown.json').exists():
        from scripts.report_optimization_components import build_component_sections
        component_lines=build_component_sections(OUT)
        position=lines.index('## 上下文SP/CP结果')
        lines[position:position]=component_lines
        component_document=['# Attention 与 MoE：整体及内部时延拆解','',*component_lines]
        (OUT/'component-report.md').write_text('\n'.join(component_document)+'\n',encoding='utf-8')
        (OUT/'component-report.html').write_text(render(component_document),encoding='utf-8')
    if (OUT/'cp-fine-operators/report.html').exists():
        position=lines.index('## 上下文SP/CP结果')
        lines[position:position]=['## SP/CP 算子细粒度扩展','',
            '已在同一CP构图器后接入原生kernel_traces并实际重跑子算子。下文原CP表保留聚合模型口径；'
            '细粒度对照显式计入QK/softmax/AV、KDA状态链、专家gate/up/activation/down及中间访存。'
            '网络与kernel启动分别控制，未知共享D仍保留预算标记。','',
            '[细粒度报告与5张图](cp-fine-operators/report.html) · [完整算子表](cp-fine-operators/operator_details.csv) · '
            '[16点数值](cp-fine-operators/summary.csv)','']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (OUT/'report.html').write_text(render(lines),encoding='utf-8')
    print(json.dumps({'report':str(OUT/'report.html'),'main':len(rows),'startup_executions':len(reductions['rows']),
                      'cp_executions':len(cp['rows']),'figures':len(list(figures.rglob('*.png')))},ensure_ascii=False))


def startup_plot(rows,directory):
    fig,axes=plt.subplots(1,2,figsize=(14.4,6.2))
    for ax,arch,title in zip(axes,('kimi-k3','deepseek-v41'),('Kimi K3','DeepSeek V4.1')):
        group=[r for r in rows if r['architecture']==arch];x=np.arange(10)
        for delta,key,label,color in [(-.24,'baseline_tp_ms','原 TP / ring','#9ca6b4'),
                                     (0,'tp_startup_50_ms','通信启动剩余 50%','#db9439'),
                                     (.24,'tp_startup_30_ms','通信启动剩余 30%','#227f8b')]:
            ax.bar(x+delta,[r[key] for r in group],.22,label=label,color=color)
        ax.plot(x,[r['target_ms'] for r in group],linestyle='--',color='#c64242',marker='x',label='原目标')
        ax.set_yscale('log');ax.set_ylim(.8,110)
        ax.set_xticks(x,[r['scenario'] for r in group]);ax.set_title(title,loc='left',fontsize=15)
        decorate(ax,'预测 TPOT / ms（对数轴）')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.5,-.015),fontsize=10)
    fig.suptitle('仅降低通信启动：部分 5 ms 场景达标，1 ms 场景仍不达标',x=.06,ha='left',fontsize=17)
    fig.text(.06,-.055,'C01–C06：960/970 的 2.8T、10T、100T；C07–C10：10T 的950、960、970、AFD 1:1。不是实测 megakernel 提速。',color='#566274')
    fig.tight_layout(rect=(0,.05,1,.95));save(fig,directory,'01_startup_reduction')


def cp_plot(data,directory):
    fig,axes=plt.subplots(1,2,figsize=(12.8,5.8))
    for ax,arch,title in zip(axes,('kimi-k3','deepseek-v41'),('Kimi K3 · 2.8T','DeepSeek V4.1 · 2.8T')):
        selected=[r for r in data['rows'] if r['architecture_id']==arch]
        for offset,alpha,label,color in [(-.25,1.,'CP 原启动','#526f91'),(0,.5,'CP 启动50%','#db9439'),(.25,.3,'CP 启动30%','#227f8b')]:
            values=[next(r['tpot_ms'] for r in selected if r['startup_factor']==alpha and hw in r['case_id']) for hw in ('960DT','970DT')]
            ax.bar(np.arange(2)+offset,values,.23,label=label,color=color)
            for x,y in zip(np.arange(2)+offset,values):ax.text(x,y+.07,f'{y:.3f}',ha='center',fontsize=10)
        ax.axhline(5,color='#c64242',linestyle='--',label='目标 5 ms')
        ax.set_xticks([0,1],['960DT · CP32','970DT · CP96']);ax.set_ylim(0,5.8)
        ax.set_title(title,loc='left',fontsize=15);decorate(ax,'预测 TPOT / ms')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.5,-.015),fontsize=10)
    fig.suptitle('纯上下文并行：4 个 2.8T 场景完成事件执行并达到 5 ms',x=.06,ha='left',fontsize=17)
    fig.text(.06,-.055,'联合方案：TP1 + 上下文分片 + owner-star 归并；共享/专家权重未按CP分片。其余16点被共享权重带宽下界排除。',color='#566274')
    fig.tight_layout(rect=(0,.05,1,.95));save(fig,directory,'02_context_parallel')


def ratio_plot(data,directory):
    fig,axes=plt.subplots(1,2,figsize=(13.6,5.8))
    for ax,arch,title in zip(axes,('kimi-k3','deepseek-v41'),('Kimi K3 · 10T AFD','DeepSeek V4.1 · 10T AFD')):
        group=[r for r in data['rows'] if r['scope']=='ratio' and r['architecture_id']==arch]
        for alpha,color in [(.5,'#db9439'),(.3,'#227f8b')]:
            values=[r for r in group if r['startup_remaining_fraction']==alpha]
            x=np.arange(len(values));ax.plot(x,[r['tpot_ms'] for r in values],'-o',color=color,label=f'启动剩余 {alpha*100:g}%')
            best=min(values,key=lambda r:r['tpot_ms']);i=values.index(best)
            ax.scatter(i,best['tpot_ms'],marker='*',s=150,color=color,zorder=5)
            ax.annotate(f'{normalized_ratio(best)} / {best["tpot_ms"]:.3f} ms',(i,best['tpot_ms']),
                        xytext=(0,16 if alpha==.5 else -25),textcoords='offset points',ha='center',fontsize=10,color=color)
        ax.axhline(1,color='#c64242',linestyle='--',label='目标 1 ms')
        ax.set_xticks(x,[normalized_ratio(r)+('*' if r['tp_a']+r['tp_f']<24 else '') for r in values])
        ax.set_ylim(bottom=0);ax.set_title(title,loc='left',fontsize=15);decorate(ax,'预测 TPOT / ms')
        ax.legend(frameon=False,loc='upper center',fontsize=10);ax.set_xlabel('970DT : LPU')
    fig.suptitle('启动优化后重新选配比，仍不足以达到 1 ms',x=.06,ha='left',fontsize=17)
    fig.text(.06,-.025,'* 4:1 留空 2048 卡；最优仅限原9个候选与当前预算分片/通信假设。',color='#566274')
    fig.tight_layout(rect=(0,0,1,.95));save(fig,directory,'03_ratio_after_startup')


def report_lines(rows,cp,reductions):
    cp_table=[]
    for r in rows:
        cp_table.append(['K3' if r['architecture']=='kimi-k3' else 'V4.1',r['scenario'],f'{r["model_T"]:g}T',r['hardware'],fmt(r['target_ms'],0),
            fmt(r['pure_cp_ms']) if r['pure_cp_ms'] is not None else '至少 '+fmt(r['cp_shared_weight_floor_ms']),
            '实际事件执行：达标' if r['pure_cp_passed'] else '必要下界已排除'])
    startup_table=[]
    for r in rows:
        startup_table.append(['K3' if r['architecture']=='kimi-k3' else 'V4.1',r['scenario'],fmt(r['target_ms'],0),fmt(r['baseline_tp_ms']),
            fmt(r['tp_startup_50_ms'])+(' ✓' if r['tp_startup_50_passed'] else ''),
            fmt(r['tp_startup_30_ms'])+(' ✓' if r['tp_startup_30_passed'] else '')])
    ratio_rows=[]
    for arch in ('kimi-k3','deepseek-v41'):
        for alpha in (.5,.3):
            candidates=[r for r in reductions['rows'] if r['scope']=='ratio' and r['architecture_id']==arch and r['startup_remaining_fraction']==alpha]
            best=min(candidates,key=lambda r:r['tpot_ms'])
            ratio_rows.append(['K3' if arch=='kimi-k3' else 'V4.1',f'{alpha*100:g}%',normalized_ratio(best),fmt(best['tpot_ms']),'不满足 1 ms'])
    return ['# SP/CP 与启动优化：哪些目标可以达成','',
        '日期：2026-11-13。沿用上轮用户输入、FP8权重/计算、FP16 KV与通信、全POD batch32、1M=1048576。',
        '**这仍是 Kimi K3 / V4.1 Attention-KV 机制模板与参数预算下的条件预测，不是完整checkpoint的实机测量。**','',
        '## 结论','',
        '- 真正按上下文切分 Attention/KV、完全去掉 TP 后，4个2.8T场景在本次owner-star方案中达到5 ms；其余16个10T/100T场景仅共享权重HBM读取下界就超过目标。',
        '- 保持原TP部署，仅假设通信启动降至50%/30%，分别有3/20和4/20条场景达标，均为V4.1的5 ms目标；包括960DT/10T在两个实验组中的重复物理配置。所有1 ms目标仍不达标。',
        '- 不能将该通信启动缩减直接称为megakernel收益。原账本的非通信kernel显式启动未校准、取值为0；标准局部融合不会自动消除逐轮链路时延。',
        '- EP已经分片路由专家。纯CP替代TP失去的是共享稠密权重的TP分片；预取可隐藏部分延迟，不能免费减少HBM读取量。','',
        '## 上下文SP/CP结果','',
        '用户已确认这里的SP是上下文分片。执行方案为TP1、CP=原TP；每请求由owner执行共享D预算、KDA全状态和MoE专家计算，context peers执行分片Attention。'
        '每层显式计query/current-KV广播、稳定softmax的max与numerator/denominator归并、V4全局索引候选选择与广播、当前KV唯一shard写入。'
        'KDA递归状态不按上下文长度缩减。通信为单SU的owner-star，计入owner注入、HBM和归并计算。','',
        '因此这比较的是 **TP+ring 与 TP1+CP+owner-star 两种联合方案**；不能把全部收益单独归因于SP。'
        '只对未被下界排除的4个2.8T场景执行完整事件图（各3档启动，共12点）；其余16点是必要下界判断，不伪称完整仿真。','',
        table(['模板','场景','模型','硬件','目标 ms','纯CP TPOT / ms','证据及结论'],cp_table),'',
        '![上下文并行](figures/02_context_parallel.png)','',
        '这里没有保留FFN TP，也没有搜索TP+CP混合方案。共享稠密D含未逐张量分离的共享FFN等预算；AFD仍沿用把D放Attention池的假设。'
        '重新划分D到LPU或更换权重布局会改变此下界，需另建模型，不能由本表推断。','',
        '## 启动开销剩余50%与30%','',
        '以下均由实际事件图重跑，只改变通信event.startup与对应归因域；FLOPs、HBM/网络字节、内存容量和依赖保持不变。'
        '“剩余30%”表示减少70%，不是减少30%。保存的startup线性公式仅用于独立交叉核验。单位ms，✓为达到该行目标。','',
        table(['模板','场景','目标','原TP','启动50%','启动30%'],startup_table),'',
        '![启动缩减](figures/01_startup_reduction.png)','',
        'V4.1 960DT/2.8T在30%时为5.136 ms，仍略超5 ms。V4.1 970DT/10T在30%时为4.052 ms，只能满足模型对比组的5 ms，不能满足硬件对比组的1 ms。','',
        '## AFD配比重新选择','',table(['模板','启动剩余','候选最优970:LPU','TPOT ms','目标判定'],ratio_rows),'',
        '![配比重选](figures/03_ratio_after_startup.png)','',
        '9个已检查配比中，K3即使消除全部通信启动，最佳残余也约2.459 ms。'
        'V4.1要达到1 ms，最宽松候选2:1要求通信启动剩余不超过约6.33%（减少至少93.67%）；这是假设固定工作量与布局的门槛，不是实际megakernel承诺。','',
        '## EP分片与权重预取','',
        '按用户给定P总参数、A激活参数、k=top-k、E专家数，共享预算 D=(A−P×k/E)/(1−k/E)。专家预算为P−D，已经按EP分片；原TP还继续分片这两类权重。','',
        table(['模型','共享D / B','专家权重/EP / B','纯TP1共享权重读取下界：950 / 960 / 970 ms'],
              [['2.8T','17.032','21.742','— / 1.774 / 1.183'],['10T','95.238','19.345','23.810 / 9.921 / 6.614'],['100T','220.472','48.720','— / 22.966 / 15.311']]),'',
        '当前所有权重已在HBM。进一步权重预取是HBM到片上缓冲，可与计算/通信重叠，但仍占同一HBM资源。'
        '若每个token都要读取该共享预算、没有足够跨token片上常驻缓存，则即使理想预取也保留D/HBM带宽下界。'
        '把预读移到首步计时之前可以缩短那个窗口，不能当作稳态每token吞吐改善。'
        '本次没有给未知片上容量或异步搬运效率伪造数值，也没有把整个权重读时间直接从TPOT中减掉。','',
        '非专家权重也可另用分布式计算或按层all-gather来分片，但需计新增通信；它不等价于只增加EP。'
        '理想预取与EP的独立数据见 [分析JSON](prefetch_ep_analysis.json)和[数据CSV](prefetch_ep_analysis.csv)（含适用条件）。','',
        '## 算子级仿真的范围','',
        '本次执行组件/算子事件图，包括Attention core、索引、归并、专家与collective；计入实际依赖、争用与内存准入。'
        'attention_dense、experts等仍是聚合预算kernel，按FLOPs、HBM访问与资源速率估计。'
        '没有芯片指令级逐周期仿真或同硬件kernel实测。原非通信kernel启动尚无校准，当前显式值设为0，故无法从当前账本定量识别“仅降低kernel launch”的真实收益。','',
        '## 数据与复跑','',
        '[20场景对照CSV](comparison.csv) · [76点启动干预](startup-reduction/summary.csv) · [12点CP执行](cp-owner-star/summary.json) · [20点CP必要下界](sp-analysis/pure-cp-bounds.csv)','',
        '原20场景、TTFT和Attention/MoE细分见[上一轮完整报告](../report.html)。本轮达标目标均为TPOT，未对无目标的TTFT再作扫描。','',
        '独立验证已通过：[76点启动干预](startup-reduction/independent-verification.json)、[12点CP执行](cp-owner-star/independent-verification.json)、[20点CP下界](sp-analysis/bounds-independent-verification.json)。'
        '所有结论受统一预算、均匀路由、单SU、所选collective算法及权重读取策略约束。',
        '', '```powershell',
        '.venv/Scripts/python.exe scripts/scaling_startup_reduction.py',
        '.venv/Scripts/python.exe scripts/analyze_scaling_sp.py',
        '.venv/Scripts/python.exe scripts/scaling_context_parallel.py',
        '.venv/Scripts/python.exe scripts/analyze_scaling_prefetch_ep.py',
        '.venv/Scripts/python.exe scripts/report_sp_megakernel.py',
        '```']


def render(lines):
    blocks=[];in_table=False;in_list=False;in_code=False
    def inline(s):
        s=html.escape(s)
        s=re.sub(r'!\[([^]]*)\]\(([^)]+)\)',r'<img alt="\1" src="\2">',s)
        s=re.sub(r'\[([^]]*)\]\(([^)]+)\)',r'<a href="\2">\1</a>',s)
        s=re.sub(r'\*\*(.+?)\*\*',r'<strong>\1</strong>',s)
        return s
    for block in lines:
        for line in block.splitlines():
            if line.startswith('```'):
                blocks.append('</code></pre>' if in_code else '<pre><code>');in_code=not in_code;continue
            if in_code:blocks.append(html.escape(line)+'\n');continue
            if in_table and not line.startswith('|'):blocks.append('</tbody></table></div>');in_table=False
            if in_list and not line.startswith('- '):blocks.append('</ul>');in_list=False
            if line.startswith('|'):
                if not line.replace('|','').replace('-','').replace(':','').strip():continue
                cells=[inline(s.strip()) for s in line.strip('|').split('|')]
                if not in_table:
                    blocks.append('<div class="table"><table><thead><tr>'+''.join('<th>'+s+'</th>' for s in cells)+'</tr></thead><tbody>');in_table=True
                else:blocks.append('<tr>'+''.join('<td>'+s+'</td>' for s in cells)+'</tr>')
            elif line.startswith('# '):blocks.append('<h1>'+inline(line[2:])+'</h1>')
            elif line.startswith('## '):blocks.append('<h2>'+inline(line[3:])+'</h2>')
            elif line.startswith('### '):blocks.append('<h3>'+inline(line[4:])+'</h3>')
            elif line.startswith('- '):
                if not in_list:blocks.append('<ul>');in_list=True
                blocks.append('<li>'+inline(line[2:])+'</li>')
            elif line:blocks.append('<p>'+inline(line)+'</p>')
    return '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SP/CP与启动优化</title><style>
    *{box-sizing:border-box}body{background:#f5f7fa;color:#233145;font:16px/1.75 "Microsoft YaHei",sans-serif;margin:0}main{max-width:1320px;margin:auto;padding:35px 40px 70px}h1{border-top:6px solid #237f8b;padding-top:24px;font-size:34px}h2{margin-top:42px;padding-bottom:8px;border-bottom:1px solid #d7e0e9;font-size:24px}img{width:100%;display:block;margin:22px 0}.table{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px;background:white}th,td{padding:10px 12px;border-bottom:1px solid #dde5ed;white-space:nowrap}th{text-align:left;background:#244056;color:white}td{font-variant-numeric:tabular-nums}tr:nth-child(even){background:#edf2f6}a{color:#137b8a}li{margin:9px 0}pre{background:#244056;color:#edf4f9;padding:20px;overflow:auto}p,li{max-width:1160px}@media(max-width:700px){main{padding:18px}h1{font-size:27px}}
    </style><main>'''+ '\n'.join(blocks)+'</main></html>'


if __name__=='__main__':main()
