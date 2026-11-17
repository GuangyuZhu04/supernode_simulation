"""Two-level Attention/MoE visualizations from saved operator attributions."""
from __future__ import annotations
from collections import OrderedDict
import json
from math import gcd
from pathlib import Path
import re

from scripts.report_scaling_study import plt, np, save, fmt, table, decorate

ARCHS=('kimi-k3','deepseek-v41')
NAMES={'kimi-k3':'Kimi K3','deepseek-v41':'DeepSeek V4.1'}
OUTER=[('attention','Attention侧（含共享稠密预算）','#2477a8'),
       ('moe','MoE侧（路由专家及其通信）','#c94444'),('other','未归属/其他','#a6adba')]
INNER=[('compute','计算限制服务','#2477a8'),('communication','通信（含启动/传输/争用）','#c94444'),
       ('memory','访存限制服务','#8bbbd1'),('other','启动/等待/其他','#a6adba')]


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def windows(payload):
    result=OrderedDict()
    for row in payload['rows']:
        identity=(row['scope'],row['case_id'],row['variant'])
        value=result.setdefault(identity,{'meta':row,'components':{}})
        value['components'][row['component']]=row
    return list(result.values())


def select(data,scope,arch=None,alpha=None):
    return [w for w in data if w['meta']['scope']==scope
            and (arch is None or w['meta']['architecture_id']==arch)
            and (alpha is None or abs(w['meta']['startup_remaining_fraction']-alpha)<1e-9)]


def ratio_label(w):
    meta=w['meta']
    a,b=meta.get('tp_a'),meta.get('tp_f')
    if a is None:
        match=re.search(r'-a(\d+)-f(\d+)',meta['case_id'])
        if not match:raise ValueError('Ratio identity missing: '+meta['case_id'])
        a,b=map(int,match.groups())
    divisor=gcd(a,b)
    return f'{a//divisor}:{b//divisor}'+('*' if a+b<24 else '')


def draw_panel(ax,group,labels,component=None,metric='TPOT'):
    x=np.arange(len(group));bottom=np.zeros(len(group))
    series=OUTER if component is None else INNER
    for key,label,color in series:
        values=[]
        for w in group:
            if component is None:
                values.append(w['components'][key]['window_share_pct'] or 0)
            else:
                values.append(w['components'][component][key+'_pct'] or 0)
        values=np.array(values)
        ax.bar(x,values,.68,bottom=bottom,color=color,label=label)
        if component is None:
            for i,v in enumerate(values):
                if v>=10:
                    ax.text(i,bottom[i]+v/2,f'{v:.1f}%',ha='center',va='center',fontsize=9,color='white')
        bottom+=values
    if component is None:
        for i,w in enumerate(group):
            total=w['meta']['window_total_ms']
            if total is None:
                ax.text(i,4,'N/A',ha='center',rotation=90,fontsize=9,color='#68798b')
            else:
                value=total/1000 if metric=='TTFT' else total
                ax.text(i,102,f'{value:.2f}',ha='center',fontsize=9,color='#33465a')
    ax.set_xticks(x,labels)
    ax.tick_params(axis='x',labelsize=9)
    ax.set_ylim(0,112 if component is None else 104)
    decorate(ax,f'占总 {metric} / %' if component is None else f'占 {"Attention侧" if component=="attention" else "MoE侧"} 时间 / %')
    if component is not None:
        for i,w in enumerate(group):
            if not w['components'][component]['total_ms']:
                ax.text(i,4,'N/A',ha='center',fontsize=8,color='#68798b')


def draw_group(directory,stem,title,columns,note,metric='TPOT'):
    fig,axes=plt.subplots(3,2,figsize=(14.8,13.4))
    for col,(subtitle,group,labels) in enumerate(columns):
        if not group:raise ValueError('Empty figure group: '+subtitle)
        draw_panel(axes[0,col],group,labels,metric=metric)
        draw_panel(axes[1,col],group,labels,'attention',metric)
        draw_panel(axes[2,col],group,labels,'moe',metric)
        unit='s' if metric=='TTFT' else 'ms'
        axes[0,col].set_title(subtitle+f'\n总时延分工；柱顶数字为 {metric} / {unit}',loc='left',fontsize=12,pad=14)
        axes[1,col].set_title('Attention侧内部：计算 / 通信 / 访存',loc='left',fontsize=12,pad=12)
        axes[2,col].set_title('MoE侧内部：计算 / 通信 / 访存',loc='left',fontsize=12,pad=12)
    outer_handles,outer_labels=axes[0,0].get_legend_handles_labels()
    inner_handles,inner_labels=axes[1,0].get_legend_handles_labels()
    fig.legend(outer_handles,outer_labels,loc='upper center',bbox_to_anchor=(.5,.947),ncol=3,frameon=False,fontsize=10)
    fig.legend(inner_handles,inner_labels,loc='lower center',bbox_to_anchor=(.5,.025),ncol=4,frameon=False,fontsize=10)
    fig.suptitle(title,x=.055,ha='left',fontsize=19,y=.991)
    fig.text(.055,.009,note,fontsize=10,color='#566274')
    fig.tight_layout(rect=(0,.055,1,.918),h_pad=2.4)
    save(fig,directory,stem)


def build_component_sections(directory):
    payload=read(directory/'component_breakdown.json')
    data=windows(payload)
    if len(data)!=108:raise ValueError(f'Expected 108 saved windows; found {len(data)}')
    figure_dir=directory/'figures/components';figure_dir.mkdir(parents=True,exist_ok=True)
    case_map={r['id']:f'C{i%10+1:02d}' for i,r in enumerate(read(directory.parent/'manifest.json')['cases'])}
    baseline=[]
    for arch in ARCHS:
        group=select(data,'baseline',arch)
        baseline.append((NAMES[arch],group,[case_map[w['meta']['case_id']] for w in group]))
    note='内层分母为各组件自身时间；计算与访存按限制资源归因，0% 不表示没有工作量。通信子项不重复相加。'
    draw_group(figure_dir,'01_baseline_components','原 TP 基线：整体时延与两个组件的内部组成',baseline,note)
    for arch in ARCHS:
        columns=[]
        for alpha in (.5,.3):
            group=select(data,'main',arch,alpha)
            columns.append((f'通信启动剩余 {alpha*100:g}%',group,[case_map[w['meta']['case_id']] for w in group]))
        draw_group(figure_dir,'02_startup_'+arch,NAMES[arch]+' · 启动优化后的两层时延拆解',columns,note)
    cp_columns=[]
    for arch in ARCHS:
        group=select(data,'cp',arch)
        labels=[('960' if '960DT' in w['meta']['case_id'] else '970')+f'\n{w["meta"]["startup_remaining_fraction"]*100:g}%' for w in group]
        cp_columns.append((NAMES[arch]+' · 2.8T / 纯CP',group,labels))
    draw_group(figure_dir,'03_cp_components','上下文 CP：Attention 与 MoE 的瓶颈变化',cp_columns,
               '横轴：设备 / 通信启动剩余比例。仅12个已执行CP结果；16个仅有必要下界的场景不生成虚构占比。')
    for arch in ARCHS:
        columns=[]
        for alpha in (.5,.3):
            group=select(data,'ratio',arch,alpha)
            columns.append((f'通信启动剩余 {alpha*100:g}%',group,[ratio_label(w) for w in group]))
        draw_group(figure_dir,'04_ratio_'+arch,NAMES[arch]+' · AFD配比与组件内部开销',columns,
                   '横轴为970DT:LPU；* 4:1闲置2048卡。外层分母为总TPOT，内部比例分别除以Attention侧或MoE侧时间。')
    archive=read(directory.parent/'component_breakdown.json')
    prefill_data=windows({'rows':[r for r in archive['rows'] if r['scope']=='prefill' and r['metric']=='ttft']})
    if len(prefill_data)!=20:raise ValueError('Archived Prefill must contain 20 evaluated cases')
    prefill_columns=[]
    for arch in ARCHS:
        group=select(prefill_data,'prefill',arch)
        prefill_columns.append((NAMES[arch]+' · 原Prefill',group,[case_map[w['meta']['case_id']] for w in group]))
    draw_group(figure_dir,'05_archived_ttft_components','上一轮 TTFT：补齐 Attention 与 MoE 的内部组成',prefill_columns,
               '仅使用原Prefill已保存账本；16个完整TTFT，4个容量失败为N/A。这不是SP/CP或启动优化后的TTFT。',metric='TTFT')
    overview=[]
    for w in select(data,'cp',alpha=1.):
        m=w['meta'];a=w['components']['attention'];e=w['components']['moe']
        for c in (a,e):
            overview.append([NAMES[m['architecture_id']],('960DT' if '960DT' in m['case_id'] else '970DT'),
                c['component_label'],fmt(c['total_ms'],4),fmt(c['window_share_pct'],2),
                fmt(c['compute_pct'],2),fmt(c['communication_pct'],2),fmt(c['memory_pct'],2),fmt(c['other_pct'],2)])
    lines=['## 整体时延与 Attention / MoE 内部占比','',
        '本节覆盖108个有完整算子账本的TPOT窗口：20个原TP基线、40个启动优化主场景、36个配比场景、12个CP场景。'
        '并对上一轮20个Prefill补齐相同展示（16个可用、4个容量失败）。没有重新计算或改变原时延，直接汇总保存的关键路径算子时间。','',
        '**外层分母＝整个TPOT；Attention内部的分母＝Attention侧时间；MoE内部的分母＝MoE侧时间。** '
        '计算、通信、访存及其他开销在各自分母下加和为100%。通信内部仍保留启动与传输明细，但不重复加到通信总时间。','',
        'Attention侧包含共享稠密预算、KDA/MLA/CSA、KV、CP广播、softmax归并和V4索引通信。'
        'MoE侧包含路由专家、dispatch/combine、专家TP归约与AFD部分和。未逐张量分离的共享专家仍在Attention侧的D预算中。','',
        '下表展示纯CP、原通信启动时的具体比例。所有场景的绝对时间和百分比见CSV。','',
        table(['模板','设备','组件','时间 ms','占整个TPOT %','内部计算 %','内部通信 %','内部访存 %','内部其他 %'],overview),'',
        '[组件明细CSV](component_breakdown.csv) · [每窗口汇总CSV](window_summary.csv) · [算子映射与来源JSON](component_breakdown.json)','',
        '计算与访存按roofline限制资源归因，不能把重叠服务重复相加；0%表示没有归入该项的关键路径时间，不表示没有对应FLOPs或访存。'
        '本轮SP/CP与启动优化只执行TPOT，没有相应TTFT的实际账本；上一轮TTFT仍见[原报告](../report.html)。','',
        '### 原TP基线','', '![原TP两层分解](figures/components/01_baseline_components.png)','',
        'C01–C06依次为960/970上的2.8T、10T、100T；C07–C10为10T的950、960、970、AFD1:1。','',
        '### 启动优化主场景','',
        '![K3启动优化内部](figures/components/02_startup_kimi-k3.png)','',
        '![V4启动优化内部](figures/components/02_startup_deepseek-v41.png)','',
        '### 上下文SP/CP','', '![CP两层分解](figures/components/03_cp_components.png)','',
        '### AFD配比','',
        '![K3配比内部](figures/components/04_ratio_kimi-k3.png)','',
        '![V4配比内部](figures/components/04_ratio_deepseek-v41.png)','',
        '### 上一轮Prefill：Attention与MoE各自内部占比','',
        '![原TTFT组件内部](figures/components/05_archived_ttft_components.png)','',
        '该图来自原Prefill执行，与本轮SP/CP分开；空值表示容量不可行，不填零。原始TTFT组件数值见[上一轮组件CSV](../component_breakdown.csv)。','',
        '本轮108个TPOT窗口的独立守恒与算子映射检查见[组件审计](component-independent-verification.json)。'
        '原TTFT的独立检查见[Prefill组件审计](../component-independent-verification.json)。','']
    return lines
