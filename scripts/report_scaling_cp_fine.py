"""Render the executed CP fine-operator matrix without changing numerics."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.report_scaling_study import plt,np,save,fmt,table,decorate
from scripts.report_sp_megakernel import render

DEFAULT=ROOT/'outputs/scaling-10case-20261113/sp-megakernel/cp-fine-operators'
COLORS={'compute':'#2477a8','communication':'#c94444','memory':'#8bbbd1','launch':'#df9a35','other':'#a6adba'}
COST_NAMES={'compute':'算力受限服务','communication':'通信算子服务','memory':'HBM受限服务','launch':'kernel启动','other':'等待/其他'}


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def json_save(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2),encoding='utf-8')
def csv_save(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        w.writerows({k:(';'.join(v) if isinstance(v,list) else v) for k,v in r.items()} for r in rows)


def key(row):return row['case_id'],row['network_startup_factor'],row['kernel_launch_us']
def label(row):return ('Kimi K3' if row['architecture_id']=='kimi-k3' else 'V4.1')+' / '+row['hardware']


def cost_rows(rows):
    components=[];operators=[]
    for row in rows:
        record=row['latency_breakdown']
        if record is None:continue
        communication_names={name for info in row['kinds'].values() for name in info['communication_operators']}
        grouped={name:defaultdict(float) for name in ('attention','moe','other')}
        for op in record['operator_breakdown']['operators']:
            c=op['component'].lower()
            component='moe' if 'moe' in c else 'attention' if 'attention' in c else 'other'
            target=grouped[component];total=op['total_s']*1000;target['total_ms']+=total
            communication=op['operator'] in communication_names
            values={k:0. for k in COLORS}
            if communication:values['communication']=total
            else:
                values.update(compute=op['components_s'].get('compute_s',0)*1000,
                    memory=op['components_s'].get('memory_access_s',0)*1000,
                    launch=op['components_s'].get('launch_s',0)*1000)
                values['other']=max(0.,total-values['compute']-values['memory']-values['launch'])
            for name,value in values.items():target[name+'_ms']+=value
            operators.append({'id':row['id'],'case_id':row['case_id'],'architecture_id':row['architecture_id'],
                'hardware':row['hardware'],'network_startup_factor':row['network_startup_factor'],
                'kernel_launch_us':row['kernel_launch_us'],'component':component,'source_component':op['component'],
                'operator':op['operator'],'total_ms':total,'window_share_pct':op['share_pct'],
                'window_total_ms':row['fine_tpot_ms'],**{k+'_ms':v for k,v in values.items()},
                **{k+'_pct':100*v/total if total else None for k,v in values.items()},
                'source_artifact':str(Path(row['result_path']).with_name('operator_breakdown.csv')).replace('\\','/')})
        for component,values in grouped.items():
            total=values['total_ms'];window=row['fine_tpot_ms']
            assert abs(sum(values[k+'_ms'] for k in COLORS)-total)<1e-7
            components.append({'id':row['id'],'case_id':row['case_id'],'architecture_id':row['architecture_id'],
                'hardware':row['hardware'],'network_startup_factor':row['network_startup_factor'],
                'kernel_launch_us':row['kernel_launch_us'],'component':component,'total_ms':total,
                'window_total_ms':window,'window_share_pct':100*total/window if window else None,
                **{k+'_ms':values[k+'_ms'] for k in COLORS},
                **{k+'_pct':100*values[k+'_ms']/total if total else None for k in COLORS}})
        assert abs(sum(v['total_ms'] for v in grouped.values())-row['fine_tpot_ms'])<1e-7
    return components,operators


def trace_rows(rows,directory):
    coverage=[];specs=[]
    for row in rows:
        for kind,info in row['kinds'].items():
            audit=info['fine_audit'];trace=info['native_operator_trace'];count=info['layers']
            strict=budget=scalar=0
            source=info.get('kernel_trace_artifact')
            if source:
                config=read(directory/source)
                for parent,traces in config['hardware']['kernel_traces'].items():
                    for item in traces:
                        for i,op in enumerate(item['operations']):
                            impl=op.get('implementation',{});shape=op.get('matmul')
                            is_budget='budget' in op['kernel'] or impl.get('op_type') in ('aggregate_projection_budget','gated_ffn_budget_matmul')
                            category='integer_reference_algebra' if shape else 'unresolved_budget_matrix' if is_budget else 'scalar_or_atomic'
                            strict+=bool(shape);budget+=is_budget and not bool(shape);scalar+=not shape and not is_budget
                            specs.append({'id':row['id'],'kind':kind,'trace_id':item['id'],'parent_kernel':parent,
                                'operation_index':i,'kernel':op['kernel'],'classification':category,
                                'flops':op['flops'],'hbm_bytes':op['memory_bytes'],'startup_us':op.get('startup_s',0)*1e6,
                                'matmul':json.dumps(shape,ensure_ascii=False),'source_artifact':source,
                                'implementation':json.dumps(impl,ensure_ascii=False)})
            coverage.append({'id':row['id'],'case_id':row['case_id'],'architecture_id':row['architecture_id'],
                'hardware':row['hardware'],'network_startup_factor':row['network_startup_factor'],
                'kernel_launch_us':row['kernel_launch_us'],'kind':kind,'layers':count,
                'eligible_parent_events':trace['eligible_events'],'matched_parent_events':trace['matched_events'],
                'coverage_fraction':trace['coverage_fraction'],'generated_trace_operations':trace['generated_operations'],
                'additional_scalar_events':audit['fine_extra_events'],
                'integer_reference_shape_specs':strict,'unresolved_budget_matrix_specs':budget,
                'scalar_or_atomic_specs':scalar,'unresolved_shared_parent_events':len(audit['budget_unresolved']),
                'added_nonparameter_flops_representative_layer':audit['added_nonparameter_flops'],
                'reassigned_conv_parameter_flops_representative_layer':audit['reassigned_conv_parameter_flops'],
                'added_hbm_bytes_representative_layer':audit['added_hbm_bytes'],
                'cross_child_buffer_count':len(audit['cross_child_buffers']),
                'cross_child_allocation_bytes_sum_not_peak':sum(b['bytes'] for b in audit['cross_child_buffers']),
                'network_events_unchanged':audit['network_events_unchanged'],
                'parameter_flops_conserved':audit['parameter_flops_conserved'],
                'source_artifact':source or ''})
    return coverage,specs


def comparison_plot(rows,directory):
    lookup={key(r):r for r in rows};base=[r for r in rows if r['network_startup_factor']==1 and r['kernel_launch_us']==0]
    fig,ax=plt.subplots(figsize=(12.8,5.8));x=np.arange(len(base))
    for offset,which,name,color in [(-.24,'coarse','Coarse 聚合模型','#9ca6b4'),
        (0,'fine0','Fine · 子kernel启动0','#237f8b'),(.24,'fine1','Fine · 每kernel 1 μs','#db9439')]:
        values=[r['coarse_tpot_ms'] if which=='coarse' else r['fine_tpot_ms'] if which=='fine0'
                else lookup[r['case_id'],1.,1.]['fine_tpot_ms'] for r in base]
        ax.bar(x+offset,values,.22,label=name,color=color)
        for p,v in zip(x+offset,values):ax.text(p,v+.045,f'{v:.3f}',ha='center',fontsize=10)
    ax.axhline(5,color='#c94444',linestyle='--',label='目标 5 ms')
    ax.set_xticks(x,[label(r) for r in base]);decorate(ax,'预测 TPOT / ms')
    ax.set_ylim(0,max(5,max(r['fine_tpot_ms'] for r in rows))*1.19)
    ax.legend(loc='lower center',bbox_to_anchor=(.5,-.23),ncol=4,frameon=False,fontsize=10)
    ax.set_title('同一 SP/CP 部署：细分访问、依赖与 kernel 启动的影响',loc='left',fontsize=17,pad=20)
    fig.text(.07,-.025,'Fine0 为本次主方案；1 μs 是每个实际child/extra kernel的独立工程假设。网络启动均保持原值。',color='#566274')
    fig.tight_layout();save(fig,directory,'01_coarse_fine_launch')


def network_plot(rows,directory):
    lookup={key(r):r for r in rows};fig,axes=plt.subplots(1,2,figsize=(13.6,5.7))
    for ax,arch,title in zip(axes,('kimi-k3','deepseek-v41'),('Kimi K3 · 2.8T','DeepSeek V4.1 · 2.8T')):
        for hw,color in [('960DT','#237f8b'),('970DT','#b64b52')]:
            case=next(r['case_id'] for r in rows if r['architecture_id']==arch and r['hardware']==hw)
            group=[lookup[case,a,0.] for a in (1.,.5,.3)]
            ax.plot(range(3),[r['coarse_tpot_ms'] for r in group],'--o',color=color,label=hw+' Coarse')
            ax.plot(range(3),[r['fine_tpot_ms'] for r in group],'-s',color=color,label=hw+' Fine0')
        ax.axhline(5,color='#566274',linestyle=':',label='目标5ms')
        ax.set_xticks(range(3),['100%','50%','30%']);ax.set_xlabel('网络启动剩余比例')
        ax.set_title(title,loc='left');decorate(ax,'预测 TPOT / ms');ax.set_ylim(bottom=0)
        ax.legend(frameon=False,fontsize=9,ncol=2)
    fig.suptitle('网络启动与 kernel 细化是两个独立变量',x=.065,ha='left',fontsize=17)
    fig.text(.065,-.025,'此图全部使用子kernel启动0；50%/30%缩减的是原网络事件启动，不冒称局部megakernel降低物理链路时延。',color='#566274')
    fig.tight_layout();save(fig,directory,'02_network_factor')


def component_plots(rows,components,directory):
    indexed={(r['id'],r['component']):r for r in components}
    for arch,title in [('kimi-k3','Kimi K3'),('deepseek-v41','DeepSeek V4.1')]:
        group=[r for r in rows if r['architecture_id']==arch and r['network_startup_factor']==1]
        fig,axes=plt.subplots(1,3,figsize=(17.2,5.9));x=np.arange(len(group))
        for ax,component in zip(axes,(None,'attention','moe')):
            bottom=np.zeros(len(group))
            series=[('attention','Attention侧','#2477a8'),('moe','MoE侧','#c94444'),('other','其他','#a6adba')] if component is None else [
                (field,COST_NAMES[field],COLORS[field]) for field in COLORS]
            for field,name,color in series:
                vals=[indexed[r['id'],field]['window_share_pct'] if component is None else
                      indexed[r['id'],component][field+'_pct'] or 0 for r in group]
                ax.bar(x,vals,.62,bottom=bottom,label=name,color=color);bottom+=np.array(vals)
            ax.set_xticks(x,[r['hardware'].replace('DT','')+f'\n{r["kernel_launch_us"]:g} μs' for r in group])
            ax.set_ylim(0,105);decorate(ax,'占 TPOT / %' if component is None else '占该组件时间 / %')
            ax.set_title('整体：Attention / MoE' if component is None else ('Attention侧：限制资源归因' if component=='attention' else 'MoE侧：限制资源归因'),loc='left')
            ax.legend(frameon=False,fontsize=8,ncol=2,loc='lower center',bbox_to_anchor=(.5,-.3))
        fig.suptitle(title+' · Fine模型两层时间占比（网络启动100%）',x=.055,ha='left',fontsize=17)
        fig.text(.055,-.055,'GEMM等数学运算kernel的总时间可归入HBM受限服务；此图不是计算单元活跃时间或利用率，0%不表示没有FLOPs。',color='#566274')
        fig.tight_layout(rect=(0,0,1,.94));save(fig,directory,'03_components_'+arch)


def operator_plot(rows,operators,directory):
    main=[r for r in rows if r['network_startup_factor']==1 and r['kernel_launch_us']==0]
    fig,axes=plt.subplots(2,2,figsize=(16.5,11.0))
    for ax,row in zip(axes.flat,main):
        values=sorted((r for r in operators if r['id']==row['id']),key=lambda r:r['total_ms'],reverse=True)[:10]
        values=values[::-1];names=[r['operator'] for r in values];left=np.zeros(len(values))
        for field in COLORS:
            amounts=np.array([r[field+'_ms'] for r in values])
            ax.barh(range(len(names)),amounts,left=left,color=COLORS[field],label=COST_NAMES[field]);left+=amounts
        ax.set_yticks(range(len(names)),names,fontsize=8)
        ax.set_xlabel('关键路径归因耗时 / ms');ax.set_title(label(row),loc='left',pad=10)
        ax.grid(axis='x',color='#e0e5eb');ax.set_axisbelow(True)
        ax.margins(x=.06)
    fig.suptitle('算子总耗时 Top10：颜色与直方图使用同一限制资源归因',x=.05,ha='left',fontsize=18)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=5,frameon=False,bbox_to_anchor=(.5,.018),fontsize=10)
    fig.text(.05,-.01,'矩阵算子不等于算力受限算子；HBM色表示其服务受访存限制。完整16场景算子表仍在operator_details.csv。',color='#566274')
    fig.tight_layout(rect=(0,.06,1,.96));save(fig,directory,'04_operator_ranking')


def interpretation_plot(rows,components,directory):
    source=directory.parent/'operator_semantics.json'
    if not source.exists():return False
    semantic={r['id']:r for r in read(source)['windows']}
    main=[r for r in rows if r['network_startup_factor']==1 and r['kernel_launch_us']==0]
    fig,axes=plt.subplots(1,2,figsize=(14.2,6.8));x=np.arange(len(main))
    left=[('matrix_ms','矩阵运算kernel','#237f8b'),('pointwise_ms','点算/状态运算kernel','#779968'),
          ('cache_ms','缓存写入kernel','#a49abd'),('communication_ms','通信算子','#c94444'),('other_ms','其他','#a6adba')]
    bottom=np.zeros(len(main))
    for key_,name,color in left:
        vals=np.array([100*semantic[r['id']][key_]/r['fine_tpot_ms'] for r in main])
        axes[0].bar(x,vals,.62,bottom=bottom,label=name,color=color)
        for i,v in enumerate(vals):
            if v>8:axes[0].text(i,bottom[i]+v/2,f'{v:.1f}%',ha='center',va='center',color='white',fontsize=11)
        bottom+=vals
    bottom=np.zeros(len(main))
    for field in COLORS:
        vals=np.array([100*sum(c[field+'_ms'] for c in components if c['id']==r['id'])/r['fine_tpot_ms'] for r in main])
        axes[1].bar(x,vals,.62,bottom=bottom,label=COST_NAMES[field],color=COLORS[field])
        for i,v in enumerate(vals):
            if v>8:axes[1].text(i,bottom[i]+v/2,f'{v:.1f}%',ha='center',va='center',color='white',fontsize=11)
        bottom+=vals
    for index,(ax,title) in enumerate(zip(axes,('按算子用途：kernel完整耗时','按限制资源：关键路径服务归因'))):
        ax.set_title(title,loc='left',fontsize=14,pad=15)
        ax.set_xticks(x,[label(r).replace(' / ','\n') for r in main]);ax.set_ylim(0,105)
        decorate(ax,'占总 TPOT / %')
        handles,labels=ax.get_legend_handles_labels()
        fig.legend(handles,labels,frameon=False,fontsize=9,ncol=2,loc='center',bbox_to_anchor=(.285 if index==0 else .77,.12))
    fig.suptitle('矩阵kernel占时大，与“算力受限时间接近零”并不矛盾',x=.055,ha='left',fontsize=17)
    fig.text(.055,.015,'同一段GEMM时间左侧可归为矩阵运算、右侧归为HBM受限；两种视图分别合计100%，不能相加。',color='#566274')
    fig.subplots_adjust(left=.065,right=.985,top=.82,bottom=.31,wspace=.18)
    save(fig,directory,'05_operator_type_vs_resource')
    return True


def report_lines(rows,coverage,directory):
    main=[r for r in rows if r['network_startup_factor']==1 and r['kernel_launch_us']==0]
    lookup={key(r):r for r in rows};fine0=[r for r in rows if r['kernel_launch_us']==0]
    fine1=[r for r in rows if r['kernel_launch_us']==1]
    concise=[]
    for r in main:
        one=lookup[r['case_id'],1.,1.]
        concise.append([label(r),str(r['cp']),fmt(r['coarse_tpot_ms']),fmt(r['fine_tpot_ms']),
            fmt(one['fine_tpot_ms']),fmt(r['peak_hbm_GB'],3),'通过' if r['target_satisfied'] else '未通过',
            '通过' if one['target_satisfied'] else '未通过'])
    links=[]
    for r in rows:
        folder=Path(r['result_path']).parent
        display=label(r)+f' · α{r["network_startup_factor"]:g} / kernel {r["kernel_launch_us"]:g}μs'
        links.append([display,f'[正常报告]({(folder/"report.md").as_posix()})',
            f'[完整算子表]({(folder/"operator_breakdown.csv").as_posix()})',
            f'[配置]({r["config_path"]})',f'[数值]({r["result_path"]})'])
    coverage_main=[c for c in coverage if c['network_startup_factor']==1 and c['kernel_launch_us']==0]
    cover_values=[[label(next(r for r in rows if r['id']==c['id'])),c['kind'],str(c['layers']),
        f'{c["matched_parent_events"]}/{c["eligible_parent_events"]}',str(c['generated_trace_operations']),
        str(c['additional_scalar_events']),str(c['integer_reference_shape_specs']),str(c['unresolved_budget_matrix_specs']),
        fmt(c['added_hbm_bytes_representative_layer']/1e6,3)] for c in coverage_main]
    evidence_cases=sum(any(info.get('evidence_artifact') and (directory/info['evidence_artifact']).exists()
                           for info in row['kinds'].values()) for row in rows)
    lines=['# SP/CP 细粒度算子仿真报告','',
        '**Fine0 为本次主模型；每kernel 1 μs仅为独立工程敏感性，未经实测校准。** '
        '在原4个2.8T owner-star CP场景上执行12个Fine0网络启动对照，另执行4个1μs子kernel启动场景。'
        '其余16个原始模型/硬件CP组合仍受此前必要下界排除，不补造执行结果。','',
        f'Fine0 的12点中 {sum(r["target_satisfied"] for r in fine0)} 点达到5ms；1μs工程敏感性的4点中 '
        f'{sum(r["target_satisfied"] for r in fine1)} 点达到5ms。更细粒度增加了显式中间访问和串行依赖，不能称为实机精度提升。','',
        '## 为什么大耗时GEMM没有进入“算力受限”占比','',
        '**算子总耗时与限制资源归因是两个统计口径。** 算子排行按kernel完整关键路径时间排序；'
        '组件直方图按该kernel的限制资源分类。GEMM、QK、AV等数学运算kernel也可能主要受HBM读取限制，因此其服务被归到HBM受限。','',
        '当前单kernel服务近似为 `max(FLOPs / 有效算力, HBM字节 / 带宽)`，再计启动与争用。'
        '这两个服务需求可以重叠，不能相加充当kernel总耗时；直方图不是实机计算单元活跃时间或利用率图。','',
        '以Kimi K3/960DT Fine0的 shared_projection_post.budget_matmul 为例：跨93层的算力独占需求约0.017032 ms，'
        'HBM独占需求约0.887335 ms，实际模型服务为0.887335 ms，因而全部归为HBM受限。'
        '全POD batch32在当前EP布局中分散到32个Attention owner，每个owner只有一个Decode query，不能按每卡batch32理解权重复用。','',
        '这些下界仍依赖当前有效算力（峰值×0.5）与HBM标称带宽假设，尚未按真实低batch矩阵形状校准。'
        '“HBM受限”是模型结论，不是实机profiler测得的资源利用率。','',
        table(['模板/硬件','CP','Coarse ms','Fine0 ms','Fine1μs ms','Fine0峰值GB','Fine0 5ms','Fine1μs 5ms'],concise),'',
        '![聚合与细粒度](figures/01_coarse_fine_launch.png)','',
        '## 网络启动对照','',
        '所有网络通信事件、payload、资源需求和启动在同一alpha下逐项保持原CP图不变。'
        'alpha=1/0.5/0.3只作用网络启动；1μs只作用实际child/extra compute kernel。两者没有相互替代。','',
        '![网络启动](figures/02_network_factor.png)','',
        '## Attention / MoE 整体与内部占比','',
        '外层分母是该场景TPOT；内层分母是对应Attention侧或MoE侧的自身时间。算力受限/HBM受限是瓶颈归因，而非算子用途分类。'
        '通信以整个通信算子关键路径时间计入，包含其网络启动和争用；kernel启动单列且不重复计入计算。'
        'Attention侧仍包含共享稠密D预算，MoE侧为路由专家及其通信。','',
        '![Kimi两层占比](figures/03_components_kimi-k3.png)','',
        '![V4两层占比](figures/03_components_deepseek-v41.png)','',
        '[完整组件时间与百分比](component_costs.csv) · [全部算子时间与内部开销](operator_details.csv)','',
        '## 算子执行深度','',
        table(['路径','执行顺序/保持条件'],[
            ['共享投影','保留pre/post预算GEMM；未知D的真实Q/K/V/输出形状未重建'],
            ['MLA/稀疏Attention','QK → FP32 local max/exp/sum → AV → owner max归并 → rescale → numerator/denominator归并与归一化'],
            ['V4索引','per-head score dot → head加权归约 → local top-k → pack → gather → global top-k → index广播'],
            ['KDA','depthwise conv/shift → SiLU → QK归一化与gate → state decay/prediction → residual/beta → rank-one update → readout → output norm/gate'],
            ['路由专家','gate预算GEMM → up预算GEMM → FP32 SiTU/SwiGLU → down预算GEMM；新增carry生命周期明确'],
            ['通信','保留原owner-star通信边界与次序，不用子kernel融合吞掉collective']]),'',
        'Kimi压缩MLA的QK/AV整数matmul注释只表示参考代数，不声明真实压缩KV binary布局。'
        'KDA core沿用coarse compute资源；FP32持久状态的读写不等于已经校准FP32算术吞吐。'
        'KDA已知depthwise-conv参数工作从共享D移入相应算子，参数FLOPs守恒；其他新增非线性工作分列。','',
        '## 原生trace覆盖与形状边界','',
        table(['模板/硬件','kind','复用层数','父事件匹配','生成child','额外scalar','整数参考shape定义','预算矩阵定义','新增HBM MB/代表层'],cover_values),'',
        '事件数针对保存的代表图，不能当作全POD硬件profiler的累计kernel数。shape定义数来自精确匹配配置中的操作定义，'
        '与匹配的事件实例数是不同分母。匹配100%不意味着所有权重形状已解析。','',
        '共享D维持未解析预算；MoE根据三矩阵恒等式推导的逻辑中间宽度可能是小数，未伪造整数专家GEMM形状。'
        '新增HBM为代表图工作量，不冒称全POD物理流量；不同中间张量的申请额不能相加当作并发峰值。'
        '实际峰值取引擎逐池观察，原coarse arena保留使容量估计偏保守。','',
        '[完整trace覆盖表](trace_coverage.csv) · [全部子算子定义与shape/implementation](trace_operation_specs.csv)','',
        '## 耗时算子排行','',
        '![算子排行](figures/04_operator_ranking.png)','',
        '图仅显示前10项；完整表无截取，单位ms，含各算子的计算/通信/访存/kernel启动/其他与各自分母。','',
        '## 输入、证据与复跑','',table(['场景','正常报告','算子表','输入','原始结果'],links),'',
        f'逐kind的精确kernel_traces输入都随case保存；实际有 {evidence_cases} 个场景保存完整代表graph和simulation的gzip证据。'
        '每个case的kinds字段列出真实文件路径，不为未导出的trace编造链接。','',
        '[运行来源与源码哈希](provenance.json) · [汇总CSV](summary.csv) · [汇总JSON](summary.json)','',
        '```powershell','.venv/Scripts/python.exe -X utf8 scripts/run_scaling_cp_fine.py',
        '.venv/Scripts/python.exe -X utf8 scripts/report_scaling_cp_fine.py','```','',
        '细粒度模型仍是解析执行图：没有kernel实测、二进制、tiling、运行时调度或质量校准。'
        '成功的warm Decode不证明CP Prefill、排队时延或生产P99可行。']
    checks=[p.name for p in directory.glob('*verification*.json') if p.is_file()]
    if checks:lines += ['', '独立校验：'+' · '.join(f'[{name}]({name})' for name in sorted(checks)),'']
    if (directory/'operator_semantics.json').exists() and (directory/'resource_service_demands.json').exists():
        demands=read(directory/'resource_service_demands.json')['rows']
        names=('shared_projection_post.budget_matmul','shared_projection_pre.budget_matmul','moe.gate.budget_gemm')
        examples=[r for r in demands if r['case_id']=='kimi-k3-model-960DT-2p8T' and r['operator'] in names]
        explanation=['## 同一批算子的两种视图','',
            '下面左图按算子用途统计kernel总时长，右图按限制资源归因。矩阵或点算kernel本身可受访存限制，'
            '所以左图的数学运算占比高，不要求右图的算力受限占比也高。','',
            '![用途与资源归因](figures/05_operator_type_vs_resource.png)','',
            'Kimi K3/960DT Fine0的代表性算子（跨全部层的同一关键路径，单位ms）：','',
            table(['算子','算力独占服务需求','HBM独占服务需求','模型实际总时间'],[
                [r['operator'],fmt(r['compute_service_ms'],6),fmt(r['hbm_service_ms'],6),fmt(r['actual_ms'],6)] for r in examples]),'',
            '**独占服务需求不是实际时间拆分，算力需求与HBM需求不能相加。** '
            '这里从保存的真实图与trace沿同一关键路径取FLOPs/资源能力、bytes/带宽；没有把所有并行rank的时间累加。','',
            '[全部代表算子双服务需求CSV](resource_service_demands.csv) · [按算子用途的窗口汇总](semantic_window_summary.csv) · '
            '[用途分类与逐算子数据](operator_semantics.json)','']
        position=lines.index('## Attention / MoE 整体与内部占比')
        lines[position:position]=explanation
    return lines


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT)
    args=parser.parse_args();directory=args.output;data=read(directory/'summary.json')
    assert data['complete'] and len(data['rows'])==16,'Wait for all16 actual fine executions'
    rows=data['rows'];components,operators=cost_rows(rows);coverage,specs=trace_rows(rows,directory)
    for name,values in [('component_costs',components),('operator_details',operators),('trace_coverage',coverage),('trace_operation_specs',specs)]:
        csv_save(directory/(name+'.csv'),values)
    json_save(directory/'component_costs.json',{'rows':components,'scope':'Executed fine critical-path ledger; all16 windows; not whole-POD accumulated work'})
    figures=directory/'figures';figures.mkdir(exist_ok=True)
    comparison_plot(rows,figures);network_plot(rows,figures);component_plots(rows,components,figures);operator_plot(rows,operators,figures)
    interpretation_plot(rows,components,figures)
    lines=report_lines(rows,coverage,directory)
    (directory/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    document=render(lines).replace('<title>SP/CP与启动优化</title>','<title>SP/CP细粒度算子仿真</title>')
    (directory/'report.html').write_text(document,encoding='utf-8')
    note=('<!-- cost-meaning:start -->\n\n'
          '## 算子总耗时与资源归因的区别\n\n'
          'GEMM等数学运算kernel的总耗时包含内部访存；表中的计算比例表示算力受限服务，访存比例表示HBM受限服务。'
          '模型使用max(FLOPs/有效算力, HBM字节/带宽)，两种需求可重叠，不能相加。0%算力受限不代表没有FLOPs或计算单元没有工作。\n\n'
          '<!-- cost-meaning:end -->')
    for row in rows:
        path=(directory/row['result_path']).with_name('report.md')
        old=path.read_text(encoding='utf-8')
        old=old.replace('| 计算 / % |','| 算力受限 / % |').replace('| 访存 / % |','| HBM受限 / % |')
        if '<!-- cost-meaning:start -->' in old:
            left,rest=old.split('<!-- cost-meaning:start -->',1);_,right=rest.split('<!-- cost-meaning:end -->',1)
            old=left+note+right
        else:old=old.rstrip()+'\n\n'+note+'\n'
        path.write_text(old,encoding='utf-8')
    from html.parser import HTMLParser
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.links=[]
        def handle_starttag(self,tag,attrs):
            for k,v in attrs:
                if k in ('src','href') and not v.startswith(('http:','https:','#')):self.links.append(v)
    p=Links();p.feed(document);missing=[x for x in p.links if not (directory/x).is_file()]
    assert not missing,missing
    audit={'status':'passed','windows':16,'component_rows':len(components),'complete_operator_rows':len(operators),
        'local_links_checked':len(p.links),'missing_links':missing,'figure_count':len(list(figures.glob('*.png'))),
        'all_component_and_operator_additivity_checked':True}
    json_save(directory/'report-verification.json',audit)
    print(json.dumps({'report':str(directory/'report.html'),**audit},ensure_ascii=False))


if __name__=='__main__':main()
