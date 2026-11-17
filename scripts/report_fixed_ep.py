"""Render fixed-POD/fixed-EP model-scaling evidence."""
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.report_scaling_study import plt,np,save,fmt,table,decorate
from scripts.report_sp_megakernel import render

OUT=ROOT/'outputs/scaling-10case-20261113/fixed-ep-128'


def main():
    payload=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    assert payload['complete'] and len(payload['rows'])==12
    rows=payload['rows']
    groups=[]
    for arch in ('kimi-k3','deepseek-v41'):
        for hw in ('960DT','970DT'):
            group=sorted([r for r in rows if r['architecture_id']==arch and r['hardware']==hw],key=lambda r:r['model_T'])
            assert len(group)==3 and all(r['ep']==128 for r in group)
            groups.append((arch,hw,group))
    fig,axes=plt.subplots(2,2,figsize=(13.5,9.5))
    for col,arch in enumerate(('kimi-k3','deepseek-v41')):
        for hw,color in [('960DT','#237c88'),('970DT','#b64b52')]:
            group=next(g for a,h,g in groups if a==arch and h==hw)
            values=[r['communication_pct'] for r in group]
            axes[0,col].plot(range(3),values,'-o',color=color,linewidth=2.3,label=f'{hw} · TP{group[0]["tp"]}')
            for x,y in enumerate(values):
                axes[0,col].annotate(f'{y:.2f}%',(x,y),xytext=(0,9 if hw=='970DT' else -20),textcoords='offset points',ha='center',color=color)
        axes[0,col].set_xticks(range(3),['2.8T','10T','100T'])
        axes[0,col].set_title('Kimi K3' if arch=='kimi-k3' else 'DeepSeek V4.1',loc='left',fontsize=15)
        axes[0,col].legend(frameon=False)
        axes[0,col].margins(x=.15,y=.3)
        decorate(axes[0,col],'总通信占 TPOT / %')
        selected=[r for hw in ('960DT','970DT') for r in next(g for a,h,g in groups if a==arch and h==hw)]
        x=np.arange(6);bottom=np.zeros(6)
        for key,label,color in [('communication_pct','通信（启动+传输）','#c94444'),
                                ('memory_pct','访存限制服务','#8bbbd1'),
                                ('compute_pct','计算限制服务','#2477a8'),('other_pct','其他','#a6adba')]:
            values=np.array([r[key] for r in selected])
            axes[1,col].bar(x,values,.65,bottom=bottom,label=label,color=color)
            bottom+=values
        axes[1,col].set_xticks(x,[f'{r["hardware"][:3]}\n{r["model_T"]:g}T' for r in selected])
        axes[1,col].set_ylim(0,106);decorate(axes[1,col],'整个 TPOT 的组成 / %')
    handles,labels=axes[1,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.5,.015))
    fig.suptitle('固定卡数、固定 EP128：总通信仍占主导，但占比随模型扩大而下降',x=.055,ha='left',fontsize=17)
    fig.text(.055,.002,'960为4096卡/TP32，970为12288卡/TP96；同硬件内EP、TP、启动时间保持不变。0%计算归因不代表无计算。',fontsize=10,color='#566274')
    fig.tight_layout(rect=(0,.06,1,.95));save(fig,OUT,'fixed_ep_comparison')
    matrix=[]
    for arch,hw,group in groups:
        matrix.append(['Kimi K3' if arch=='kimi-k3' else 'V4.1',hw,f'TP{group[0]["tp"]}/EP128',
                       *[fmt(r['communication_pct'],2)+'%' for r in group]])
    detail=[]
    for r in rows:
        detail.append(['K3' if r['architecture_id']=='kimi-k3' else 'V4.1',r['hardware'],f'{r["model_T"]:g}T',
                       fmt(r['tpot_ms']),fmt(r['communication_ms']),fmt(r['startup_ms']),fmt(r['transfer_ms'],4),
                       fmt(r['communication_pct'],2),fmt(r['memory_pct'],2),fmt(r['compute_pct'],2)])
    lines=['# 固定卡数、固定EP后的模型扩展对照','',
        '**在本次EP128对照中，总通信占比：2.8T > 10T > 100T；三个规模都是通信占比高于访存与计算归因。**','',
        '这不是“通信时间减少”：绝对通信传输量和传输时间上升，但共享/专家权重访存增长得更快，因而总通信占比下降。'
        '相较原实验，本轮TP固定，不再用“TP下降导致启动减少”解释趋势。','',
        table(['模板','硬件','固定并行度','2.8T通信占比','10T通信占比','100T通信占比'],matrix),'',
        '![固定EP对照](fixed_ep_comparison.png)','',
        '## 各项明细','',
        table(['模板','硬件','模型','TPOT ms','通信 ms','启动 ms','传输 ms','通信 %','访存 %','计算 %'],detail),'',
        '通信比例的分母为整个TPOT，通信包含其启动和传输/争用。计算与访存按限制资源归因，'
        '计算列0%不表示没有FLOPs；所有kernel仍计计算需求。','',
        '## 控制条件与来源','',
        '固定EP=128，沿用原2.8T基线；960DT固定4096卡、TP32，970DT固定12288卡、TP96。'
        '保持用户三组P/A/top-k/专家数、1M上下文、全POD batch32、各架构Attention/KV模板、精度、硬件速率与ring/direct算法。'
        '每种模板/硬件内层数与缓存模板不变，只改变指定模型预算和MoE隐藏宽度/top-k/专家数。','',
        '4个2.8T点与原场景完全相同，复用已保存执行证据；其余8点实际逐kind执行原离散事件引擎。'
        '本结果是该预算模板、固定EP128与执行策略下的预测，不是所有固定EP或真实checkpoint的普遍规律。','',
        '[全部数据CSV](summary.csv) · [原始数值及来源](summary.json) · [上一轮报告](../report.html)','',
        '```powershell','.venv/Scripts/python.exe scripts/scaling_fixed_ep.py',
        '.venv/Scripts/python.exe scripts/report_fixed_ep.py','```']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (OUT/'report.html').write_text(render(lines),encoding='utf-8')
    print(json.dumps({'report':str(OUT/'report.html'),'cases':len(rows)},ensure_ascii=False))


if __name__=='__main__':main()
