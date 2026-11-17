"""Render the ten-case scaling study from saved engine evidence only."""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

FONT = Path('C:/Windows/Fonts/msyh.ttc')
if FONT.exists():
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(FONT)).get_name()
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.unicode_minus': False, 'figure.facecolor': '#f5f7fa',
                     'axes.facecolor': 'white', 'savefig.facecolor': '#f5f7fa',
                     'svg.fonttype': 'none'})
COLORS = {'compute': '#2477a8', 'memory': '#8bbbd1', 'comm': '#c94444',
          'transfer': '#df9a35', 'other': '#a6adba', '960': '#237c88', '970': '#b64b52'}


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def fmt(value, precision=3):
    return '—' if value is None else f'{value:,.{precision}f}'


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                       '|' + '|'.join('---' for _ in headers) + '|'] +
                     ['| ' + ' | '.join(str(v) for v in row) + ' |' for row in rows])


def csv_out(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)


def network_operator(row):
    meta = ' '.join(str(row.get(k, '')).lower() for k in ('operator', 'component'))
    return (row.get('components_s', {}).get('communication_s', 0) > 0 or
            any(v in meta for v in ('dispatch', 'combine', 'all_reduce', 'allreduce', 'tp_index_scores',
                                    'reduce_scatter', 'all_gather', 'tp_attention', 'tp_ffn')))


def records(directory):
    manifest = load(directory / 'manifest.json')
    split_path = directory/'communication_breakdown.json'
    split_rows = load(split_path).get('totals',[]) if split_path.exists() else []
    split_by_case = {r['case_id']:r for r in split_rows if r['scope']=='main'}
    prefill_path = directory/'prefill-study/summary.json'
    prefill_rows = load(prefill_path).get('rows',[]) if prefill_path.exists() else []
    prefill_by_case = {r['case_id']:r for r in prefill_rows}
    out = []
    for item in manifest['cases']:
        path = directory / item['result_path']
        summary_path = path.with_name('summary.json')
        result = load(summary_path if summary_path.exists() else path)
        config = result['config']
        model, hw, exe = (config[k] for k in ('model', 'hardware', 'execution'))
        metrics = result.get('metrics', {})
        rec = metrics.get('latency_breakdown', {}).get('tpot')
        total = metrics.get('tpot_s')
        operators = (rec or {}).get('operator_breakdown', {}).get('operators', [])
        communication = sum(o['total_s'] for o in operators if network_operator(o)) if rec else None
        if rec and not operators:
            raise ValueError(f'{item["id"]}: missing operator attribution')
        costs = (rec or {}).get('operator_breakdown', {}).get('components_s', {})
        service = sum(v for k, v in (rec or {}).get('components_s', {}).items()
                      if k.startswith('communication_')) if rec else None
        study = result.get('study', {})
        memory = result.get('memory_breakdown', {}).get('pools', [])
        peaks = result.get('simulation', {}).get('memory_peaks_bytes',
                result.get('simulation', {}).get('peak_memory_bytes', {}))
        peak = max((v for k, v in peaks.items() if 'hbm' in k), default=0)
        if memory:
            peak = max(peak, max((p.get('peak_bytes', p.get('resident_bytes', 0)) or 0
                                 for p in memory if p.get('pool_kind') == 'hbm'), default=0))
        row = {'id': item['id'], 'group': item['group'],
               'architecture': item.get('architecture_id', item.get('architecture', 'baseline')),
               'architecture_label': item.get('architecture_label', item.get('architecture_id', 'baseline')),
               'model': item.get('model_label', model['name']),
               'hardware': item.get('hardware_label', hw.get('name')),
               'model_T': model['total_params'] / 1e12,
               'active_B': model['active_params'] / 1e9,
               'top_k':model['top_k'],'num_experts':model['num_experts'],'layers':model['layers'],
               'cards': hw['npu_count'], 'ep': exe['ep'], 'tp': (study.get('afd') or {}).get('tp', exe['tp']),
               'target_ms': item['target_tpot_ms'],
               'status': result['status'], 'decode_status': study.get('decode', {}).get('status'),
               'ttft_s': metrics.get('ttft_s'), 'tpot_ms': total * 1000 if total is not None else None,
               'comm_ms': communication * 1000 if communication is not None else None,
               'comm_pct': 100 * communication / total if total and communication is not None else None,
               'comm_service_ms': service * 1000 if service is not None else None,
               'comm_service_pct': 100 * service / total if total and service is not None else None,
               'compute_ms': 1000 * sum(o['components_s'].get('compute_s', 0) for o in operators if not network_operator(o)) if rec else None,
               'memory_ms': 1000 * sum(o['components_s'].get('memory_access_s', 0) for o in operators if not network_operator(o)) if rec else None,
               'peak_hbm_GB': peak / 1e9 if peak else None,
               'hbm_capacity_GB': hw['npu']['hbm_bytes'] / 1e9,
               'result_path': item['result_path'],
               'report_path': str(Path(item['result_path']).with_name('report.md')).replace('\\', '/')}
        row['other_ms'] = max(0., row['tpot_ms'] - row['comm_ms'] - row['compute_ms'] - row['memory_ms']) if rec else None
        row['target_ratio'] = row['tpot_ms'] / row['target_ms'] if total else None
        row['comm_budget_pct'] = row['comm_ms'] / row['target_ms'] * 100 if rec else None
        if item['id'] in split_by_case:
            split = split_by_case[item['id']]
            row.update(comm_startup_ms=split['startup_ms'], comm_transfer_ms=split['transfer_ms'],
                       comm_startup_pct=100*split['startup_ms']/row['tpot_ms'] if row['tpot_ms'] else None,
                       comm_transfer_pct=100*split['transfer_ms']/row['tpot_ms'] if row['tpot_ms'] else None)
        if item['id'] in prefill_by_case:
            prefill = prefill_by_case[item['id']]
            row.update(ttft_s=prefill.get('ttft_s'),prefill_status=prefill['status'])
            row['_prefill'] = prefill
        row['_result'] = {'config': config, 'metrics': metrics}
        row['_item'] = item
        out.append(row)
    return manifest, out


def save(fig, directory, stem):
    fig.savefig(directory / (stem + '.png'), dpi=160, bbox_inches='tight')
    fig.savefig(directory / (stem + '.svg'), bbox_inches='tight')
    plt.close(fig)


def decorate(ax, ylabel):
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', color='#e0e5eb', linewidth=.7)
    ax.set_axisbelow(True)


def scaling_plot(rows, directory):
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.7))
    for name, group in [('960', rows[:3]), ('970', rows[3:6])]:
        x = np.arange(3)
        vals = [r['comm_pct'] if r['comm_pct'] is not None else np.nan for r in group]
        axes[0].plot(x, vals, '-o', linewidth=2.5, markersize=7,
                     label=f'昇腾{name}DT · {group[0]["cards"]:,}卡', color=COLORS[name])
        for px, value in zip(x, vals):
            if np.isfinite(value):
                axes[0].annotate(f'{value:.1f}%', (px, value), xytext=(0, -18 if name == '960' else 9),
                                 textcoords='offset points', ha='center', color=COLORS[name])
        vals = [r['comm_ms'] if r['comm_ms'] is not None else np.nan for r in group]
        axes[1].plot(x, vals, '-o', linewidth=2.5, markersize=7, label=f'昇腾{name}DT', color=COLORS[name])
    for ax in axes:
        ax.set_xticks(range(3), ['2.8T / EP128', '10T / EP512', '100T / EP2048'])
        ax.margins(x=.14, y=.25)
        ax.legend(loc='best', frameon=False)
    decorate(axes[0], '通信算子占实际 TPOT / %')
    decorate(axes[1], '通信算子关键路径时间 / ms')
    fig.suptitle('模型扩展：通信占比与绝对通信时间', fontsize=18, x=.06, ha='left', y=1.01)
    fig.text(.06, -.025, '目标 TPOT 均为 5 ms；分母使用实际预测 TPOT。通信含通信算子自身的资源争用。', color='#566274')
    fig.tight_layout()
    save(fig, directory, '01_model_scaling')


def breakdown_plot(rows, directory):
    fig, axes = plt.subplots(1, 2, figsize=(14.3, 6.2), gridspec_kw={'width_ratios': [1.5, 1]})
    for ax, group, title in [(axes[0], rows[:6], '模型扩展 · 目标 5 ms'),
                              (axes[1], rows[6:], '硬件变化 · 同一 10T 模型')]:
        x = np.arange(len(group))
        bottom = np.zeros(len(group))
        components = [('compute_ms', '计算限制服务', COLORS['compute']),
                      ('memory_ms', '访存限制服务', COLORS['memory'])]
        if all('comm_startup_ms' in r for r in group):
            components += [('comm_startup_ms','通信启动',COLORS['comm']),
                           ('comm_transfer_ms','通信传输（含争用）',COLORS['transfer'])]
        else:
            components += [('comm_ms', '通信算子（含其争用）', COLORS['comm'])]
        components += [('other_ms', '其他开销 / 等待', COLORS['other'])]
        for key,label,color in components:
            vals = np.array([r[key] or 0 for r in group])
            ax.bar(x, vals, bottom=bottom, color=color, width=.62, label=label)
            bottom += vals
        for i, row in enumerate(group):
            if row['tpot_ms'] is None:
                ax.text(i, .2, '容量\n不可行', ha='center', va='bottom', color=COLORS['comm'])
            else:
                ax.text(i, row['tpot_ms'] + max(bottom) * .025, f'{row["tpot_ms"]:.2f}', ha='center', fontsize=10)
            ax.plot([i-.33, i+.33], [row['target_ms']]*2, color='#172738', linestyle='--', linewidth=1.7)
        labels = [f'{r["model_T"]:g}T\n' + ('960DT' if '960' in r['hardware'] else '970DT') for r in group] if len(group) == 6 else ['950DT\n1K', '960DT\n4K', '970DT\n12K', '970+LPU\n各6K']
        ax.set_xticks(x, labels)
        ax.set_title(title, loc='left', pad=15)
        decorate(ax, '实际 TPOT / ms')
        ax.set_ylim(0, max(max(bottom) * 1.23, max(r['target_ms'] for r in group) * 1.2))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), frameon=False, bbox_to_anchor=(.5, -.03))
    fig.suptitle('每 token 延迟：瓶颈组成与目标差距', fontsize=18, x=.06, ha='left', y=1.01)
    fig.text(.06, -.07, '黑色虚线为目标；容量失败不填入有限 TPOT。计算与访存按限制资源归因，避免重复相加。', color='#566274')
    fig.tight_layout(rect=(0, .05, 1, 1))
    save(fig, directory, '02_latency_breakdown')


def memory_plot(rows, directory):
    fig, ax = plt.subplots(figsize=(13.2, 5.5))
    x = np.arange(len(rows))
    peaks = [r['peak_hbm_GB'] or 0 for r in rows]
    caps = [r['hbm_capacity_GB'] for r in rows]
    ax.bar(x, peaks, width=.63, color=[COLORS['comm'] if p > c else COLORS['960'] for p, c in zip(peaks, caps)])
    for i, (p, c) in enumerate(zip(peaks, caps)):
        ax.plot([i-.38, i+.38], [c, c], 'k--', linewidth=1.5)
        ax.text(i, p + 4, f'{p:.1f}', ha='center', fontsize=10)
    ax.set_xticks(x, [f'C{i+1:02d}\n{r["model_T"]:g}T\n{r["hardware"].replace("DT", "")}' for i,r in enumerate(rows)])
    ax.set_ylim(0, max(max(peaks), max(caps)) * 1.18)
    decorate(ax, '最重单卡 HBM 占用 / GB（十进制）')
    ax.set_title('Decode 容量：长上下文 KV 与共享权重的单卡压力', loc='left', pad=20, fontsize=17)
    fig.text(.065, -.015, '黑色虚线为单卡容量；失败场景显示申请需求或驻留需求，详见逐场景内存表；不能以全 POD 总容量代替单卡准入。', color='#566274')
    fig.tight_layout()
    save(fig, directory, '03_memory_capacity')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/scaling-10case-20261113')
    args = parser.parse_args()
    directory = args.output
    manifest, rows = records(directory)
    verify_results(directory, rows)
    figures = directory / 'figures'
    figures.mkdir(exist_ok=True)
    architectures = list(dict.fromkeys(r['architecture'] for r in rows))
    for architecture in architectures:
        group = [r for r in rows if r['architecture'] == architecture]
        location = figures / architecture
        location.mkdir(exist_ok=True)
        scaling_plot(group, location)
        breakdown_plot(group, location)
        memory_plot(group, location)
    if len(architectures) == 2:
        architecture_plot(rows, figures)
    if (directory / 'sensitivity.json').exists():
        sensitivity_plot(rows, load(directory / 'sensitivity.json'), figures)
    if (directory / 'ratio-search/summary.json').exists():
        ratio_data = load(directory / 'ratio-search/summary.json')
        if not ratio_data['complete']:
            raise ValueError('AFD ratio search is not complete')
        ratio_plot(ratio_data, figures)
    if all('comm_startup_ms' in r for r in rows):
        communication_split_plot(rows, figures)
    if (directory/'component_breakdown.json').exists():
        component_data=load(directory/'component_breakdown.json')
        component_plots(rows,component_data,figures)
    public_rows = [{k:v for k,v in r.items() if not k.startswith('_')} for r in rows]
    csv_out(directory / 'comparison.csv', public_rows)
    (directory / 'comparison.json').write_text(json.dumps(public_rows, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    write_report(directory, manifest, rows)
    print(json.dumps({'report': str(directory / 'report.html'), 'cases': len(rows), 'figures': len(list(figures.rglob('*.png')))}, ensure_ascii=False))


def verify_results(directory, rows):
    checks = []
    if len(rows) != 20 or len({r['id'] for r in rows}) != 20:
        raise ValueError('The final study requires exactly 20 distinct result points')
    for row in rows:
        result = row['_result']
        model = result['config']['model']
        assert result['config']['workload']['batch_size'] == 32
        assert result['config']['workload']['input_length'] == 1048576
        if row['tpot_ms'] is not None:
            record = result['metrics']['latency_breakdown']['tpot']
            assert abs(sum(record['components_s'].values()) - record['total_s']) < 1e-10
            assert abs(sum(o['total_s'] for o in record['operator_breakdown']['operators']) - record['total_s']) < 1e-10
            assert 0 <= row['comm_pct'] <= 100 + 1e-8
        assert result['metrics']['ttft_s'] is None, 'Original baseline source is warm-decode-only; Prefill comes from its independent artifact'
        if row.get('_prefill',{}).get('latency_breakdown'):
            prefill_record = row['_prefill']['latency_breakdown']
            if 'total_s' in prefill_record:
                assert abs(sum(prefill_record['components_s'].values())-prefill_record['total_s']) < 1e-6
        assert model['total_params'] in (2.8e12, 1e13, 1e14)
        assert model['active_params'] in (104e9, 250e9, 1e12)
        checks.append({'case': row['id'], 'input_and_additivity': 'passed'})
    for architecture in dict.fromkeys(r['architecture'] for r in rows):
        group = [r for r in rows if r['architecture'] == architecture]
        for hw in ('960DT', '970DT'):
            pair = [r for r in group if r['hardware'] == hw and r['model_T'] == 10]
            assert len(pair) == 2
            for key in ('tpot_ms', 'comm_ms', 'peak_hbm_GB'):
                if pair[0][key] is None or pair[1][key] is None:
                    assert pair[0][key] == pair[1][key]
                else:
                    assert abs(pair[0][key] - pair[1][key]) < 1e-8
            checks.append({'case': architecture+'-'+hw, 'duplicate_physical_scenario': 'passed'})
    (directory / 'result-audit.json').write_text(json.dumps({'status':'passed','checks':checks}, indent=2), encoding='utf-8')


def sensitivity_plot(rows, sensitivity, directory):
    variants = {(r['case'],r['variant']):r for r in sensitivity}
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.8))
    archs = list(dict.fromkeys(r['architecture'] for r in rows))
    for ax, architecture in zip(axes, archs):
        group = [r for r in rows if r['architecture'] == architecture]
        x = np.arange(len(group))
        ax.plot(x, [r['tpot_ms'] or np.nan for r in group], '-o', label='主基线：2.1 μs 启动', color='#b64b52')
        for variant, color, label in [('zero_network_startup','#237c88','仅链路启动设为 0'),
                                       ('half_network_bandwidth','#79759b','仅互联带宽减半')]:
            values = [variants.get((r['id'],variant),{}).get('tpot_s') for r in group]
            ax.plot(x, [v*1000 if v is not None else np.nan for v in values], '-o', label=label, color=color)
        ax.set_title(group[0]['architecture_label'], loc='left')
        ax.set_xticks(x, [f'C{i+1:02d}' for i in x])
        ax.legend(frameon=False, fontsize=10)
        decorate(ax, '实际 TPOT / ms')
    fig.suptitle('网络假设敏感性：分离启动开销与带宽限制', x=.06, ha='left', fontsize=18)
    fig.text(.06,-.025,'零启动是诊断性假设，不代表可实现的硬件；带宽减半也可用于观察“双向总和”口径的影响。',color='#566274')
    fig.tight_layout()
    save(fig, directory, '05_network_sensitivity')


def architecture_plot(rows, directory):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.9))
    archs = list(dict.fromkeys(r['architecture'] for r in rows))
    groups = [[r for r in rows if r['architecture'] == a] for a in archs]
    for index, group in enumerate(groups):
        x = np.arange(len(group)) + (index-.5) * .35
        color = ['#237c88', '#b64b52'][index]
        label = group[0]['architecture_label']
        for ax, key in zip(axes, ('tpot_ms', 'comm_pct')):
            vals = [r[key] if r[key] is not None else 0 for r in group]
            ax.bar(x, vals, .33, label=label, color=color)
            for pos, val, r in zip(x, vals, group):
                if r[key] is None:
                    ax.text(pos, 0, 'N/A', rotation=90, va='bottom', ha='center', fontsize=8)
        labels = [f'C{i+1:02d}' for i in range(len(group))]
    for ax in axes:
        ax.set_xticks(range(len(groups[0])), labels)
        ax.legend(frameon=False)
        ax.margins(y=.2)
    decorate(axes[0], '实际 TPOT / ms')
    decorate(axes[1], '通信算子占实际 TPOT / %')
    fig.suptitle('同一负载与资源：两种 Attention / KV 模板对比', x=.06, ha='left', fontsize=18)
    fig.text(.06, -.025, 'C01–C06：960/970 的 2.8T、10T、100T；C07–C10：10T 的 950、960、970、970+LPU。', color='#566274')
    fig.tight_layout()
    save(fig, directory, '04_architecture_comparison')


def ratio_plot(data, directory):
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.1))
    for ax, architecture, label in zip(axes, ('kimi-k3','deepseek-v41'), ('Kimi K3 Attention/KV','DeepSeek V4.1 Attention/KV')):
        base = [r for r in data['rows'] if r['architecture_id']==architecture and r['variant']=='baseline']
        zero = {(r['tp_a'],r['tp_f']):r for r in data['rows'] if r['architecture_id']==architecture and r['variant']=='zero_network_startup'}
        x = np.arange(len(base))
        for variant, color, name in [('baseline','#b64b52','主基线：2.1 μs 启动'),
                                      ('zero_network_startup','#237c88','诊断：零启动开销')]:
            plotted = base if variant=='baseline' else [zero[r['tp_a'],r['tp_f']] for r in base]
            values = [r['tpot_ms'] if r['tpot_ms'] is not None else np.nan for r in plotted]
            ax.plot(x, values, '-o', label=name, color=color, linewidth=2.2)
            best = data['best_by_architecture'][architecture][variant]
            index = next(i for i,r in enumerate(base) if (r['tp_a'],r['tp_f'])==(best['tp_a'],best['tp_f']))
            ax.scatter(index,best['tpot_ms'],marker='*',s=180,color=color,zorder=5)
            ax.annotate(f'候选最优 {best["ratio"]}\n{best["tpot_ms"]:.3f} ms',
                        (index,best['tpot_ms']),xytext=(0,18 if variant=='baseline' else 26),
                        textcoords='offset points',ha='center',fontsize=10,color=color)
        ax.axhline(1,color='#364b60',linestyle='--',linewidth=1,label='目标 1 ms')
        ax.set_xticks(x,[r['ratio']+('*' if r['idle_cards'] else '') for r in base])
        ax.set_xlabel('970DT : LPU 卡数配比（离散候选）')
        ax.set_title(label,loc='left',pad=15)
        decorate(ax,'实际 TPOT / ms')
        ax.legend(frameon=False,fontsize=9,loc='upper left')
        ax.margins(y=.25)
        ax.set_ylim(bottom=0)
    fig.suptitle('AFD 配比搜索：推荐结果取决于通信启动开销',x=.06,ha='left',fontsize=18)
    fig.text(.06,-.03,'* 4:1 使用 8192 张 970DT + 2048 张 LPU，余 2048 卡空闲；其余候选用满 12288 卡。仅在已检查候选中选优。',color='#566274')
    fig.tight_layout()
    save(fig,directory,'06_afd_ratio_search')


def communication_split_plot(rows, directory):
    fig, axes = plt.subplots(2,2,figsize=(14.2,9.1))
    for col,architecture in enumerate(('kimi-k3','deepseek-v41')):
        group=[r for r in rows if r['architecture']==architecture]
        x=np.arange(len(group))
        for row_index,key,label,color in [(0,'comm_startup_ms','通信启动开销 / ms',COLORS['comm']),
                                           (1,'comm_transfer_ms','通信传输开销 / ms（含带宽争用）',COLORS['transfer'])]:
            ax=axes[row_index,col]
            values=[r[key] or 0 for r in group]
            ax.bar(x,values,width=.65,color=color)
            for i,value in enumerate(values):
                ax.text(i,value+max(values)*.025,f'{value:.3f}',ha='center',fontsize=9)
            ax.set_xticks(x,[f'C{i+1:02d}' for i in x])
            ax.set_ylim(0,max(values)*1.22)
            decorate(ax,label)
            if row_index==0:
                ax.set_title(group[0]['architecture_label'],loc='left',fontsize=14,pad=14)
    fig.suptitle('通信启动与传输分别计量：上下两行使用不同纵轴尺度',x=.06,ha='left',fontsize=18)
    fig.text(.06,-.015,'启动由事件图的网络启动时间逐层累计；传输包含实际共享带宽争用。C01–C10 场景编号与总表一致。',color='#566274')
    fig.tight_layout()
    save(fig,directory,'07_communication_startup_transfer')


def component_lookup(data):
    return {(r['case_id'],str(r['metric']).lower(),r['component']):r for r in data['rows']
            if r['scope'] in ('main','prefill') and r.get('variant','baseline')=='baseline'}


def component_plots(rows,data,directory):
    lookup=component_lookup(data)
    for architecture in ('kimi-k3','deepseek-v41'):
        group=[r for r in rows if r['architecture']==architecture]
        location=directory/architecture
        for inner,stem in [(False,'08_attention_moe_shares'),(True,'09_moe_internal_shares')]:
            fig,axes=plt.subplots(1,2,figsize=(14.4,6))
            for ax,metric,title in zip(axes,('ttft','tpot'),('TTFT：Prefill 首 token','TPOP / TPOT：首步 Decode')):
                x=np.arange(len(group));bottom=np.zeros(len(group))
                if inner:
                    series=[('communication_startup_ms','通信启动','#c94444'),
                            ('communication_transfer_ms','通信传输（含争用）','#df9a35'),
                            ('compute_ms','计算限制服务','#2477a8'),
                            ('memory_ms','访存限制服务','#8bbbd1'),
                            ('other_ms','启动 / 等待 / 其他','#a6adba')]
                else:
                    series=[('attention','Attention侧（含共享稠密预算）','#2477a8'),
                            ('moe','MoE侧','#c94444'),('other','未归类 / 其他','#a6adba')]
                for key,label,color in series:
                    values=[]
                    for r in group:
                        if inner:
                            rec=lookup.get((r['id'],metric,'moe'),{})
                            denom=rec.get('total_ms')
                            values.append(100*(rec.get(key) or 0)/denom if denom else 0)
                        else:
                            rec=lookup.get((r['id'],metric,key),{})
                            values.append(rec.get('window_share_pct') or 0)
                    ax.bar(x,values,.64,bottom=bottom,label=label,color=color)
                    bottom+=np.array(values)
                for i,r in enumerate(group):
                    rec=lookup.get((r['id'],metric,'moe'),{})
                    if not rec.get('window_total_ms'):
                        ax.text(i,3,'N/A',rotation=90,ha='center',color='#68798b',fontsize=9)
                ax.set_xticks(x,[f'C{i+1:02d}' for i in x])
                ax.set_ylim(0,108)
                ax.set_title(title,loc='left')
                decorate(ax,'占 MoE 时间 / %' if inner else '占该阶段总时延 / %')
            handles,labels=axes[0].get_legend_handles_labels()
            fig.legend(handles,labels,loc='lower center',ncol=len(labels),frameon=False,fontsize=10,bbox_to_anchor=(.5,-.015))
            fig.suptitle(group[0]['architecture_label']+' · '+('MoE 内部时间组成' if inner else 'Attention / MoE 时间分工'),x=.06,ha='left',fontsize=17)
            fig.text(.06,-.055,'N/A 表示该阶段不可行或没有完整执行结果；计算和访存按限制资源归因，通信的启动与传输只计一次。',color='#566274')
            fig.tight_layout(rect=(0,.05,1,.95))
            save(fig,location,stem)


def write_report(directory, manifest, rows):
    # Populated from saved evidence; independent assumptions and limitations remain explicit.
    headers = ['架构模板', '场景', '模型', '硬件', 'EP / TP', '目标 ms', '实际 TPOT ms', '通信 ms', '通信占比 %']
    values = [[('Kimi K3' if 'kimi' in r['architecture'] else 'V4.1'), f'C{i%10+1:02d}',
               f'{r["model_T"]:g}T', r['hardware'], f'{r["ep"]} / {r["tp"]}',
               fmt(r['target_ms'], 0), fmt(r['tpot_ms']), fmt(r['comm_ms']), fmt(r['comm_pct'], 2)] for i,r in enumerate(rows)]
    architectures = list(dict.fromkeys(r['architecture'] for r in rows))
    evidence_sections = []
    for architecture in architectures:
        group = [r for r in rows if r['architecture'] == architecture]
        label = group[0]['architecture_label']
        evidence_sections += [f'## {label}：模型扩展', '',
            f'![{label}模型扩展通信对比](figures/{architecture}/01_model_scaling.png)', '',
            *scaling_findings(group), '',
            f'## {label}：延迟与容量', '',
            f'![{label}TPOT分解](figures/{architecture}/02_latency_breakdown.png)', '',
            f'![{label}容量](figures/{architecture}/03_memory_capacity.png)', '']
    sensitivity_section = ['## 网络假设敏感性', '', '![网络敏感性](figures/05_network_sensitivity.png)', '',
        '主场景使用全部 POD 卡数，TP=N/EP，并采用环形 TP 归约。TP 增大也会增加逐轮启动开销，'
        '因此更多设备可能在当前小 batch 部署中预测更慢。这不是该硬件的最优性能，也不是硬件退步的证据。'
        '零启动只用于诊断启动敏感性；带宽减半保持模型/卡数不变。完整结果见 [sensitivity.csv](sensitivity.csv)。', ''] if (directory/'sensitivity.json').exists() else []
    ratio_section = []
    ratio_findings = []
    ratio_path = directory/'ratio-search/summary.json'
    if ratio_path.exists():
        ratios = load(ratio_path)
        ratio_rows = []
        for architecture in architectures:
            label = 'Kimi K3' if architecture=='kimi-k3' else 'DeepSeek V4.1'
            best = ratios['best_by_architecture'][architecture]['baseline']
            zero_best = ratios['best_by_architecture'][architecture]['zero_network_startup']
            ratio_findings.append(f'- {label} 的已检查配比中，主基线最优为 **970:LPU={best["ratio"]}**，'
                f'{best["attention_cards"]:,}+{best["ffn_cards"]:,} 卡，TPOT **{best["tpot_ms"]:.3f} ms**；'
                f'零启动诊断最优变为 {zero_best["ratio"]}、{zero_best["tpot_ms"]:.3f} ms。')
            for target in ('1:1','2:1','4:1',best['ratio']):
                r = next(r for r in ratios['rows'] if r['architecture_id']==architecture and r['variant']=='baseline' and r['ratio']==target)
                ratio_rows.append([label,target+('（候选最优）' if target==best['ratio'] else ''),
                    f'{r["attention_cards"]:,} / {r["ffn_cards"]:,}',f'{r["tp_a"]} / {r["tp_f"]}',
                    str(r['idle_cards']),fmt(r['tpot_ms']),fmt(r['comm_pct'],2)])
        ratio_section = ['## AFD：970DT 与 LPU 配比优化','',
            '在 10T、250B 激活、top-32、2048 专家、EP512、全 POD batch32、1M 上下文下，'
            '以容量可行且实际 TPOT 最小为目标。每种模板评估 9 个主候选和 9 个零启动诊断候选。','',
            *ratio_findings,'',
            table(['模板','970:LPU','970卡 / LPU卡','TP_A / TP_F','空闲卡','TPOT ms','通信占比 %'],ratio_rows),'',
            '![AFD配比](figures/06_afd_ratio_search.png)','',
            '候选为 TP_A=1/2/3/4/6/8/12/16、TP_F=24−TP_A，并补充 TP_A16/TP_F4。'
            '各池 EP=512；2:1 可用满 12K，精确 4:1 使用 10K、空闲 2K。最优仅限已评估集合，不是所有部署策略的全局最优。','',
            '不等 TP 桥接显式计费：A 侧先归约为完整 hidden，每个 F 分片从唯一 A lane 接收；F partial 返回 A 后归约广播。'
            '每方向全 POD MoE payload 为 batch × top-k × hidden_bytes × TP_F，不重复乘 TP_A。'
            'TP_F>TP_A 的额外部分和、两池状态与容量均计入。1:1 与20点主基线已做一致性对照。','',
            '主基线的两个最优候选仍未达到 1 ms。零启动是诊断性条件，不能把其达标结果当成真实 runtime 已达标；'
            '它说明配比选择需结合实测 collective 启动开销。','',
            '[全部36个配比结果](ratio-search/summary.csv) · [配比配置及原始数值](ratio-search/summary.json)','']
    control_rows = []
    for architecture in architectures:
        control_path = directory/'controls'/f'{architecture}-afd-970-ffn-control.json'
        if control_path.exists():
            control = load(control_path)
            group = [r for r in rows if r['architecture']==architecture]
            t_control = control['metrics']['tpot_s'] * 1000
            control_rows.append([group[0]['architecture_label'],fmt(group[8]['tpot_ms']),fmt(t_control),
                                 fmt(group[9]['tpot_ms']),fmt((1-group[9]['tpot_ms']/t_control)*100,3)+'%'])
    if control_rows:
        ratio_section += ['## 分离 LPU 芯片收益与部署划分收益','',
            table(['模板','同构970 TP24 ms','AFD 1:1、F仍970 ms','AFD 1:1、F为LPU ms','仅更换FFN芯片的延迟下降'],control_rows),'',
            '中间列保持 A/F 各6144卡、TP12，只把 FFN 芯片恢复为970DT。'
            '因此最后两列比较隔离了 LPU 本身的收益；同构970到AFD的全部变化还包含 TP 和通信调度变化。'
            '不能将 LPU 的2×算力或10×HBM带宽直接换算成整模型加速倍率。','']
    communication_section = []
    if all('comm_startup_ms' in r for r in rows):
        split_values = [[('Kimi K3' if 'kimi' in r['architecture'] else 'V4.1'),f'C{i%10+1:02d}',
                         fmt(r['comm_ms'],4),fmt(r['comm_startup_ms'],4),fmt(r['comm_transfer_ms'],4),
                         fmt(r['comm_startup_pct'],3)+'%',fmt(r['comm_transfer_pct'],3)+'%'] for i,r in enumerate(rows)]
        communication_section = ['## 通信拆解：启动与传输','',
            '**通信总开销 = 启动开销 + 传输开销**（本次没有额外通信调度/内存等待项）。'
            '启动取实际事件的 startup 时间并按层数累计，涵盖逐轮网络启动/链路时延；传输是其余实际网络执行时间，'
            '包含共享带宽与端点访存争用。该拆解没有使用“主结果减零启动结果”作为计量方法。','',
            table(['模板','场景','通信总 ms','启动 ms','传输 ms','启动/TPOT','传输/TPOT'],split_values),'',
            '![通信启动传输](figures/07_communication_startup_transfer.png)','',
            '四组模型扩展中，总通信占比下降，但**传输时间和传输占 TPOT 的比例均上升**。'
            '降低的是 TP 变小时的环形归约启动开销；因此不能仅看总通信占比就断言模型扩展减少了传输压力。','',
            '[通信聚合拆解 CSV](communication_breakdown.csv) · [逐通信算子与证据 JSON](communication_breakdown.json)','']
        ttft_splits={r['case_id']:r for r in load(directory/'communication_breakdown.json')['totals'] if r['scope']=='prefill'}
        if ttft_splits:
            ttft_split_values=[]
            for i,r in enumerate(rows):
                s=ttft_splits.get(r['id'],{})
                def seconds(key):
                    value=s.get(key)
                    return fmt(value/1000,5) if value is not None else '—'
                ttft_split_values.append([('Kimi K3' if 'kimi' in r['architecture'] else 'V4.1'),f'C{i%10+1:02d}',
                    seconds('communication_total_ms'),seconds('startup_ms'),seconds('transfer_ms'),
                    fmt(s.get('startup_pct_of_window'),3)+'%',fmt(s.get('transfer_pct_of_window'),3)+'%'])
            communication_section += ['TTFT 的通信使用同样的拆法，但下面绝对时间单位为**秒**；百分比分母为 TTFT。','',
                table(['模板','场景','通信总 s','启动 s','传输 s','启动/TTFT','传输/TTFT'],ttft_split_values),'']
    component_section = []
    component_path=directory/'component_breakdown.json'
    if component_path.exists():
        components=load(component_path)
        lookup=component_lookup(components)
        shares=[]
        for i,r in enumerate(rows):
            def share(metric,component):
                return lookup.get((r['id'],metric,component),{}).get('window_share_pct')
            shares.append([('Kimi K3' if 'kimi' in r['architecture'] else 'V4.1'),f'C{i%10+1:02d}',
                           fmt(r['ttft_s']),r.get('prefill_status','未执行'),fmt(share('ttft','attention'),2),
                           fmt(share('ttft','moe'),2),fmt(share('tpot','attention'),2),fmt(share('tpot','moe'),2)])
        component_section=['## TTFT / TPOP：Attention 与 MoE 分解','',
            '**第一层分母是对应阶段总时延；第二层分母是该阶段的 MoE 时间。** '
            'TPOP 在本报告中与 TPOT 同义，都是首步 Decode 间隔。Attention侧包含当前模型的共享稠密预算；'
            'MoE侧统计路由专家及其通信，共享专家等未逐张量分离的D预算仍在A侧。'
            '这是按执行算子归类，没有把未知的共享权重逐张量伪拆成纯Attention。','',
            table(['模板','场景','TTFT / s','Prefill状态','TTFT Attention %','TTFT MoE %','TPOP Attention %','TPOP MoE %'],shares),'',
            'MoE 内部采用可加口径：通信 + 计算限制服务 + 访存限制服务 + 其他开销。'
            '通信再拆成启动与传输，不能把这两项和通信总量重复相加。'
            '计算/访存的0%只表示它不是该算子的限制资源，不表示没有FLOPs或HBM访问。','']
        for architecture in architectures:
            label='Kimi K3' if architecture=='kimi-k3' else 'DeepSeek V4.1'
            component_section += [f'![{label}阶段占比](figures/{architecture}/08_attention_moe_shares.png)','',
                                  f'![{label}MoE内部占比](figures/{architecture}/09_moe_internal_shares.png)','']
        component_section += ['[完整组件分解 CSV](component_breakdown.csv) · [算子映射与分母 JSON](component_breakdown.json)','']
    prefill_completed=sum(r.get('ttft_s') is not None for r in rows)
    prefill_attempted=sum('_prefill' in r for r in rows)
    if prefill_attempted:
        prefill_note=(f'已执行 {prefill_attempted} 个 Prefill 场景，其中 {prefill_completed} 个完成并得到 TTFT。'
            'TTFT 使用显式流式通信暂存：源 activation 和目的 expert buffer 全量保留，每peer只暂存64KiB；'
            'payload、HBM工作量、逻辑启动和依赖保持不变。原整消息暂存策略的准入失败同时保存在 Prefill 原始结果中。'
            '未完成场景的 TTFT 与占比留空。此策略没有缩短1M上下文或扩大HBM。')
    else:
        prefill_note='当前仅有预置1M上下文状态的首步Decode；TTFT尚未执行，暂不可用。'
    passed = sum(r['target_ratio'] is not None and r['target_ratio'] <= 1 for r in rows)
    missing = sum(r['tpot_ms'] is None for r in rows)
    findings = [f'- 20 个结果点中，{passed} 个满足目标 TPOT，{missing} 个容量不可行或指标不可用。'
                '相同目标 TPOT 与实际达到相同 TPOT 是两个条件，本报告不会用目标值替换预测值。']
    for architecture in architectures:
        group = [r for r in rows if r['architecture'] == architecture]
        findings += [f'- {group[0]["architecture_label"]}，960DT 的通信占比为 ' +
                     ' → '.join(fmt(r['comm_pct'],2)+'%' for r in group[:3]) +
                     '；970DT 为 ' + ' → '.join(fmt(r['comm_pct'],2)+'%' for r in group[3:6]) + '。']
        base, lpu = group[8], group[9]
        if base['tpot_ms'] and lpu['tpot_ms']:
            findings += [f'- {group[0]["architecture_label"]}，10T 的 970DT 全池 {base["tpot_ms"]:.3f} ms → '
                         f'AFD+LPU {lpu["tpot_ms"]:.3f} ms，端到端 Decode 速度比 {base["tpot_ms"]/lpu["tpot_ms"]:.2f}×。'
                         '同时发生 TP/池划分变化，不能将此倍数全部归因于 LPU。']
    lines = ['# 10 场景 × 2 种架构：模型扩展与硬件对比仿真', '', '日期：2026-11-13（Asia/Shanghai）', '',
             '**结果性质：参数预算模型的条件性预测，未经实机校准。** 本实验使用当前 supernode-modeling skill 和共享事件调度引擎；'
             '大规模同构/AFD 情况由独立实验适配器构图。原始输入、假设、状态和算子归因均随结果保留。', '',
             '## 主要结果', '', *findings, '', table(headers, values), '',
             '“—”表示指标未定义或容量不可行，不表示零耗时。TPOT 目标是达标约束，不用于替代预测值。'
             '独立 Decode 成功只证明缓存预置后的执行可行；若 Prefill 失败，不能据此声称完整请求可行。', '',
             prefill_note,'',
             'TTFT 和首步 Decode 使用各自的阶段执行图；预测不包含请求排队、冷启动、持续批处理或 P99。','',
             '## 两种架构对比', '', '![架构对比](figures/04_architecture_comparison.png)', '',
             *component_section, *communication_section, *ratio_section, *evidence_sections, *sensitivity_section,
             '## 输入与解释边界', '',
             table(['规模','激活参数','top-k','专家数','EP'],[['2.8T','104B','16','512','128'],['10T','250B','32','2048','512'],['100T','1T','64','8192','2048']]),'',
             table(['设备','POD规模','FP8峰值 PFLOP/s','单卡HBM GB','HBM TB/s','互联 TB/s'],
                   [['950DT','1024','1','144','4','2'],['960DT','4096','2','288','9.6','2.2'],
                    ['970DT','12288','4','288','14.4','4'],['FFN LPU','与970共享12K总额','8','288（假设）','144','4']]),'',
             '- 用户确认：bs=32 为整个 POD 的总并发；AFD总12K。1:1是原始基线，后续增加2:1、4:1及其他配比搜索。',
             '- POD 按 1K=1024、4K=4096、12K=12288；1M 按 1048576 token。容量与带宽使用十进制 GB/TB。',
             '- 保留用户的精确总参数、激活参数、top-k、专家数和 EP。第一组基线是用户列出的 2.8T，非另造 1T。',
             '- 按用户后续指定，分别采用 Kimi K3 与 DeepSeek V4.1 的 Attention/KV 机制模板。'
             '精确 P/A/top-k/专家数是用户的假设扩展，不能称为原始官方 checkpoint 或完整架构复现。',
             '- Kimi K3 参考为 93 层、69 KDA + 24 MLA；MLA 使用 512+64 压缩 KV 优化布局，KDA 使用 recurrent/conv 状态。'
             '官方 K3 的专家数与用户指定值不同，本实验优先保留用户输入。',
             '- DeepSeek V4.1 使用仓库 Flash40 的 CSA2 Attention/KV 参考，保留窗口、索引、共享 KV 的机制；'
             '为保持统一的用户激活参数预算，每步仍执行 40 个预算层，不复现 CED 的 Prefill/Decode 跳层。'
             '不直接套用仓库另一份 64 层 Pro10T/155B 活跃参数实验。模板参数与执行范围见 profiles/deepseek-v41.json。',
             '- Kimi 的投影、AttnRes、embedding 与 gating 未逐张量重建；V4.1 的 Engram 表及 host lookup、mHC 图未建模。'
             '本结果回答所选 Attention/KV 机制下的资源与通信变化，不能替代完整架构部署仿真。',
             '- V4.1 原始 o_groups=8 不支持本表所有 TP 值直接按输出组切分；高 TP 结果采用预算级理论分片，'
             '不能据此认定该并行布局已由真实 runtime 实现。',
             '- 配比搜索中的 FFN TP21/20 等也属于参数预算分片，尚未验证整数张量切分和原生 kernel 支持；'
             '因此1:7/1:5是当前候选集合中的理论最优。',
             '- FP8 权重/计算，FP16 activation/通信/KV；FP8 权重 scale 元数据另计。',
             '- 固定卡数下 TP 随 EP 变化：960 的 TP=32/8/2，970 的 TP=96/24/6。硬件变化组 TP=2/8/24，AFD 每池 TP12、EP512。',
             '- 硬件速率按用户假设；网络带宽暂解释为每卡每方向。全 POD 为逻辑非阻塞 SU；未提供真实交换网络层级或实测 collective 数据。',
             '- 计算效率 0.5、网络效率 0.8；单逻辑路径启动 2.1 μs；每卡 runtime 8 GB、workspace 256 MB。'
             '这些沿用仓库基线的参数均为假设，HBM 带宽按标称值。',
             '- direct A2A、均匀分数路由；小 batch 下的整数路由热点没有建模。TP 使用环形归约。'
             '对称约简保留所建模阶段资源竞争，不能直接外推到超售网络或不同拓扑。',
             '- AFD 两池各 6144 卡，LPU=8 PFLOP/s FP8、144 TB/s HBM、4 TB/s 互联；HBM 容量沿用 288 GB 假设。'
             '这里描述1:1基线；配比搜索的卡数单独列出。两池均计权重/运行时和跨池 dispatch/combine，未施加固定加速倍率。',
             '- Prefill另采用显式流式通信暂存；V4.1以统一保守选择arena及2×触达预算计HBM，'
             '不是每kind的最小工作区或真实kernel访问trace。完整策略与失败解释见 [Prefill方法](prefill-study/methodology.md)。','',
             '## 通信占比的分母与归因', '',
             '主图通信占比 = 通信算子在关键路径上的时间（含其启动、调度/资源争用）÷ 实际 TPOT。'
             '原生 latency_breakdown 中的纯通信服务与资源争用单独保留；不能将资源争用遗漏后把纯服务占比当成通信的全部影响。'
             'comparison.csv 同时提供 comm_pct、comm_service_pct 和 comm_budget_pct（通信时间÷目标 TPOT）。'
             'kernel 内计算/访存按限制资源归因；它们的工作需求可能同时存在，图中时间不重复相加。', '',
             '新增 comm_startup_ms / comm_transfer_ms 分别表示启动与传输时间；对应 *_pct 分母为实际 TPOT。'
             'communication_breakdown 中的 *_pct_of_comm 分母则是通信总时间，两种百分比分开命名。','',
             '## 验证与来源', '',
             '小规模事件图与原生预算引擎逐项对照；另核对两种 Attention/KV 的状态容量、工作量和 V4.1 索引归约依赖。'
             '正式结果检查 20 个场景完整性、用户参数、时延/算子分项加和，以及相同物理场景在两组实验中的数值一致性。'
             '验证记录见 [原生引擎对照](verification-legacy-budget.json)、[Attention机制校验](verification-attention.json)'
             '和 [正式结果审计](audit.json)。通信、组件、配比和Prefill另有独立审计文件，均已通过。', '',
             'Kimi 来源：仓库 [官方配置快照](references/kimi_k3_config.json)及 [来源说明](references/kimi_sources.md)。'
             'V4.1 来源：仓库 [Flash40参考配置](references/v41_flash40_config.json)及 target.py 的 KV/index/attention 公式。'
             '各结果 provenance 保留代码与模板哈希；这些检查是软件一致性校验，不是实机精度校准。', '',
             '## 原始结果与复跑', '',
             table(['场景', '正常报告', '原始数值'], [[r['id'], f'[report.md]({r["report_path"]})', f'[result.json]({r["result_path"]})'] for r in rows]), '',
             '完整关键路径和算子表保存在每场景 latency_breakdown.csv、operator_breakdown.csv；汇总值见 [comparison.csv](comparison.csv)。', '',
             'TTFT 的实际执行与原始准入诊断见 [Prefill汇总](prefill-study/summary.json)；各case对应 report.md 和代表层证据保存在 prefill-study/cases/。','',
             '```powershell', '.venv/Scripts/python.exe scripts/scaling_study_engine.py --prepare --run --output outputs/scaling-10case-20261113',
             '.venv/Scripts/python.exe scripts/scaling_study_engine.py --sensitivity --sensitivity-variant zero_network_startup --output outputs/scaling-10case-20261113',
             '.venv/Scripts/python.exe scripts/scaling_afd_ratios.py --output outputs/scaling-10case-20261113',
             '.venv/Scripts/python.exe scripts/scaling_prefill_study.py --output outputs/scaling-10case-20261113',
             '.venv/Scripts/python.exe scripts/split_scaling_communication.py',
             '.venv/Scripts/python.exe scripts/split_scaling_communication.py --append-prefill',
             '.venv/Scripts/python.exe scripts/split_scaling_components.py --additional-ledgers outputs/scaling-10case-20261113/prefill-study/component-ledgers.json',
             '.venv/Scripts/python.exe scripts/report_scaling_study.py --output outputs/scaling-10case-20261113', '```', '',
             '半带宽敏感性的8个独立配置保存在 sensitivity-half/；20主场景的资源CSV派生利用率收尾使用 scripts/finalize_scaling_study_outputs.py。'
             'AFD、Attention模板和流式Prefill均有侧车参数，复跑须使用相应实验脚本，不能只把原生Config交给普通simulate替代。','',
             '环境、输入与引擎来源见 manifest.json 和各场景配置。图表提供 PNG 与 SVG 两种格式。'
             '每种架构的完整计算图仅保存代表层的压缩执行证据；未生成完整模型逐事件回放。']
    (directory / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    # Simple Markdown rendering for the fixed report vocabulary, with no external dependencies.
    html_lines, in_table, in_list, in_code = [], False, False, False
    import re
    def inline(text):
        text = html.escape(text)
        text = re.sub(r'!\[([^]]*)\]\(([^)]+)\)', r'<img alt="\1" src="\2">', text)
        text = re.sub(r'\[([^]]*)\]\(([^)]+)\)', r'<a href="\2">\1</a>', text)
        text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
        text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
        return text
    for line in lines:
        for line in line.splitlines():
            if line.startswith('```'):
                html_lines.append('</code></pre>' if in_code else '<pre><code>')
                in_code = not in_code
                continue
            if in_code:
                html_lines.append(html.escape(line) + '\n')
                continue
            if in_table and not line.startswith('|'):
                html_lines.append('</tbody></table></div>'); in_table = False
            if in_list and not line.startswith('- '):
                html_lines.append('</ul>'); in_list = False
            if line.startswith('|'):
                if set(line.replace('|','').replace('-','').replace(':','').strip()) == set():
                    continue
                cells = line.strip('|').split('|')
                if not in_table:
                    html_lines.append('<div class="table"><table><thead><tr>' + ''.join('<th>'+inline(c.strip())+'</th>' for c in cells) + '</tr></thead><tbody>')
                    in_table = True
                else:
                    html_lines.append('<tr>' + ''.join('<td>'+inline(c.strip())+'</td>' for c in cells) + '</tr>')
            elif line.startswith('# '):
                html_lines.append('<header><p class="eyebrow">SUPERNODE SIMULATION · 2026.11</p><h1>'+inline(line[2:])+'</h1></header>')
            elif line.startswith('## '):
                html_lines.append('<h2>'+inline(line[3:])+'</h2>')
            elif line.startswith('- '):
                if not in_list: html_lines.append('<ul>'); in_list=True
                html_lines.append('<li>'+inline(line[2:])+'</li>')
            elif line:
                html_lines.append('<p>'+inline(line)+'</p>')
    document = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>10 场景仿真报告</title><style>
    *{box-sizing:border-box}body{margin:0;background:#f5f7fa;color:#233145;font:16px/1.75 "Microsoft YaHei",sans-serif}main{max-width:1320px;margin:auto;padding:36px 42px 80px}header{border-top:6px solid #c94444;padding:25px 0 10px}.eyebrow{letter-spacing:3px;font-size:12px;color:#65778a}h1{font-size:34px;line-height:1.35;max-width:950px}h2{margin-top:48px;border-bottom:1px solid #dbe2ea;padding-bottom:10px;font-size:24px}p,li{max-width:1120px}img{display:block;width:100%;margin:24px 0;background:white}a{color:#137b8a;text-decoration:none}a:hover{text-decoration:underline}.table{overflow:auto}table{width:100%;border-collapse:collapse;font-size:14px;background:white}th{background:#1d364b;color:white;text-align:left;white-space:nowrap}th,td{padding:11px 12px;border-bottom:1px solid #e2e7ed}tr:nth-child(even){background:#edf2f6}td{font-variant-numeric:tabular-nums}pre{padding:22px;background:#1d364b;color:#e6f0f6;overflow:auto;font-size:14px}li{margin:10px 0}@media(max-width:720px){main{padding:18px}h1{font-size:27px}}@media print{body{background:white}main{padding:0}img,table{break-inside:avoid}h2{break-after:avoid}}
    </style><main>''' + '\n'.join(html_lines) + '</main></html>'
    (directory / 'report.html').write_text(document, encoding='utf-8')


def scaling_findings(rows):
    lines = []
    for name, group in [('960DT', rows[:3]), ('970DT', rows[3:6])]:
        pcts = [r['comm_pct'] for r in group]
        if all(v is not None for v in pcts):
            trend = '单调上升' if pcts[0] < pcts[1] < pcts[2] else '未呈单调上升'
            lines.append(f'- {name}：通信占比 {" → ".join(fmt(v, 2)+"%" for v in pcts)}，{trend}；实际 TPOT 为 '+
                         ' → '.join(fmt(r['tpot_ms'])+' ms' for r in group)+'。')
        else:
            lines.append(f'- {name}：存在不可行或未定义场景，不能给出完整通信占比趋势。')
    lines.append('上述趋势同时包含参数量、top-k、EP、TP 与隐藏维度的变化，不能归因为总参数量的单独作用。')
    return lines


if __name__ == '__main__':
    main()
