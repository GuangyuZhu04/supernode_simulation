"""Audit mixed precision and compare each request with the preserved FP8 run."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from math import ceil, isclose
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import summarize_kimi_10t_950dt_dspark as common
from run_kimi_10t_950dt_dspark import read, save, table, parameter_inventory
from run_kimi_10t_950dt_dspark_fp4 import OUTPUT, numerical_config


def compare(directory, payload):
    manifest = read(directory / 'manifest.json')
    baseline = Path(manifest['fp8_baseline_directory'])
    old_summary = read(baseline / 'summary.json')
    old_rows = {row['case']: row for row in old_summary['rows']}
    items = {item['name']: item for item in manifest['cases']}
    comparisons, phase_comparisons, weights = [], [], {}
    for row in payload['rows']:
        name = row['case']
        old = old_rows[name]
        item = items[name]
        original, mixed = read(baseline / item['config']), read(directory / item['config'])
        restored = copy.deepcopy(mixed)
        assert restored['execution']['precision']['weight'] == 'fp8'
        assert restored['execution']['precision']['routed_expert_weight'] == 'fp4'
        restored['execution']['precision']['routed_expert_weight'] = None
        assert numerical_config(restored) == numerical_config(original), name
        completed_old = old['full_status'] in ('feasible', 'slo_failed')
        completed_new = row['full_status'] in ('feasible', 'slo_failed')
        transition = ('both_complete' if completed_old and completed_new else
            'restored_full_request' if completed_new else
            'lost_full_request' if completed_old else 'both_infeasible')
        pair = {k: row[k] for k in ('case', 'group', 'variant', 'cards', 'tp', 'ep',
            'global_batch', 'context', 'input_tokens_per_sequence', 'output_tokens_per_sequence',
            'su_size', 'collective', 'accepted_length', 'cost_multiplier')}
        pair.update(transition=transition,
            fp8_status=old['full_status'], mixed_status=row['full_status'],
            fp8_ttft_s=old['ttft_s'], mixed_ttft_s=row['ttft_s'],
            fp8_tpot_ms=old['tpot_ms'], mixed_tpot_ms=row['tpot_ms'],
            fp8_hbm_GB=old['full_max_hbm_GB'], mixed_hbm_GB=row['full_max_hbm_GB'],
            fp8_peak_basis=old['hbm_peak_basis'], mixed_peak_basis=row['hbm_peak_basis'],
            mixed_ttft_speedup=old['ttft_s'] / row['ttft_s'] if completed_old and completed_new else None,
            mixed_decode_speedup=old['tpot_ms'] / row['tpot_ms'] if completed_old and completed_new else None,
            completed_peak_reduction_GB=old['full_max_hbm_GB'] - row['full_max_hbm_GB']
                if completed_old and completed_new else None,
            fp8_warm_status=old['warm_status'], mixed_warm_status=row['warm_status'],
            fp8_warm_tpot_ms=old['warm_tpot_ms'], mixed_warm_tpot_ms=row['warm_tpot_ms'])
        comparisons.append(pair)
        row.update(precision_variant='routed_fp4_other_fp8', fp8_status=old['full_status'],
            fp8_tpot_ms=old['tpot_ms'], fp8_ttft_s=old['ttft_s'],
            precision_transition=transition, mixed_decode_speedup=pair['mixed_decode_speedup'])
        layout = (row['cards'], row['tp'], row['ep'])
        if layout not in weights:
            x, f = mixed['execution'], mixed['execution']['features']
            routed, other, scales = 0, 0, 0
            for tensor in parameter_inventory(mixed['model']):
                count = tensor.local_elements(x['tp'], x['ep'])
                if tensor.ep_shard:
                    routed += count
                else:
                    other += count
                scales += ceil(count / f['weight_scale_block']) * f['scale_bytes'] if count else 0
            weights[layout] = {'cards': row['cards'], 'tp': row['tp'], 'ep': row['ep'],
                'fp8_weight_bytes_per_rank': routed + other,
                'mixed_weight_bytes_per_rank': routed * 0.5 + other,
                'unchanged_weight_scales_bytes_per_rank': scales,
                'weight_saving_bytes_per_rank': routed * 0.5,
                'fp8_weights_with_scales_GB_per_rank': (routed + other + scales) / 1e9,
                'mixed_weights_with_scales_GB_per_rank': (routed * 0.5 + other + scales) / 1e9,
                'basis': 'Static weights + weight scales only, excludes KV/runtime/draft/temporaries'}
        # Compare like scopes; unavailable/partial work is never used for ratios.
        for scope in ('full', 'warm-decode'):
            a_path, b_path = (p / 'cases' / name / scope / 'result.json' for p in (baseline, directory))
            if not a_path.exists() or not b_path.exists():
                continue
            a, b = read(a_path), read(b_path)
            if a['status'] == b['status'] == 'feasible':
                expected_saving = weights[layout]['weight_saving_bytes_per_rank']
                for pool, value in a['simulation']['memory_final_bytes'].items():
                    if pool.endswith('.hbm'):
                        assert isclose(value - b['simulation']['memory_final_bytes'][pool],
                            expected_saving, rel_tol=1e-10, abs_tol=0.1), (name, scope, pool)
            for phase in a['phase_metrics'].keys() & b['phase_metrics'].keys():
                ap, bp = a['phase_metrics'][phase], b['phase_metrics'][phase]
                if ap['status'] != 'feasible' or bp['status'] != 'feasible':
                    continue
                aw, bw = ap['resource_work'], bp['resource_work']
                # Matrix work and transport do not change when only storage changes.
                matrix = lambda work: sum(v for k, v in work.items() if k.endswith('.compute'))
                scalar = lambda work: sum(v for k, v in work.items() if '.compute.fp32' in k)
                hbm = lambda work: sum(v for k, v in work.items() if k.endswith('.hbm'))
                assert isclose(matrix(aw), matrix(bw), rel_tol=1e-9, abs_tol=1), (name, phase)
                ac, bc = ap['communication'], bp['communication']
                assert isclose(ac['transport_payload_bytes'], bc['transport_payload_bytes'],
                               rel_tol=1e-9, abs_tol=1), (name, phase)
                extra_scalar = scalar(bw) - scalar(aw)
                extra_hbm = hbm(bw) - hbm(aw)
                # Each touched routed element adds 2 scalar FLOPs and 1.5 HBM
                # bytes: packed read + FP8 buffer write + FP8 kernel read vs FP8 read.
                assert isclose(extra_hbm, 0.75 * extra_scalar, rel_tol=1e-8, abs_tol=1), (name, scope, phase)
                phase_comparisons.append({'case': name, 'scope': scope, 'phase': phase,
                    'fp8_duration_s': ap['duration_s'], 'mixed_duration_s': bp['duration_s'],
                    'fp8_hbm_access_bytes': hbm(aw), 'mixed_hbm_access_bytes': hbm(bw),
                    'fp8_scalar_flops': scalar(aw), 'mixed_scalar_flops': scalar(bw),
                    'additional_conversion_scalar_flops': extra_scalar,
                    'additional_hbm_bytes': extra_hbm,
                    'conversion_traffic_matches_explicit_buffer_model': True,
                    'matrix_flops_unchanged': True, 'transport_payload_unchanged': True,
                    'work_basis': 'Both matching phases complete; full request may still fail in another phase'})
    main = [p for p in comparisons if p['group'] == 'main' and p['variant'] == 'dspark']
    paired = [p for p in main if p['transition'] == 'both_complete']
    stats = {'paired_runs': len(comparisons), 'all_numeric_config_deltas_checked': True,
        'main_dspark_transitions': dict(Counter(p['transition'] for p in main)),
        'all_run_transitions': dict(Counter(p['transition'] for p in comparisons)),
        'fp8_main_status': old_summary['aggregate']['main_status'],
        'mixed_main_status': payload['aggregate']['main_status'],
        'restored_main_cases': [p['case'] for p in main if p['transition'] == 'restored_full_request'],
        'lost_main_cases': [p['case'] for p in main if p['transition'] == 'lost_full_request'],
        'paired_main_tpot_ratio_min': min((p['mixed_decode_speedup'] for p in paired), default=None),
        'paired_main_tpot_ratio_max': max((p['mixed_decode_speedup'] for p in paired), default=None),
        'sp_main_transitions': {str(tp): dict(Counter(p['transition'] for p in main if p['tp'] == tp)) for tp in (1, 8, 16)},
        'interpretation': 'Ratio = FP8 / mixed; >1 means mixed is faster. Only complete full requests are paired.',
        'matched_complete_phase_count': len(phase_comparisons),
        'completed_scope_final_hbm_saving_matches_routed_weight_shapes': True}
    result = {'aggregate': stats, 'rows': comparisons, 'weights_by_layout': list(weights.values()),
              'precision_assumptions': read(directory / 'precision-assumptions.json')}
    payload['precision_comparison'] = result
    save(directory / 'precision_comparison.json', result)
    table(directory / 'precision_comparison.csv', comparisons)
    table(directory / 'precision_phase_comparison.csv', phase_comparisons)
    table(directory / 'precision_weights_by_layout.csv', sorted(weights.values(), key=lambda w: (w['cards'], w['tp'])))
    save(directory / 'precision-audit.json', {'passed': payload['aggregate']['complete'],
        'scope': 'complete matrix' if payload['aggregate']['complete'] else 'available cases only; partial preview', **stats})
    save(directory / 'summary.json', payload)
    table(directory / 'summary.csv', payload['rows'])
    return result


def render(directory, payload):
    comparison = compare(directory, payload)
    write_report(directory, payload, comparison)


def write_report(directory, payload, comparison):
    # The numerical audit is shared with FP8; the prose and precision view are distinct.
    template = (ROOT / 'scripts/kimi_10t_950dt_dspark_fp4_report.html').read_text(encoding='utf-8')
    (directory / 'report.html').write_text(template.replace('__DATA__', json.dumps(payload,
        ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')), encoding='utf-8')
    a, c, q = payload['aggregate'], comparison['aggregate'], comparison['precision_assumptions']
    fmt = lambda v: '—' if v is None else f'{v:.3f}'
    representatives = sorted((p for p in comparison['rows'] if p['group'] == 'main' and
        p['variant'] == 'dspark' and p['tp'] in (8, 16) and p['context'] == '128K' and
        p['global_batch'] in (16, 32, 64)), key=lambda p: (p['cards'], p['tp'], p['global_batch']))
    lines = ['# Kimi K3 10T / 950DT / DSpark：路由专家 FP4，其余权重 FP8', '',
        f"完成 {a['completed_runs']} / {a['expected_runs']} 组。主矩阵 DSpark 180 组：{a['main_status']}。",
        f"相对 FP8 主矩阵的容量变化：{c['main_dspark_transitions']}。", '',
        '[交互报告](report.html) · [FP8/FP4 逐组对照](precision_comparison.csv) · [精度对照图](precision_comparison.png) · [审计](verification.json)', '',
        f"混合精度下 SP8 主矩阵状态：{a['sp_main_status']['8']}；SP16：{a['sp_main_status']['16']}。",
        f"相对同负载 SP8，SP16 恢复完整执行 {len(a['sp16_restored_full_cases'])} 组、失去完整执行 {len(a['sp16_lost_full_cases'])} 组。SP16 并不保证更快或更省峰值显存，EP减小会增加每lane请求数及TP复制的MLA KV需求。",
        'SP16 相对 SP8 失去完整执行的场景：' + ('、'.join(a['sp16_lost_full_cases']) or '无') + '。详见 [完整并行策略对照](sp_comparison.csv)。', '',
        '本轮沿用原 392 组输入：64/128/256 卡，TP1（SP关闭）与 SP8/TP8、SP16/TP16，batch=1/16/32/64/256，32K/128K/1M/4M；包含 DSpark 开/关、拓扑和投机敏感性。原 FP8 输出完整保留。',
        '唯一数值配置修改：execution.precision.routed_expert_weight 从 null 改为 fp4；execution.precision.weight 保持 fp8。仅 ep_shard 路由专家矩阵量化，shared expert、router、Dense、attention、Embedding/Output 等其余权重仍为 FP8。计算、激活、通信、KV 均为 FP8，KDA recurrent 为 FP32。',
        '量化元数据保持 weight_scale_block=128、scale_bytes=4，KV 元数据不变。这是受控的存储/转换建模假设，不声称复现某个 MXFP4 checkpoint。DSpark 草稿模型仍按原 FP8 资源代理计费。', '',
        f"逻辑目标参数 {q['total_parameters']:,}，其中路由专家 {q['routed_parameters']:,}。不含 scale 的权重本体 {q['fp8_logical_weight_bytes_excluding_scales']/1e12:.6f} TB → {q['mixed_logical_weight_bytes_excluding_scales']/1e12:.6f} TB，减少 {q['logical_weight_saving_bytes']/1e12:.6f} TB。", '',
        '引擎显式执行 FP4→FP8 weight_conversion：读取压缩权重和 scale、写临时 FP8 缓冲、计入标量计算与缓冲容量，后续矩阵 kernel 再读 FP8 缓冲。因此驻留下降不保证 TPOT 下降；没有假设融合解量化或 FP4 算力翻倍。precision_phase_comparison.csv 给出同阶段完整执行的访存/标量工作对照。', '',
        '| 卡数 | SP/TP | EP | B | FP8 状态 | FP4 状态 | FP8 TPOT ms | FP4 TPOT ms | FP8/FP4 | FP8 TTFT s | FP4 TTFT s |',
        '|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|']
    for p in representatives:
        lines.append(f"| {p['cards']} | {p['tp']} | {p['ep']} | {p['global_batch']} | {p['fp8_status']} | {p['mixed_status']} | {fmt(p['fp8_tpot_ms'])} | {fmt(p['mixed_tpot_ms'])} | {fmt(p['mixed_decode_speedup'])} | {fmt(p['fp8_ttft_s'])} | {fmt(p['mixed_ttft_s'])} |")
    lines += ['', '以上代表点均为上下文128K、输出8、DSpark开启。倍数>1表示FP4更快；缺失指标不补零。feasible=完整执行且TPOT≤30ms；slo_failed=完成但超过30ms；infeasible=容量失败。TTFT未设目标。', '',
        '每卡静态权重与量化元数据（GB，十进制；不含 KV、草稿、runtime、activation 和转换临时缓冲）：', '',
        '| 卡数 | TP×EP | FP8 GB/卡 | 混合 FP4 GB/卡 | 权重减少 GB/卡 |',
        '|---:|---|---:|---:|---:|']
    for w in sorted(comparison['weights_by_layout'], key=lambda w: (w['cards'], w['tp'])):
        lines.append(f"| {w['cards']} | {w['tp']}×{w['ep']} | {w['fp8_weights_with_scales_GB_per_rank']:.3f} | {w['mixed_weights_with_scales_GB_per_rank']:.3f} | {w['weight_saving_bytes_per_rank']/1e9:.3f} |")
    lines += ['', '模型仍是既有 synthetic Kimi 10.171T 文本设计：3328专家/top32、93层（69 KDA+24 MLA）、92 MoE+1 Dense、AttnRes。硬件逐项沿用原 DSK：144GB HBM/卡、4TB/s，FP8 1PFLOP/s、50%效率，8卡/host，SU/SO和物理拓扑不变。',
        'SP共用TP ranks，EP=卡数/TP；DP=PP=FSDP=1。batch为每DP副本全局请求数。主矩阵为单逻辑SU，拓扑敏感性保留双SU和direct/pairwise。',
        'DSpark继续使用已认可的显式假设：草稿7、每轮提交目标4（含bonus/replacement）、1B recurrent草稿代理。输出8实际verify7/commit4和verify3/commit3；另有接受2/8、输出33、附加成本2倍敏感性。成本、驻留、prompt准备、快照和回放按原配置计入，未实测接受率和质量。',
        '容量失败保留失败原因；独立 warm decode 排除 prompt 准备，不能替代完整请求。失败内存仅为初始需求或已观察前缀，与完成峰值不可直接比较。组件同一时刻峰值构成见 max_card_memory_composition.csv；component/layer_state_resident.csv 为完整请求计划最终驻留，失败时不代表已执行。', '',
        '复跑（项目根目录）：', '```powershell',
        '.\\.venv\\Scripts\\python.exe scripts/run_kimi_10t_950dt_dspark_fp4.py --prepare --run',
        '.\\.venv\\Scripts\\python.exe scripts/summarize_kimi_10t_950dt_dspark_fp4.py',
        '.\\.venv\\Scripts\\python.exe scripts/render_kimi_fp4_comparison.py', '```',
        '冻结代码单例重放：`python reproduce.py --case n128-tp16-ep8-b32-128K`；写入 replayed，不覆盖原结果。',
        '逐组 cases/<case>/full 与失败后的 warm-decode 保存结果JSON、resolved config、组件时间线、可加和时延拆解、峰值构成、资源、参数和状态表。逐事件临时张量生命周期未保留。',
        '全部结果为未校准的架构/资源预测；当前显式转换路径不等于设备上的融合FP4 kernel，绝对时延和量化质量未实机验证。']
    (directory / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (directory / 'README.txt').write_text('\n'.join(lines[:13]) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--partial', action='store_true', help='Explicitly label an incomplete preview')
    args = parser.parse_args()
    common.render = render
    common.summarize(args.output, partial=args.partial)
