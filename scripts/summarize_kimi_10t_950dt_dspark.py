"""Audit and render the complete Kimi 10T / 950DT / DSpark matrix."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
from datetime import datetime, timezone, timedelta
from math import ceil, isclose
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from run_kimi_10t_950dt_dspark import (
    DEFAULT_OUTPUT, REFERENCE, MODEL_SOURCE, read, save, table, sha, code_hashes)
from supernode_sim.config import Config, HardwareSpec
from supernode_sim.components import attention_state, parameter_inventory
from supernode_sim.core import DTYPE_BYTES


def status(result):
    if result['status'] != 'feasible':
        return result['status']
    return 'feasible' if result['metrics']['tpot_s'] <= 0.03 else 'slo_failed'


def peak(result):
    values = {p: v for p, v in result['simulation']['memory_peaks_bytes'].items() if p.endswith('.hbm')}
    pool = max(values, key=values.get, default=None)
    return pool, values.get(pool, 0)


def write_case_report(directory, config, result):
    """Keep an incomplete execution distinct from the complete model design."""
    speculative = result['model_metadata'].get('speculative_decode')
    rounds = speculative['round_count'] if speculative else config.workload.output_length - 1
    planned = config.model.layers * (rounds + (result['phase'] == 'all'))
    completed = sum(c['id'].endswith('.layer_done') for c in result['chunks'])
    m = result['metrics']
    fmt = lambda value: '不可用' if value is None else f'{value:.9g}'
    message = ('本范围完整执行完成。' if result['status'] == 'feasible' else
               '执行因容量检查中止；仅报告实际完成的部分，未完成层不计为已执行。')
    text = (f"# Kimi K3 10T / 950DT — {result['phase']}\n\n"
        f"资源执行状态：{result['status']}。{message}\n\n"
        f"已完成整层调用：{completed} / {planned}（每个目标轮次 {config.model.layers} 层）。\n\n"
        f"TTFT：{fmt(m['ttft_s'])} s；TPOT：{fmt(m['tpot_s'])} s。"
        "30ms目标判定见总表 summary.csv 的 full_status / warm_status。\n\n"
        f"DSpark：{'开启' if speculative else '关闭'}。原生算子、通信和资源准入保持；宏缓存仅复用匹配且可容纳的调度。\n\n"
        "component_timeline.csv 记录已执行组件窗口；失败的最后一个窗口可能只有部分执行。"
        "失败峰值为初始需求或已观察前缀；warm-decode 预置所需状态，排除prompt准备，TTFT为空，不能替代完整请求。\n\n"
        "这是显式假设驱动的资源预测，不是实机或质量验证。逐事件临时张量生命周期未保留。"
        "resources.csv 中 compute 行为FLOP，存储/链路行为byte，busy_s为秒。\n")
    (directory / 'report.md').write_text(text, encoding='utf-8')


def resident_inventory(config):
    """Independent final-state ledger; no execution is skipped by this report."""
    c = config.model_dump()
    model, x, h, w = (c[k] for k in ('model', 'execution', 'hardware', 'workload'))
    p, f = x['precision'], x['features']
    fixed = defaultdict(float)
    layers = defaultdict(float)
    for t in parameter_inventory(model):
        dtype = (p['routed_expert_weight'] if t.ep_shard else None) or p['weight']
        elements = t.local_elements(x['tp'], x['ep'])
        for state, value in [('weight', elements * DTYPE_BYTES[dtype]),
                ('quantization_metadata', ceil(elements / f['weight_scale_block']) * f['scale_bytes']
                 if elements and dtype in ('fp4', 'fp8', 'int8') else 0)]:
            fixed[t.component, state] += value
            layers[t.layer, t.component, state] += value * h['npu_count']
    fixed['System', 'runtime'] += h['npu']['runtime_bytes']
    fixed['System', 'workspace'] += h['npu']['workspace_bytes']
    if f['speculative_decode']['enabled']:
        fixed['DraftModel', 'draft_resident'] += f['speculative_decode']['draft']['resident_bytes_per_rank']
    # Per-lane max differs when B < EP or B is not divisible by EP.
    totals, components = {}, defaultdict(float)
    per_local_batch = {}
    for rank in range(h['npu_count']):
        lane = rank // x['tp']
        owned = len(range(lane, w['batch_size'], x['ep']))
        if owned not in per_local_batch:
            state_ledger, layer_ledger = defaultdict(float), defaultdict(float)
            length = w['input_lengths'][0] + w['output_lengths'][0] - 1
            for layer in range(model['layers']):
                s = attention_state(model, layer, owned, length, tp=x['tp'])
                if s['kind'] == 'recurrent':
                    state_values = {'recurrent_state': s['recurrent_elements'] * DTYPE_BYTES[p['recurrent']],
                        'convolution_state': s['convolution_elements'] * DTYPE_BYTES[p['activation']]}
                else:
                    pages = owned * ceil(length / f['kv_page_tokens'])
                    elements = pages * f['kv_page_tokens'] * s['elements_per_token']
                    state_values = {'kv_cache': elements * DTYPE_BYTES[p['kv']],
                        'kv_scales': ceil(elements / f['kv_scale_block']) * f['scale_bytes']
                            if p['kv'] in ('fp4', 'fp8', 'int8') else 0,
                        'page_table': pages * f['kv_page_entry_bytes']}
                for state, value in state_values.items():
                    state_ledger[s['component'], state] += value
                    layer_ledger[layer, s['component'], state] += value
            per_local_batch[owned] = state_ledger, layer_ledger
        dynamic, by_layer = per_local_batch[owned]
        for key in fixed.keys() | dynamic.keys():
            components[key] += fixed.get(key, 0) + dynamic.get(key, 0)
        for key, value in by_layer.items():
            layers[key] += value
        totals[f'npu{rank}.hbm'] = sum(fixed.values()) + sum(dynamic.values())
    return totals, components, layers


def summarize(directory, partial=False):
    manifest = read(directory / 'manifest.json')
    model_hash = sha(Config.model_validate(read(MODEL_SOURCE)).model.model_dump())
    current_code = code_hashes()
    rows, details, failures, checks = [], {}, [], []
    latency_rows, peak_rows, phase_rows, inventory_rows, layer_rows = [], [], [], [], []
    missing = []
    reference_rows = {r['case']: r for r in read(REFERENCE / 'summary.json')['rows']}
    for item in manifest['cases']:
        dest = directory / 'cases' / item['name']
        if not (dest / 'case.json').exists():
            missing.append(item['name'])
            continue
        index = read(dest / 'case.json')
        config = Config.model_validate(read(directory / item['config']))
        ref = read(REFERENCE / 'configs' / (item['reference_case'] + '.json'))
        assert config.hardware == HardwareSpec.model_validate(ref['hardware']), item['name']
        assert sha(config.model.model_dump()) == model_hash
        assert index['fingerprint'] == sha({'config': config.model_dump(), 'source': current_code})
        assert sha(config.model_dump()) == item['config_sha256']
        if item['tp'] in (8, 16):
            assert config.execution.sequence_parallel
            assert config.execution.tp == item['tp'] and config.execution.ep == item['cards'] // item['tp']
            assert config.execution.dp == config.execution.pp == 1
        planned, component_states, layer_states = resident_inventory(config)
        for (component, state), value in component_states.items():
            inventory_rows.append({'case': item['name'], 'component': component, 'state': state,
                'planned_final_resident_bytes': value, 'basis': 'complete-request planned resident; not a peak'})
        for (layer, component, state), value in layer_states.items():
            layer_rows.append({'case': item['name'], 'layer': layer, 'component': component, 'state': state,
                'planned_final_resident_bytes': value})
        full, warm = read(dest / 'full/result.json'), None
        enabled = item['variant'] == 'dspark'
        feature = next(v for v in full['model_metadata']['feature_plan']['features'] if v['name'] == 'speculative_decode')
        assert feature['selected'] == enabled
        assert bool(full['model_metadata'].get('speculative_decode', {}).get('enabled')) == enabled
        if (dest / 'warm-decode/result.json').exists():
            warm = read(dest / 'warm-decode/result.json')
        pool, maximum = peak(full)
        m = full['metrics']
        row = {'case': item['name'], 'reference_case': item['reference_case'], 'group': item['group'],
            'comparison_case': item.get('comparison_case', item['reference_case']),
            'sp_parent_case': item.get('sp_parent_case'),
            'variant': item['variant'], 'cards': item['cards'], 'tp': item['tp'], 'ep': item['ep'],
            'sp': item['tp'] if item['tp'] > 1 else 1, 'global_batch': item['batch'],
            'context': item['context'], 'input_tokens_per_sequence': item['length'],
            'output_tokens_per_sequence': item['output_length'], 'su_size': item['su_size'],
            'collective': item['algorithm'], 'accepted_length': item['accepted_length'],
            'cost_multiplier': item['cost_multiplier'], 'full_status': status(full),
            'ttft_s': m['ttft_s'], 'tpot_ms': m['tpot_s'] * 1000 if m['tpot_s'] is not None else None,
            'decode_tokens_s': item['batch'] / m['tpot_s'] if m['tpot_s'] else None,
            'full_max_hbm_GB': maximum / 1e9, 'max_hbm_pool': pool,
            'warm_status': status(warm) if warm else None,
            'warm_tpot_ms': warm['metrics']['tpot_s'] * 1000 if warm and warm['metrics']['tpot_s'] else None,
            'warm_hbm_GB': peak(warm)[1] / 1e9 if warm else None,
            'wall_s': index['wall_s'], 'dspark_speedup': None,
            'dsk_reference_tpot_ms': reference_rows[item['reference_case']]['tpot_ms']
                if item['batch'] == ref['workload']['batch_size'] and item['output_length'] == ref['workload']['output_length'] and item['tp'] == ref['execution']['tp'] else None,
            'dsk_reference_status': reference_rows[item['reference_case']]['full_status']
                if item['batch'] == ref['workload']['batch_size'] and item['output_length'] == ref['workload']['output_length'] and item['tp'] == ref['execution']['tp'] else None}
        row['hbm_peak_basis'] = ('initial_resident_requirement' if any(d['code'] == 'initial_memory_exceeded'
            for d in full['simulation']['diagnostics']) else
            'complete_execution_peak' if full['status'] == 'feasible' else 'observed_prefix')
        detail = {'config': item['config'], 'hardware': config.hardware.model_dump(),
            'speculative': config.execution.features.speculative_decode.model_dump(),
            'rounds': full['model_metadata'].get('speculative_decode', {}).get('rounds', []),
            'phases': {}, 'timeline': [], 'latency': {}, 'peaks': {}, 'diagnostics': {}}
        for scope, result in [('full', full), ('warm-decode', warm)]:
            if result is None:
                continue
            write_case_report(dest / scope, config, result)
            if item['tp'] in (8, 16):
                sp_meta = result['model_metadata']['sequence_parallel']
                assert sp_meta['enabled'] and sp_meta['degree'] == item['tp']
                for step in sp_meta.get('steps', []):
                    for lane, valid in step['valid_source_tokens_by_ep'].items():
                        padded = ceil(valid / item['tp'])
                        assert step['padded_local_rows_by_ep'][lane] == padded
                        assert step['gathered_padded_tokens_by_ep'][lane] == padded * item['tp']
                if result['status'] == 'feasible':
                    for phase in result['phase_metrics'].values():
                        traffic = phase['communication']['payload_by_collective_bytes']
                        assert traffic.get('all_gather', 0) > 0
                        assert traffic.get('reduce_scatter', 0) > 0
            metric = result['metrics']
            for name in ('ttft', 'tpot'):
                part = metric['latency_breakdown'].get(name)
                if part:
                    assert isclose(part['total_s'], metric[name + '_s'], rel_tol=1e-8, abs_tol=1e-9)
                    assert isclose(sum(part['components_s'].values()), part['total_s'], rel_tol=1e-8, abs_tol=1e-9)
                latency_rows.append({'case': item['name'], 'scope': scope, 'metric': name,
                    'available': part is not None, 'total_s': part['total_s'] if part else None,
                    **(part['components_s'] if part else {})})
            detail['latency'][scope] = metric['latency_breakdown']
            peak_pool, peak_bytes = peak(result)
            composition = result['simulation']['memory_categories_at_peak_bytes'].get(peak_pool, {})
            assert isclose(sum(composition.values()), peak_bytes, rel_tol=1e-8, abs_tol=0.1), (item['name'], scope)
            detail['peaks'][scope] = {'pool': peak_pool, 'peak_bytes': peak_bytes, 'composition': composition}
            detail['diagnostics'][scope] = result['simulation']['diagnostics']
            peak_rows.append({'case': item['name'], 'scope': scope, 'pool': peak_pool,
                'peak_bytes': peak_bytes, 'execution_complete': result['status'] == 'feasible', **composition})
            for diag in result['simulation']['diagnostics']:
                failures.append({'case': item['name'], 'scope': scope, **diag})
            if result['status'] == 'feasible':
                for pool_name, size in planned.items():
                    assert isclose(result['simulation']['memory_final_bytes'].get(pool_name, 0), size,
                        rel_tol=1e-9, abs_tol=1), (item['name'], scope, pool_name, size)
                rounds = len(detail['rounds']) if item['variant'] == 'dspark' else item['output_length'] - 1
                expected_layers = 93 * (rounds + (scope == 'full'))
                actual_layers = sum(c['id'].endswith('.layer_done') for c in result['chunks'])
                assert actual_layers == expected_layers, (item['name'], scope, actual_layers, expected_layers)
                if scope == 'full':
                    assert metric['output_tokens'] == item['batch'] * item['output_length']
                    assert len(result['model_metadata']['token_completion_events']['0']) == item['output_length']
            else:
                assert metric['ttft_s'] is metric['tpot_s'] is None
            for phase, value in result['phase_metrics'].items():
                traffic = value['communication']
                work = value['resource_work']
                phase_row = {'case': item['name'], 'scope': scope, 'phase': phase,
                    'status': value['status'], 'duration_s': value['duration_s'],
                    'communication_basis': 'complete_phase' if value['status'] == 'feasible' else 'completed_chunks_only_excludes_failed_chunk',
                    'transport_payload_bytes': traffic['transport_payload_bytes'],
                    **{'physical_' + key + '_bytes': amount for key, amount in traffic['physical_bytes_by_domain'].items()},
                    'hbm_access_bytes': sum(v for k, v in work.items() if k.endswith('.hbm')),
                    'compute_flops': sum(v for k, v in work.items() if '.compute' in k),
                    'peak_hbm_bytes': value['peak_hbm_bytes']}
                phase_rows.append(phase_row)
                detail['phases'][scope + '/' + phase] = phase_row
            # Full macro timeline is retained as a linked CSV; show per-layer durations in HTML.
            layer_durations = defaultdict(float)
            for chunk in result['chunks']:
                match = re.search(r'\.s(\d+)\.l(\d+)\.', chunk['id'])
                if match:
                    layer_durations[int(match[1]), int(match[2])] += chunk['duration_s']
            detail['timeline'].extend({'scope': scope, 'step': step, 'layer': layer + 1, 'duration_s': value}
                for (step, layer), value in layer_durations.items())
        details[item['name']] = detail
        rows.append(row)
        checks.append({'case': item['name'], 'hardware_exact_match': True, 'model_exact_match': True,
            'sp_degree_padding_and_collectives_checked': item['tp'] in (8, 16),
            'dspark_feature_selection_matches_variant': True, 'full_execution_completed': full['status'] == 'feasible',
            'code_and_config_hash_match': True, 'latency_additive': True,
            'peak_composition_conserved': True, 'complete_execution_final_residency_matches_shapes': True,
            'all_layers_and_output_tokens_checked': True})
    if missing and not partial:
        raise ValueError(f'{len(missing)} cases are incomplete; use --partial only for an explicitly partial preview')
    lookup = {r['case']: r for r in rows}
    paired = []
    for row in rows:
        if row['variant'] != 'dspark':
            continue
        suffix = '-output33-baseline' if row['output_tokens_per_sequence'] == 33 else '-baseline'
        baseline = lookup.get(row['comparison_case'] + suffix)
        if baseline:
            on_config = read(directory / details[row['case']]['config'])
            off_config = read(directory / details[baseline['case']]['config'])
            on_execution, off_execution = copy.deepcopy(on_config['execution']), copy.deepcopy(off_config['execution'])
            on_execution['features'].pop('speculative_decode')
            off_execution['features'].pop('speculative_decode')
            assert on_execution == off_execution, row['case']
            for key in ('model', 'hardware', 'workload', 'targets'):
                assert on_config[key] == off_config[key], (row['case'], key)
            row['baseline_tpot_ms'] = baseline['tpot_ms']
            row['baseline_ttft_s'] = baseline['ttft_s']
            row['baseline_status'] = baseline['full_status']
            row['baseline_warm_tpot_ms'] = baseline['warm_tpot_ms']
            row['warm_dspark_speedup'] = (baseline['warm_tpot_ms'] / row['warm_tpot_ms']
                if baseline['warm_tpot_ms'] and row['warm_tpot_ms'] else None)
            if row['tpot_ms'] and baseline['tpot_ms']:
                row['dspark_speedup'] = baseline['tpot_ms'] / row['tpot_ms']
            paired.append({k: row.get(k) for k in ['case', 'group', 'full_status', 'baseline_status',
                'tpot_ms', 'baseline_tpot_ms', 'dspark_speedup', 'ttft_s', 'baseline_ttft_s',
                'warm_status', 'warm_tpot_ms', 'baseline_warm_tpot_ms', 'warm_dspark_speedup']})
    main = [r for r in rows if r['variant'] == 'dspark' and r['group'] == 'main']
    sp_pairs = []
    for row in rows:
        if not row['sp_parent_case']:
            continue
        parent = lookup[row['sp_parent_case']]
        old_config = read(directory / details[parent['case']]['config'])
        new_config = read(directory / details[row['case']]['config'])
        for key in ('model', 'hardware', 'workload', 'targets'):
            assert old_config[key] == new_config[key], (row['case'], key)
        old_x, new_x = copy.deepcopy(old_config['execution']), copy.deepcopy(new_config['execution'])
        for key in ('tp', 'ep', 'sequence_parallel'):
            old_x.pop(key)
            new_x.pop(key)
        old_feature = old_x['features'].pop('speculative_decode')
        new_feature = new_x['features'].pop('speculative_decode')
        assert old_x == new_x
        for key in ('enabled', 'method', 'draft_tokens', 'accepted_length', 'accepted_lengths'):
            assert old_feature[key] == new_feature[key]
        pair = {'sp8_case': parent['case'], 'sp16_case': row['case'], 'group': row['group'],
            'variant': row['variant'], 'cards': row['cards'], 'batch': row['global_batch'],
            'context': row['context'], 'output_length': row['output_tokens_per_sequence'],
            'su_size': row['su_size'], 'algorithm': row['collective'], 'sp8_ep': parent['ep'], 'sp16_ep': row['ep'],
            'sp8_status': parent['full_status'], 'sp16_status': row['full_status'],
            'sp8_ttft_s': parent['ttft_s'], 'sp16_ttft_s': row['ttft_s'],
            'sp8_tpot_ms': parent['tpot_ms'], 'sp16_tpot_ms': row['tpot_ms'],
            'sp8_peak_GB': parent['full_max_hbm_GB'], 'sp16_peak_GB': row['full_max_hbm_GB'],
            'sp8_peak_basis': parent['hbm_peak_basis'], 'sp16_peak_basis': row['hbm_peak_basis'],
            'sp16_ttft_speedup': parent['ttft_s'] / row['ttft_s'] if parent['ttft_s'] and row['ttft_s'] else None,
            'sp16_decode_speedup': parent['tpot_ms'] / row['tpot_ms'] if parent['tpot_ms'] and row['tpot_ms'] else None,
            'sp8_warm_tpot_ms': parent['warm_tpot_ms'], 'sp16_warm_tpot_ms': row['warm_tpot_ms'],
            'warm_sp16_decode_speedup': parent['warm_tpot_ms'] / row['warm_tpot_ms'] if parent['warm_tpot_ms'] and row['warm_tpot_ms'] else None}
        sp_pairs.append(pair)
    primary = [r for r in rows if r['variant'] == 'dspark' and r['group'] != 'dspark_sensitivity']
    baseline = [r for r in rows if r['variant'] == 'baseline' and r['group'] == 'main']
    aggregate = {'complete': not missing, 'expected_runs': len(manifest['cases']), 'completed_runs': len(rows),
        'main_cases': len(main), 'primary_cases': len(primary),
        'reference_workload_cases': sum(r['comparison_case'] == r['reference_case'] for r in primary),
        'primary_status': dict(Counter(r['full_status'] for r in primary)),
        'main_status': dict(Counter(r['full_status'] for r in main)),
        'baseline_main_status': dict(Counter(r['full_status'] for r in baseline)),
        'main_tpot_threshold_counts': {str(x): sum(r['tpot_ms'] is not None and r['tpot_ms'] <= x for r in main) for x in (10, 20, 30, 50)},
        'main_speedup_min': min((r['dspark_speedup'] for r in main if r['dspark_speedup']), default=None),
        'main_speedup_max': max((r['dspark_speedup'] for r in main if r['dspark_speedup']), default=None),
        'total_wall_s': sum(r['wall_s'] for r in rows), 'units': 'SI bytes; binary context token counts; TPOT ms / TTFT s'}
    aggregate['sp_main_status'] = {str(sp): dict(Counter(r['full_status'] for r in main if r['sp'] == sp)) for sp in (8, 16)}
    aggregate['sp_completed_pairs'] = len(sp_pairs)
    requested_pairs = [p for p in sp_pairs if p['group'] == 'main' and p['variant'] == 'dspark']
    aggregate['sp16_restored_full_cases'] = [p['sp16_case'] for p in requested_pairs
        if p['sp8_status'] == 'infeasible' and p['sp16_status'] != 'infeasible']
    aggregate['sp16_lost_full_cases'] = [p['sp16_case'] for p in requested_pairs
        if p['sp8_status'] != 'infeasible' and p['sp16_status'] == 'infeasible']
    limits = []
    for batch in manifest.get('batches', (1, 32, 256)):
        for context in ('32K', '128K', '1M', '4M'):
            feasible = [r for r in main if r['global_batch'] == batch and r['context'] == context and r['full_status'] == 'feasible']
            chosen = min(feasible, key=lambda r: (r['cards'], r['tpot_ms'])) if feasible else None
            limits.append({'global_batch': batch, 'context': context, 'case': chosen['case'] if chosen else None,
                'minimum_tested_cards': chosen['cards'] if chosen else None,
                'tp': chosen['tp'] if chosen else None, 'ep': chosen['ep'] if chosen else None,
                'ttft_s': chosen['ttft_s'] if chosen else None, 'tpot_ms': chosen['tpot_ms'] if chosen else None,
                'scope': 'tested primary candidates only; full request and TPOT <=30ms; no TTFT target'})
    aggregate['minimum_cards_by_workload'] = limits
    payload = {'aggregate': aggregate, 'rows': rows, 'details': details, 'failures': failures,
        'sp_comparisons': sp_pairs,
        'updated_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'dspark_assumptions': read(directory / 'dspark-assumptions.json')}
    save(directory / 'summary.json', payload)
    save(directory / 'audit.json', {'passed': not missing, 'audited_cases': len(checks), 'missing': missing,
        'checks': checks, 'sp_pairs_audited': len(sp_pairs),
        'comparison_scope': 'Matched DSpark/baseline and SP8/SP16 pairs; DSK has different model/precision.'})
    save(directory / 'sp_comparison.json', sp_pairs)
    for name, values in [('summary.csv', rows), ('dspark_comparison.csv', paired),
            ('latency_breakdown_all.csv', latency_rows), ('max_card_memory_composition.csv', peak_rows),
            ('phase_metrics.csv', phase_rows), ('component_state_resident.csv', inventory_rows),
            ('layer_state_resident.csv', layer_rows), ('minimum_cards.csv', limits), ('sp_comparison.csv', sp_pairs)]:
        table(directory / name, values)
    render(directory, payload)
    print(aggregate, flush=True)


def render(directory, payload):
    import json
    template = (ROOT / 'scripts/kimi_10t_950dt_dspark_report.html').read_text(encoding='utf-8')
    (directory / 'report.html').write_text(template.replace('__DATA__', json.dumps(payload,
        ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')), encoding='utf-8')
    a = payload['aggregate']
    main = [r for r in payload['rows'] if r['variant'] == 'dspark' and r['group'] == 'main']
    representative = sorted([r for r in main if r['global_batch'] in (16, 32, 64) and r['tp'] in (8, 16) and r['context'] == '128K'], key=lambda r: (r['cards'], r['global_batch'], r['tp']))
    lines = ['# Kimi K3 10T / Ascend 950DT / DSpark', '',
        f"已完成 {a['completed_runs']}/{a['expected_runs']} 组；主矩阵 DSpark 状态：{a['main_status']}。", '',
        '先看 [交互报告](report.html)；全部原始结果、输入和可重跑代码保存在本目录。', '',
        '模型沿用既有 10.171385035872T 文本设计：3328 专家、top32、每token活跃约154.414B、93 层（69 KDA＋24 MLA）、92 MoE＋1 Dense，含 AttnRes。',
        '硬件逐场景严格复用 DSK 参考：64/128/256 卡、144 GB HBM/卡、4 TB/s、FP8 1 PFLOP/s、50% 计算效率、SU 1 TB/s/方向与80%效率。',
        '精度沿用上一轮 Kimi F_kv：权重/计算/激活/通信/KV 为 FP8，KDA recurrent 为 FP32；压缩 MLA，TP>1 时启用 SP。', '',
        '新增 SP16/TP16，对应64/128/256卡的EP为4/8/16；SP8/TP8对应EP为8/16/32。SP共用TP ranks，不额外乘进卡数；TP1旧结果保留。',
        'SP16沿用真实物理拓扑，TP组跨两个8卡host。更换EP会改变每lane请求数；草稿预算按相同公式重算，MLA KV仍按TP复制，KDA heads按TP分片。',
        '[SP8/SP16对照CSV](sp_comparison.csv) · [并行策略对照图](sp8_sp16_comparison.png) · [扩展前258组报告](report.before-sp16.html)', '',
        'DSpark 显式假设：草稿长度7、每轮提交4 token（含 bonus/replacement），假设的1B recurrent草稿；成本、驻留、prompt准备、候选KV、快照和回滚均计入。',
        '输出8的主矩阵实际执行 verify7/commit4 和 verify3/commit3；另有输出33、接受2/8与投机附加成本2倍（草稿/提交/回放）敏感性。未使用固定加速倍数。', '',
        '| 卡数 | TP×EP | B | 上下文 | 状态 | TTFT s | TPOT ms | baseline/DSpark | 峰值/需求 GB |',
        '|---:|---|---:|---|---|---:|---:|---:|---:|']
    fmt = lambda v: f'{v:.3f}' if v is not None else '—'
    for r in representative:
        lines.append(f"| {r['cards']} | {r['tp']}×{r['ep']} | {r['global_batch']} | {r['context']} | {r['full_status']} | {fmt(r['ttft_s'])} | {fmt(r['tpot_ms'])} | {fmt(r['dspark_speedup'])} | {fmt(r['full_max_hbm_GB'])} |")
    lines += ['', 'feasible 表示完整执行且 TPOT≤30ms；slo_failed 表示完整执行但未达30ms；infeasible 表示容量失败。TTFT未设目标。',
        '上述达标状态使用 summary.full_status；原生 result.json.status 仅表示资源执行状态，不单独判定30ms目标。',
        '完整请求失败时不填入有限 TTFT/TPOT；独立 warm decode 仅表示目标与草稿状态预先驻留后的诊断，不代表端到端成功。',
        '失败时内存值可能是初始驻留需求或中止前峰值，详见 hbm_peak_basis。不同组件的独立峰值不能相加；max_card_memory_composition.csv 是同一时刻构成。',
        '专家路由是均匀期望，pairwise按完整轮次展开；小batch下可能比真实稀疏路由的启动成本更保守。',
        '这是未实机校准的算子/资源预测；扩容、top-k、量化与投机接受假设没有质量验证。DSK模型及精度不同，仅复用硬件与矩阵，不将两模型差异称为DSpark收益。', '',
        '复跑：', '```powershell',
        '.\\.venv\\Scripts\\python.exe scripts/run_kimi_10t_950dt_dspark.py --prepare',
        '.\\.venv\\Scripts\\python.exe scripts/extend_kimi_dspark_batches.py',
        '.\\.venv\\Scripts\\python.exe scripts/run_kimi_10t_950dt_dspark.py --run',
        '.\\.venv\\Scripts\\python.exe scripts/extend_kimi_dspark_sp.py',
        '.\\.venv\\Scripts\\python.exe scripts/run_kimi_10t_950dt_dspark.py --run',
        '.\\.venv\\Scripts\\python.exe scripts/summarize_kimi_10t_950dt_dspark.py', '```', '',
        f"共{a['expected_runs']}组：原258组保留，新增134组SP16结果，与已有134组SP8逐一对应（包括开/关、拓扑与投机敏感性）。运行器按代码和配置指纹复用已完成结果；顺序执行，时序缓存32MiB。",
        '逐组 cases/<case>/full 和 warm-decode 保存 result.json、resolved config、组件时间线、时延拆解、峰值构成、资源与模型参数。',
        'component_state_resident.csv / layer_state_resident.csv 是形状推导的完整请求最终驻留账本；失败场景仅为计划需求，不冒充实际已完成状态。',
        f"audit.json 已核验 {a['completed_runs']} 组的硬件/模型一致性、指纹、时延加和、峰值构成以及成功执行的最终状态与层/token计数。"]
    (directory / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (directory / 'README.txt').write_text('\n'.join(lines[:12]) + '\n\n完整说明见 report.md；图表与筛选见 report.html。\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    summarize(args.output, args.partial)
