"""Check the two attention templates against retained reference mechanisms.

This validates state, work accounting, dependencies and budget conservation.
It does not validate model quality, native TP deployability or hardware latency.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.scaling_kimi_attention import default_profile as kimi_profile, attention_state as kimi_state, attention_work as kimi_work
from scripts.scaling_v41_attention import default_profile as v41_profile, attention_state as v41_state, attention_work as v41_work
from scripts.scaling_study_graph import attribution_graph, build_layer
from scripts.verify_scaling_study import base, close
from supernode_sim.case_studies.v41_fp4.target import index_geometry
from supernode_sim.components import attention_state as native_state
from supernode_sim.config import Config, load_config
from supernode_sim.engine import simulate
from supernode_sim.inference import kv_layout_bytes
from supernode_sim.latency_breakdown import window_breakdown


def check_kimi():
    profile = kimi_profile()
    reference = load_config(ROOT / 'examples/kimi_k3_ascend950_fp8.yaml').model.model_dump()
    kinds = reference['architecture']['attention_by_layer']
    counts = {kind: kinds.count(kind) for kind in set(kinds)}
    if profile['counts'] != counts or sum(counts.values()) != 93:
        raise AssertionError('Kimi layer mix differs from retained source')
    checks = []
    for tp in (1, 2, 6, 12, 24, 32, 96):
        for context in (16, 257, 1048577):
            result = kimi_state(profile, context, tp)
            kda = native_state(reference, kinds.index('KDA'), 1, context, tp=tp)
            expected_recurrent = kda['recurrent_elements'] * 4 * 69
            expected_conv = kda['convolution_elements'] * 2 * 69
            mla = kv_layout_bytes(reference, 'fp16', [context], layers=24, tp=tp)
            expected = expected_recurrent + expected_conv + mla['total_bytes']
            close(result['total_bytes'], expected, 'Kimi full persistent state')
            kda_work = kimi_work(profile, 'KDA', reference, context, tp, 'decode')
            close(kda_work['extra_compute_flops'], 8 * 96 * 128 * 128 / tp, 'KDA work')
            close(kda_work['memory_bytes'], (expected_recurrent + expected_conv) / 69, 'KDA state read')
            close(kda_work['cache_write_bytes'], kda_work['memory_bytes'], 'KDA state update')
            mla_work = kimi_work(profile, 'MLA', reference, context, tp, 'decode')
            close(mla_work['extra_compute_flops'], 2 * 96 * (128 + 64 + 128) * (context + 1) / tp,
                  'MLA score/value work')
            checks.append({'tp': tp, 'context': context, 'state_bytes': expected})
    return {'name': 'Kimi_reference_state_and_work', 'status': 'passed', 'configurations': checks}


def check_v41():
    profile = v41_profile()
    source = json.loads((ROOT / 'supernode_sim/case_studies/v41_fp4/reference_config.json').read_text())
    if profile['layers'] != source['n_layers'] or sum(profile['counts'].values()) != source['n_layers']:
        raise AssertionError('V4.1 Flash40 layer count mismatch')
    for name in ('kv_source_layers', 'index_source_layers'):
        if profile[name] != source[name]:
            raise AssertionError(f'V4.1 {name} mismatch')
    ratios = [source['compress_ratios'][layer] for layer in source['kv_source_layers']]
    if ratios != profile['kv_source_ratios']:
        raise AssertionError('V4.1 owner compression ratios mismatch')
    checks = []
    context = 1048576
    # Both cached vector/index pools are shared at four owner layers. Reuse
    # by eight index layers does not create eight resident copies.
    final = context + 1
    rows = [math.ceil(math.ceil(final / ratio) / 16) * 16 for ratio in ratios]
    expected = 40 * 128 * 512 * 2 + sum(rows) * (512 + 128) * 2
    expected += sum(rows) / 16 * 8 * 2 + sum(2 * ratio * 512 * 4 for ratio in ratios if ratio > 1)
    expected += final * 64 * 4
    for tp in (1, 2, 6, 12, 24, 32, 96):
        close(v41_state(profile, final, tp)['total_bytes'], expected, 'V4.1 shared persistent state')
        for kind in profile['counts']:
            work = v41_work(profile, kind, {}, context, tp, 'decode')
            ratio = 2 if 'ENCODER' in kind else 1
            if 'INDEX' in kind:
                geometry = index_geometry(1, context, ratio, 'CANDIDATE' in kind, False)
                close(work['index_visible_rows'], geometry['pairs'], 'V4.1 reference index pairs')
                close(work['index_reduce_bytes'], geometry['pairs'] * 4, 'V4.1 FP32 score payload')
                close(work['index_reduce_repeats'], geometry['tiles'], 'V4.1 score tile startups')
                close(work['index_flops'], 2 * geometry['pairs'] * 32 / tp * 128, 'V4.1 index dot work')
            else:
                close(work['index_reduce_bytes'], 0, 'V4.1 non-index layers have no score collective')
            pairs = 128 + (min(512, (context + 1) // ratio) if kind != 'SWA' else 0)
            close(work['sparse_attention_pairs'], pairs, 'V4.1 sparse attention pairs')
        checks.append({'tp': tp, 'persistent_state_bytes': expected})
    return {'name': 'V4.1_Flash40_shared_state_and_index_geometry', 'status': 'passed', 'configurations': checks}


def check_graph(profile, module_work):
    data = base()
    data['hardware']['npu']['flops']['fp32'] = 5e13
    data['model']['layers'] = profile['layers']
    data['workload']['input_length'] = 256
    config = Config.model_validate(data)
    rows = []
    linear_flops = 0.0
    for kind, count in profile['counts'].items():
        graph = build_layer(config, attention_profile=profile, attention_kind=kind)
        result = simulate(graph)
        if result['status'] != 'feasible':
            raise AssertionError(result['diagnostics'])
        ledger = window_breakdown(attribution_graph(graph), result, 'infer.d0.token1')
        close(sum(ledger['components_s'].values()), ledger['total_s'], 'New attention latency sum')
        close(sum(x['total_s'] for x in ledger['operator_breakdown']['operators']), ledger['total_s'], 'New attention operator sum')
        nonparam = module_work(profile, kind, config.model.model_dump(), 256, 2, 'decode')
        parameter_flops = 0.0
        for event in graph.events:
            if event.metadata.get('operation') == 'attention_dense':
                current = event.work.get(event.id.split('.')[0] + '.compute', 0)
                if profile['family'] == 'kimi_k3':
                    current -= nonparam['extra_compute_flops']
                parameter_flops += current * event.metadata['symmetry_multiplicity']
            elif event.metadata.get('operation') == 'experts':
                parameter_flops += event.metadata['flops'] * event.metadata['symmetry_multiplicity']
        linear_flops += parameter_flops * count
        if profile['family'] == 'deepseek_v41' and nonparam['index_reduce_bytes']:
            times = {row['id']: row for row in result['trace']}
            reduce_id = 'tp_index_scores.ring_envelope'
            # Scores must be globally reduced before top-k, followed by sparse
            # QK/AV. This checks actual emitted dependencies/times.
            if reduce_id not in times:
                raise AssertionError('Missing index-score collective')
            for event in graph.events:
                if event.metadata.get('kernel') == 'csa2_index_select':
                    if times[event.id]['start_s'] < times[reduce_id]['end_s']:
                        raise AssertionError('Top-k starts before index-score reduction')
            expected_bytes = 2 * (config.execution.tp - 1) * config.workload.batch_size * nonparam['index_reduce_bytes']
            close(graph.metadata['study']['per_semantic_bytes_total']['tp_index_scores'], expected_bytes,
                  'Score tiling does not duplicate total payload')
        rows.append({'kind': kind, 'count': count, 'layer_seconds': ledger['total_s'], 'status': 'passed'})
    close(linear_flops, 2 * config.workload.batch_size * config.model.active_params,
          'User active parameter budget conserved across all executed layers')
    return {'name': profile['family'] + '_graph_and_active_budget', 'status': 'passed',
            'parameter_flops_per_batch_step': linear_flops, 'kinds': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/scaling-10case-20261113/verification-attention.json')
    args = parser.parse_args()
    checks = [check_kimi(), check_v41(), check_graph(kimi_profile(), kimi_work), check_graph(v41_profile(), v41_work)]
    modules = ['scripts/scaling_study_graph.py', 'scripts/scaling_kimi_attention.py', 'scripts/scaling_v41_attention.py']
    payload = {'status': 'passed', 'scope': 'reference-mechanism accounting and graph consistency; not native model reconstruction, model quality, TP deployability or hardware timing validation',
               'source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in modules},
               'checks': checks}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'passed', 'groups': len(checks), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
