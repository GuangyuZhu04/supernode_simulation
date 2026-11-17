"""Independent small-case checks of the scaling-study experiment adapter.

Compare the representative-layer execution to the unmodified native complete
budget-model graph. These are numerical consistency checks, not validation
against real Ascend hardware measurements.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from supernode_sim.analysis import compile_graph, execution_metrics
from supernode_sim.config import Config
from supernode_sim.engine import simulate
from supernode_sim.latency_breakdown import window_breakdown
from scripts.scaling_study_graph import attribution_graph, build_layer


def base():
    return {
        'model': {'name': 'independent-small-budget-check', 'total_params': 2e8,
                  'active_params': 65e6, 'layers': 3, 'hidden_dim': 384,
                  'intermediate_dim': 1024, 'num_heads': 6, 'kv_heads': 6,
                  'head_dim': 64, 'attention_kind': 'latent',
                  'kv_latent_dim': 8, 'kv_rope_dim': 4, 'num_experts': 8,
                  'top_k': 2, 'moe_layer_fraction': 1,
                  'expert_parameter_fraction': 0.9},
        'hardware': {'npu_count': 8, 'su_size': 8, 'npus_per_host': 2,
                     'topology': 'full_mesh',
                     'npu': {'hbm_bytes': 2e9, 'hbm_Bps': 4e12,
                             'flops': {'fp8': 1e15, 'fp16': 5e14},
                             'efficiency': 0.5, 'runtime_bytes': 1e7,
                             'workspace_bytes': 1e6},
                     'scale_up': {'bandwidth_Bps': 2e12,
                                  'injection_Bps': 2e12,
                                  'efficiency': 0.8, 'latency_s': 2.1e-6}},
        'workload': {'scenario': 'inference', 'batch_size': 2,
                     'input_length': 8, 'output_length': 2},
        'execution': {'tp': 2, 'ep': 4, 'all_to_all_algorithm': 'direct',
                      'precision': {'weight': 'fp8', 'compute': 'fp8',
                                    'kv': 'fp16', 'activation': 'fp16',
                                    'communication': 'fp16'}},
    }


def close(actual, expected, label):
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-16):
        raise AssertionError(f'{label}: {actual} != {expected}')


def compare_native(name, data):
    config = Config.model_validate(data)
    full_graph = compile_graph(config)
    full_sim = simulate(full_graph)
    if full_sim['status'] != 'feasible':
        raise AssertionError(f'native {name}: {full_sim["diagnostics"]}')
    metrics = execution_metrics(config, full_graph, full_sim)
    rows = []
    for phase, metric in [('prefill', 'ttft'), ('decode', 'tpot')]:
        graph = build_layer(config, phase=phase)
        sim = simulate(graph)
        if sim['status'] != 'feasible':
            raise AssertionError(f'adapter {name}: {sim["diagnostics"]}')
        record = window_breakdown(attribution_graph(graph), sim, 'infer.d0.token1')
        expected = metrics[metric + '_s']
        actual = record['total_s'] * config.model.layers
        close(actual, expected, f'{name}/{metric}')
        expected_parts = metrics['latency_breakdown'][metric]['components_s']
        # In a uniform fabric, endpoint-host attribution may choose another
        # equally long critical path. Compare combined network time and every
        # non-network component, not arbitrary tie-breaking across host IDs.
        network_keys = [key for key in expected_parts if key.startswith('communication_')]
        close(sum(record['components_s'][k] for k in network_keys) * config.model.layers,
              sum(expected_parts[k] for k in network_keys), f'{name}/{metric}/network')
        for key in expected_parts:
            if key not in network_keys:
                close(record['components_s'][key] * config.model.layers,
                      expected_parts[key], f'{name}/{metric}/{key}')
        operators = record['operator_breakdown']['operators']
        close(sum(row['total_s'] for row in operators), record['total_s'],
              f'{name}/{metric}/operator_sum')
        for row in operators:
            close(sum(row['components_s'].values()), row['total_s'],
                  f'{name}/{metric}/{row["operator"]}/internal_sum')
        # Independently count full-POD bytes: local MoE diagonals are free;
        # two TP all-reduces each contain RS + AG over TP - 1 ring rounds.
        tokens = config.workload.batch_size * (config.workload.input_length if phase == 'prefill' else 1)
        vector_bytes = config.model.hidden_dim * 2
        ep, tp = config.execution.ep, config.execution.tp
        expected_moe = 2 * tokens * config.model.top_k * vector_bytes * tp * (1 - 1 / ep)
        expected_tp = 4 * (tp - 1) * tokens * vector_bytes
        close(graph.metadata['study']['transport_payload_bytes_total'],
              expected_moe + expected_tp, f'{name}/{metric}/payload')
        rows.append({'case': name, 'phase': phase, 'native_seconds': expected,
                     'adapter_seconds': actual, 'absolute_difference_seconds': abs(actual - expected),
                     'all_latency_and_operator_checks': 'passed',
                     'full_pod_transport_bytes_per_layer': expected_moe + expected_tp})
    return rows


def check_afd():
    data = base()
    data['execution'].update(ep=2, tp=4)
    config = Config.model_validate(data)
    afd = {'attention_count': 4, 'ffn_count': 4, 'tp': 2, 'ep': 2,
           'lpu_flops': 2e15, 'lpu_hbm_Bps': 4e13, 'lpu_hbm_bytes': 2e9}
    graph = build_layer(config, phase='decode', afd=afd)
    result = simulate(graph)
    if result['status'] != 'feasible':
        raise AssertionError(result['diagnostics'])
    study = graph.metadata['study']
    expected_each_direction = 2 * 2 * 384 * 2 * 2
    close(study['per_semantic_bytes_total']['dispatch'], expected_each_direction,
          'AFD dispatch includes same-EP bridge')
    close(study['per_semantic_bytes_total']['combine'], expected_each_direction,
          'AFD combine includes same-EP bridge')
    for alloc in graph.initial_allocations:
        if alloc.category == 'kv' and int(alloc.pool.split('.')[0][3:]) >= 4:
            raise AssertionError('AFD FFN pool must not own KV')
    close(graph.resources['npu4.compute'].capacity, 1e15, 'AFD effective LPU compute')
    close(graph.resources['npu4.hbm'].capacity, 4e13, 'AFD LPU HBM rate')
    # An unrealistically small capacity must stay infeasible, not become a
    # finite successful token time by averaging memory over the POD.
    too_small = copy.deepcopy(data)
    too_small['hardware']['npu']['hbm_bytes'] = 1e6
    infeasible = simulate(build_layer(Config.model_validate(too_small), phase='decode'))
    if infeasible['status'] != 'infeasible':
        raise AssertionError('Expected resident-capacity rejection')
    return {'case': 'AFD resource_and_bridge_checks', 'status': 'passed',
            'dispatch_bytes_per_layer': expected_each_direction,
            'combine_bytes_per_layer': expected_each_direction,
            'same_ep_cross_pool_transfers_checked': True,
            'resident_capacity_rejection_checked': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'outputs/scaling-10case-20261113/verification-legacy-budget.json')
    args = parser.parse_args()
    variants = [('base', base())]
    full_batch = base()
    full_batch['workload']['batch_size'] = 4
    variants.append(('all_EP_lanes_active', full_batch))
    tp_one = base()
    tp_one['execution'].update(tp=1, ep=8)
    variants.append(('TP1_no_reductions', tp_one))
    all_experts = base()
    all_experts['model'].update(top_k=8, active_params=2e8)
    variants.append(('topk_equals_experts', all_experts))
    bandwidth_limited = base()
    bandwidth_limited['hardware']['scale_up'].update(
        bandwidth_Bps=1e8, injection_Bps=1e8, latency_s=0)
    variants.append(('network_bandwidth_and_contention', bandwidth_limited))
    endpoint_limited = base()
    endpoint_limited['hardware']['npu']['hbm_Bps'] = 1e8
    endpoint_limited['hardware']['scale_up']['latency_s'] = 0
    variants.append(('endpoint_HBM_contention', endpoint_limited))
    results = [item for name, data in variants for item in compare_native(name, data)]
    results.append(check_afd())
    payload = {'status': 'passed', 'scope': 'legacy uniform-attention budget graph and shared communication adapter consistency checks against unmodified native engine; not validation of the new Kimi/V4.1 attention templates or hardware',
               'checks': results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'passed', 'checks': len(results), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
