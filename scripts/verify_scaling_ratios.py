"""Independent input, endpoint-flow and resident-state checks for ratio search."""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.scaling_afd_ratios import CANDIDATES, build_layer


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    directory = ROOT / 'outputs/scaling-10case-20261113'
    manifest = load(directory / 'manifest.json')
    checks = []
    for item in manifest['cases']:
        if not item.get('afd_path'):
            continue
        config = load(directory / item['config_path'])
        profile = load(directory / item['attention_profile_path'])
        ep, batch = config['execution']['ep'], config['workload']['batch_size']
        model = config['model']
        dense = model['total_params'] * (1-model['expert_parameter_fraction'])
        routed = model['total_params'] - dense
        kind = next(iter(profile['counts']))
        for tp_a, tp_f in CANDIDATES:
            graph, a_count, f_count = build_layer(config, profile, kind, tp_a, tp_f)
            assert a_count == ep*tp_a and f_count == ep*tp_f
            assert len(graph.memory_caps) == 12288
            mapping = Counter((e*tp_f+t) % tp_a for e in range(ep) for t in range(tp_f))
            assert len(mapping) == tp_a and len(set(mapping.values())) == 1
            expected_payload = batch * model['top_k'] * model['hidden_dim'] * 2 * tp_f
            for operation in ('dispatch', 'combine'):
                event = next(e for e in graph.events if e.id == operation)
                actual_tx = sum(v for k,v in event.work.items() if k.endswith('.SU.tx'))
                actual_rx = sum(v for k,v in event.work.items() if k.endswith('.SU.rx'))
                assert math.isclose(actual_tx, expected_payload, rel_tol=1e-12)
                assert math.isclose(actual_rx, expected_payload, rel_tol=1e-12)
                assert math.isclose(sum(v for k,v in event.work.items() if k.endswith('.hbm')), 2*expected_payload, rel_tol=1e-12)
                assert all(value > 0 for value in event.work.values())
            initial = defaultdict(lambda: defaultdict(float))
            for allocation in graph.initial_allocations:
                initial[allocation.pool][allocation.category] += allocation.size_bytes
            total_weight = sum(categories['weight'] for categories in initial.values())
            assert math.isclose(total_weight, ep*dense+routed, rel_tol=1e-12)
            assert math.isclose(initial['npu0.hbm']['weight'], dense/tp_a, rel_tol=1e-12)
            assert math.isclose(initial[f'npu{a_count}.hbm']['weight'], routed/(ep*tp_f), rel_tol=1e-12)
            state_categories = set(graph.metadata['ratio_study']['attention_state_bytes_per_owned_rank']['categoriesbytes'])
            for index in range(a_count, a_count+f_count):
                assert not any(initial[f'npu{index}.hbm'].get(category, 0) for category in state_categories)
                assert graph.resources[f'npu{index}.hbm'].capacity == 144e12
                assert graph.resources[f'npu{index}.compute'].capacity == 4e15
            for index in range(a_count+f_count, 12288):
                assert not initial[f'npu{index}.hbm']
            event_ids = [event.id for event in graph.events]
            assert event_ids.index('tp_attention') < event_ids.index('dispatch') if tp_a > 1 else 'tp_attention' not in event_ids
            assert event_ids.index('dispatch') < event_ids.index('experts') < event_ids.index('combine')
            if tp_a > 1:
                assert event_ids.index('combine') < event_ids.index('tp_ffn')
            if tp_f > tp_a:
                assert 'afd_additional_partial_merge' in event_ids
            else:
                assert 'afd_additional_partial_merge' not in event_ids
            checks.append({'architecture': item['architecture_id'], 'tp_a':tp_a, 'tp_f':tp_f,
                           'active_cards':a_count+f_count,'idle_cards':12288-a_count-f_count,
                           'mapping_flow_residency_and_dependency':'passed',
                           'one_layer_each_direction_payload_bytes':expected_payload})
    result_path = directory / 'ratio-search/summary.json'
    if result_path.exists():
        summary = load(result_path)
        for row in summary['rows']:
            record = row['latency_breakdown']
            if record:
                assert math.isclose(record['total_s']*1000, row['tpot_ms'], rel_tol=1e-12)
                assert math.isclose(sum(record['components_s'].values()),record['total_s'], rel_tol=1e-12)
                assert math.isclose(sum(o['total_s'] for o in record['operator_breakdown']['operators']),record['total_s'], rel_tol=1e-12)
                assert max(row['peak_a_GB'],row['peak_f_GB']) <= row['hbm_capacity_GB']
                assert row['status'] == ('feasible' if row['tpot_ms'] <= 1 else 'slo_failed')
        result_checks = len(summary['rows'])
        complete = summary['complete']
    else:
        result_checks, complete = 0, False
    payload = {'status':'passed','scope':'Independent physical budget, explicit lane mapping, endpoint byte conservation, state placement, dependencies and saved result arithmetic; not hardware validation',
               'candidate_graph_checks':checks,'result_rows_checked':result_checks,'complete_search_checked':complete,
               'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('scripts/scaling_afd_ratios.py','scripts/verify_scaling_ratios.py')}}
    path = directory / 'ratio-search/independent-verification.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':'passed','graph_checks':len(checks),'result_rows':result_checks,'complete_search':complete}))


if __name__ == '__main__':
    main()
