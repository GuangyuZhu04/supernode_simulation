"""Reference bounds preserve units, scope and physical traffic accounting."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from supernode_sim.analysis import compile_graph
from supernode_sim.config import Config, KernelTrace, KernelTraceOperation
from supernode_sim.david_v100 import build_demands, operator_document
from supernode_sim.engine import simulate
from supernode_sim.streaming import run_streaming
from tests.test_kernel_environment import context
from tests.test_operator_trace import scoped_trace
from tests.test_structured import configuration


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'skills/supernode-modeling/scripts/build_david_v100_demands.py'
ASSET = ROOT / 'skills/supernode-modeling/assets/david-v100-workload.synthetic.json'


def workload():
    return json.loads(ASSET.read_text(encoding='utf-8'))


def test_fp32_complete_work_is_1024_cycles_not_the_reported_92():
    report = build_demands(workload())
    cube = report['resource_demands'][0]
    assert cube['service_s'] == pytest.approx(1024 / 1e9)
    assert report['resource_lower_bound_s'] == cube['service_s']
    assert report['model_kind'] == 'analytical_lower_bound'
    assert report['accounting']['padding_ops'] == 0
    assert '§2.2' in cube['source']


def test_explicit_padding_and_core_count_change_service_not_useful_flops():
    data = workload()
    data['cube'].update(executed_ops=2 * 524288, active_aic=2)
    result = build_demands(data)
    document = operator_document(result, kernel='synthetic.mm', trace_id='synthetic')
    op = KernelTraceOperation.model_validate(document['operations'][0])
    assert result['accounting']['padding_ops'] == 524288
    assert op.flops == 524288
    assert result['resource_demands'][0]['service_s'] == pytest.approx(1024e-9)
    assert op.memory_bytes == 49152  # L2 bytes are not added to HBM.


@pytest.mark.parametrize('level,r,w,expected', [
    ('hbm', 100, 0, 1506e9), ('hbm', 0, 100, 1546e9), ('hbm', 100, 100, 1289e9),
    ('l2', 100, 0, 5305e9), ('l2', 0, 100, 5235e9), ('l2', 100, 100, 5445e9),
])
def test_memory_direction_uses_corresponding_total_bandwidth(level, r, w, expected):
    data = workload()
    data['cube'] = None
    data['memory'] = [dict(level=level, access_path='simt', read_bytes=r, write_bytes=w)]
    result = build_demands(data)
    assert result['resource_demands'][0]['service_s'] == pytest.approx((r + w) / expected)
    assert result['accounting']['memory'][0]['effective_Bps'] == expected


def test_mte_and_mixed_paths_need_their_own_evidence_and_share_the_level_resource():
    data = workload()
    data['memory'][0]['access_path'] = 'mte'
    with pytest.raises(ValueError, match='require effective_Bps'):
        build_demands(data)
    data['memory'][0].update(effective_Bps=1e12, source='Synthetic mixed-traffic fixture, not measured')
    one = build_demands(data)['resource_demands'][1]
    data['memory'][0]['access_path'] = 'mixed'
    two = build_demands(data)['resource_demands'][1]
    assert one['resource'] == two['resource'] == 'david_hbm'
    assert one['service_s'] == two['service_s'] == pytest.approx(49152 / 1e12)
    assert one['evidence_kind'] == 'assumed'


@pytest.mark.parametrize('patch', [
    {'device_sku': 'DavidV200'}, {'device_sku': 'Ascend950'},
    {'enabled_ai_cores': 36}, {'enabled_ai_cores': True},
    {'cann_version': 'different'}, {'frequency_hz': None},
    {'frequency_hz': float('inf')}, {'frequency_hz': True},
    {'dependency_floor_s': 1},
])
def test_rejects_unverified_scope_frequency_and_dependency_boundaries(patch):
    with pytest.raises(ValueError):
        build_demands({**workload(), **patch})


def test_rejects_duplicate_levels_underreported_work_and_derived_overflow():
    data = workload()
    data['memory'].append(deepcopy(data['memory'][0]))
    with pytest.raises(ValueError, match='Aggregate'):
        build_demands(data)
    data = workload()
    data['cube']['executed_ops'] = 1
    with pytest.raises(ValueError, match='include all useful'):
        build_demands(data)
    data = workload()
    data['frequency_hz'] = 1e308
    with pytest.raises(ValueError, match='finite'):
        build_demands(data)


def test_dependency_bound_uses_max_and_requires_an_explicit_source():
    data = workload()
    data.update(dependency_floor_s=2e-6, dependency_source='Synthetic dependent chain excluding launch')
    result = build_demands(data)
    assert result['resource_lower_bound_s'] == 2e-6
    assert sum(x['service_s'] for x in result['resource_demands']) + 2e-6 > result['resource_lower_bound_s']


def scoped_config():
    # All unspecified environment and work details are synthetic test input.
    runtime = {**context(), 'device_sku': 'DavidV100', 'cann_version': '9.1.T560'}
    base = configuration({'hardware': {'calibration_context': runtime},
                          'workload': {'scenario': 'inference', 'output_length': 5}})
    parent = next(e for e in compile_graph(base).events
                  if e.metadata.get('kernel') == 'dense' and e.metadata.get('phase') == 'decode')
    trace = scoped_trace(base, parent)
    data = workload()
    data['cube'].update(mode=base.execution.precision.compute,
                        useful_ops=trace['flops'], executed_ops=trace['flops'] * 2)
    data['memory'] = [dict(level='hbm', access_path='simt', read_bytes=trace['memory_bytes'], write_bytes=0),
                      dict(level='l2', access_path='simt', read_bytes=1e9, write_bytes=1e9)]
    result = build_demands(data)
    trace['operations'] = operator_document(result, kernel='synthetic.david', trace_id=trace['id'])['operations']
    trace['calibration_context'] = runtime
    patched = base.model_dump()
    patched['hardware']['kernel_traces'] = {'dense': [trace]}
    return base, Config.model_validate(patched)


@pytest.mark.parametrize('patch', [
    {'calibration_context': {}},
    {'calibration_context': {**context(), 'device_sku': 'Ascend950', 'cann_version': '9.1.T560'}},
    {'calibration_context': {**context(), 'device_sku': 'DavidV100', 'cann_version': 'different'}},
    {'dtype': 'fp32'},
])
def test_generated_operator_cannot_be_bound_to_another_device_runtime_or_dtype(patch):
    _, config = scoped_config()
    trace = config.hardware.kernel_traces['dense'][0].model_dump()
    with pytest.raises(ValueError):
        KernelTrace.model_validate({**trace, **patch})


def test_full_streaming_and_cache_replay_preserve_hbm_and_new_services():
    base, refined = scoped_config()
    original = simulate(compile_graph(base))
    graph = compile_graph(refined)
    full = simulate(graph)
    assert full['duration_s'] > original['duration_s']
    assert full['resource_work']['npu0.hbm'] == original['resource_work']['npu0.hbm']
    assert full['resource_work']['npu0.compute'] == original['resource_work']['npu0.compute']
    assert graph.metadata['fidelity']['hardware_performance'] == 'uncalibrated'
    cache = {}
    for _ in range(2):
        streamed = run_streaming(refined, cache=cache)
        assert streamed['simulation']['duration_s'] == pytest.approx(full['duration_s'])
        assert streamed['model_metadata']['operator_trace'] == graph.metadata['operator_trace']
        for key, value in full['resource_work'].items():
            assert streamed['simulation']['resource_work'].get(key, 0) == pytest.approx(value)
    assert any(chunk['timing_cache_hit'] for chunk in streamed['chunks'])
    changed = refined.model_dump()
    changed['hardware']['kernel_traces']['dense'][0]['operations'][0]['resource_demands'][2]['service_s'] *= 2
    changed_full = simulate(compile_graph(changed))
    changed_stream = run_streaming(changed, cache=cache)
    assert changed_full['duration_s'] > full['duration_s']
    assert changed_stream['simulation']['duration_s'] == pytest.approx(changed_full['duration_s'])


def test_cli_generates_both_reports_without_overwriting_input(tmp_path):
    report, operators = tmp_path / 'demands.json', tmp_path / 'operators.json'
    subprocess.run([sys.executable, str(SCRIPT), str(ASSET), '-o', str(report),
                    '--operators-output', str(operators), '--kernel', 'synthetic.mm',
                    '--trace-id', 'synthetic'], check=True, capture_output=True, text=True)
    assert json.loads(report.read_text())['model_kind'] == 'analytical_lower_bound'
    document = json.loads(operators.read_text())
    KernelTraceOperation.model_validate(document['operations'][0])
    assert document['source']['status'] == 'assumed'
    rejected = subprocess.run([sys.executable, str(SCRIPT), str(report), '-o', str(report)],
                              capture_output=True, text=True)
    assert rejected.returncode != 0
    assert 'different files' in rejected.stderr
