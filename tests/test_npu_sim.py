"""NPUSim imports use one launch window and exact implemented-operator scope."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from supernode_sim.analysis import compile_graph
from supernode_sim.config import Config
from supernode_sim.core import Event, Graph, Resource
from supernode_sim.engine import simulate
from supernode_sim.npu_sim import analyze_summary, build_calibration, list_targets
from supernode_sim.streaming import run_streaming
from tests.test_kernel_environment import context
from tests.test_operator_trace import scoped_trace
from tests.test_structured import configuration


SCRIPT = Path(__file__).resolve().parents[1] / 'skills/supernode-modeling/scripts/build_npu_sim_calibration.py'


def summary():
    # Official field layout with deliberately overlapping pipelines. These
    # values are synthetic; the clocks are not a hardware measurement.
    return {'kernel_info': {'kernel_total_clocks': '20000000', 'frequency_ghz': '2',
                            'ai_core_active': 8},
            'pipe_utilization': {'pipeline_util_summary': {
                'cube': {'mean': .8, 'min': .7, 'max': .9, 'per_core': [.8] * 8},
                'mte2': {'mean': .9, 'min': .8, 'max': 1, 'per_core': [.9] * 8}}},
            'bandwidth': {'l1_to_l0a': {'synthetic_value': 999}},
            'pipeline_overlap': {'cube_mte2': .7}}


def implementation():
    return {'op_type': 'SyntheticDense', 'kernel_name': 'synthetic_dense', 'tiling_key': '17',
            'binary_sha256': 'b' * 64, 'block_dim': 8,
            'inputs': [{'shape': [2, 16], 'dtype': 'fp16', 'format': 'ND'}],
            'outputs': [{'shape': [2, 32], 'dtype': 'fp16', 'format': 'ND'}]}


@pytest.fixture
def case(tmp_path):
    base = configuration({'workload': {'scenario': 'inference', 'output_length': 4},
                          'hardware': {'calibration_context': context()}})
    parent = next(event for event in compile_graph(base).events
                  if event.metadata.get('kernel') == 'dense' and event.metadata.get('phase') == 'decode')
    trace = scoped_trace(base, parent)
    trace['operations'][0].update(implementation=implementation(), memory_bytes=trace['memory_bytes'],
                                  startup_s=2e-6)
    trace['operations'][1]['memory_bytes'] = 0
    data = base.model_dump()
    data['hardware']['kernel_traces'] = {'dense': [trace]}
    config = Config.model_validate(data)
    target = list_targets(config)[0]
    report_path = tmp_path / 'summary.json'
    report_path.write_text(json.dumps(summary()), encoding='utf-8')
    manifest = dict(mode='ca', correctness='passed', capture_scope='all_active_cores', expected_active_cores=8,
                    hardware_name=config.hardware.name, simulator_version='synthetic-test-1',
                    simulator_revision='synthetic-revision', command='npusim --config synthetic.json',
                    summary_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
                    calibration_context=context(), target=target, evidence_kind='synthetic',
                    source={'status': 'assumed', 'description': 'Synthetic NPUSim fixture; not measured hardware.'})
    return config, report_path, manifest, parent.id


def imported_config(config, built):
    data = config.model_dump()
    for kernel, profiles in built['calibration']['hardware']['kernel_profiles'].items():
        data['hardware']['kernel_profiles'].setdefault(kernel, []).extend(profiles)
    return Config.model_validate(data)


def test_summary_uses_kernel_clock_window_without_core_or_pipeline_accumulation():
    document = summary()
    report = analyze_summary(document)
    assert report['evidence_kind'] == 'unspecified'  # A summary alone does not establish provenance.
    assert report['duration_s'] == pytest.approx(.01)
    assert report['duration_us'] == pytest.approx(10000)
    assert report['ai_core_active'] == 8
    document['kernel_info']['ai_core_active'] = 32
    document['pipe_utilization']['pipeline_util_summary']['cube']['per_core'] *= 4
    assert analyze_summary(document)['duration_s'] == report['duration_s']
    assert report['diagnostics']['bandwidth'] == document['bandwidth']


@pytest.mark.parametrize('field,value', [('kernel_total_clocks', 0), ('frequency_ghz', 'nan'),
                                         ('frequency_ghz', True), ('ai_core_active', 1.5)])
def test_invalid_summary_cannot_supply_timing(field, value):
    document = summary()
    document['kernel_info'][field] = value
    with pytest.raises(ValueError):
        analyze_summary(document)


@pytest.mark.parametrize('change,error', [
    ({'mode': 'pv'}, 'mode'),
    ({'command': 'npusim --pv --config synthetic.json'}, 'PV'),
    ({'correctness': 'failed'}, 'correctness'),
    ({'capture_scope': 'selected_cores'}, 'capture_scope'),
    ({'summary_sha256': '0' * 64}, 'summary_sha256'),
    ({'expected_active_cores': 4}, 'Active core'),
    ({'hardware_name': 'another-chip'}, 'hardware/environment'),
    ({'source': {'status': 'measured', 'description': 'incorrect simulator attribution'}}, 'not hardware'),
])
def test_manifest_refuses_invalid_or_mismatched_evidence(case, change, error):
    config, report_path, manifest, _ = case
    with pytest.raises(ValueError, match=error):
        build_calibration(config, report_path, {**manifest, **change})


def test_import_binds_expanded_operator_keeps_startup_and_preserves_inputs(case):
    config, report_path, manifest, parent_id = case
    original = config.model_dump()
    built = build_calibration(config, report_path, manifest)
    assert config.model_dump() == original
    kernel = manifest['target']['kernel']
    profile = built['calibration']['hardware']['kernel_profiles'][kernel][0]
    assert profile['duration_s'] == pytest.approx(.01)
    assert profile['memory_bytes'] == manifest['target']['memory_bytes']
    assert profile['metadata_match']['implementation'] == implementation()
    assert profile['evidence_kind'] == 'synthetic'
    assert profile['provenance']['summary_sha256'] == manifest['summary_sha256']
    assert built['report']['startup_s'] == manifest['target']['original_startup_s']
    assert built['report']['startup_policy'] == 'preserved_event_startup'
    graph = compile_graph(imported_config(config, built))
    event = next(event for event in graph.events if event.id == manifest['target']['event_id'])
    assert event.metadata['calibration']['service_time_s'] == pytest.approx(.01)
    assert next(event for event in graph.events if event.id == parent_id).metadata['operator_trace_completion']
    wrong = deepcopy(manifest)
    wrong['target']['event_id'] = parent_id
    with pytest.raises(ValueError, match='expanded operator'):
        build_calibration(config, report_path, wrong)
    replaced = build_calibration(config, report_path, {**manifest, 'startup_s': 0})
    assert replaced['report']['startup_s'] == 0


def test_changed_implementation_and_workload_cannot_reuse_same_work_profile(case):
    config, report_path, manifest, _ = case
    built = build_calibration(config, report_path, manifest)
    calibrated = imported_config(config, built)
    for key, value in [('tiling_key', '18'), ('binary_sha256', 'c' * 64),
                       ('outputs', [{'shape': [4, 16], 'dtype': 'fp16', 'format': 'ND'}])]:
        data = calibrated.model_dump()
        data['hardware']['kernel_traces']['dense'][0]['operations'][0]['implementation'][key] = value
        changed = Config.model_validate(data)
        with pytest.raises(ValueError, match='target differs'):
            build_calibration(changed, report_path, manifest)
        graph = compile_graph(changed)
        assert graph.metadata['kernel_calibration']['matched_events'] == 0
        assert graph.metadata['kernel_calibration']['scope_mismatch_reasons']['metadata_mismatch'] > 0
    wrong = deepcopy(manifest)
    wrong['target']['metadata_match']['workload_shape']['context_tokens'] += 1
    with pytest.raises(ValueError, match='target differs'):
        build_calibration(config, report_path, wrong)


def test_profile_rejects_impossible_physical_time_and_ambiguous_reimport(case):
    config, report_path, manifest, _ = case
    built = build_calibration(config, report_path, manifest)
    with pytest.raises(ValueError, match='Ambiguous kernel calibration'):
        build_calibration(imported_config(config, built), report_path, manifest)
    document = summary()
    document['kernel_info'].update(kernel_total_clocks=1, frequency_ghz=1e20)
    report_path.write_text(json.dumps(document), encoding='utf-8')
    manifest['summary_sha256'] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='physical'):
        build_calibration(config, report_path, manifest)


def test_import_retains_hbm_contention_instead_of_using_a_fixed_delay(case):
    config, report_path, manifest, _ = case
    graph = compile_graph(imported_config(config, build_calibration(config, report_path, manifest)))
    event = deepcopy(next(event for event in graph.events if event.id == manifest['target']['event_id']))
    event.deps, event.allocs, event.releases, event.slots = (), (), (), ()
    event.startup_s = 0
    hbm = next(key for key in event.work if key.endswith('.hbm'))
    amount = event.work[hbm]
    resources = {key: deepcopy(graph.resources[key]) for key in event.work}
    resources[hbm] = Resource(hbm, amount / .0075, 'byte/s', 'bandwidth')
    isolated = Graph(resources=resources, events=[event])
    solo = simulate(isolated)
    assert solo['duration_s'] == pytest.approx(.01)
    isolated.events.append(Event('concurrent_hbm_transfer', work={hbm: amount}))
    contended = simulate(isolated)
    row = next(row for row in contended['trace'] if row['id'] == event.id)
    assert row['end_s'] > solo['duration_s']
    assert contended['resource_work'][hbm] == 2 * amount


def test_full_and_streaming_cache_replay_preserve_imported_timing(case):
    config, report_path, manifest, _ = case
    config = imported_config(config, build_calibration(config, report_path, manifest))
    graph = compile_graph(config)
    full = simulate(graph)
    cache = {}
    for _ in range(2):
        streamed = run_streaming(config, cache=cache)
        assert streamed['simulation']['duration_s'] == pytest.approx(full['duration_s'])
        assert streamed['model_metadata']['kernel_calibration'] == graph.metadata['kernel_calibration']
        assert streamed['model_metadata']['fidelity'] == graph.metadata['fidelity']
        assert streamed['model_metadata']['fidelity']['kernel_measurement_matched_events'] == 0
        for key in ('resource_work', 'memory_peaks_bytes', 'memory_final_bytes'):
            for resource in set(streamed['simulation'][key]) | set(full[key]):
                assert streamed['simulation'][key].get(resource, 0) == pytest.approx(full[key].get(resource, 0))
    assert any(chunk['timing_cache_hit'] for chunk in streamed['chunks'])


def test_cli_list_analyze_and_import_are_consistent_and_protect_inputs(case, tmp_path):
    config, report_path, manifest, _ = case
    config_path, manifest_path, output = (tmp_path / name for name in ('config.json', 'manifest.json', 'calibration.json'))
    config_path.write_text(config.model_dump_json(), encoding='utf-8')
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
    def call(*args):
        return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True)
    listed = call(config_path, '--list')
    assert listed.returncode == 0, listed.stderr
    assert json.loads(listed.stdout)['events'][0] == manifest['target']
    analyzed = call('--analyze', '--summary', report_path)
    assert analyzed.returncode == 0, analyzed.stderr
    assert json.loads(analyzed.stdout)['duration_s'] == pytest.approx(.01)
    built = call(config_path, '--summary', report_path, '--manifest', manifest_path, '-o', output)
    assert built.returncode == 0, built.stderr
    assert json.loads(output.read_text()) == build_calibration(config, report_path, manifest)['calibration']
    assert output.with_suffix('.report.json').is_file()
    before = report_path.read_bytes()
    refused = call('--analyze', '--summary', report_path, '-o', report_path)
    assert refused.returncode == 2
    assert report_path.read_bytes() == before
