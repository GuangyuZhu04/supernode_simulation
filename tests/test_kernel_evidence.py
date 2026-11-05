"""Simulator predictions retain provenance without becoming hardware measurements."""
from copy import deepcopy

import pytest

from supernode_sim.analysis import compile_graph, model_fidelity
from supernode_sim.calibration import (apply_kernel_profiles, calibration_summary,
                                      merge_calibration_summaries)
from supernode_sim.config import Config, KernelProfile
from supernode_sim.engine import simulate
from supernode_sim.streaming import run_streaming
from tests.test_kernel_environment import fixture
from tests.test_structured import configuration


@pytest.mark.parametrize('kind,label', [
    ('measurement', 'partially_kernel_calibrated'),
    ('simulator', 'simulator_profiled'),
    ('synthetic', 'synthetic_profiled'),
    ('unspecified', 'partially_kernel_calibrated'),
])
def test_applied_profile_reports_evidence_and_provenance_without_changing_service(kind, label):
    graph, data = fixture()
    provenance = {'tool': 'example', 'revision': 'test-1', 'artifact_sha256': 'a' * 64,
                  'selection': {'run': 0, 'kernel': ['gemm']}}
    profile = data['hardware']['kernel_profiles']['test_gemm'][0]
    profile.update(evidence_kind=kind, provenance=provenance)
    KernelProfile.model_validate(profile)
    apply_kernel_profiles(graph, data)
    assert simulate(graph)['duration_s'] == pytest.approx(3.1)
    record = graph.events[0].metadata['calibration']
    assert record['evidence_kind'] == kind
    assert record['provenance'] == provenance
    summary = graph.metadata['kernel_calibration']
    assert sum(summary['evidence_kind_counts'].values()) == summary['matched_events'] == 1
    assert summary['evidence_kind_counts'][kind] == 1
    assert summary['by_phase']['decode']['evidence_kind_counts'] == summary['evidence_kind_counts']
    fidelity = model_fidelity(configuration(), summary)
    assert fidelity['hardware_performance'] == label
    assert fidelity['hardware_performance_by_phase']['decode'] == label
    assert fidelity['kernel_timing_evidence'] == summary['evidence_kind_counts']
    assert fidelity['kernel_measurement_matched_events'] == (1 if kind == 'measurement' else 0)


def test_legacy_profiles_and_cached_summaries_remain_unspecified():
    graph, data = fixture()
    profile = KernelProfile.model_validate(data['hardware']['kernel_profiles']['test_gemm'][0])
    assert profile.evidence_kind == 'unspecified'
    assert profile.provenance == {}
    apply_kernel_profiles(graph, data)
    legacy = deepcopy(graph.metadata['kernel_calibration'])
    del legacy['evidence_kind_counts']
    for phase in legacy['by_phase'].values():
        del phase['evidence_kind_counts']
    merged = calibration_summary()
    merge_calibration_summaries(merged, legacy)
    merge_calibration_summaries(merged, legacy)
    assert merged['evidence_kind_counts']['unspecified'] == 2
    assert merged['by_phase']['decode']['evidence_kind_counts']['unspecified'] == 2
    assert model_fidelity(configuration(), legacy)['hardware_performance'] == 'partially_kernel_calibrated'
    assert model_fidelity(configuration(), legacy)['kernel_measurement_matched_events'] == 0


def test_evidence_merge_counts_all_types_and_phases():
    merged = calibration_summary()
    for kind in ('simulator', 'synthetic', 'simulator'):
        graph, data = fixture()
        data['hardware']['kernel_profiles']['test_gemm'][0]['evidence_kind'] = kind
        apply_kernel_profiles(graph, data)
        merge_calibration_summaries(merged, graph.metadata['kernel_calibration'])
    assert merged['evidence_kind_counts'] == {
        'measurement': 0, 'simulator': 2, 'synthetic': 1, 'unspecified': 0}
    assert merged['by_phase']['decode']['evidence_kind_counts'] == merged['evidence_kind_counts']
    assert model_fidelity(configuration(), merged)['hardware_performance'] == 'simulator_and_synthetic_profiled'


@pytest.mark.parametrize('change', [
    {'evidence_kind': 'measured'},
    {'provenance': {'elapsed_s': float('nan')}},
    {'provenance': {'objects': {1, 2}}},
    {'provenance': []},
])
def test_schema_rejects_unknown_evidence_and_non_json_provenance(change):
    _, data = fixture()
    profile = data['hardware']['kernel_profiles']['test_gemm'][0]
    with pytest.raises(ValueError):
        KernelProfile.model_validate({**profile, **change})


def test_simulator_evidence_survives_streaming_cache_replay():
    config = configuration({'workload': {'scenario': 'inference', 'output_length': 4}})
    parent = next(event for event in compile_graph(config).events
                  if event.metadata.get('kernel') == 'dense' and event.metadata.get('phase') == 'decode')
    compute = next(key for key in parent.work if '.compute' in key)
    rank = compute.split('.')[0]
    data = config.model_dump()
    data['hardware']['kernel_profiles'] = {'dense': [dict(
        flops=parent.work[compute], memory_bytes=parent.work[rank + '.hbm'], duration_s=.01,
        phase='decode', dtype=config.execution.precision.compute, hardware_name=config.hardware.name,
        metadata_match={'model_shape_signature': parent.metadata['model_shape_signature']},
        evidence_kind='simulator', provenance={'tool': 'synthetic-simulator-fixture'})]}
    config = Config.model_validate(data)
    full_graph = compile_graph(config)
    expected = simulate(full_graph)
    cache = {}
    for _ in range(2):
        streamed = run_streaming(config, cache=cache)
        assert streamed['simulation']['duration_s'] == pytest.approx(expected['duration_s'])
        assert streamed['model_metadata']['kernel_calibration'] == full_graph.metadata['kernel_calibration']
        assert streamed['model_metadata']['fidelity'] == full_graph.metadata['fidelity']
        assert streamed['model_metadata']['fidelity']['hardware_performance'] == 'simulator_profiled'
        assert streamed['model_metadata']['fidelity']['kernel_measurement_matched_events'] == 0
    assert any(chunk['timing_cache_hit'] for chunk in streamed['chunks'])
