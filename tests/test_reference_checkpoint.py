"""Bounded checks of resumable gzip checkpoints and streaming report paths."""
import copy
import csv
import gc
import json

import pytest

from scripts import run_reference_experiment as reference
from supernode_sim.analysis import run
from supernode_sim.config import Config
from supernode_sim.core import Allocation, Event, Graph
from supernode_sim.engine import simulate
from supernode_sim.memory_breakdown import build_memory_breakdown
import supernode_sim.report as report
from test_integration import temporary_directory


@pytest.fixture
def workspace():
    with temporary_directory('checkpoint_') as path:
        yield path


def small_config():
    return {
        'model': {'name': 'synthetic-checkpoint-test', 'total_params': 1000, 'layers': 1,
                  'hidden_dim': 4, 'num_heads': 2, 'kv_heads': 1, 'head_dim': 2},
        'hardware': {'npu_count': 1, 'su_size': 1, 'npus_per_host': 1,
                     'npu': {'hbm_bytes': 1e7, 'runtime_bytes': 0, 'workspace_bytes': 0}},
        'workload': {'scenario': 'training', 'batch_size': 1, 'sequence_length': 2,
                     'gradient_accumulation': 1, 'training_steps': 1},
    }


def checkpoint(directory, fingerprint='known-fingerprint'):
    data = {'trace': [{'id': 'first', 'start_s': 0, 'end_s': 1}],
            'memory_log': [{'sequence': 0, 'time_s': 0, 'action': 'allocate',
                            'allocation_id': 'weight', 'pool': 'npu0.hbm', 'bytes': 10}]}
    manifest = {'fingerprint': fingerprint, 'allocation_count': 1,
                'simulation': {'status': 'feasible', 'duration_s': 1}, 'file_sizes': {}}
    for key, records in data.items():
        path = directory / (key + '.jsonl.gz')
        manifest[key + '_count'] = reference.save_records(path, records)
        manifest['file_sizes'][key] = path.stat().st_size
    (directory / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    return manifest, data


def test_gzip_jsonl_roundtrip_is_repeatable_and_checks_declared_record_count(workspace):
    path = workspace / 'records.jsonl.gz'
    values = [{'name': '模型', 'bytes': 12.5}, {'nested': {'shape': [2, 3]}, 'flag': True}]
    assert reference.save_records(path, iter(values)) == 2
    records = reference.JSONLRecords(path, 2)
    assert len(records) == 2
    assert list(records) == values
    assert list(records) == values  # A new pass reopens the file.
    assert not path.with_suffix(path.suffix + '.tmp').exists()
    with pytest.raises(ValueError, match='record count mismatch'):
        list(reference.JSONLRecords(path, 3))


def test_truncated_compressed_record_file_is_rejected(workspace):
    path = workspace / 'truncated.jsonl.gz'
    reference.save_records(path, [{'id': i} for i in range(4)])
    path.write_bytes(path.read_bytes()[:-8])
    with pytest.raises((EOFError, OSError)):
        list(reference.JSONLRecords(path, 4))


def test_checkpoint_load_rejects_fingerprint_and_file_size_mismatch(workspace):
    manifest, values = checkpoint(workspace)
    loaded = reference.load_checkpoint(workspace, manifest['fingerprint'])
    assert loaded['trace'] == values['trace']
    assert loaded['memory_log'] == values['memory_log']
    with pytest.raises(ValueError, match='configuration or numerical engine differs'):
        reference.load_checkpoint(workspace, 'different-config-or-engine')
    path = workspace / 'trace.jsonl.gz'
    path.write_bytes(path.read_bytes() + b'x')
    with pytest.raises(ValueError, match='file size mismatch'):
        reference.load_checkpoint(workspace, manifest['fingerprint'])


def test_large_checkpoint_load_returns_sized_repeatable_streams(workspace, monkeypatch):
    monkeypatch.setattr(reference, 'STREAM_THRESHOLD', 0)
    manifest, values = checkpoint(workspace)
    loaded = reference.load_checkpoint(workspace, manifest['fingerprint'])
    assert isinstance(loaded['trace'], reference.JSONLRecords)
    assert isinstance(loaded['memory_log'], reference.JSONLRecords)
    assert list(loaded['trace']) == values['trace']
    assert list(loaded['memory_log']) == values['memory_log']


def memory_graph():
    weight = Allocation('weight', 'npu0.hbm', 20, 'weights', 'fp16',
                        metadata={'component': 'Attention', 'state': 'weights'})
    activation = Allocation('activation', 'npu0.hbm', 80, 'activations', 'fp16',
                            metadata={'component': 'Attention', 'state': 'activation'})
    buffer = Allocation('buffer', 'npu0.hbm', 200, 'communication_buffer',
                        metadata={'component': 'System', 'state': 'buffers'})
    return Graph(memory_caps={'npu0.hbm': 1000}, initial_allocations=[weight], events=[
        Event('model', allocs=(activation,), releases=(activation.id,)),
        Event('transfer', deps=('model',), allocs=(buffer,), releases=(buffer.id,))])


def test_streamed_memory_replay_matches_materialized_numbers_and_lifetimes(workspace):
    graph = memory_graph()
    simulation = simulate(graph)
    expected = build_memory_breakdown(graph, simulation, include_ephemeral_details=False)
    path = workspace / 'memory_log.jsonl.gz'
    count = reference.save_records(path, simulation['memory_log'])
    streamed = {**simulation, 'memory_log': reference.JSONLRecords(path, count)}
    actual = build_memory_breakdown(graph, streamed, include_ephemeral_details=False)
    assert actual == expected
    assert actual['pools'][0]['peak_bytes'] == 220
    assert actual['pools'][0]['state_bytes_at_peak']['buffers'] == 200


@pytest.mark.parametrize('damage,match', [('time', 'not chronological'), ('sequence', 'unique and increasing')])
def test_streamed_memory_replay_rejects_ordering_errors(workspace, damage, match):
    graph = memory_graph()
    simulation = simulate(graph)
    records = copy.deepcopy(simulation['memory_log'])
    if damage == 'time':
        records[1]['time_s'] = -1
    else:
        records[1]['sequence'] = -1
    path = workspace / 'bad_memory.jsonl.gz'
    count = reference.save_records(path, records)
    simulation['memory_log'] = reference.JSONLRecords(path, count)
    with pytest.raises(ValueError, match=match):
        build_memory_breakdown(graph, simulation, include_ephemeral_details=False)


def count_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return sum(1 for _ in csv.DictReader(stream))


def test_streamed_compact_export_preserves_summary_and_csv_counts(workspace, monkeypatch):
    monkeypatch.setattr(report, 'AUTO_DETAIL_THRESHOLD', 1)
    result = run(small_config())
    streamed = {**result, 'simulation': dict(result['simulation'])}
    for key in ('trace', 'memory_log'):
        path = workspace / (key + '.jsonl.gz')
        count = reference.save_records(path, result['simulation'][key])
        streamed['simulation'][key] = reference.JSONLRecords(path, count)
    materialized_dir, streamed_dir = workspace / 'materialized', workspace / 'streamed'
    report.write_report(result, materialized_dir)
    report.write_report(streamed, streamed_dir)
    expected = json.loads((materialized_dir / 'result.json').read_text(encoding='utf-8'))
    actual = json.loads((streamed_dir / 'result.json').read_text(encoding='utf-8'))
    assert actual == expected
    for name in ('events.csv', 'tensors.csv', 'component_memory.csv', 'state_memory.csv', 'memory_peaks.csv'):
        assert count_csv(streamed_dir / name) == count_csv(materialized_dir / name)
    timeline = json.loads((streamed_dir / 'timeline.json').read_text(encoding='utf-8'))
    assert len(timeline['traceEvents']) == actual['artifact_export']['events']['exported_rows']
    with pytest.raises(ValueError, match='Streamed checkpoint records require compact export'):
        report.write_report(streamed, workspace / 'full', full_detail=True)


def test_fingerprint_tracks_config_changes_without_recomputing_physics():
    config = Config.model_validate(small_config())
    first = reference.fingerprint(config)
    assert first == reference.fingerprint(Config.model_validate(config.model_dump()))
    changed = config.model_dump()
    changed['workload']['sequence_length'] *= 2
    assert reference.fingerprint(Config.model_validate(changed)) != first


def test_resume_uses_checkpoint_without_calling_simulator_again(workspace, monkeypatch, capsys):
    monkeypatch.setattr(reference, 'STREAM_THRESHOLD', 0)
    monkeypatch.setattr(report, 'AUTO_DETAIL_THRESHOLD', 1)
    config_path, output = workspace / 'case.json', workspace / 'output'
    config_path.write_text(json.dumps(small_config()), encoding='utf-8')
    original_simulate = reference.analysis.simulate_graph
    original_compile = reference.analysis.compile_graph
    physics_calls = []

    def count_physics(graph, **kwargs):
        physics_calls.append(len(graph.events))
        return original_simulate(graph, **kwargs)

    monkeypatch.setattr(reference.analysis, 'simulate_graph', count_physics)
    assert reference.main([str(config_path), '--output', str(output)]) == 0
    first = json.loads((output / 'result.json').read_text(encoding='utf-8'))
    assert len(physics_calls) == 1
    assert len(list((output / 'checkpoint').glob('*/manifest.json'))) == 1
    assert reference.analysis.compile_graph is original_compile
    assert reference.analysis.simulate_graph is count_physics

    def fail_if_recomputed(*args, **kwargs):
        raise AssertionError('Completed checkpoint must bypass numerical simulation')

    monkeypatch.setattr(reference.analysis, 'simulate_graph', fail_if_recomputed)
    was_enabled = gc.isenabled()
    assert reference.main([str(config_path), '--output', str(output)]) == 0
    second = json.loads((output / 'result.json').read_text(encoding='utf-8'))
    assert second == first
    assert reference.analysis.compile_graph is original_compile
    assert reference.analysis.simulate_graph is fail_if_recomputed
    assert gc.isenabled() == was_enabled
    stages = [json.loads(line)['stage'] for line in capsys.readouterr().out.splitlines()]
    assert stages.count('simulation_started') == 1
    assert stages.count('checkpoint_committed') == 1
    assert stages.count('loading_completed_simulation_checkpoint') == 1
    assert stages.count('complete') == 2
