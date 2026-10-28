"""Real spawned workers preserve search results, diagnostics and ordering."""
import gc
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from supernode_sim.config import Config
from supernode_sim.feedback import feedback
from supernode_sim.parallel import ordered_map, worker_count
from supernode_sim.search import size_su, sweep
from tests.test_feedback import inference_configuration, single_search


@pytest.mark.parametrize('mode', ['simulate', 'estimate'])
def test_spawned_sweep_matches_serial_including_invalid_and_memory_failures(mode):
    config = inference_configuration()
    original = config.model_dump()
    axes = {'hardware.npu.hbm_bytes': [80e9, 1], 'execution.tp': [1, 3]}
    expected = sweep(config, axes, mode)
    actual = sweep(config, axes, mode, workers=2)
    assert actual == expected
    assert config.model_dump() == original
    statuses = [row['result']['status'] for row in actual['candidates']]
    assert statuses[1] == statuses[3] == 'invalid_config'
    if mode == 'simulate':
        assert statuses[0] == 'feasible'
        assert statuses[2] == 'infeasible'


@pytest.mark.parametrize('hbm_bytes', [80e9, 1])
def test_spawned_su_search_preserves_best_order_and_pruning(hbm_bytes):
    config = inference_configuration({
        'hardware': {'npu_count': 2, 'su_size': 2, 'npu': {'hbm_bytes': hbm_bytes}},
        'execution': {'tp': 2},
        'targets': {'candidate_npu_counts': [2], 'candidate_su_sizes': [1, 2], 'total_npus': 2},
    })
    expected = size_su(config)
    actual = size_su(config, workers=2)
    assert actual == expected
    if hbm_bytes == 1:
        assert all(row['result']['pruned'] == 'resident_memory_lower_bound'
                   for result in actual['searches'].values() for row in result['candidates'])


@pytest.mark.parametrize('engine', ['simulate', 'streaming'])
def test_spawned_feedback_preserves_ranking_frontier_and_invalid_candidates(engine):
    config = inference_configuration()
    search = single_search(prefill_lengths=[4, 8], tp_sizes=[1, 2], engine=engine)
    expected = feedback(config, search)
    actual = feedback(config, search, workers=2)
    assert actual == expected
    assert [row['id'] for row in actual['candidates']] == list(range(4))
    assert actual['counts']['feasible'] == actual['counts']['invalid_config'] == 2


@pytest.mark.parametrize('workers', [-1, 1.5, True, '2'])
def test_invalid_host_workers_are_rejected(workers):
    config = inference_configuration()
    with pytest.raises(ValueError, match='workers'):
        sweep(config, {'workload.input_length': [4]}, workers=workers)
    with pytest.raises(ValueError, match='workers'):
        size_su(config, workers=workers)
    with pytest.raises(ValueError, match='workers'):
        feedback(config, single_search(), workers=workers)


def test_automatic_workers_respect_available_cpu_and_candidate_limits(monkeypatch):
    monkeypatch.setattr(os, 'process_cpu_count', lambda: 128, raising=False)
    assert worker_count(0, 20) == 4
    assert worker_count(0, 2) == 2
    monkeypatch.setattr(os, 'process_cpu_count', lambda: None)
    assert worker_count(0, 20) == 1
    assert worker_count(8, 1) == 1


def test_spawned_task_failure_is_propagated_and_caller_gc_state_is_preserved():
    initial = gc.isenabled()
    with pytest.raises(TypeError):
        list(ordered_map(abs, [-1, None, -2], workers=2, count=3))
    assert gc.isenabled() == initial
    # A second invocation after failure is independent and returns input order.
    assert list(ordered_map(abs, [-3, -1, -2], workers=2, count=3)) == [3, 1, 2]


def test_parallel_submission_does_not_eagerly_consume_all_candidates():
    consumed = []

    def inputs():
        for index in range(100):
            consumed.append(index)
            yield -index

    results = ordered_map(abs, inputs(), workers=2, count=100)
    try:
        assert next(results) == 0
        assert len(consumed) <= 4
    finally:
        results.close()


def test_cli_spawn_writes_same_candidate_results_and_rejects_negative_workers(tmp_path):
    config = tmp_path / 'scenario.json'
    axes = tmp_path / 'axes.json'
    config.write_text(json.dumps(inference_configuration().model_dump()), encoding='utf-8')
    axes.write_text(json.dumps({'workload.input_length': [4, 8]}), encoding='utf-8')
    output = tmp_path / 'result'
    command = [sys.executable, '-m', 'supernode_sim', 'sweep', str(config),
               '--axes', str(axes), '--workers', '2', '-o', str(output)]
    completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[1],
                               capture_output=True, text=True, encoding='utf-8', timeout=60)
    assert completed.returncode == 0, completed.stderr
    actual = json.loads((output / 'result.json').read_text(encoding='utf-8'))
    expected = sweep(Config.model_validate(json.loads(config.read_text(encoding='utf-8'))),
                     {'workload.input_length': [4, 8]})
    assert actual['candidates'] == json.loads(json.dumps(expected['candidates']))
    command[command.index('--workers') + 1] = '-1'
    completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[1],
                               capture_output=True, text=True, encoding='utf-8', timeout=60)
    assert completed.returncode == 2
    assert 'workers' in json.loads(completed.stderr)['message']
