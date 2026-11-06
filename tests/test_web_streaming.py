"""Memory-safe web execution keeps numerical semantics and honest artifact scope."""
import json
from pathlib import Path

import pytest

from supernode_sim.analysis import run
from supernode_sim.host_memory import MIB
from supernode_sim.web_streaming import run_for_web, should_stream, write_streaming_report
from supernode_sim.web_worker import compact_result
from tests.test_structured import configuration


@pytest.mark.parametrize('patch', [
    {'workload': {'scenario': 'inference', 'input_lengths': [8, 16], 'output_lengths': [4, 2]},
     'hardware': {'npu_count': 4, 'su_size': 2}, 'execution': {'tp': 2, 'ep': 2}},
    {'workload': {'scenario': 'inference', 'output_length': 3},
     'hardware': {'npu_count': 4, 'su_size': 4, 'topology': 'clos'}, 'execution': {'tp': 2, 'ep': 2}},
    {'workload': {'scenario': 'inference'}, 'hardware': {'npu': {'hbm_bytes': 100}}},
])
def test_web_streaming_conserves_metrics_memory_traffic_and_artifacts(tmp_path, patch):
    config = configuration(patch)
    expected = run(config)
    actual = run_for_web(config, memory_limit=512 * MIB)
    assert actual['status'] == expected['status']
    assert actual['config'] == expected['config']
    assert actual['config_sha256'] == expected['config_sha256']
    for key in ('duration_s', 'ttft_s', 'tpot_s'):
        if expected['metrics'].get(key) is not None:
            assert actual['metrics'][key] == pytest.approx(expected['metrics'][key], rel=1e-10)
    assert actual['simulation']['memory_peaks_bytes'] == pytest.approx(expected['simulation']['memory_peaks_bytes'])
    if actual['status'] == 'feasible':
        assert actual['estimate']['communication']['transport_payload_bytes'] == pytest.approx(expected['estimate']['communication']['transport_payload_bytes'])
    view = compact_result(actual)
    assert view['execution_summary']['basis'] == 'completed_chunks'
    assert view['trace']['retained'] is False
    assert view['trace']['events'] == []
    assert view['operator_execution']['basis'] == 'not_retained'
    assert view['estimate']['duration_lower_bound_s'] is None
    write_streaming_report(actual, tmp_path)
    assert json.loads((tmp_path / 'result.json').read_text(encoding='utf-8'))['config'] == expected['config']
    assert {'result.json', 'config.resolved.json', 'report.md', 'memory_pools.csv', 'resources.csv',
            'streaming_chunks.csv', 'operator_breakdown.csv', 'latency_breakdown.csv'} <= {p.name for p in tmp_path.iterdir()}
    assert not (tmp_path / 'events.csv').exists()


def test_retention_strategy_depends_on_host_budget_and_feature_support():
    config = configuration({'workload': {'scenario': 'inference', 'output_lengths': [256, 256]},
                            'hardware': {'npu_count': 4, 'su_size': 2}, 'execution': {'tp': 2, 'ep': 2}})
    assert should_stream(config, 'simulate', 128 * MIB)
    assert not should_stream(config, 'simulate', 64 * 1024 * MIB)
    assert not should_stream(config, 'estimate', 128 * MIB)
    config.execution.features.megakernel.enabled = True
    assert not should_stream(config, 'simulate', 128 * MIB)
    config.execution.features.megakernel.enabled = False
    config.execution.dp = 2
    assert not should_stream(config, 'simulate', 128 * MIB)


def test_worker_uses_streaming_and_preserves_failed_slo(tmp_path, monkeypatch):
    from supernode_sim import web_worker, web_streaming

    config = configuration({'workload': {'scenario': 'inference'}, 'targets': {'ttft_s': 1e-30}})
    (tmp_path / 'input.config.json').write_text(config.model_dump_json(), encoding='utf-8')
    monkeypatch.setattr(web_worker, 'apply_worker_memory_limit', lambda limit: None)
    monkeypatch.setattr(web_streaming, 'should_stream', lambda *args: True)
    monkeypatch.setattr(web_worker, 'run', lambda *args, **kwargs: pytest.fail('must not build full graph'))
    assert web_worker.main([str(tmp_path), 'simulate']) == 0
    result = json.loads((tmp_path / 'ui.result.json').read_text(encoding='utf-8'))
    assert result['status'] == 'slo_failed'
    assert result['execution_summary']['complete'] is True
    assert result['metrics']['ttft_s'] > 0


def test_memory_failure_releases_job_and_next_worker_can_run(tmp_path, monkeypatch):
    import subprocess
    import sys
    import time
    from supernode_sim.web import JobManager
    from supernode_sim.config import load_config

    monkeypatch.setenv('SUPERNODE_WEB_MEMORY_MIB', '128')
    original = subprocess.Popen
    first = True

    def start(args, **kwargs):
        nonlocal first
        if first and 'supernode_sim.web_worker' in args:
            first = False
            code = '''
import sys
from supernode_sim import web_worker
def allocate(*args, **kwargs):
    return bytearray(256 * 1024 * 1024)
web_worker.run = allocate
raise SystemExit(web_worker.main(sys.argv[1:]))
'''
            args = [sys.executable, '-c', code, *args[-2:]]
        return original(args, **kwargs)

    monkeypatch.setattr(subprocess, 'Popen', start)
    manager = JobManager(tmp_path)
    config = load_config(Path(__file__).resolve().parents[1] / 'examples/demo_inference.yaml').model_dump()
    try:
        for expected in ('failed', 'completed'):
            job = manager.submit(config, 'estimate')
            deadline = time.monotonic() + 30
            while job['status'] in ('queued', 'running'):
                assert time.monotonic() < deadline
                time.sleep(.05)
                job = manager.get(job['id'])
            assert job['status'] == expected, job
            if expected == 'failed':
                assert '主机内存预算' in job['error']
                assert (manager.directory(job['id']) / 'input.config.json').is_file()
            assert not manager.processes
    finally:
        manager.close()
