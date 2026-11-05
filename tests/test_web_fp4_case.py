"""Check the promoted case against frozen results from the original FP4 study."""
import json
from pathlib import Path

import pytest

from supernode_sim.case_studies.v41_fp4 import CaseConfig, simulate_case
from supernode_sim.web import JobManager
from test_web import client, finish


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = json.loads((ROOT / 'tests/fixtures/v41_fp4_512_reference.json').read_text(encoding='utf-8'))


@pytest.mark.parametrize('reference', REFERENCE['cases'], ids=['950dt', '950pr'])
def test_original_fp4_checkpoint_equivalence(reference):
    result = simulate_case(reference['config'])
    row = reference['row']
    assert result['status'] == row['joint_status']
    assert result['metrics']['ttft_s'] == pytest.approx(row['ttft_ms'] / 1000, rel=1e-12)
    assert result['metrics']['tpot_s'] == pytest.approx(row['tpot_ms'] / 1000, rel=1e-12)
    assert result['metrics']['decode_tokens_s'] == pytest.approx(row['decode_tokens_s'], rel=1e-12)
    for phase, original in reference['phases'].items():
        actual = result['phases'][phase]
        assert actual['status'] == original['status']
        assert actual['native_completed_events'] == original['native_completed_events']
        assert actual['memory'] == original['memory']
        assert actual['capacity_check'] == original['capacity_check']
        for field in ('duration_s', 'seed_s', 'breakdown_s', 'physical_work', 'component_latency_s'):
            assert actual[field] == pytest.approx(original[field], rel=1e-12), (phase, field)
        assert len(actual['stages']) == actual['native_completed_events']


@pytest.mark.parametrize('changes', [
    {'cards': 1024}, {'ep': 64}, {'tp': 8}, {'tp': 2, 'sp': 4},
    {'accepted': 6, 'block': 4}, {'context_tokens': 10**10}, {'decode_batch': 100_000},
    {'tp': True}, {'tp': '4'}, {'prefill_batch': 2}, {'chip': 'other'},
    {'module_path': '/tmp/untrusted.py'},
])
def test_case_rejects_unsupported_or_unbounded_inputs(client, changes):
    config = CaseConfig().model_dump() | changes
    client.json('POST', '/api/validate', {'mode': 'pod_dflash', 'config': config}, status=400)
    client.json('POST', '/api/jobs', {'mode': 'pod_dflash', 'config': config}, status=400)
    assert client.json('GET', '/api/jobs')['jobs'] == []


def test_catalog_worker_snapshots_artifacts_and_restart(client):
    catalog = client.json('GET', '/api/case-presets')
    assert [p['id'] for p in catalog['presets']] == ['950dt', '950pr']
    for preset in catalog['presets']:
        example = ROOT / f"examples/deepseek_v41_fp4_512_ep128_{preset['id']}.json"
        assert json.loads(example.read_text()) == preset['config']
    config = catalog['presets'][0]['config']
    validated = client.json('POST', '/api/validate', {'config': config, 'mode': 'pod_dflash'})
    assert validated['summary']['parallel_product'] == 512
    job = client.json('POST', '/api/jobs', {'config': config, 'mode': 'pod_dflash'}, status=202)
    config['chip'] = '950pr'  # Subsequent edits must not change the saved input.
    job = finish(client, job)
    assert job['result_status'] == 'feasible'
    assert client.json('GET', job['config_url'])['config']['chip'] == '950dt'
    result = client.json('GET', f"/api/jobs/{job['id']}/result")
    assert result['metrics']['tpot_s'] == pytest.approx(0.008880084223796166, rel=1e-12)
    assert result['provenance']['source_fingerprint'] == REFERENCE['fingerprint']
    assert all('stages' not in value for value in result['phases'].values())
    artifacts = {x['name']: x['url'] for x in job['artifacts']}
    assert {'result.json', 'report.md', 'phases.csv', 'latency_breakdown.csv', 'config.resolved.json', 'provenance.json'} <= artifacts.keys()
    full = client.json('GET', artifacts['result.json'])
    assert len(full['phases']['prefill']['stages']) == 1237
    assert full['hardware']['npu_count'] == full['hardware']['su_size'] == 512
    assert full['hardware']['npu']['hbm_bytes'] == 96e9
    reloaded = JobManager(client.server.manager.output)
    try:
        assert reloaded.get(job['id'])['status'] == 'completed'
    finally:
        reloaded.close()


def test_long_context_capacity_failure_keeps_independent_decode_metrics():
    result = simulate_case(CaseConfig(context_tokens=4194304).model_dump())
    assert result['status'] == result['metrics']['prefill_status'] == 'infeasible'
    assert result['metrics']['ttft_s'] is None
    assert result['metrics']['latency_breakdown']['ttft'] is None
    assert result['metrics']['prefill_hbm_GB'] > 96
    assert result['metrics']['decode_status'] == 'feasible'
    assert result['metrics']['tpot_s'] > 0


def test_case_assets_and_main_dashboard_entry(client):
    for path in ('/fp4-case.html', '/fp4-case.js', '/fp4-case.css'):
        status, content, _ = client.request('GET', path)
        assert status == 200 and content
    _, content, _ = client.request('GET', '/')
    assert b'/fp4-case.html' in content
