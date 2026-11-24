"""Frozen-baseline scaling comparisons and cross-scale measurement provenance."""
import json
from pathlib import Path

import pytest

from supernode_sim.cli import main
from supernode_sim.config import load_config
from supernode_sim.training_scaling import run_training_scaling


def setup_study(directory, mode='strong', *, hbm=None):
    source = Path(__file__).resolve().parents[1] / 'examples' / 'training_scaling_moe.yaml'
    data = load_config(source).model_dump()
    data['hardware'].update(npu_count=2, su_size=2, npus_per_host=2)
    data['execution'].update(dp=1, dp_domain_size=None)
    if hbm is not None:
        data['hardware']['npu']['hbm_bytes'] = hbm
    (directory / 'base.json').write_text(json.dumps(data), encoding='utf-8')
    changes = {'hardware.npu_count': 4, 'execution.dp': 2}
    if mode == 'strong':
        changes['workload.batch_size'] = 2
    manifest = {'config': 'base.json', 'scaling': mode, 'baseline': {'name': 'base'},
                'holdouts': [{'name': 'target', 'changes': changes}]}
    path = directory / 'study.json'
    path.write_text(json.dumps(manifest), encoding='utf-8')
    return path, manifest


@pytest.mark.parametrize('mode', ['strong', 'weak'])
def test_native_prediction_and_measured_holdout_are_independent(tmp_path, mode):
    path, manifest = setup_study(tmp_path, mode)
    raw = run_training_scaling(path)
    base, target = raw['cases']
    pred = target['prediction']
    assert raw['measured_holdouts_compared'] == 0
    assert target['validation_status'] == 'prediction_only'
    assert not target['native_errors'] and 'baseline_adjusted_step_time_s' not in target
    assert target['comparability'] == 'matched'
    assert pred['tokens_per_step'] / base['prediction']['tokens_per_step'] == (2 if mode == 'weak' else 1)
    assert target['scaling']['efficiency'] == pytest.approx(pred['throughput_tokens_s'] / base['prediction']['throughput_tokens_s'] / 2)
    assert pred['critical_path_comm_service_s_per_step'] <= pred['step_time_s']
    assert pred['exposed_comm_s_per_rank_per_step'] <= pred['step_time_s']
    assert pred['ep_sent_bytes_per_rank_per_step_by_phase']['forward']['dispatch'] > 0
    assert target['coverage']['kernel_calibration']['matched_events'] == 0
    assert raw['sources']['*']['status'] == 'assumed'

    # Controlled test observations exercise the scoring path; they are not benchmark data.
    manifest['baseline']['observation'] = {'kind': 'measured', 'source': 'unit-test baseline',
                                          'step_time_s': 2 * base['prediction']['step_time_s']}
    obs = {'kind': 'measured', 'source': 'unit-test holdout', 'step_time_s': 4 * pred['step_time_s'],
           'peak_hbm_bytes': pred['peak_hbm_bytes'], 'exposed_comm_s_per_rank_per_step': 0}
    manifest['holdouts'][0]['observation'] = obs
    path.write_text(json.dumps(manifest), encoding='utf-8')
    study = run_training_scaling(path)
    target = study['cases'][1]
    assert study['measured_holdouts_compared'] == 1 and target['prediction'] == pred
    assert target['native_errors']['step_time_s']['relative_pct'] == pytest.approx(-75)
    assert target['native_errors']['peak_hbm_bytes']['delta'] == 0
    assert target['native_errors']['exposed_comm_s_per_rank_per_step']['relative_pct'] is None
    assert target['baseline_adjusted_step_time_s'] == pytest.approx(2 * pred['step_time_s'])
    assert target['baseline_adjusted_step_error']['relative_pct'] == pytest.approx(-50)
    assert target['scaling']['measured_efficiency'] == pytest.approx(target['scaling']['efficiency'] / 2)


def test_synthetic_observation_and_mismatched_workload(tmp_path):
    path, manifest = setup_study(tmp_path)
    manifest['holdouts'][0]['changes'].pop('workload.batch_size')
    obs = {'kind': 'synthetic', 'source': 'example model', 'step_time_s': 4}
    manifest['baseline']['observation'] = manifest['holdouts'][0]['observation'] = obs
    path.write_text(json.dumps(manifest), encoding='utf-8')
    assert main(['training-scaling', str(path), '-o', str(tmp_path)]) == 3
    result = json.loads((tmp_path / 'training_scaling.json').read_text(encoding='utf-8'))
    baseline, target = result['cases']
    for row in result['cases']:
        assert json.loads((tmp_path / row['summary_file']).read_text(encoding='utf-8'))['status'] == 'complete'
        assert '<svg' in (tmp_path / row['chart_file']).read_text(encoding='utf-8')
    assert baseline['validation_status'] == target['validation_status'] == 'synthetic_reference'
    assert target['comparability'] == 'workload_mismatch' and target['prediction'] is not None
    assert result['measured_holdouts_compared'] == 0 and target['native_errors'] == {}
    assert 'scaling' not in target and 'baseline_adjusted_step_time_s' not in target


def test_infeasible_case_retains_diagnostic_but_has_no_scaling_or_errors(tmp_path):
    path, _ = setup_study(tmp_path, hbm=64)
    target = run_training_scaling(path)['cases'][1]
    assert target['prediction'] is None and 'scaling' not in target and target['native_errors'] == {}
    assert target['comparability'] == 'incomplete_simulation'
    assert target['diagnostics'][0]['code'] == 'initial_memory_exceeded'


def test_holdout_cannot_refit_hardware_runtime_or_change_model(tmp_path):
    path, manifest = setup_study(tmp_path)
    manifest['holdouts'][0]['changes']['hardware.npu.efficiency'] = .5
    path.write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError, match='Frozen training settings'):
        run_training_scaling(path)


def test_steady_observations_require_multiple_simulated_steps(tmp_path):
    path, manifest = setup_study(tmp_path)
    manifest['holdouts'][0]['observation'] = {'kind': 'measured', 'source': 'test steady window', 'steady_step_time_s': 1}
    path.write_text(json.dumps(manifest), encoding='utf-8')
    single = run_training_scaling(path)
    assert single['measured_holdouts_compared'] == 0 and single['cases'][1]['native_errors'] == {}
    base_path = tmp_path / 'base.json'
    data = json.loads(base_path.read_text(encoding='utf-8'))
    data['workload']['training_steps'] = 2
    base_path.write_text(json.dumps(data), encoding='utf-8')
    study = run_training_scaling(path)
    target = study['cases'][1]
    assert study['measured_holdouts_compared'] == 1
    assert target['native_errors']['steady_step_time_s']['delta'] == target['prediction']['steady_step_time_s'] - 1
    assert 'baseline_adjusted_step_time_s' not in target


def test_repeating_baseline_does_not_count_as_scaling_evidence(tmp_path):
    path, manifest = setup_study(tmp_path)
    manifest['holdouts'][0].update(changes={}, observation={'kind': 'measured', 'source': 'same point', 'step_time_s': 1})
    path.write_text(json.dumps(manifest), encoding='utf-8')
    study = run_training_scaling(path)
    assert study['measured_holdouts_compared'] == 0 and study['cases'][1]['comparability'] == 'same_configuration'


def test_duplicate_measured_holdouts_count_one_configuration(tmp_path):
    path, manifest = setup_study(tmp_path)
    holdout = manifest['holdouts'][0]
    holdout['observation'] = {'kind': 'measured', 'source': 'same test run', 'step_time_s': 1}
    manifest['holdouts'].append({**holdout, 'name': 'duplicate'})
    path.write_text(json.dumps(manifest), encoding='utf-8')
    result = run_training_scaling(path)
    assert result['measured_holdouts_compared'] == 1
    assert result['cases'][1]['native_errors'] == result['cases'][2]['native_errors']


def test_invalid_layout_preserves_other_predictions_and_reports(tmp_path):
    path, manifest = setup_study(tmp_path)
    manifest['holdouts'].insert(0, {'name': 'bad-layout', 'changes': {'execution.dp': 3}})
    path.write_text(json.dumps(manifest), encoding='utf-8')
    result = run_training_scaling(path, output=tmp_path)
    invalid, good = result['cases'][1:]
    assert invalid['simulation_status'] == 'invalid_config' and invalid['native_errors'] == {}
    assert invalid['comparability'] == 'incomplete_simulation' and 'chart_file' not in invalid
    assert json.loads((tmp_path / invalid['summary_file']).read_text(encoding='utf8'))['status'] == 'unavailable'
    assert good['comparability'] == 'matched' and good['prediction']['step_time_s'] > 0
