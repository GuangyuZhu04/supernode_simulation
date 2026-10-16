"""Feedback searches preserve the fixed system and verify whole inference runs."""
import copy
import csv
import importlib
import json
import math

import pytest

from supernode_sim.analysis import evaluate_targets, run
from supernode_sim.cli import main
from supernode_sim.config import Config
from supernode_sim.feedback import FeedbackSpec, feedback
from tests.test_structured import configuration


feedback_module = importlib.import_module('supernode_sim.feedback')


def inference_configuration(patch=None):
    from supernode_sim.config import deep_merge
    return configuration(deep_merge({'workload': {'scenario': 'inference'}}, patch or {}))


def single_search(**overrides):
    return {'ttft_s': 1.0, 'tpot_s': 1.0, 'prefill_lengths': [4],
            'decode_batch_sizes': [1], 'tp_sizes': [1], 'ep_sizes': [1],
            'sp_enabled': [False], 'engine': 'simulate', **overrides}


def decode_throughput(metrics):
    completions = [request['token_completion_s'] for request in metrics['requests']]
    return sum(len(times) - 1 for times in completions) / (
        max(times[-1] for times in completions) - min(times[0] for times in completions))


def synthetic_result(config, *_args, **_kwargs):
    """Batch 2 wins only when ranking actual decode completions, not total tokens."""
    batch = config.workload.batch_size
    if batch == 2:
        completions = [[10.0, 10.1, 10.2], [10.05, 10.2, 10.4]]
    else:
        completions = [[1.0, 2.0, 3.0] for _ in range(batch)]
    metrics = {
        'duration_s': max(times[-1] for times in completions),
        'ttft_s': max(times[0] for times in completions),
        'tpot_s': max((times[-1] - times[0]) / 2 for times in completions),
        'requests': [{'token_completion_s': times} for times in completions],
        'output_tokens': 3 * batch,
    }
    metrics['throughput_tokens_s'] = metrics['output_tokens'] / metrics['duration_s']
    return {'status': 'feasible', 'config': config.model_dump(), 'metrics': metrics,
            'slo': evaluate_targets(config, metrics, True),
            'simulation': {'status': 'feasible', 'memory_peaks_bytes': {'npu0.hbm': 100},
                           'diagnostics': []}}


@pytest.mark.parametrize('field', ['prefill_lengths', 'decode_batch_sizes', 'tp_sizes', 'ep_sizes'])
@pytest.mark.parametrize('values', [[], [0], [-1], [1.5], [True], ['2']])
def test_search_axes_require_nonempty_strict_positive_integers(field, values):
    with pytest.raises(ValueError):
        FeedbackSpec.model_validate({field: values})


@pytest.mark.parametrize('field', ['ttft_s', 'tpot_s', 'tpop_s'])
@pytest.mark.parametrize('value', [0, -1, math.inf, -math.inf, math.nan])
def test_latency_targets_require_positive_finite_values(field, value):
    with pytest.raises(ValueError):
        FeedbackSpec.model_validate({field: value})


def test_spec_normalizes_axes_and_accepts_tpop_spelling():
    spec = FeedbackSpec.model_validate({'ttft_s': 1, 'tpop_s': 0.1,
                                       'prefill_lengths': [8, 1, 8, 4],
                                       'decode_batch_sizes': [4, 2, 4]})
    assert spec.tpot_s == 0.1
    assert spec.prefill_lengths == [1, 4, 8]
    assert spec.decode_batch_sizes == [2, 4]
    with pytest.raises(ValueError):
        FeedbackSpec.model_validate({'tpot_s': 0.1, 'tpop_s': 0.2})


@pytest.mark.parametrize('values', [[], [0], [1], ['true']])
def test_sp_axis_requires_actual_boolean_values(values):
    with pytest.raises(ValueError):
        FeedbackSpec.model_validate({'sp_enabled': values})


@pytest.mark.parametrize('output_length', [0, 1, 2.5, True])
def test_decode_requires_at_least_two_output_tokens(output_length):
    with pytest.raises(ValueError):
        FeedbackSpec.model_validate({'output_length': output_length})


def test_feedback_requires_inference_and_both_targets():
    with pytest.raises(ValueError, match='inference'):
        feedback(configuration(), single_search())
    for targets in ({}, {'ttft_s': 1}, {'tpot_s': 1}):
        with pytest.raises(ValueError):
            feedback(inference_configuration(), targets)


def test_config_targets_are_inherited_and_explicit_targets_override():
    config = inference_configuration({'targets': {'ttft_s': 1e-30, 'tpot_s': 1e-30}})
    axes = {key: value for key, value in single_search().items() if key not in ('ttft_s', 'tpot_s')}
    inherited = feedback(config, axes)
    assert inherited['status'] == 'no_feasible_candidate'
    overridden = feedback(config, single_search())
    assert overridden['status'] == 'feasible'
    assert overridden['best']['config']['targets']['ttft_s'] == 1.0
    assert overridden['best']['config']['targets']['tpot_s'] == 1.0


def test_default_grid_includes_powers_of_two_and_actual_ragged_endpoint(monkeypatch):
    monkeypatch.setattr(feedback_module, 'run', synthetic_result)
    config = inference_configuration({'workload': {'batch_size': 3, 'input_length': 99,
                                                   'input_lengths': [3, 7, 5]}})
    result = feedback(config, {'ttft_s': 100, 'tpot_s': 100, 'engine': 'simulate'})
    candidates = result['candidates']
    assert {row['parameters']['prefill_length'] for row in candidates} == {1, 2, 4, 7}
    assert {row['parameters']['decode_batch_size'] for row in candidates} == {1, 2, 3}
    assert len(candidates) == 12
    assert all(row['parameters']['sp'] == 1 for row in candidates)


def test_ragged_output_lengths_need_explicit_uniform_output_length():
    config = inference_configuration({'workload': {'output_lengths': [2, 3]}})
    with pytest.raises(ValueError, match='output'):
        feedback(config, single_search())
    result = feedback(config, single_search(output_length=4))
    assert result['best']['config']['workload']['output_lengths'] == [4]


def test_candidate_limit_is_checked_before_any_simulation(monkeypatch):
    calls = []

    def record_run(*args, **kwargs):
        calls.append(args)
        return synthetic_result(*args, **kwargs)

    monkeypatch.setattr(feedback_module, 'run', record_run)
    monkeypatch.setattr(feedback_module, 'run_streaming', record_run)
    with pytest.raises(ValueError, match='max_candidates'):
        feedback(inference_configuration(), single_search(
            prefill_lengths=[1, 2, 4], decode_batch_sizes=[1, 2], max_candidates=5))
    assert calls == []


def test_explicit_invalid_layouts_remain_auditable():
    config = inference_configuration({'hardware': {'npu_count': 2, 'su_size': 2},
                                      'execution': {'tp': 2}})
    result = feedback(config, single_search(tp_sizes=[1, 2], sp_enabled=[False, True]))
    assert len(result['candidates']) == 4
    invalid = [row for row in result['candidates'] if row['status'] == 'invalid_config']
    assert len(invalid) == 2
    assert all(row['diagnostics'] for row in invalid)
    assert result['status'] == 'feasible'
    assert result['search_complete'] is True


@pytest.mark.parametrize('engine', ['simulate', 'streaming'])
def test_real_search_preserves_system_and_replay_matches_best(engine):
    config = inference_configuration({
        'hardware': {'npu_count': 4, 'su_size': 2, 'npus_per_host': 2},
        'execution': {'tp': 2, 'ep': 2, 'rank_order': [3, 2, 1, 0],
                      'precision': {'kv': 'fp8'}, 'features': {'kv_offload': 'host'}},
    })
    original = copy.deepcopy(config.model_dump())
    result = feedback(config, {'ttft_s': 1, 'tpot_s': 1, 'prefill_lengths': [4, 8],
                               'decode_batch_sizes': [1, 2], 'engine': engine})
    assert result['status'] == 'feasible'
    assert result['objective'] == 'decode_throughput_tokens_s'
    assert result['search_complete'] is True
    assert config.model_dump() == original
    assert result['config'] == original
    best = result['best']
    replay_config = Config.model_validate(best['config'])
    for key in ('model', 'hardware'):
        assert best['config'][key] == original[key]
    for key in ('pp', 'dp', 'fsdp', 'rank_order', 'precision', 'features'):
        assert best['config']['execution'][key] == original['execution'][key]
    assert replay_config.workload.input_lengths == [best['parameters']['prefill_length']] * replay_config.workload.batch_size
    assert replay_config.workload.output_lengths == [3] * replay_config.workload.batch_size
    assert replay_config.workload.batch_size == best['parameters']['decode_batch_size']
    replay = run(replay_config, full_detail=False)
    assert replay['status'] == 'feasible'
    for name in ('ttft_s', 'tpot_s', 'duration_s'):
        assert best['metrics'][name] == pytest.approx(replay['metrics'][name], rel=1e-10)
    assert best['metrics']['decode_throughput_tokens_s'] == pytest.approx(decode_throughput(replay['metrics']), rel=1e-10)
    assert best['memory_peaks_bytes'] == pytest.approx(replay['simulation']['memory_peaks_bytes'])
    assert best['slo']['satisfied'] is True
    assert {check['metric'] for check in best['slo']['checks']} >= {'ttft_s', 'tpot_s'}
    assert 'requests' not in best['metrics']
    assert all('config' not in row and 'requests' not in row['metrics'] for row in result['candidates'])
    feasible = [row for row in result['candidates'] if row['status'] == 'feasible']
    assert best['metrics']['decode_throughput_tokens_s'] == max(row['metrics']['decode_throughput_tokens_s'] for row in feasible)
    assert result['pareto_frontier']
    assert set(result['pareto_frontier']) <= {row['id'] for row in feasible}
    assert result['feasible_boundaries']


@pytest.mark.parametrize('engine', ['simulate', 'streaming'])
@pytest.mark.parametrize('failed_target', ['ttft_s', 'tpot_s'])
def test_each_latency_target_is_enforced_including_streaming(engine, failed_target):
    result = feedback(inference_configuration(), single_search(engine=engine, **{failed_target: 1e-30}))
    assert result['status'] == 'no_feasible_candidate'
    assert result['best'] is None
    assert result['pareto_frontier'] == []
    assert result['feasible_boundaries'] == []
    candidate = result['candidates'][0]
    assert candidate['status'] == 'slo_failed'
    checks = {check['metric']: check['passed'] for check in candidate['slo']['checks']}
    assert checks[failed_target] is False
    assert checks['tpot_s' if failed_target == 'ttft_s' else 'ttft_s'] is True


@pytest.mark.parametrize('engine', ['simulate', 'streaming'])
def test_memory_limit_is_checked_through_decode_not_only_prefill(engine):
    patch = {'workload': {'batch_size': 1, 'input_length': 1, 'output_length': 1},
             'execution': {'features': {'kv_page_tokens': 1}}}
    prefill_config = inference_configuration(patch)
    prefill = run(prefill_config, full_detail=False)
    patch['workload']['output_length'] = 16
    full_config = inference_configuration(patch)
    full = run(full_config, full_detail=False)
    prefill_peak = prefill['simulation']['memory_peaks_bytes']['npu0.hbm']
    decode_peak = full['simulation']['memory_peaks_bytes']['npu0.hbm']
    assert decode_peak > prefill_peak
    cap = (prefill_peak + decode_peak) / 2
    patch['hardware'] = {'npu': {'hbm_bytes': cap}}
    patch['workload']['output_length'] = 1
    assert run(inference_configuration(patch), full_detail=False)['status'] == 'feasible'
    patch['workload']['output_length'] = 16
    result = feedback(inference_configuration(patch), single_search(prefill_lengths=[1], engine=engine))
    assert result['status'] == 'no_feasible_candidate'
    assert result['best'] is None
    assert result['candidates'][0]['status'] == 'infeasible'
    assert result['candidates'][0]['diagnostics']


def test_sequence_parallel_on_and_off_are_real_distinct_simulations():
    config = inference_configuration({'hardware': {'npu_count': 2, 'su_size': 2},
                                      'execution': {'tp': 2}})
    result = feedback(config, single_search(tp_sizes=[2], sp_enabled=[False, True], prefill_lengths=[8]))
    assert len(result['candidates']) == 2
    rows = {row['parameters']['sequence_parallel']: row for row in result['candidates']}
    assert rows[False]['parameters']['sp'] == 1
    assert rows[True]['parameters']['sp'] == 2
    assert all(row['status'] == 'feasible' for row in rows.values())
    assert rows[False]['metrics']['duration_s'] != rows[True]['metrics']['duration_s']
    for enabled, row in rows.items():
        data = config.model_dump()
        data['execution']['sequence_parallel'] = enabled
        data['workload'].update(batch_size=1, input_lengths=[8], output_lengths=[3])
        replay = run(Config.model_validate(data), full_detail=False)
        assert row['metrics']['duration_s'] == pytest.approx(replay['metrics']['duration_s'])


def test_nonmonotonic_decode_throughput_wins_over_batch_size_and_end_to_end_throughput(monkeypatch):
    monkeypatch.setattr(feedback_module, 'run', synthetic_result)
    result = feedback(inference_configuration(), single_search(
        ttft_s=100, tpot_s=100, decode_batch_sizes=[1, 2, 4]))
    assert len(result['candidates']) == 3
    assert result['best']['parameters']['decode_batch_size'] == 2
    assert result['best']['metrics']['decode_throughput_tokens_s'] == pytest.approx(10)
    assert result['best']['metrics']['throughput_tokens_s'] < result['candidates'][0]['metrics']['throughput_tokens_s']


@pytest.mark.parametrize('axis', ['tp_sizes', 'ep_sizes'])
def test_invalid_single_axis_is_retained_when_other_axis_is_derived(axis):
    config = inference_configuration({'hardware': {'npu_count': 2, 'su_size': 2},
                                      'execution': {'tp': 2}})
    result = feedback(config, {'ttft_s': 1, 'tpot_s': 1, axis: [3],
                               'prefill_lengths': [4], 'decode_batch_sizes': [1],
                               'sp_enabled': [False], 'engine': 'simulate'})
    assert result['status'] == 'no_feasible_candidate'
    assert result['search_complete'] is True
    assert result['counts']['total'] == result['counts']['invalid_config'] == 1
    candidate = result['candidates'][0]
    assert candidate['parameters']['tp' if axis == 'tp_sizes' else 'ep'] == 3
    assert candidate['status'] == 'invalid_config'
    assert candidate['diagnostics']


def test_engine_failure_retains_other_candidates_but_search_is_incomplete(monkeypatch):
    def fail_one_candidate(config, *args, **kwargs):
        if config.workload.batch_size == 1:
            raise ValueError('synthetic engine could not finish this candidate')
        return synthetic_result(config, *args, **kwargs)

    monkeypatch.setattr(feedback_module, 'run', fail_one_candidate)
    result = feedback(inference_configuration(), single_search(
        ttft_s=100, tpot_s=100, decode_batch_sizes=[1, 2]))
    assert result['status'] == 'error'
    assert result['search_complete'] is False
    assert result['counts']['total'] == 2
    assert result['counts']['error'] == result['counts']['feasible'] == 1
    assert result['best']['parameters']['decode_batch_size'] == 2
    failed = next(row for row in result['candidates'] if row['status'] == 'error')
    assert any('synthetic engine' in diagnostic for diagnostic in failed['diagnostics'])


def test_boundary_maxima_do_not_imply_a_jointly_feasible_candidate(monkeypatch):
    def coupled_latency(config, *_args, **_kwargs):
        batch, length = config.workload.batch_size, config.workload.input_length
        first = float(batch * length)
        metrics = {'ttft_s': first, 'tpot_s': 1.0, 'duration_s': first + 2,
                   'throughput_tokens_s': 3 * batch / (first + 2),
                   'requests': [{'token_completion_s': [first, first + 1, first + 2]}
                                for _ in range(batch)]}
        return {'status': 'feasible', 'metrics': metrics,
                'simulation': {'status': 'feasible', 'diagnostics': [], 'memory_peaks_bytes': {}}}

    monkeypatch.setattr(feedback_module, 'run', coupled_latency)
    result = feedback(inference_configuration(), single_search(
        ttft_s=10, tpot_s=1, prefill_lengths=[4, 8], decode_batch_sizes=[1, 2]))
    boundary, = result['feasible_boundaries']
    assert (boundary['max_prefill_length'], boundary['max_decode_batch_size']) == (8, 2)
    by_id = {row['id']: row for row in result['candidates']}
    assert all(by_id[identifier]['status'] == 'feasible' for identifier in boundary['candidate_ids'])
    assert {(by_id[identifier]['parameters']['prefill_length'],
             by_id[identifier]['parameters']['decode_batch_size'])
            for identifier in result['pareto_frontier']} == {(4, 2), (8, 1)}
    assert result['best']['parameters']['prefill_length'] == 4
    assert result['best']['parameters']['decode_batch_size'] == 2
    assert next(row for row in result['candidates'] if row['parameters']['prefill_length'] == 8
                and row['parameters']['decode_batch_size'] == 2)['status'] == 'slo_failed'


@pytest.mark.parametrize('parallelism', ['dp', 'pp'])
def test_auto_engine_preserves_data_and_pipeline_parallelism(parallelism):
    config = inference_configuration({'hardware': {'npu_count': 2, 'su_size': 2},
                                      'execution': {parallelism: 2}})
    result = feedback(config, single_search(engine='auto'))
    assert result['status'] == 'feasible'
    best = result['best']
    assert best['engine'] == 'simulate'
    assert best['config']['execution'][parallelism] == 2
    expected = run(Config.model_validate(best['config']), full_detail=False)
    assert best['metrics']['decode_throughput_tokens_s'] == pytest.approx(decode_throughput(expected['metrics']))


def test_cli_search_file_overrides_exports_and_no_feasible_exit(tmp_path, capsys):
    config_path = tmp_path / 'config.json'
    search_path = tmp_path / 'feedback.yaml'
    output = tmp_path / 'report'
    config_path.write_text(json.dumps(inference_configuration().model_dump()), encoding='utf-8')
    search_path.write_text('ttft_s: 1\ntpop_s: 0.000000000000000000000000000001\n'
                           'tp_sizes: [1]\nep_sizes: [1]\nsp_enabled: [false]\n'
                           'prefill_lengths: [4]\ndecode_batch_sizes: [1]\n', encoding='utf-8')
    args = ['feedback', str(config_path), '--search', str(search_path), '--tpop-s', '1',
            '--engine', 'simulate', '--output', str(output)]
    assert main(args) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary['status'] == 'feasible'
    assert summary['search_complete'] is True
    assert {'result.json', 'report.md', 'candidates.csv', 'config.resolved.json', 'best.config.json'} <= {
        path.name for path in output.iterdir()}
    result = json.loads((output / 'result.json').read_text(encoding='utf-8'))
    best_config = json.loads((output / 'best.config.json').read_text(encoding='utf-8'))
    assert best_config == result['best']['config']
    assert best_config['targets']['tpot_s'] == 1
    assert run(Config.model_validate(best_config), full_detail=False)['status'] == 'feasible'
    with (output / 'candidates.csv').open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == result['counts']['total'] == 1
    assert float(rows[0]['decode_throughput_tokens_s']) == result['best']['metrics']['decode_throughput_tokens_s']
    assert json.loads(rows[0]['slo'])['satisfied'] is True
    failed_args = ['feedback', str(config_path), '--search', str(search_path),
                   '--engine', 'simulate', '--output', str(output)]
    assert main(failed_args) == 3
    assert json.loads(capsys.readouterr().out)['status'] == 'no_feasible_candidate'
    assert not (output / 'best.config.json').exists()
    assert json.loads((output / 'result.json').read_text(encoding='utf-8'))['best'] is None


def test_cli_rejects_conflicting_tpot_aliases_before_running(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        main(['feedback', str(tmp_path / 'not_needed.json'), '--tpot-s', '0.1', '--tpop-s', '0.2'])
    assert error.value.code == 2
    assert 'must specify the same value' in capsys.readouterr().err


def test_cli_candidate_limit_returns_invalid_input_without_export(tmp_path, capsys):
    config_path = tmp_path / 'config.json'
    output = tmp_path / 'report'
    config_path.write_text(json.dumps(inference_configuration().model_dump()), encoding='utf-8')
    assert main(['feedback', str(config_path), '--ttft-s', '1', '--tpot-s', '1',
                 '--prefill-lengths', '1', '2', '--decode-batch-sizes', '1', '2',
                 '--max-candidates', '3', '--output', str(output)]) == 2
    assert 'max_candidates' in json.loads(capsys.readouterr().err)['message']
    assert not output.exists()

