"""Feature-branch execution remains compatible with main's search and latency APIs."""
import pytest

from supernode_sim.analysis import run
from supernode_sim.config import Config, load_config
from supernode_sim.feedback import feedback


@pytest.fixture(params=['dspark', 'megakernel', 'pd'])
def accelerated(request):
    return load_config(f'examples/features/synthetic-{request.param}.yaml')


def search_options(config, **overrides):
    return {'ttft_s': 100, 'tpot_s': 100,
            'prefill_lengths': [max(config.workload.input_lengths)],
            'decode_batch_sizes': [config.workload.batch_size],
            'sp_enabled': [False], **overrides}


def test_auto_feedback_runs_accelerations_and_replays_the_selected_candidate(accelerated):
    result = feedback(accelerated, search_options(accelerated))
    assert result['status'] == 'feasible'
    assert result['search_complete']
    assert result['counts']['total'] == result['counts']['feasible'] == 1
    best = result['best']
    assert best['engine'] == 'simulate'
    config = Config.model_validate(best['config'])
    assert config.hardware == accelerated.hardware
    assert config.execution.features == accelerated.execution.features
    replay = run(config)
    for name in ('ttft', 'tpot'):
        metric = best['metrics'][name + '_s']
        assert metric == pytest.approx(replay['metrics'][name + '_s'])
        breakdown = best['metrics']['latency_breakdown'][name]
        assert breakdown['total_s'] == pytest.approx(metric)
        assert sum(breakdown['components_s'].values()) == pytest.approx(metric)
    assert best['metrics']['decode_tokens'] == sum(n - 1 for n in config.workload.output_lengths)


def test_explicit_streaming_feedback_rejects_unsupported_accelerations(accelerated):
    with pytest.raises(ValueError, match='ordinary simulate'):
        feedback(accelerated, search_options(accelerated, engine='streaming'))


def test_pd_handoff_remains_in_latency_communication_breakdown():
    result = run(load_config('examples/features/synthetic-pd.yaml'))
    assert result['metrics']['pd_handoff_time_s'] > 0
    parts = result['metrics']['latency_breakdown']['tpot']['components_s']
    assert parts['communication_scaleout_s'] > 0


def test_deepinfer_stage_example_runs_with_streaming_latency_breakdown():
    from supernode_sim.streaming import run_streaming

    config = load_config('examples/features/synthetic-deepinfer-stages.yaml')
    for phase, metric, missing in [('prefill', 'ttft', 'tpot'), ('decode', 'tpot', 'ttft')]:
        result = run_streaming(config, phase=phase)
        assert result['status'] == 'feasible'
        assert result['metrics'][missing + '_s'] is None
        record = result['metrics']['latency_breakdown'][metric]
        assert sum(record['components_s'].values()) == pytest.approx(result['metrics'][metric + '_s'])
