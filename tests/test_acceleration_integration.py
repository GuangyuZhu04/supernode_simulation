"""Public execution, search and report contracts for composable accelerations."""
import pytest

from supernode_sim.analysis import run
from supernode_sim.config import Config
from supernode_sim.report import markdown_report
from supernode_sim.search import size_su, sweep
from supernode_sim.streaming import run_streaming
from tests.test_disaggregated import specification


def fused_pd():
    data = specification().model_dump()
    data['execution']['features']['megakernel'] = {'enabled': True, 'groups': [{
        'name': 'norm_mla', 'kernels': ['prenorm', 'mla'], 'fused_startup_s': 0,
        'source': {'status': 'assumed', 'description': 'Synthetic no-savings fusion control'},
    }]}
    return Config.model_validate(data)


def test_megakernel_pd_composition_preserves_handoff_tokens_state_and_report():
    baseline = run(specification())
    result = run(fused_pd())
    assert result['status'] == baseline['status'] == 'feasible'
    fusion = result['model_metadata']['megakernel']
    assert fusion['matched_groups'] == 4  # prefill + three decode passes on MLA layer 0
    assert fusion['matched_events'] == 8
    pd = result['model_metadata']['prefill_decode_disaggregation']
    assert pd['handoff_payload_bytes'] == baseline['model_metadata']['prefill_decode_disaggregation']['handoff_payload_bytes']
    assert result['metrics']['output_tokens'] == baseline['metrics']['output_tokens'] == 8
    assert result['simulation']['memory_final_bytes'] == baseline['simulation']['memory_final_bytes']
    # No launch or HBM savings: fusion must not invent a roofline-overlap gain.
    assert result['metrics']['duration_s'] == pytest.approx(baseline['metrics']['duration_s'])
    metrics = result['metrics']
    assert metrics['pd_prefill_time_s'] == pytest.approx(metrics['ttft_s'])
    assert metrics['pd_first_decode_gap_s'] == pytest.approx(metrics['requests'][0]['itl_s'][0])
    assert metrics['duration_s'] == pytest.approx(metrics['pd_prefill_time_s'] + metrics['pd_handoff_time_s'] + metrics['pd_decode_time_s'])
    text = markdown_report(result)
    assert 'PD 分离' in text and 'Megakernel 图变换' in text
    assert Config.model_validate(result['config']).model_dump() == result['config']


def test_pd_search_counts_both_pools_without_inventing_dp_replicas():
    data = specification().model_dump()
    data['targets'].update(candidate_npu_counts=[2], candidate_su_sizes=[1, 2], total_npus=2)
    result = size_su(Config.model_validate(data))
    assert result['searches']['single']['candidate_count'] == 1
    assert result['searches']['multi']['candidate_count'] == 2
    for search in result['searches'].values():
        assert search['status'] == 'feasible'
        for row in search['candidates']:
            assert row['parallelism']['dp'] == row['parallelism']['pp'] == 1
            assert row['npu_count'] == 2 * row['parallelism']['tp'] * row['parallelism']['ep']


def test_pd_sweep_keeps_invalid_physical_mapping_as_diagnostic():
    result = sweep(specification(), {'hardware.npu_count': [2, 4]})
    assert result['candidates'][0]['result']['status'] == 'feasible'
    assert result['candidates'][1]['result']['status'] == 'invalid_config'


@pytest.mark.parametrize('feature', ['prefill_decode_disaggregation', 'megakernel'])
def test_streaming_refuses_new_features_instead_of_silently_dropping_them(feature):
    data = fused_pd().model_dump()
    if feature == 'megakernel':
        data['execution']['features']['prefill_decode_disaggregation']['enabled'] = False
        data['hardware'].update(npu_count=1, su_size=1)
    else:
        data['execution']['features']['megakernel']['enabled'] = False
    with pytest.raises(ValueError, match='ordinary simulate'):
        run_streaming(Config.model_validate(data))
