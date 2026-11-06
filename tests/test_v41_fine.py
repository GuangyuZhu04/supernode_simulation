import math

import pytest

from supernode_sim.case_studies.v41_fp4.fine import run_phase


def test_trace_covers_compute_and_real_mlp_order():
    r = run_phase('950dt',4,128,1,32,1024,8,'verify',keep_stages=True)
    trace = r['operator_trace']
    assert trace['coverage_fraction'] == 1
    assert trace['generated_operations'] > trace['matched_events']
    assert r['duration_s'] == pytest.approx(sum(r['breakdown_s'].values()))
    rows = [x for x in r['stages'] if x['layer']==0]
    names = [x['name'] for x in rows]
    assert names.index('gemm.expert.up') < names.index('SwiGLU') < names.index('gemm.expert.down')
    assert names.index('gemm.shared.up') < names.index('shared_swiglu') < names.index('gemm.shared.down') < names.index('shared_expert_combine')
    assert len({x['startup_s'] for x in rows if x['startup_s']}) >= 8
    for row in rows:
        shape=row['matmul']
        if shape:
            assert sum(v for k,v in row['work'].items() if 'compute' in k) == pytest.approx(2*math.prod(shape.values()))


@pytest.mark.parametrize('phase,committed',[('prefill',1),('draft',1),('verify',1),('commit',6)])
def test_launch_sensitivity_excludes_network(phase,committed):
    args=('950pr',4,128,1,4 if phase!='prefill' else 1,1024,8,phase,committed)
    nominal=run_phase(*args)
    double=run_phase(*args,launch_multiplier=2)
    assert nominal['status']==double['status']=='feasible'
    assert double['duration_s']-nominal['duration_s'] == pytest.approx(nominal['breakdown_s']['operator_launch_s'])
    assert double['breakdown_s'].get('communication_startup_s',0) == nominal['breakdown_s'].get('communication_startup_s',0)
    assert nominal['zero_operator_launch_s'] == pytest.approx(nominal['duration_s']-nominal['breakdown_s']['operator_launch_s'])


def test_failed_capacity_never_creates_timing():
    r=run_phase('950dt',1,2,1,1,1024,4,'prefill')
    assert r['status']=='infeasible'
    assert r['capacity_check']=='persistent_lower_bound'
    assert r['duration_s'] is None and r['breakdown_s'] is None
    assert r['native_completed_events']==0
    assert r['memory']['hbm_peak_bytes']>96e9


def test_long_context_independent_decode_does_not_prove_prefill():
    pre=run_phase('950dt',4,128,1,1,4194304,8,'prefill')
    verify=run_phase('950dt',4,128,1,32,4194304,8,'verify')
    assert pre['status']=='infeasible' and pre['duration_s'] is None
    assert verify['status']=='feasible' and verify['duration_s']>0
