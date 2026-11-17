"""Pure budget-safe MoE/shared-projection recipes for CP operator refinement.

The CP adapter owns graph guards, native apply_kernel_traces, cross-operation
allocation lifetimes and FP32 insertion. This module never changes a graph.
It intentionally does not create MatmulShape for an aggregate parameter
budget or for expected fractional routed-token/expert dimensions.
"""
from __future__ import annotations

import math

from supernode_sim.config import KernelTraceOperation


RECIPE_SOURCE = (
    'Synthetic CP fine-operator budget recipe. Three equally sized gate/up/down '
    'matrices follow the gated-FFN parameter identity 3*H*I. Exact aggregate '
    'parameter FLOPs and weight reads are conserved; expected routing geometry '
    'is not an executable integer tensor shape. Added FP16 materialization and '
    'FP32 gating work are explicit uncalibrated implementation assumptions.'
)


def _data(config):
    return config.model_dump() if hasattr(config, 'model_dump') else config


def _parent_work(event):
    compute = [key for key in event.work if key.endswith(('.compute', '.compute.fp32'))]
    if len(compute) != 1:
        raise ValueError('Recipe requires one NPU compute resource')
    rank = compute[0].split('.')[0]
    if set(event.work)-{compute[0], rank+'.hbm'}:
        raise ValueError('Recipe requires only local compute/HBM work')
    return float(event.work[compute[0]]), float(event.work.get(rank+'.hbm', 0.))


def _operation(kernel, flops, memory, implementation):
    raw={'kernel':kernel,'flops':flops,'memory_bytes':memory,
         'startup_s':0.,'workspace_bytes':0.,'implementation':implementation}
    # No matmul field is supplied: strict native shape validation must never
    # mistake a budget-equivalent width for an observed matrix layout.
    return KernelTraceOperation.model_validate(raw).model_dump(exclude_none=True)


def _validate(recipe, parent_flops, parent_memory):
    if not math.isclose(math.fsum(op['flops'] for op in recipe['operations']), parent_flops,
                        rel_tol=1e-12, abs_tol=1e-6):
        raise ValueError('Matrix recipe must conserve parent FLOPs')
    if math.fsum(op['memory_bytes'] for op in recipe['operations']) < parent_memory-max(1e-6,parent_memory*1e-12):
        raise ValueError('Matrix recipe cannot remove parent HBM reads')
    for buffer in recipe['carry_buffers']:
        if not math.isfinite(buffer['size_bytes']) or buffer['size_bytes'] < 0:
            raise ValueError('Carry buffer must be finite and nonnegative')
    for extra in recipe['extra_after_ops']:
        if any(not math.isfinite(extra[key]) or extra[key] < 0 for key in ('flops','memory_bytes')):
            raise ValueError('Additional scalar work must be finite and nonnegative')
    return recipe


def operations_for(event, config, profile, kind):
    """Return a recipe for experts/shared projections, or None otherwise.

    operations: native KernelTraceOperation dictionaries, same dtype as parent.
    carry_buffers: additive allocations owned by the adapter. Indices refer to
      the matrix operations. Gate/up/activated buffers are conservatively held
      through the down projection. The activation buffer is allocated before
      the up projection even though the following scalar event produces it.
    extra_after_ops: independent FP32 events to insert after a native operation
      and before its next operation. These are additional FLOPs, explicitly
      outside the parent's conserved matrix FLOPs.
    """
    c=_data(config);m=c['model'];w=c['workload'];x=c['execution']
    original=event.metadata.get('kernel') or event.metadata.get('operation')
    if original not in ('experts','shared_projection_pre','shared_projection_post'):
        return None
    flops,memory=_parent_work(event)
    common={'recipe_source':RECIPE_SOURCE,'evidence_kind':'synthetic',
            'physical_binary_or_tiling_known':False,'native_integer_matmul_shape_available':False,
            'parent_kernel':original,'attention_template_family':profile['family']}
    if original in ('shared_projection_pre','shared_projection_post'):
        operation=_operation(original+'.budget_matmul',flops,memory,
            {**common,'op_type':'aggregate_projection_budget',
             'dimension_basis':'Exact shared-D aggregate budget; constituent projection/norm/gating tensor shapes are unknown.',
             'parameter_flops':flops,'preserved_parent_hbm_bytes':memory,
             'no_QKV_shape_reconstruction':True})
        return _validate({'operations':[operation],'carry_buffers':[], 'extra_after_ops':[],
            'logical_geometry':{'parent_parameter_flops':flops,'parent_hbm_bytes':memory},
            'dimension_basis':'Shared D remains one exact budget matmul, not invented Q/K/V or output projection shapes.',
            'source':RECIPE_SOURCE,'notes':[
                'Shared pre/post allocation is inherited unchanged from the existing coarse CP scheduling envelope.',
                'No projection tensor shape, new intermediate HBM traffic or scalar cost is inferred from D.']},flops,memory)

    hidden=float(m['hidden_dim']);layers=int(m['layers']);experts=int(m['num_experts'])
    ep=int(x['ep']);batch=int(w['batch_size']);topk=int(m['top_k'])
    dense=m['total_params']*(1-m['expert_parameter_fraction'])
    expert_params=m['total_params']-dense
    expert_active=m['active_params']-dense
    width=expert_params/(3*hidden*experts*layers)
    routed_rows=batch*topk/ep
    expected_flops=6*routed_rows*hidden*width
    if not math.isclose(flops,expected_flops,rel_tol=1e-9,abs_tol=1e-6):
        raise ValueError('Parent expert FLOPs do not match the declared three-matrix gated budget')
    if not math.isclose(expert_active,expert_params*topk/experts,rel_tol=1e-9,abs_tol=1e-6):
        raise ValueError('Active expert budget conflicts with top-k / expert count')
    touched=-math.expm1(batch*math.log1p(-topk/experts)) if topk<experts else 1.
    expected_weight_reads=expert_params*touched/(layers*ep)
    if not math.isclose(memory,expected_weight_reads,rel_tol=1e-9,abs_tol=1e-6):
        raise ValueError('Parent experts HBM is no longer the baseline FP8 weight-read budget')
    element_bytes=2.
    input_bytes=routed_rows*hidden*element_bytes
    intermediate_bytes=routed_rows*width*element_bytes
    geometry={'hidden_dim':hidden,'logical_intermediate_width':width,
        'expected_routed_token_assignments_per_rank':routed_rows,
        'local_resident_experts':experts/ep,'expected_distinct_expert_fraction':touched,
        'expert_parameter_budget':expert_params,'layers':layers,'num_experts':experts,'top_k':topk,
        'geometry_kind':'expected_fractional_budget_geometry_not_a_physical_tensor_shape',
        'width_formula':'expert_params / (3 * hidden_dim * num_experts * layers)',
        'rows_formula':'whole_POD_batch * top_k / EP',
        'matrix_flops_identity':'3 * 2 * expected_rows * H * I == original expert matrix FLOPs'}
    matrix_flops=[flops/3,flops/3,flops-2*(flops/3)]
    matrix_weights=[memory/3,memory/3,memory-2*(memory/3)]
    operations=[]
    for index,name in enumerate(('gate','up','down')):
        # Separate GEMMs reread X for gate/up and materialize G/U; down reads
        # the gated activation and writes Y. Original weight reads are intact.
        added=input_bytes+intermediate_bytes
        operations.append(_operation('moe.'+name+'.budget_gemm',matrix_flops[index],matrix_weights[index]+added,
            {**common,'op_type':'gated_ffn_budget_matmul','matrix_role':name,
             'logical_geometry':geometry,'dimension_basis':'Three equal parameter-count HxI/IxH matrices; expected routing, no integer batch/expert token layout.',
             'preserved_weight_read_bytes':matrix_weights[index],
             'added_activation_hbm_bytes':added,'activation_dtype':'fp16',
             'added_io_formula':'2 * expected_rows * (H + I) bytes',
             'materialization':'Unfused gate/up/down activation buffers; original coarse arena is retained.'}))
    if profile['family']=='kimi_k3':
        activation='situ_gate_mul';scalar_per_element=12.
        formula='beta*tanh(gate/beta)*sigmoid(gate) * (linear_beta*tanh(up/linear_beta))'
        scalar_basis=('Explicit symbolic proxy: nine scalar expression operations plus three casts per element; '
                      'tanh/sigmoid each count as one proxy operation, not measured instruction latency or true FLOP throughput.')
        source='docs/sources/modeling_kimi_linear.py.source.txt:SituAndMul; twelve-operation symbolic FP32 proxy is an assumption'
    else:
        activation='swiglu';scalar_per_element=8.
        formula='silu(gate) * up'
        scalar_basis='Eight FP32 proxy FLOPs per gated element, following the existing V4.1 analytical kernel budget; not calibrated nonlinear throughput.'
        source='supernode_sim/case_studies/v41_fp4/target.py:Phase.moe SwiGLU 8*a*i/t'
    scalar_elements=routed_rows*width
    extra={'after_index':1,'kernel':'moe.'+activation,'flops':scalar_per_element*scalar_elements,
        'memory_bytes':3*intermediate_bytes,'fp32':True,'workspace_bytes':0.,
        'implementation':{**common,'op_type':'gated_activation','activation_formula':formula,
            'scalar_cost_basis':scalar_basis,'source':source,'logical_geometry':geometry,
            'added_flops_scope':'Additional pointwise operations, not subtracted from conserved parent matrix FLOPs.',
            'hbm_formula':'Read gate + read up + write activated output: 6 * expected_rows * I bytes',
            'internal_fp32_compute':'FP16 inputs are converted within the kernel; FP32 casts are not materialized to HBM.'}}
    carries=[{'name':'gate_output','size_bytes':intermediate_bytes,'dtype':'fp16','producer_index':0,'consumer_index':2},
             {'name':'up_output','size_bytes':intermediate_bytes,'dtype':'fp16','producer_index':1,'consumer_index':2},
             {'name':'gated_activation_output','size_bytes':intermediate_bytes,'dtype':'fp16','producer_index':1,'consumer_index':2}]
    return _validate({'operations':operations,'carry_buffers':carries,'extra_after_ops':[extra],
        'logical_geometry':geometry,'dimension_basis':geometry['geometry_kind'],'source':RECIPE_SOURCE,
        'additional_parameter_flops':0.,'additional_scalar_flops':extra['flops'],
        'additional_activation_hbm_bytes':math.fsum(o['memory_bytes'] for o in operations)-memory+extra['memory_bytes'],
        'additional_workspace_bytes':sum(b['size_bytes'] for b in carries),
        'notes':['Native trace matrix FLOPs conserve the parent exactly; pointwise gating is a separate explicitly added FP32 event.',
                 'Budget-derived I and expected token rows may be fractional. No native matmul shape or official expert width is claimed.',
                 'Gate/up/activated carries use conservative lifetimes through down completion; the original coarse activation arena remains allocated.',
                 'Per-child launch overhead is supplied separately by the adapter; the recipe adds no network startup or fixed timing multiplier.']},flops,memory)
