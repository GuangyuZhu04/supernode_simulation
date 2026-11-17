"""Kimi K3 attention/KV template for an exact-parameter-budget study.

Only the attention layout and its non-parameter workload come from K3.
Projection weights, norms and MoE remain the user's exact declared budgets;
this is not a reconstruction of an official expanded checkpoint.
"""
from __future__ import annotations

from supernode_sim.core import DTYPE_BYTES
from supernode_sim.inference import kv_layout_bytes


def default_profile():
    return {
        'family': 'kimi_k3', 'architecture_id': 'kimi-k3', 'display_label': 'Kimi K3 Attention/KV',
        'layers': 93, 'counts': {'KDA': 69, 'MLA': 24},
        'num_heads': 96, 'kv_latent_dim': 512, 'kv_rope_dim': 64,
        'mla': {'qk_nope_head_dim':128, 'qk_rope_head_dim':64, 'v_head_dim':128},
        'kda': {'num_heads':96, 'head_k_dim':128, 'head_v_dim':128, 'conv_state_tokens':4, 'state_dtype':'fp32'},
        'precision': {'compute':'fp8','kv':'fp16','activation':'fp16','recurrent':'fp32'},
        'sources': ['examples/kimi_k3_ascend950_fp8.yaml', 'supernode_sim/components.py:attention_state',
                    'supernode_sim/structured.py:Builder.kernel'],
        'scope': 'Official reference Attention/KV template only. Fixed96heads and69KDA+24MLA; exact user total/active/expert budgets are synthetic scale variants. No AttnRes/embedding/gating tensor reconstruction.',
    }


def _kv(profile, context, layers, tp):
    return kv_layout_bytes({'attention_kind':'latent','kv_latent_dim':profile['kv_latent_dim'],
                            'kv_rope_dim':profile['kv_rope_dim']},
                           profile.get('precision',{}).get('kv','fp16'), [context], layers=layers, tp=tp)


def _recurrent(profile, tp):
    spec=profile['kda']
    matrix=spec['num_heads']*spec['head_k_dim']*spec['head_v_dim']/tp
    convolution=spec['num_heads']*(2*spec['head_k_dim']+spec['head_v_dim'])*spec['conv_state_tokens']/tp
    return matrix*DTYPE_BYTES[spec.get('state_dtype','fp32')], convolution*2


def attention_state(profile, context, tp):
    """Persistent bytes for one owned sequence on one attention TP rank."""
    counts=profile['counts']
    kv=_kv(profile,context,counts['MLA'],tp)
    recurrent,convolution=_recurrent(profile,tp)
    categories={'kv_cache':kv['data_bytes'],'kv_scales':kv['scale_bytes'],
                'page_table':kv['page_table_bytes'],
                'recurrent_state':recurrent*counts['KDA'],
                'convolution_state':convolution*counts['KDA']}
    return {'categoriesbytes':categories, 'total_bytes':sum(categories.values()),
            'categories_bytes':categories, 'scope':'all93layers_one_sequence_one_TP_rank'}


def attention_work(profile, kind, model, context, tp, phase):
    """Non-parameter work of one attention layer on one sequence-owning rank.

    The graph separately adds exact dense-budget parameter FLOPs/reads and
    generic residual activations. Here memory_bytes is cache/state reads and
    cache_write_bytes is writes; this avoids double counting shared weights.
    """
    tokens=context if phase=='prefill' else 1
    result={'query_tokens':tokens,'extra_compute_flops':0.,'extra_fp32_flops':0.,
            'memory_bytes':0.,'cache_write_bytes':0.,'workspace_bytes':0.,
            'index_reduce_bytes':0.,'index_reduce_repeats':0,
            'kernel_name':kind.lower(),'kind':kind,
            'cache_conversion_flops':0.,'cache_raw_bytes':0.,'cache_storage_bytes':0.}
    if kind=='KDA':
        spec=profile['kda']
        recurrent,convolution=_recurrent(profile,tp)
        result.update(extra_compute_flops=8*tokens*spec['num_heads']*spec['head_k_dim']*spec['head_v_dim']/tp,
                      memory_bytes=recurrent+convolution,cache_write_bytes=recurrent+convolution)
    elif kind=='MLA':
        spec=profile['mla']
        pairs=context*(context+1)/2 if phase=='prefill' else context+1
        width=profile['kv_latent_dim']+profile['kv_rope_dim']
        kv=_kv(profile,context,1,tp)
        reads=0 if phase=='prefill' else kv['data_bytes']+kv['scale_bytes']
        raw=tokens*width
        stored=tokens*width*2
        result.update(extra_compute_flops=2*profile['num_heads']*(spec['qk_nope_head_dim']+spec['qk_rope_head_dim']+spec['v_head_dim'])*pairs/tp,
                      memory_bytes=reads,cache_write_bytes=raw,
                      cache_conversion_flops=tokens*width,cache_raw_bytes=raw,cache_storage_bytes=stored,
                      workspace_bytes=raw)
    else:
        raise ValueError(f'Unknown Kimi attention kind: {kind}')
    return result
