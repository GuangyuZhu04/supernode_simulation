"""DeepSeek V4.1 Flash40 Attention/KV mechanism template.

This preserves CSA2 sharing/index geometry from the repository reference.
It intentionally does not claim a full Flash checkpoint, CED skip execution,
Engram/mHC operators, native grouped TP layout, or quality-equivalent scaling.
"""
from __future__ import annotations
from math import ceil


def default_profile():
    return {
        'family': 'deepseek_v41', 'architecture_id': 'deepseek-v41',
        'display_label': 'DeepSeek V4.1 Attention/KV', 'layers': 40,
        'counts': {'SWA': 2, 'CSA2_ENCODER': 15, 'CSA2_ENCODER_INDEX': 3,
                   'CSA2_DECODER': 15, 'CSA2_DECODER_FULL_INDEX': 1,
                   'CSA2_DECODER_CANDIDATE_INDEX': 4},
        'num_heads': 64, 'head_dim': 512, 'rope_head_dim': 64,
        'window_size': 128, 'index_num_heads': 32, 'index_head_dim': 128,
        'index_topk': 512, 'candidate_rows': 2048 * 8, 'score_tile_rows': 65536,
        'kv_source_layers': [2, 8, 14, 20], 'kv_source_ratios': [2, 2, 2, 1],
        'index_source_layers': [2, 8, 14, 20, 24, 28, 32, 36],
        'precision': {'compute': 'fp8', 'kv': 'fp16', 'activation': 'fp16'},
        'sources': ['supernode_sim/case_studies/v41_fp4/reference_config.json',
                    'supernode_sim/case_studies/v41_fp4/target.py:kv_ledger,index_geometry,attention'],
        'scope': 'Flash40 Attention/KV mechanism template with exact synthetic user MoE/dense parameter budgets; no DSpark, no FP4. All40budget layers execute per decode. CED encoder-only prefill/decoder-only decode skipping is not reproduced because it would change the supplied active budget.',
        'layout_status': 'logical_tensor_budget_sharding_only; reference o_groups8 does not establish deployability of requested TP values, including TP>8 and non-divisors of8',
        'omissions': ['Exact projection/router/normalization tensor inventory', 'Engram table state and host lookup traffic',
                      'mHC operator graph', 'CED phase-specific layer skipping', 'native group-parallel layout validation'],
    }


def attention_state(profile, context, tp):
    d, index_d = profile['head_dim'], profile['index_head_dim']
    window = profile['layers'] * profile['window_size'] * d * 2
    pools = index_pools = pages = recurrent = 0
    for ratio in profile['kv_source_ratios']:
        rows = ceil(ceil(context / ratio) / 16) * 16
        pools += rows * d * 2
        index_pools += rows * index_d * 2
        pages += rows / 16 * 8 * 2
        if ratio > 1:
            recurrent += 2 * ratio * d * 4
    categories = {'sliding_window_kv': window, 'shared_compressed_kv': pools,
                  'shared_index_cache': index_pools, 'page_table': pages,
                  'compressor_state': recurrent, 'shared_rope_table': context * profile['rope_head_dim'] * 4}
    return {'categoriesbytes': categories, 'categories_bytes': categories,
            'total_bytes': sum(categories.values()),
            'scope': 'all40windows_plus4sharedKV/indexowners_per_sequence_perTP_rank; shared caches counted once'}


def attention_work(profile, kind, model, context, tp, phase):
    if phase != 'decode':
        raise ValueError('V4.1 mechanism-template main experiment is warm decode only; CED TTFT is not implemented')
    if kind not in profile['counts']:
        raise ValueError('Unknown V4.1 attention kind')
    d, heads = profile['head_dim'], profile['num_heads'] / tp
    ratio = 2 if 'ENCODER' in kind else 1
    pairs = min(context + 1, profile['window_size'])
    if kind != 'SWA':
        pairs += min(profile['index_topk'], (context + 1) // ratio)
    keys = (context + 1) // ratio if 'INDEX' in kind else 0
    if 'CANDIDATE' in kind:
        keys = min(keys, profile['candidate_rows'])
    index_heads = profile['index_num_heads'] / tp
    index_d = profile['index_head_dim']
    # Index rows are FP16 in this controlled comparison, not source FP8.
    index_flops = 2 * keys * index_heads * index_d
    index_memory = keys * index_d * 2 + keys * 4 + index_heads * index_d * 2
    selection = keys * (3 * index_heads + 9 + 2)
    selection_memory = keys * 8 + profile['index_topk'] * 4 + profile['candidate_rows'] / 8 * 4 if keys else 0
    # Every layer updates its rolling window; only four owners append shared
    # pools. Boundary-dependent compressor writes are charged on source kinds.
    shared_emit = int('FULL_INDEX' in kind or ('ENCODER_INDEX' in kind and (context + 1) % 2 == 0))
    cache_write = d * 2 + shared_emit * (d + index_d) * 2
    return {'query_tokens': 1, 'kind': kind, 'kernel_name': kind.lower(),
            'extra_compute_flops': 4 * pairs * heads * d,
            'extra_fp32_flops': pairs * heads * 5 + (heads + 1) * d * 10,
            'memory_bytes': 2 * pairs * d * 2 + heads * d * 2,
            'fp32_memory_bytes': (heads + 1) * d * 2,
            'cache_write_bytes': cache_write,
            'workspace_bytes': min(keys, profile['score_tile_rows']) * 4 + (profile['index_topk'] + profile['candidate_rows'] / 8) * 4,
            'index_reduce_bytes': keys * 4,
            'index_reduce_repeats': ceil(keys / profile['score_tile_rows']) if keys else 0,
            'index_flops': index_flops, 'index_memory_bytes': index_memory if keys else 0,
            'index_selection_flops': selection, 'index_selection_memory_bytes': selection_memory,
            'cache_conversion_flops': 0., 'cache_raw_bytes': 0., 'cache_storage_bytes': 0.,
            'index_visible_rows': keys, 'sparse_attention_pairs': pairs,
            'precision_note': 'FP8 projection/dot compute and FP16 persistent KV/index; FP32 scores, selection and softmax. No native FP4 or DSpark.'}
