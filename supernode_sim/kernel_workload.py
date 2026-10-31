"""Compact workload identities for phase- and shape-scoped operator evidence.

These describe the compiler inputs, not new serving or attention algorithms.
In training the sequence population is one DP microbatch, before EP division;
in inference it is the live requests owned by an EP source rank. MoE balanced
expert work and SP padding can make operator_tokens differ from query_tokens.
"""
from __future__ import annotations

import hashlib
import json


def kernel_workload_shape(*, phase: str, tokens: float, lengths: list[int],
                          query_lengths: list[int] | None = None,
                          training: bool = False, ep: int = 1) -> dict:
    if query_lengths is None:
        if phase == 'verify':
            raise ValueError('verify workload shape requires explicit query lengths')
        query_lengths = [1] * len(lengths) if phase == 'decode' else list(lengths)
    if len(query_lengths) != len(lengths) or any(
            type(s) is not int or type(q) is not int or not 0 < q <= s
            for s, q in zip(lengths, query_lengths)):
        raise ValueError('workload shape requires matching positive query/context lengths with query <= context')
    if type(ep) is not int or ep < 1:
        raise ValueError('workload shape requires a positive integer EP degree')
    pairs = sum(q * (s - q) + q * (q + 1) // 2 for s, q in zip(lengths, query_lengths))
    signature = hashlib.sha256(json.dumps(list(zip(lengths, query_lengths)),
        separators=(',', ':')).encode('utf-8')).hexdigest()
    return {
        'version': 1, 'phase': phase, 'operator_tokens': tokens,
        'sequence_scope': 'dp_microbatch_before_ep' if training else 'ep_source_live_requests',
        'source_batch_size': len(lengths), 'context_tokens': sum(lengths),
        'query_tokens': sum(query_lengths),
        'kv_cache_tokens_before': sum(s - q for s, q in zip(lengths, query_lengths)),
        'context_length_squared_sum': sum(s * s for s in lengths),
        'query_length_squared_sum': sum(q * q for q in query_lengths),
        'context_length_min': min(lengths, default=0),
        'context_length_max': max(lengths, default=0),
        'dense_causal_pairs_reference': pairs,
        'training_ep_divisor': ep if training else 1,
        'sequence_signature': 'sequence-shape-v1:sha256:' + signature,
    }
