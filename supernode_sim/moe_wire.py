"""MoE wire precision and explicitly declared per-assignment metadata.

Uniform routing remains an expectation. Each routed token copy carries one
hidden vector and the supplied metadata bytes; no expert/rank deduplication,
FP8 scale layout, padding or codec implementation is inferred. Metadata is
copied unchanged when an activation is packed/unpacked, not converted as tensor
elements. Zero metadata is the backward-compatible tensor-only lower bound.
"""
from __future__ import annotations

from .core import DTYPE_BYTES


def moe_wire_metadata(precision: dict, operation: str, width: int, *, top_k: int | None = None,
                      num_experts: int | None = None) -> dict:
    dtype = precision.get(f'moe_{operation}') or precision['communication']
    extra = precision.get(f'moe_{operation}_metadata_bytes_per_token', 0)
    metadata = {'wire_dtype': dtype, 'buffer_dtype': dtype,
            'moe_hidden_width': width, 'moe_tensor_bytes_per_token': width * DTYPE_BYTES[dtype],
            'moe_metadata_bytes_per_token': extra,
            'moe_wire_metadata_policy': 'explicit per routed token copy; scales, indices and padding are not inferred'}
    if top_k is not None:
        metadata['moe_top_k'] = top_k
    if num_experts is not None:
        metadata['moe_num_experts'] = num_experts
    return metadata


def moe_token_bytes(precision: dict, operation: str, width: int) -> float:
    meta = moe_wire_metadata(precision, operation, width)
    return meta['moe_tensor_bytes_per_token'] + meta['moe_metadata_bytes_per_token']


def moe_payload_matrix(precision: dict, operation: str, source_tokens: list[float],
                       width: int, top_k: int) -> list[list[float]]:
    size = moe_token_bytes(precision, operation, width)
    n = len(source_tokens)
    matrix = [[tokens * top_k * size / n for _ in source_tokens] for tokens in source_tokens]
    return list(map(list, zip(*matrix))) if operation == 'combine' else matrix


def wire_payload_parts(payload: float, dtype: str, metadata: dict) -> tuple[float, float]:
    """Return tensor element count and unchanged metadata bytes in a payload."""
    extra_per_token = metadata.get('moe_metadata_bytes_per_token', 0)
    if extra_per_token:
        copies = payload / (metadata['moe_tensor_bytes_per_token'] + extra_per_token)
        return copies * metadata['moe_hidden_width'], copies * extra_per_token
    return payload / DTYPE_BYTES[dtype], 0.0
