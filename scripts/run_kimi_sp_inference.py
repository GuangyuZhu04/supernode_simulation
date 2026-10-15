"""Repeat the archived Kimi 10.171T sweep with Megatron sequence parallelism.

SP shares the TP ranks: TP16/SP16/EP64 and TP32/SP32/EP32, both 1024 cards.
The previous TP8 results and topology remain archived in their original folder.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from supernode_sim.config import load_config, read_data
from run_kimi_10t_inference import BATCHES, CONTEXTS, TOPKS, run_cases, save


def prepare(directory, baseline):
    profile = read_data(baseline / 'hardware.profile.json')
    save(directory / 'hardware.profile.json', profile)
    manifest = {
        'contexts': CONTEXTS, 'batches': BATCHES, 'top_ks': TOPKS, 'sp_degrees': [16, 32],
        'total_cards': 1024, 'model_scope': 'same 3328-expert Kimi K3 text backbone as archived TP8 run',
        'baseline_directory': str(baseline.resolve()),
        'parallelism': [{'tp': degree, 'sp': degree, 'ep': 1024 // degree, 'pp': 1, 'dp': 1}
                        for degree in (16, 32)], 'cases': [],
    }
    baseline_manifest = read_data(baseline / 'manifest.json')
    manifest['model_counts'] = baseline_manifest['model_counts']
    group_index = 0
    for degree in (16, 32):
        ep = 1024 // degree
        for topk in TOPKS:
            for context, length in CONTEXTS.items():
                # Rotate batches so equally sized shards share expensive cases.
                offset = group_index % len(BATCHES)
                group_index += 1
                batches = BATCHES[offset:] + BATCHES[:offset]
                for batch in batches:
                    base_name = f'top{topk}-{context}-b{batch}'
                    raw = copy.deepcopy(read_data(baseline / 'configs' / f'{base_name}.json'))
                    raw['calibration'] = '../hardware.profile.json'
                    raw['execution'].update(tp=degree, ep=ep, pp=1, dp=1, fsdp=1,
                                            sequence_parallel=True)
                    raw['sources']['execution.parallelism'] = {
                        'status': 'user', 'description': f'User-confirmed Megatron SP: TP{degree}/SP{degree} share ranks, EP{ep}, PP1 DP1, total1024. SP is not an additional multiplicative axis. Contiguous TP group spans {degree//8} hosts within one64-card SU rack.',
                    }
                    raw['sources']['execution.sequence_parallel'] = {
                        'status': 'user', 'description': 'Megatron-style sequence parallelism confirmed by user. Norm/AttnRes sequence shards, all-gather before attention/FFN, reduce-scatter afterward; boundary embedding RS and selected-logit-input AG.',
                    }
                    raw['assumptions'] = [
                        'Same synthetic3328-expert,93-layer,10.171385035872T text backbone and user-confirmed FP8/FP16 hardware/precision as archived TP8 baseline; top-k remains32or64.',
                        f'Megatron SP{degree} shares TP{degree} ranks. World=TP{degree}*EP{ep}=1024, PP=DP=1. Every TP/SP group spans{degree//8} eight-card nodes within a64-card scale-up rack.',
                        'Norms, AttnRes FP32 workspace and retained residual blocks use ceil(valid_source_tokens/TP) rows per rank. Attention/FFN kernels use valid full-source tokens with tensor-sharded heads/matrices.',
                        'All-gather restores full padded hidden input before attention and FFN/MoE. Gathered activation buffer stays live through the component. Output-side reduce-scatter replaces the previous TP all-reduce.',
                        'Vocab-sharded embedding has a reduce-scatter boundary. Vocab-sharded output gathers only selected final hidden rows in prefill, not all input positions; no extra full-logit gather or sampling cost is modeled.',
                        'Equal-sized SP collectives pad source rows to a TP multiple. Prefill lengths are divisible; decode with one query per active EP lane uses16or32 padded rows. KV and KDA caches count valid sequences/tokens only.',
                        'Compressed MLA cache remains576FP16 elements/token/layer replicated over TP. KDA recurrent heads are tensor-sharded. This is sequence-parallel activation partitioning, not context-parallel KV/ring attention.',
                        f'Balanced fractional MoE dispatch retains full3584-dimensional FP16 latent on every TP lane over EP{ep}, using{ep-1} pairwise rounds. No DeepEP dispatch fusion or sparse token-routing trace is modeled.',
                        'Shared/router/latent MoE suboperators retain the existing balanced EP approximation. Hardware efficiency/kernel times remain uncalibrated.',
                        'Contexts128k/256k/512k/1M are131072/262144/524288/1048576 valid tokens per sequence. Output2 measures TTFT and one first-decode-step TPOT, with no arrival queue or cold-weight loading.',
                        'Prefill capacity failures remain failures. If necessary, separate decode seeds persistent KV/KDA state and does not imply end-to-end feasibility.',
                    ]
                    name = f'sp{degree}-{base_name}'
                    file = directory / 'configs' / f'{name}.json'
                    save(file, raw)
                    manifest['cases'].append({
                        'name': name, 'sp': degree, 'tp': degree, 'ep': ep, 'top_k': topk,
                        'context': context, 'input_length': length, 'batch_size': batch,
                        'config': str(file.relative_to(directory)), 'baseline_case': base_name,
                    })
    for item in manifest['cases']:
        config = load_config(directory / item['config'])
        assert config.execution.sequence_parallel
        assert config.execution.tp * config.execution.ep == 1024
        assert config.model.total_params == 10171385035872
    save(directory / 'manifest.json', manifest)
    print(json.dumps({'stage': 'prepared_and_validated', 'cases': len(manifest['cases']),
                      'parallelism': manifest['parallelism']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/kimi-k3-10t-sp')
    parser.add_argument('--baseline', type=Path, default=ROOT / 'outputs/kimi-k3-10t-1024')
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--case', action='append')
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--cache-mib', type=int, default=64,
                        help='Shared timing-cache accounting budget per worker in MiB; 0 disables caching')
    args = parser.parse_args()
    if args.shards < 1 or not 0 <= args.shard_index < args.shards:
        parser.error('require shards>=1 and0<=shard-index<shards')
    if args.cache_mib < 0:
        parser.error('require cache-mib >= 0')
    if args.prepare:
        prepare(args.output, args.baseline)
    if args.run:
        enabled = gc.isenabled()
        gc.disable()
        try:
            run_cases(args.output, set(args.case or []), args.shard_index, args.shards,
                      cache_mib=args.cache_mib)
        finally:
            if enabled:
                gc.enable()


if __name__ == '__main__':
    main()
