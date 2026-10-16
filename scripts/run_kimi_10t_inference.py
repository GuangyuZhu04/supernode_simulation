"""Reproducible Kimi-K3 expert-expansion inference experiment.

Run from the repository root with its .venv Python. All quantities are SI bytes
and seconds; context labels use binary token counts (128k = 131072).
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from supernode_sim.config import load_config, read_data

CONTEXTS = {'128k': 131072, '256k': 262144, '512k': 524288, '1M': 1048576}
BATCHES = (1, 2, 4, 8, 16, 32)
TOPKS = (32, 64)


def save(path, value, *, compact=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, indent=None if compact else 2,
                  separators=(',', ':') if compact else None,
                  ensure_ascii=False, allow_nan=False)
    tmp.replace(path)


def prepare(directory):
    base = read_data(ROOT / 'examples/kimi_k3_ascend950_fp8.yaml')
    topo = read_data(directory / 'topology/hardware.json')
    hw = read_data(ROOT / 'profiles/hardware/ascend950dt_64.yaml')['hardware']
    hw.update(topo['hardware'])
    save(directory / 'hardware.profile.json', {'hardware': hw, 'source': {
        'status': 'assumed', 'description': 'User-confirmed repository Ascend950DT baseline, compiled 1024-card topology; no measured calibration.',
    }})
    base['calibration'] = '../hardware.profile.json'
    base.pop('hardware', None)
    base['model']['name'] = 'Kimi-K3-text-10.171T-expert-expansion-FP8'
    base['model']['architecture']['moe']['num_experts'] = 3328
    base['model']['reported_total_params'] = 1e13
    base['model'].pop('reported_active_params', None)
    base['execution'].update(tp=8, ep=128, pp=1, dp=1, fsdp=1)
    base['sources'].update(topo.get('sources', {}))
    base['sources'].update({
        'model.architecture': {'status': 'public', 'description': 'Unchanged Kimi K3 backbone dimensions use the repository official-source snapshot. Synthetic num_experts and user top_k override those two fields as separately sourced.', 'uri': 'https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json', 'revision': 'c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721'},
        'model.architecture.moe.num_experts': {'status': 'assumed', 'description': '3328 is the nearest EP128-divisible expert count to 10T shape-derived total parameters, preserving other K3 dimensions.'},
        'model.architecture.moe.top_k': {'status': 'user', 'description': 'User requested separate top-32 and top-64 experiments.'},
        'model.reported_total_params': {'status': 'user', 'description': '10T expansion target, not an official Kimi checkpoint parameter count.'},
        'hardware.npu_count': {'status': 'user', 'description': '1024 cards total in one scale-out supernode.'},
        'hardware.su_size': {'status': 'user', 'description': '64 cards per scale-up rack, 16 racks total.'},
        'hardware.npus_per_host': {'status': 'user', 'description': '8 cards per node, 128 nodes total.'},
        'execution.parallelism': {'status': 'assumed', 'description': 'TP8 within each node, EP128 across nodes, PP1 DP1. Whole sequences placed round-robin on EP lanes; no context parallelism.'},
        'execution.precision': {'status': 'user', 'description': 'User confirmed repository baseline FP8 weights/compute and FP16 activation/communication/KV; KDA recurrent FP32.'},
        'workload': {'status': 'user', 'description': '4 contexts x 6 batches x 2 top-k; output_length=2 measures prefill/TTFT and one subsequent decode step. Binary context token counts are an explicit interpretation.'},
    })
    # Replace inherited short-case assumptions with experiment-specific facts.
    base['assumptions'] = [
        'Synthetic expert expansion of Kimi K3 text backbone; not an official released 10T model. 93 layers remain, 69 KDA and 24 MLA, 92 MoE layers.',
        'Contexts: 128k=131072, 256k=262144, 512k=524288, 1M=1048576 tokens per sequence.',
        'Preloaded weights. TTFT excludes arrival queue, cold loading and tokenizer/sampling costs; output_length=2 measures one first-decode-step TPOT, not steady multi-token or P99 latency.',
        'Uniform fractional expert/token routing with pairwise EP exchange over 127 rounds; small-batch decode startup can be pessimistic versus sparse token-specific routing.',
        '3328/128=26 complete experts per EP lane. 10T target rounds toapproximately10.171T because expert width/layers are fixed and EP128 requires expert divisibility.',
        'Compressed MLA cache is the optimized576-element/token/layer assumption. Expanded HF-reference cache capacity is separately reported; kernel performance is uncalibrated.',
        'No context parallelism or chunked prefill. Whole-sequence prefill retains AttnRes blocks and uses reference FP32 workspace; infeasible cases are preserved.',
        'Standalone decode, when prefill is infeasible, seeds resident KV/KDA state and reports its own feasibility; it does not imply end-to-end service feasibility.',
        'Hardware rates and efficiency assumptions inherited from the user-confirmed repository baseline; no hardware benchmark was supplied.',
    ] + topo.get('assumptions', [])
    manifest = {'contexts': CONTEXTS, 'batches': BATCHES, 'top_ks': TOPKS,
                'total_cards': 1024, 'parallelism': {'tp': 8, 'ep': 128, 'pp': 1, 'dp': 1},
                'model_scope': 'text backbone only; no vision/projector/training MTP', 'cases': []}
    for topk in TOPKS:
        for context, length in CONTEXTS.items():
            for batch in BATCHES:
                raw = copy.deepcopy(base)
                raw['model']['architecture']['moe']['top_k'] = topk
                raw['workload'] = {'scenario': 'inference', 'batch_size': batch,
                                   'input_length': length, 'output_length': 2}
                name = f'top{topk}-{context}-b{batch}'
                file = directory / 'configs' / f'{name}.json'
                save(file, raw)
                manifest['cases'].append({'name': name, 'top_k': topk, 'context': context,
                                          'input_length': length, 'batch_size': batch,
                                          'config': str(file.relative_to(directory))})
    # Full strict validation of every generated workload and shared topology.
    counts = {}
    for item in manifest['cases']:
        config = load_config(directory / item['config'])
        counts[str(item['top_k'])] = {'total_params': config.model.total_params,
                                     'active_params': config.model.active_params}
    manifest['model_counts'] = counts
    save(directory / 'manifest.json', manifest)
    print(json.dumps({'stage': 'prepared_and_validated', 'cases': len(manifest['cases']), 'counts': counts}), flush=True)
    return manifest


def fingerprint(config):
    digest = hashlib.sha256(json.dumps(config.model_dump(), sort_keys=True).encode())
    for module in ('core', 'config', 'analysis', 'structured', 'components', 'topology', 'engine',
                   'streaming', 'latency_breakdown', 'calibration', 'cache'):
        digest.update((ROOT / 'supernode_sim' / f'{module}.py').read_bytes())
    return digest.hexdigest()


def run_cases(directory, selected, shard_index=0, shards=1, *, cache_mib=64):
    from supernode_sim.cache import BoundedCache
    from supernode_sim.streaming import run_streaming
    manifest = read_data(directory / 'manifest.json')
    cache = BoundedCache(max_bytes=cache_mib * 1024 * 1024)
    started = time.perf_counter()
    for case_index, item in enumerate(manifest['cases']):
        if case_index % shards != shard_index:
            continue
        if selected and item['name'] not in selected:
            continue
        name = item['name']
        config = load_config(directory / item['config'])
        key = fingerprint(config)
        output = directory / 'cases' / f'{name}.json'
        if output.exists() and read_data(output).get('fingerprint') == key:
            print(json.dumps({'stage': 'reuse_completed_case', 'case': name}), flush=True)
            continue
        def progress(chunk):
            if chunk['status'] != 'feasible' or chunk['id'].endswith('token0') or chunk['id'].endswith('token1') or '.l0.' in chunk['id'] or '.l92.' in chunk['id']:
                print(json.dumps({'stage': 'chunk', 'case': name, 'wall_s': round(time.perf_counter()-started, 2), **chunk}), flush=True)
        print(json.dumps({'stage': 'case_started', 'case': name}), flush=True)
        # A shared exact schedule cache is optional in earlier module revisions.
        import inspect
        kwargs = {'progress': progress}
        if 'cache' in inspect.signature(run_streaming).parameters:
            kwargs['cache'] = cache
        result = run_streaming(config, phase='all', **kwargs)
        decode = None
        if result['status'] != 'feasible':
            decode = run_streaming(config, phase='decode', **kwargs)
        artifact = {'case': item, 'fingerprint': key, 'config_sha256': hashlib.sha256(json.dumps(config.model_dump(), sort_keys=True).encode()).hexdigest(),
                    'all': result, 'standalone_decode': decode}
        save(output, artifact, compact=True)
        print(json.dumps({'stage': 'case_complete', 'case': name, 'status': result['status'], 'metrics': result['metrics'],
                          'decode_metrics': decode['metrics'] if decode else None, 'wall_s': round(time.perf_counter()-started, 2)}), flush=True)
        # Do not retain the previous case during the next graph compilation.
        del artifact, result, decode, config
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/kimi-k3-10t-1024')
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--case', action='append')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--cache-mib', type=int, default=64,
                        help='Shared timing-cache accounting budget per worker in MiB; 0 disables caching')
    args = parser.parse_args()
    if args.shards < 1 or not 0 <= args.shard_index < args.shards:
        parser.error('require shards >= 1 and 0 <= shard-index < shards')
    if args.cache_mib < 0:
        parser.error('require cache-mib >= 0')
    if args.prepare:
        prepare(args.output)
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
