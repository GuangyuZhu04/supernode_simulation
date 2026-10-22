"""Reproduce the DSK 950DT matrix with the existing Kimi 10T design and DSpark.

Run serially. Exact component-boundary streaming conserves the native graph;
no model weights are loaded and no hardware measurements are implied.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import csv
from datetime import datetime, timezone, timedelta
import gc
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from supernode_sim.config import Config, load_config
from supernode_sim.cache import BoundedCache
from supernode_sim.components import attention_state, parameter_inventory
from supernode_sim.core import DTYPE_BYTES
from supernode_sim.streaming import run_streaming

REFERENCE = ROOT / 'outputs/dsk-10t-950dt-full/dsk-10t-950dt-full'
MODEL_SOURCE = ROOT / 'outputs/kimi-k3-10t-1m-acceleration-20261019/configs/950dt-F_kv-b1-k32-o8.json'
DEFAULT_OUTPUT = ROOT / 'outputs/kimi-k3-10t-950dt-dspark'


def read(file):
    return json.loads(Path(file).read_text(encoding='utf-8'))


def save(file, value):
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    temporary = file.with_suffix(file.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    temporary.replace(file)


def table(file, rows):
    file.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with file.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def code_hashes():
    files = sorted((ROOT / 'supernode_sim').glob('*.py')) + [Path(__file__)]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def dspark_budget(raw, acceptance=4, multiplier=1):
    """Declared recurrent 1B draft resource proxy, not a released checkpoint.

    Scalar budgets cover ceil(B/EP) on every rank. This deliberately overcharges
    idle/smaller lanes; target KV/KDA ownership remains exact.
    """
    b, ep, tp = raw['workload']['batch_size'], raw['execution']['ep'], raw['execution']['tp']
    local = (b + ep - 1) // ep
    length = raw['workload']['input_length']
    draft_weights, draft_state = 1_031_250_000, 64 * 1024**2 * local
    recurrent, conv = 0, 0
    for layer in range(raw['model']['layers']):
        state = attention_state(raw['model'], layer, local, length, tp=tp)
        if state['kind'] == 'recurrent':
            recurrent += state['recurrent_elements']
            conv += state['convolution_elements']
    state_bytes = recurrent * 4 + conv  # FP32 recurrent / FP8 convolution.
    source = {'status': 'assumed', 'description':
        'Explicit analytical resource budget, not a measured DSpark/Kimi draft. '
        '1B recurrent draft, 1.03125 GB FP8 weights+scales/rank, 64 MiB state/local request; '
        'ceil(B/EP) budget charged on all ranks. Target replay uses stored projections/state proxy.'}
    def cost(flops, memory):
        return {'flops': flops * multiplier, 'memory_bytes': memory * multiplier,
                'startup_s': 2e-6 * multiplier, 'source': source}
    return {'enabled': True, 'method': 'dspark', 'draft_tokens': 7,
        'accepted_length': acceptance,
        'acceptance_source': {'status': 'assumed', 'description':
            f'User accepted an explicit assumption: up to {acceptance} committed tokens/round, '
            'including bonus/replacement. No measured acceptance or quality guarantee.'},
        'draft': {'resident_bytes_per_rank': draft_weights + draft_state + 256_000_000,
            'workspace_bytes_per_rank': 128_000_000,
            'prefilled': False, 'cost_unit': 'per_step',
            'cost': cost(2e9 * local, draft_weights + 2 * draft_state),
            'prefill_cost': cost(2e9 * local * length,
                draft_weights + 4 * 24 * 2048 * local * length + 2 * draft_state)},
        'commit_cost': cost(2 * 163840 * 8 * local, 4 * 163840 * 8 * local),
        'replay_cost': cost(8 * recurrent * acceptance, 2 * state_bytes * acceptance),
        'snapshot_bytes_per_rank': 0}


def prepare(directory):
    base = read(MODEL_SOURCE)
    reference = read(REFERENCE / 'manifest.json')
    items = []
    for original in reference['cases']:
        for variant in ('dspark', 'baseline'):
            items.append({**original, 'reference_case': original['name'],
                'name': original['name'] + ('' if variant == 'dspark' else '-baseline'),
                'variant': variant, 'accepted_length': 4 if variant == 'dspark' else None,
                'output_length': 8, 'cost_multiplier': 1,
                'group': 'main' if original['name'][-1] not in ('t', 'u') else 'topology_sensitivity'})
    # Reference primary cases are the first 72; use membership, not name parsing.
    main_names = {item['name'] for item in reference['cases'][:72]}
    for item in items:
        item['group'] = 'main' if item['reference_case'] in main_names else 'topology_sensitivity'
    for name in ('n128-tp8-ep16-b32-128K', 'n256-tp8-ep32-b32-128K'):
        original = next(item for item in reference['cases'] if item['name'] == name)
        for suffix, variant, accepted, output, factor in (
            ('accept2', 'dspark', 2, 8, 1), ('accept8', 'dspark', 8, 8, 1),
            ('output33', 'dspark', 4, 33, 1), ('output33-baseline', 'baseline', None, 33, 1),
            ('draft-cost2x', 'dspark', 4, 8, 2)):
            items.append({**original, 'reference_case': name, 'name': name + '-' + suffix,
                'variant': variant, 'accepted_length': accepted, 'output_length': output,
                'cost_multiplier': factor, 'group': 'dspark_sensitivity'})
    source_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in [MODEL_SOURCE, REFERENCE / 'manifest.json']}
    for item in items:
        original = read(REFERENCE / 'configs' / (item['reference_case'] + '.json'))
        raw = copy.deepcopy(base)
        raw['calibration'] = None
        raw['hardware'] = copy.deepcopy(original['hardware'])
        raw['execution'].update(tp=item['tp'], ep=item['ep'], pp=1, dp=1, fsdp=1,
            sequence_parallel=item['tp'] > 1, rank_order=[], all_to_all_algorithm=item['algorithm'])
        raw['workload'] = {'scenario': 'inference', 'batch_size': item['batch'],
            'input_length': item['length'], 'output_length': item['output_length']}
        raw['targets'] = {'tpot_s': 0.03}
        raw['sources'] = {
            'model': {'status': 'user', 'description': 'Reuse previous Kimi 10T synthetic design exactly.', 'uri': str(MODEL_SOURCE)},
            'hardware': {'status': 'user', 'description': 'All explicit hardware fields copied unchanged from this reference case.',
                'uri': str(REFERENCE / 'configs' / (item['reference_case'] + '.json'))},
            'execution.precision': {'status': 'assumed', 'description': 'Reuse prior F_kv FP8 weights/compute/activation/communication/KV; FP32 recurrent. Compressed MLA; SP when TP>1.'},
            'execution.features.speculative_decode': {'status': 'assumed', 'description': 'Explicit user-accepted draft/acceptance assumptions; see dspark-assumptions.json.'}}
        raw['assumptions'] = [
            'Synthetic Kimi K3 text expansion: 10.171385035872T, 3328 experts, top32, 93 layers (69 KDA + 24 MLA), 92 MoE + 1 Dense; AttnRes retained. No vision/projector/MTP.',
            'Hardware/topology copied unchanged per reference case. No weight download or real NPU benchmark.',
            'Prior F_kv precision retained; uniform FP8 is a study design, not the DSK mixed FP4 design or a quality-validated checkpoint.',
            'Global fixed batch B; EP lanes own whole requests round-robin. No context parallelism, chunked prefill, prefix reuse or arrival queue.',
            'DSpark draft7/acceptance assumption is a resource schedule, not an acceptance prediction or quality validation. Final round truncates to requested output count.',
            'Draft prompt preparation is included before target prefill. Warm decode excludes all prompt preparation and requires externally resident target/draft state.',
            'Full-request capacity failure remains failure even when an independent warm decode succeeds. Failure peaks are observed execution prefixes.',
            'Scalar draft/commit/replay budgets conservatively cover maximum local batch on all ranks, including idle EP lanes. Snapshot/KV ownership uses actual local counts.',
            'Direct all-to-all is a sensitivity with concurrent sends constrained by shared endpoints/links; no pairwise round barriers or exclusive tx/rx slots.',
            'TPOT target 30ms is copied from reference; no TTFT target or production P99 guarantee. All absolute timing is uncalibrated.',
        ]
        if item['variant'] == 'dspark':
            raw['execution']['features']['speculative_decode'] = dspark_budget(raw, item['accepted_length'], item['cost_multiplier'])
        else:
            raw['execution']['features']['speculative_decode'] = {'enabled': False}
        config = Config.model_validate(raw)
        assert config.model.model_dump() == Config.model_validate(base).model.model_dump()
        assert all(config.hardware.model_dump()[key] == value for key, value in original['hardware'].items()
                   if not isinstance(value, dict))
        item.update(config='configs/' + item['name'] + '.json', config_sha256=sha(config.model_dump()),
                    hardware_sha256=sha(config.hardware.model_dump()), model_sha256=sha(config.model.model_dump()))
        save(directory / item['config'], config.model_dump())
    manifest = {'study_date': '2026-10-21', 'timezone': 'Asia/Shanghai',
        'reference_directory': str(REFERENCE), 'model_source': str(MODEL_SOURCE),
        'source_sha256': source_hashes, 'cases': items,
        'counts': {'reference_workloads': 76, 'main_workloads': 72, 'paired_reference_runs': 152,
                   'additional_sensitivity_runs': 10, 'total_runs': len(items)}}
    save(directory / 'manifest.json', manifest)
    save(directory / 'model.design.config.json', base['model'])
    save(directory / 'dspark-assumptions.json', {'draft_model': 'hypothetical 1B recurrent resource proxy',
        'draft_tokens': 7, 'accepted_length': 4, 'acceptance_sensitivity': [2, 8],
        'draft_weights_with_scales_bytes_per_rank': 1_031_250_000,
        'draft_state_bytes_per_local_sequence': 64 * 1024**2,
        'draft_runtime_bytes_per_rank': 256_000_000, 'draft_workspace_bytes_per_rank': 128_000_000,
        'cost_formulas': 'See dspark_budget() and each resolved feature. All profile work is analytical, not measured.',
        'long_output_sensitivity': 33, 'cost_multiplier_sensitivity': 2,
        'short_window_note': 'Output8 leaves 7 decode tokens: accept4 executes verify7/commit4 and verify3/commit3; the configured draft7 budget is EOS-truncated.'})
    print(json.dumps({'stage': 'prepared_and_validated', **manifest['counts']}), flush=True)


def export_case(directory, config, result):
    directory.mkdir(parents=True, exist_ok=True)
    save(directory / 'config.resolved.json', config.model_dump())
    save(directory / 'result.json', result)
    table(directory / 'component_timeline.csv', result['chunks'])
    latency = []
    for metric in ('ttft', 'tpot'):
        record = result['metrics']['latency_breakdown'].get(metric)
        latency.append({'metric': metric, 'total_s': record['total_s'] if record else None,
            **(record['components_s'] if record else {}), 'available': record is not None})
    table(directory / 'latency_breakdown.csv', latency)
    table(directory / 'memory_peaks.csv', ({'pool': pool, 'peak_bytes': value,
        'final_bytes': result['simulation']['memory_final_bytes'].get(pool),
        'basis': 'complete_execution' if result['status'] == 'feasible' else 'observed_prefix',
        **result['simulation']['memory_categories_at_peak_bytes'].get(pool, {})}
        for pool, value in result['simulation']['memory_peaks_bytes'].items()))
    table(directory / 'resources.csv', ({'resource': key, 'work': value,
        'busy_s': result['simulation']['resource_busy_s'].get(key)}
        for key, value in result['simulation']['resource_work'].items() if value))
    parameters = []
    components = defaultdict(lambda: defaultdict(float))
    p = config.execution.precision
    f = config.execution.features
    for tensor in parameter_inventory(config.model.model_dump()):
        dtype = (p.routed_expert_weight if tensor.ep_shard else None) or p.weight
        count = tensor.local_elements(config.execution.tp, config.execution.ep)
        size = count * DTYPE_BYTES[dtype]
        from math import ceil
        scales = ceil(count / f.weight_scale_block) * f.scale_bytes if dtype in ('fp4', 'fp8', 'int8') and count else 0
        parameters.append({**tensor.to_dict(), 'shape': str(tensor.shape), 'weight_dtype': dtype,
            'weight_bytes_per_rank': size, 'scale_bytes_per_rank': scales,
            'weight_physical_bytes': size * config.hardware.npu_count})
        components[tensor.component]['logical_weight_bytes'] += tensor.accounted_elements * DTYPE_BYTES[dtype]
        components[tensor.component]['physical_weight_bytes'] += size * config.hardware.npu_count
        components[tensor.component]['physical_weight_scale_bytes'] += scales * config.hardware.npu_count
    table(directory / 'model_parameters.csv', parameters)
    table(directory / 'state_memory.csv', ({'component': c, **values, 'basis': 'static_weight_inventory_not_peak'}
        for c, values in components.items()))
    m = result['metrics']
    text = (f"# Kimi K3 10T / 950DT — {result['phase']}\n\n"
        f"执行状态：{result['status']}；TTFT：{m['ttft_s']} s；TPOT：{m['tpot_s']} s。\n\n"
        "全部 93 层按原生算子、通信与资源事件执行；component_timeline.csv 为完整组件边界时间线。"
        "宏缓存只复用完全匹配且内存可容纳的调度，不跳过模型层。\n\n"
        "失败峰值仅代表中止前观察值；warm-decode 不证明完整请求可行。"
        "DSpark 为显式假设驱动的 draft/verify/commit 资源模型，不是实测或模型质量验证。\n\n"
        "逐事件临时张量生命周期未保留；同一设备峰值时刻的状态构成见 memory_peaks.csv。\n")
    (directory / 'report.md').write_text(text, encoding='utf-8')


def run_cases(directory, selected):
    manifest = read(directory / 'manifest.json')
    hashes = code_hashes()
    cache = BoundedCache(max_bytes=32 * 1024**2, max_entries=256)
    completed = []
    for item in manifest['cases']:
        if selected and item['name'] not in selected:
            continue
        config = load_config(directory / item['config'])
        fingerprint = sha({'config': config.model_dump(), 'source': hashes})
        destination = directory / 'cases' / item['name']
        index_file = destination / 'case.json'
        if index_file.exists() and read(index_file).get('fingerprint') == fingerprint:
            print(json.dumps({'stage': 'reuse', 'case': item['name']}), flush=True)
            completed.append(item['name'])
            continue
        started = time.perf_counter()
        print(json.dumps({'stage': 'started', 'case': item['name']}), flush=True)
        def progress(chunk):
            if chunk['status'] != 'feasible' or '.l0.' in chunk['id'] or chunk['id'].startswith('infer.d0.token'):
                print(json.dumps({'case': item['name'], 'wall_s': round(time.perf_counter() - started, 2), **chunk}), flush=True)
        gc.disable()
        try:
            result = run_streaming(config, phase='all', cache=cache, progress=progress)
            export_case(destination / 'full', config, result)
            failed = result['status'] != 'feasible'
            full_metrics = result['metrics']
            full_status = result['status']
            del result
            gc.collect()
            warm_metrics = None
            if failed:
                warm = run_streaming(config, phase='decode', cache=cache, progress=progress)
                export_case(destination / 'warm-decode', config, warm)
                warm_metrics = warm['metrics']
                del warm
            wall = time.perf_counter() - started
            save(index_file, {'case': item, 'fingerprint': fingerprint, 'wall_s': wall,
                'full_status': full_status, 'metrics': full_metrics, 'warm_metrics': warm_metrics})
            completed.append(item['name'])
            save(directory / 'progress.json', {'last_completed': item['name'], 'completed_this_invocation': completed,
                'updated_at': datetime.now(timezone(timedelta(hours=8))).isoformat()})
            print(json.dumps({'stage': 'complete', 'case': item['name'], 'status': full_status,
                'ttft_s': full_metrics['ttft_s'], 'tpot_s': full_metrics['tpot_s'], 'wall_s': round(wall, 3)}), flush=True)
        finally:
            gc.enable()
            gc.collect()
    save(directory / 'reproducibility.json', {'study_date': '2026-10-21', 'python': sys.version,
        'platform': platform.platform(), 'source_sha256': hashes,
        'source_inputs': manifest['source_sha256'], 'execution': 'serial, 32 MiB bounded timing cache',
        'scope': 'Uncalibrated architecture/resource simulation, not real NPU measurements.',
        'command': 'python scripts/run_kimi_10t_950dt_dspark.py --prepare --run'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--case', action='append')
    args = parser.parse_args()
    if args.prepare:
        prepare(args.output)
    if args.run:
        run_cases(args.output, set(args.case or []))
