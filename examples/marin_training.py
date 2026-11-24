"""Compare training work with the public Marin 535B-A23B run.

Run from the repository root with python -m examples.marin_training --help.
"""
import argparse
import json
from pathlib import Path

from supernode_sim.analysis import run
from supernode_sim.config import Config
from supernode_sim.moe_wire import moe_payload_matrix
from supernode_sim.report import write_report
from supernode_sim.structured import Builder
from supernode_sim.topology import Topology
from supernode_sim.training_attention import causal_pairs


TOKENS_PER_GPU = 65_536
PEAK = 2.5e15
RUN = 'https://github.com/marin-community/marin/issues/8435'
EP_BLOG = 'https://marin.community/blog/2026/09/29/expert-parallelism/'
WANDB = 'https://wandb.ai/marin-community/marin/reports/535B-A23B-18T-Token-Hero-Run-Scaling-Ladder--VmlldzoxNzc2MDM5Ng'
SOURCE = 'https://github.com/marin-community/marin/blob/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/'
FIXED = {'mode': 'fixed_capacity', 'sender_capacity_factor': 1.15, 'expert_capacity_factor': 1.15,
         'assignment_drop_fraction': 0.034}
RAGGED = {'mode': 'ragged', 'assignment_drop_fraction': 0.0}


def logger_flops_per_token(sequence):
    """Marin launch-revision analytic F+B counter for this published architecture."""
    hidden, heads, head_dim, intermediate, latent = 6144, 48, 128, 3072, 3072
    # Matrix MACs: latent down/up, selected experts, shared experts, router, output projection.
    common_macs = (2 * hidden * latent + 3 * intermediate * (8 * latent + 2 * hidden)
                   + hidden * 384 + hidden * hidden)
    forward = 2 * hidden * 128_256  # vocabulary readout
    for layer in range(48):
        local = sequence > 2048 and (layer + 1) % 4 != 0
        kv_heads = 12 if local or sequence <= 2048 else 6
        qkv_macs = hidden * (heads + 2 * kv_heads) * head_dim
        span = min(sequence, 2048) if local else sequence
        forward += 2 * (common_macs + qkv_macs) + heads * span * (4 * head_dim + 3)
    return 3 * forward


def configuration(sequence, routing, *, slice_run=False):
    layers, ep, dp = (4, 8, 2) if slice_run else (48, 64, 11)
    model = {
        'name': 'Marin local operator slice' if slice_run else 'Marin 535B-A23B training proxy',
        'layers': layers, 'hidden_dim': 6144, 'num_heads': 48, 'kv_heads': 12, 'head_dim': 128,
        'architecture': {'vocab_size': 128_256, 'attention_by_layer': ['GQA'] * layers,
            'ffn_by_layer': ['MoE'] * layers, 'moe': {'num_experts': 6 * ep, 'top_k': 8,
                'intermediate_dim': 3072, 'num_shared_experts': 2, 'latent_dim': 3072, 'latent_norm': True}}}
    execution = {'ep': ep, 'dp': dp, 'training_moe_routing': routing,
        'training_attention_windows': [2048, 2048, 2048, None] * (layers // 4),
        'training_active_kv_heads': [12, 12, 12, 6] * (layers // 4),
        'optimizer': {'kind': 'muon'},
        'features': {'activation_checkpointing': True, 'optimizer_offload': not slice_run},
        'precision': dict.fromkeys(('compute', 'weight', 'activation', 'communication', 'gradient', 'weight_transfer'), 'bf16')}
    if not slice_run:
        model.update(reported_total_params=535.3e9, reported_active_params=22.76e9)
        execution['training_mfu_reference'] = {'name': 'Marin GRUG 12d8b6f', 'dtype': 'bf16',
            'sequence_length': sequence, 'model_flops_per_token': logger_flops_per_token(sequence)}
    return Config.model_validate({'model': model, 'execution': execution,
        'hardware': {'name': 'GB200 analytical profile', 'npu_count': ep * dp, 'su_size': ep, 'npus_per_host': 4,
            'npu': {'hbm_bytes': 185 * 2**30, 'hbm_Bps': 8e12, 'efficiency': 0.60,
                    'flops': {'bf16': PEAK, 'fp32': 80e12}},
            'scale_up': {'bandwidth_Bps': 900e9, 'latency_s': 1e-6},
            'scale_out': {'bandwidth_Bps': 50e9, 'latency_s': 5e-6}},
        'workload': {'scenario': 'training', 'sequence_length': sequence,
                     'batch_size': ep * TOKENS_PER_GPU // sequence},
        'sources': {'model': {'status': 'public', 'description': 'Marin hero architecture and training mesh', 'uri': RUN},
                    'hardware.scale_up': {'status': 'public', 'description': 'Per-GPU one-direction physical peak', 'uri': EP_BLOG},
                    '*': {'status': 'assumed', 'description': 'Roofline efficiency, latency, checkpointing and optimizer placement/timing are proxies.'}}})


def representative_work(sequence, routing):
    """Count one symmetric GPU's native F+B operators at the full training shape."""
    cfg = configuration(sequence, routing).model_dump()
    builder = Builder(cfg, Topology(cfg['hardware'], cfg['execution']['precision']))
    for layer in range(builder.m['layers']):
        for component in builder.component_order(layer):
            for phase in ('forward', 'backward'):
                name = f'{layer}.{component}.{phase}'
                builder.kernel(name, 'npu0', layer, component, TOKENS_PER_GPU, (),
                               backward=phase == 'backward', phase=phase, lengths=builder.train_lengths)
    executed = sum(e.work.get('npu0.compute', 0) for e in builder.g.events)
    padding = sum(e.metadata.get('expert_padding_flops', 0) for e in builder.g.events)
    rows = [TOKENS_PER_GPU * builder.wire_factor] * builder.ep
    matrix = moe_payload_matrix(builder.p, 'dispatch', rows, builder.a['moe']['latent_dim'], builder.m['top_k'])
    pairs = sum(causal_pairs(sequence, window) for window in builder.windows)
    return {'sequence_length': sequence, 'mode': routing['mode'],
            'logger_fb_flops_per_token': logger_flops_per_token(sequence),
            'executed_fb_flops_per_gpu': executed, 'useful_fb_flops_per_gpu': executed - padding,
            'expert_padding_fb_flops_per_gpu': padding,
            'outbound_dispatch_bytes_per_gpu_per_layer': sum(matrix[0][1:]),
            'attention_pairs_fraction_of_full': pairs / (builder.m['layers'] * causal_pairs(sequence))}


def comparison():
    fixed = representative_work(4096, FIXED)
    cases = [fixed, *(representative_work(length, RAGGED) for length in (4096, 16384, 65536))]
    # One observed anchor. Only the physical FB work and logger numerator change.
    for case in cases:
        case['fixed_kernel_rate_mfu_pct'] = (21.5 * fixed['executed_fb_flops_per_gpu'] / case['executed_fb_flops_per_gpu']
            * case['logger_fb_flops_per_token'] / fixed['logger_fb_flops_per_token'])
    prediction = cases[1]['fixed_kernel_rate_mfu_pct']
    # A separately observed late 4K point includes intervening kernel optimizations.
    # Transfer this rate only across context lengths, keeping tokens per step fixed.
    final_projection = 29.0 * cases[-1]['fixed_kernel_rate_mfu_pct'] / cases[1]['fixed_kernel_rate_mfu_pct']
    return {'sources': {'run': RUN, 'ep': EP_BLOG, 'milestones': WANDB,
            'trainer': SOURCE + 'experiments/grug/moe_hero_ep/train.py',
            'logger': SOURCE + 'lib/levanter/src/levanter/utils/flop_utils.py',
            'peak': SOURCE + 'lib/fray/src/fray/device_flops.py',
            'midpoint': 'https://x.com/percyliang/status/2107502170191155633'},
        'mesh': {'active_gpus': 704, 'ep': 64, 'dp': 11, 'tokens_per_step': 704 * TOKENS_PER_GPU},
        'observed_mfu_pct': {'fixed_4k': 21.5, 'ragged_4k_step_81716': 23, 'fa4_and_mask_free_mlp_step_146139': 26,
            'midpoint_4k': 27, 'kernel_stack_4k_step_231464': 29, 'final_65k': 33},
        'full_shape_work': cases, 'matched_4k_holdout': {'anchor_pct': 21.5, 'observed_pct': 23,
            'old_undifferentiated_prediction_pct': 21.5, 'predicted_pct': prediction,
            'signed_error_pp': prediction - 23},
        'late_context_holdout': {'anchor_4k_pct': 29.0, 'observed_final_65k_pct': 33.0,
            'predicted_65k_pct': final_projection, 'signed_error_pp': final_projection - 33.0,
            'assumption': 'Same effective kernel rates from the late 4K stack through 65K. Other runtime changes may contribute.'},
        'basis': 'One fixed-EP production MFU anchors an assumed constant time per executed forward/backward FLOP. '
                 'Equal per-rank work under balanced routing. This projection has no independent optimizer, recompute, '
                 'communication or host timing. The first anchor evaluates the same-context ragged switch. '
                 '16K and 65K are counterfactuals with unchanged kernel rates.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='write comparison and slice metrics to JSON')
    parser.add_argument('--slice-reports', type=Path, help='also save normal simulator reports and charts for both slices')
    parser.add_argument('--full-config', type=Path, help='export the 704-GPU fixed 4K input as JSON')
    args = parser.parse_args()
    result = comparison()
    result['slice_runs'] = {}
    for routing in (FIXED, RAGGED):
        spec = configuration(4096, routing, slice_run=True)
        sim = run(spec)
        if sim['simulation']['status'] != 'feasible':
            raise ValueError(sim['simulation'].get('diagnostics'))
        summary = sim['training_summary']
        accounting = sim['metrics']['training_accounting']
        result['slice_runs'][routing['mode']] = {
            'layers': spec.model.layers, 'ep': spec.execution.ep, 'dp': spec.execution.dp,
            'tokens_per_gpu': TOKENS_PER_GPU, 'step_s': summary['duration_s'],
            'model_mfu': accounting['model_mfu'], 'executed_compute_utilization': accounting['executed_compute_utilization'],
            'npu_flops_by_role': accounting['npu_flops_by_role'], 'peak_hbm_bytes': summary['peak_hbm']['peak_bytes'],
            'rank_time_percent': {s['name']: s['percent'] for s in summary['groups']['rank_time']['segments']},
            'exposed_communication_ops_percent': {s['name']: s['percent'] for s in summary['groups']['exposed_ops']['segments']}}
        if args.slice_reports:
            write_report(sim, args.slice_reports / routing['mode'])
    if args.full_config:
        args.full_config.parent.mkdir(parents=True, exist_ok=True)
        args.full_config.write_text(json.dumps(configuration(4096, FIXED).model_dump(), indent=2), encoding='utf-8')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + chr(10), encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('matched_4k_holdout', 'late_context_holdout')}, indent=2))


if __name__ == '__main__':
    main()
