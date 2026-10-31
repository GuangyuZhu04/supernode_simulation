#!/usr/bin/env python3
"""Reproduce synthetic training/inference trace refinements; no accuracy claim."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_operator_trace import ROOT, aggregate_graph, build_calibration
from supernode_sim.analysis import run
from supernode_sim.config import Config, load_config
from supernode_sim.streaming import run_streaming


def save(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def inference_document(event):
    tokens = int(event.metadata['tokens'])
    operations = []
    for name, n, k in [('up', 32, 16), ('gate', 32, 16), ('down', 16, 32)]:
        operations.append({'kernel': f'dense.{name}', 'flops': 2 * tokens * n * k,
            'memory_bytes': max((tokens * k + k * n + tokens * n) * 2, event.work['npu0.hbm'] / 3),
            'matmul': {'m': tokens, 'n': n, 'k': k}, 'startup_s': 1e-6, 'workspace_bytes': 512})
    return {'id': 'synthetic-dense-' + event.metadata['phase'],
            'source': {'status': 'assumed', 'description':
                'Synthetic three-GEMM Dense proxy; bytes=max(operand reads/writes, baseline/3). '
                'Serial order, 1us launch and 512B workspace are assumptions, not measurements.'},
            'operations': operations}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, default=Path('outputs/operator-traces'))
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    base = load_config(args.project_root / 'examples/features/synthetic-training.yaml')
    rows = []
    for scenario in ('training', 'inference'):
        data = base.model_dump()
        if scenario == 'inference':
            data['workload'].update(scenario='inference', input_length=8, output_length=3,
                                    input_lengths=[8], output_lengths=[3])
            data['execution']['features'].update(optimizer_offload=False, activation_checkpointing=False)
        config = Config.model_validate(data)
        refined = config.model_dump()
        traces = []
        for event in aggregate_graph(config).events:
            if event.metadata.get('kernel') != 'dense':
                continue
            phase = event.metadata['phase']
            if scenario == 'training':
                if phase != 'backward':
                    continue
                document = json.loads((Path(__file__).resolve().parents[1] /
                                      'assets/operator-trace.synthetic.json').read_text(encoding='utf-8'))
            else:
                document = inference_document(event)
                # Decode steps have equal GEMM work but different context guards.
                document['id'] += f'-{len(traces)}'
            traces.extend(build_calibration(config, event.id, document)['hardware']['kernel_traces']['dense'])
        refined['hardware']['kernel_traces'] = {'dense': traces}
        refined['sources']['hardware.kernel_traces'] = {
            'status': 'assumed', 'description': 'Synthetic operator refinement; no measured accuracy claim.'}
        refined_config = Config.model_validate(refined)
        before, after = run(config), run(refined_config)
        save(args.output / f'{scenario}.baseline.config.json', config.model_dump())
        save(args.output / f'{scenario}.operators.config.json', refined_config.model_dump())
        save(args.output / f'{scenario}.baseline.result.json', before)
        save(args.output / f'{scenario}.operators.result.json', after)
        row = {'scenario': scenario, 'status_before': before['status'], 'status_after': after['status'],
            'duration_before_s': before['simulation']['duration_s'],
            'duration_after_s': after['simulation']['duration_s'],
            'metrics_before': before['metrics'], 'metrics_after': after['metrics'],
            'compute_flops_before': before['simulation']['resource_work']['npu0.compute'],
            'compute_flops_after': after['simulation']['resource_work']['npu0.compute'],
            'hbm_bytes_before': before['simulation']['resource_work']['npu0.hbm'],
            'hbm_bytes_after': after['simulation']['resource_work']['npu0.hbm'],
            'operator_trace': after['model_metadata']['operator_trace'],
            'hardware_performance': after['model_metadata']['fidelity']['hardware_performance']}
        if scenario == 'inference':
            streaming = run_streaming(refined_config)
            save(args.output / 'inference.streaming.result.json', streaming)
            row['streaming_duration_s'] = streaming['simulation']['duration_s']
            row['streaming_duration_abs_diff_s'] = abs(row['streaming_duration_s'] - row['duration_after_s'])
        rows.append(row)
    report = {'evidence_kind': 'synthetic', 'real_hardware_accuracy_claimed': False,
              'purpose': 'Check phase-specific refinement, conserved FLOPs, explicit bytes/startup and streaming parity.',
              'cases': rows}
    save(args.output / 'comparison.json', report)
    print(json.dumps({'output': str(args.output.resolve()), 'evidence_kind': 'synthetic',
        'cases': [{key: row[key] for key in ('scenario', 'status_after', 'duration_before_s', 'duration_after_s')}
                  for row in rows]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
