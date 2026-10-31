#!/usr/bin/env python3
"""Reproduce local refinements inspired by the verified reference simulators.

The examples are synthetic. This does not run an upstream simulator or compare
against hardware measurements, and cannot estimate real prediction error.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if (ROOT / 'supernode_sim').is_dir():
    sys.path.insert(0, str(ROOT))

from supernode_sim.analysis import run
from supernode_sim.config import Config, load_config, read_data
from supernode_sim.report import write_report
from supernode_sim.serving import simulate_serving
from supernode_sim.serving_report import write_serving_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, default=Path('outputs/reference-simulators'))
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    comparison = {'evidence_kind': 'synthetic', 'real_hardware_accuracy_claimed': False,
                  'training': [], 'serving': []}
    base = load_config(args.project_root / 'examples/features/synthetic-training.yaml').model_dump()
    for checkpoint in (False, True):
        data = deepcopy(base)
        data['execution']['features']['activation_checkpointing'] = checkpoint
        result = run(Config.model_validate(data))
        write_report(result, args.output / f'training-checkpoint-{int(checkpoint)}')
        comparison['training'].append({'checkpoint': checkpoint, 'status': result['status'],
            'step_time_s': result['metrics'].get('step_time_s'),
            'accounting': result['metrics'].get('training_accounting')})
    config = load_config(args.project_root / 'examples/serving_synthetic_model.yaml')
    serving = read_data(args.project_root / 'examples/serving_synthetic_requests.yaml')
    for instances in (1, 2):
        workload = deepcopy(serving)
        workload['instance_count'] = instances
        result = simulate_serving(config, workload)
        write_serving_report(result, args.output / f'serving-{instances}-instances')
        comparison['serving'].append({key: result[key] for key in
            ('status', 'hardware', 'metrics', 'throughput', 'goodput', 'time_window_s')})
    (args.output / 'comparison.json').write_text(json.dumps(comparison,
        ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(args.output.resolve()), 'evidence_kind': 'synthetic',
                      'training_cases': 2, 'serving_cases': 2}, ensure_ascii=False))


if __name__ == '__main__':
    main()
