"""Run synthetic task examples and cost sensitivities; never a hardware accuracy claim."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if (ROOT / 'supernode_sim').is_dir():
    sys.path.insert(0, str(ROOT))

from supernode_sim.analysis import run
from supernode_sim.config import Config, load_config
from supernode_sim.report import write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, default=Path('outputs/training-tasks'))
    args = parser.parse_args()
    configs = {name: load_config(args.project_root / 'examples' / f'training_task_{name}.yaml')
               for name in ('pretrain', 'sft', 'dpo', 'grpo', 'ppo')}
    variants = [('sft_padded', 'sft', lambda t: t['data'].update(batching='padded')),
                ('sft_fewer_labels', 'sft', lambda t: t['sft'].update(supervised_tokens=[1, 1])),
                ('dpo_precomputed', 'dpo', lambda t: t['dpo'].update(reference_log_probs='precomputed')),
                ('grpo_one_epoch', 'grpo', lambda t: t['rl'].update(update_epochs=1))]
    for name, base, modify in variants:
        data = configs[base].model_dump()
        modify(data['workload']['training'])
        configs[name] = Config.model_validate(data)
    rows = {}
    for name, config in configs.items():
        result = run(config)
        write_report(result, args.output / name)
        rows[name] = {'status': result['status'], 'metrics': result['metrics'],
                      'memory_peaks_bytes': result.get('simulation', {}).get('memory_peaks_bytes'),
                      'report': str(args.output / name / 'report.md')}
        print(f"{name}: {result['status']}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'comparison.json').write_text(json.dumps({
        'evidence': 'synthetic', 'accuracy_claim': None,
        'purpose': 'Task accounting, sensitivity and reproducibility; not calibrated platform performance.',
        'cases': rows}, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    if any(row['status'] != 'feasible' for row in rows.values()):
        raise SystemExit('Some cases failed; inspect their reports before interpreting comparisons.')


if __name__ == '__main__':
    main()
