#!/usr/bin/env python3
"""Reproduce NPUSim import with synthetic data; requires no CANN installation."""
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
from supernode_sim.npu_sim import build_calibration


def evaluate(example: Path, output: Path) -> dict:
    config = load_config(example / 'synthetic.config.json')
    imported = build_calibration(config, example / 'summary.synthetic.json',
                                 json.loads((example / 'manifest.synthetic.json').read_text(encoding='utf-8')))
    data = config.model_dump()
    for kernel, entries in imported['calibration']['hardware']['kernel_profiles'].items():
        data['hardware']['kernel_profiles'].setdefault(kernel, []).extend(entries)
    refined = Config.model_validate(data)
    baseline = run(config, mode='simulate')
    profiled = run(refined, mode='simulate')
    before = baseline['simulation']
    after = profiled['simulation']
    comparison = {
        'evidence_kind': 'synthetic', 'hardware_accuracy_validated': False,
        'baseline_duration_s': before['duration_s'], 'profiled_duration_s': after['duration_s'],
        'imported_kernel_duration_s': imported['report']['duration_s'],
        'imported_profile_matched_events': imported['report']['matched_events'],
        'baseline_status': before['status'], 'profiled_status': after['status'],
        'hbm_work_preserved': {key: value for key, value in before['resource_work'].items() if key.endswith('.hbm')}
            == {key: value for key, value in after['resource_work'].items() if key.endswith('.hbm')},
        'memory_peaks_preserved': before['memory_peaks_bytes'] == after['memory_peaks_bytes'],
        'memory_final_preserved': before['memory_final_bytes'] == after['memory_final_bytes'],
        'fidelity': profiled['model_metadata']['fidelity'],
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, value in {
        'baseline.config.json': config.model_dump(), 'profiled.config.json': refined.model_dump(),
        'kernel.calibration.json': imported['calibration'], 'kernel.report.json': imported['report'],
        'baseline.result.json': baseline, 'profiled.result.json': profiled, 'comparison.json': comparison,
    }.items():
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return comparison


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--example', type=Path, default=ROOT / 'examples/npu_sim')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/npu-sim')
    args = parser.parse_args(argv)
    try:
        comparison = evaluate(args.example, args.output)
        print(json.dumps(comparison, indent=2, ensure_ascii=False, allow_nan=False))
        return 0 if comparison['profiled_status'] == 'feasible' else 3
    except (OSError, ValueError) as error:
        parser.exit(2, f'npusim evaluation: {error}\n')


if __name__ == '__main__':
    raise SystemExit(main())
