#!/usr/bin/env python3
"""Inspect operators or bind one NPUSim CA summary to an explicit suboperator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if (ROOT / 'supernode_sim').is_dir():
    sys.path.insert(0, str(ROOT))

from supernode_sim.config import load_config
from supernode_sim.npu_sim import analyze_summary, build_calibration, list_targets


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path, nargs='?')
    parser.add_argument('--list', action='store_true', help='List expanded operators with implementation identities')
    parser.add_argument('--analyze', action='store_true', help='Inspect a summary without creating a timing profile')
    parser.add_argument('--summary', type=Path, help='One explicit kernel_*_reports/summary.json')
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--limit', type=int, default=20)
    parser.add_argument('-o', '--output', type=Path)
    parser.add_argument('--report-output', type=Path)
    args = parser.parse_args(argv)
    try:
        report = None
        if args.list:
            if not args.config or args.analyze or args.summary or args.manifest or args.report_output or args.limit < 1:
                raise ValueError('--list requires config and a positive --limit, without import/analyze options')
            targets = list_targets(load_config(args.config))
            output = {'total_matching_events': len(targets), 'events': targets[:args.limit],
                      'truncated': len(targets) > args.limit, 'timing_profiles_applied': False}
        elif args.analyze:
            if not args.summary or args.config or args.manifest or args.report_output:
                raise ValueError('--analyze requires only --summary and optional --output')
            output = analyze_summary(json.loads(args.summary.read_text(encoding='utf-8-sig')))
        else:
            if not args.config or not args.summary or not args.manifest:
                raise ValueError('Import requires config, --summary and --manifest')
            built = build_calibration(load_config(args.config), args.summary,
                                     json.loads(args.manifest.read_text(encoding='utf-8-sig')))
            output, report = built['calibration'], built['report']
        # Prevent accidental truncation of input evidence or target config.
        report_path = args.report_output
        if report is not None and args.output and report_path is None:
            report_path = args.output.with_suffix('.report.json')
        destinations = [p.resolve() for p in (args.output, report_path) if p is not None]
        inputs = {p.resolve() for p in (args.config, args.summary, args.manifest) if p is not None}
        if len(set(destinations)) != len(destinations) or any(p in inputs for p in destinations):
            raise ValueError('Output paths must be distinct and must not overwrite inputs')
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        else:
            print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))
        if report_path and report is not None:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        return 0
    except (OSError, ValueError) as error:
        parser.exit(2, f'npusim calibration: {error}\n')


if __name__ == '__main__':
    raise SystemExit(main())
