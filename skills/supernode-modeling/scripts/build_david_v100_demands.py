#!/usr/bin/env python3
"""Build opt-in DavidV100 analytical demands from explicit work budgets.

No hardware configuration or measured kernel profile is changed. The optional
operator document is accepted by build_operator_trace.py after scope checks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if (ROOT / 'supernode_sim').is_dir():
    sys.path.insert(0, str(ROOT))

from supernode_sim.david_v100 import build_demands, operator_document


def write_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workload', type=Path)
    parser.add_argument('-o', '--output', type=Path)
    parser.add_argument('--operators-output', type=Path)
    parser.add_argument('--kernel')
    parser.add_argument('--trace-id')
    args = parser.parse_args(argv)
    try:
        if bool(args.operators_output) != bool(args.kernel and args.trace_id):
            raise ValueError('--operators-output requires both --kernel and --trace-id')
        if not args.operators_output and (args.kernel or args.trace_id):
            raise ValueError('--kernel/--trace-id require --operators-output')
        paths = [path.resolve() for path in (args.workload, args.output, args.operators_output) if path]
        if len(set(paths)) != len(paths):
            raise ValueError('Input, report and operator output must be different files')
        result = build_demands(json.loads(args.workload.read_text(encoding='utf-8-sig')))
        operators = (operator_document(result, kernel=args.kernel, trace_id=args.trace_id)
                     if args.operators_output else None)
        if args.output:
            write_json(args.output, result)
        else:
            print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        if operators is not None:
            write_json(args.operators_output, operators)
        return 0
    except (OSError, ValueError) as error:
        parser.exit(2, f'DavidV100 demands: {error}\n')


if __name__ == '__main__':
    raise SystemExit(main())
