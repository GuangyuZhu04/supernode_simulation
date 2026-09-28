"""Command line interface for reproducible scenario files."""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

from .analysis import run
from .config import Config, import_hf_config, load_config, read_data
from .report import write_report
from .search import size_su, sweep


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog='snmodel', description='超节点通信与资源数值建模')
    root.add_argument('--version', action='version', version='snmodel 0.1.0')
    sub = root.add_subparsers(dest='command', required=True)
    for name in ('validate', 'estimate', 'simulate', 'sweep', 'size-su'):
        p = sub.add_parser(name)
        p.add_argument('config', type=Path)
        if name != 'validate':
            p.add_argument('--output', '-o', type=Path, default=Path('outputs') / name)
        if name in ('estimate', 'simulate'):
            p.add_argument('--full-detail', action='store_true',
                           help='导出全部原始事件、张量和内存日志；大图默认精简明细，保留完整数值汇总')
        if name == 'sweep':
            p.add_argument('--axes', required=True, type=Path, help='JSON/YAML mapping dotted paths to candidate lists')
            p.add_argument('--mode', choices=['estimate', 'simulate'], default='simulate')
        if name == 'size-su':
            p.add_argument('--mode', choices=['single', 'multi', 'both'], default='both')
    p = sub.add_parser('import-hf')
    p.add_argument('config', type=Path)
    p.add_argument('--overrides', type=Path)
    p.add_argument('--output', '-o', required=True, type=Path)
    p = sub.add_parser('schema')
    p.add_argument('--output', '-o', type=Path)
    return root


def _main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='backslashreplace')
    args = parser().parse_args(argv)
    try:
        if args.command == 'schema':
            data = json.dumps(Config.model_json_schema(), ensure_ascii=False, indent=2)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(data, encoding='utf-8')
                print(json.dumps({'status': 'written', 'path': str(args.output)}, ensure_ascii=False))
            else:
                print(data)
            return 0
        if args.command == 'import-hf':
            result = import_hf_config(args.config, read_data(args.overrides) if args.overrides else None)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps({'status': 'missing_input' if result['missing_fields'] else 'imported', 'missing_fields': result['missing_fields'], 'path': str(args.output)}, ensure_ascii=False))
            return 2 if result['missing_fields'] else 0
        config = load_config(args.config)
        if args.command == 'validate':
            print(json.dumps({'status': 'valid', 'schema_version': config.schema_version, 'model': config.model.name}, ensure_ascii=False))
            return 0
        if args.command == 'sweep':
            result = sweep(config, read_data(args.axes), args.mode)
        elif args.command == 'size-su':
            result = size_su(config, args.mode)
        else:
            result = run(config, args.command, full_detail=args.full_detail)
        output = write_report(result, args.output, full_detail=getattr(args, 'full_detail', False))
        summary = {'status': result['status'], 'output': str(output), 'metrics': {k: v for k, v in result.get('metrics', {}).items() if k != 'requests'}}
        if args.command == 'size-su':
            summary['searches'] = {k: {'status': v['status'], 'minimum_su_size': v.get('minimum_su_size')} for k, v in result['searches'].items()}
        print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
        return 2 if result['status'] in ('error', 'missing_input') else (3 if result['status'] in ('infeasible', 'slo_failed') else 0)
    except (ValueError, KeyError, OSError) as exc:
        print(json.dumps({'status': 'invalid_input', 'message': str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


def main(argv: list[str] | None = None) -> int:
    # Scenario graphs contain IDs rather than object back-links. Reference
    # counting reclaims these acyclic ledgers; repeated full cyclic-GC scans of
    # million-event graphs otherwise dominate command execution. Always restore
    # the caller's collector state, including parser and validation failures.
    enabled = gc.isenabled()
    gc.disable()
    try:
        return _main(argv)
    finally:
        if enabled:
            gc.enable()


if __name__ == '__main__':
    raise SystemExit(main())
