"""Command line interface for reproducible scenario files."""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

from .analysis import run
from .config import Config, import_hf_config, load_config, read_data
from .feedback import FeedbackSpec, feedback
from .features import build_feature_plan, feature_catalog
from .report import write_report
from .search import size_su, sweep


class _ConsistentTPOT(argparse.Action):
    """Accept both spellings while rejecting conflicting explicit targets."""

    def __call__(self, parser, namespace, values, option_string=None):
        current = getattr(namespace, self.dest, None)
        if current is not None and current != values:
            parser.error('--tpot-s and --tpop-s must specify the same value when repeated')
        setattr(namespace, self.dest, values)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog='snmodel', description='超节点通信与资源数值建模')
    root.add_argument('--version', action='version', version='snmodel 0.1.0')
    sub = root.add_subparsers(dest='command', required=True)
    web = sub.add_parser('web', help='启动本地仿真可视化工作台')
    web.add_argument('--host', default='127.0.0.1')
    web.add_argument('--port', type=int, default=8765)
    web.add_argument('--output', '-o', type=Path, default=Path('outputs') / 'web')
    for name in ('validate', 'estimate', 'simulate', 'sweep', 'size-su', 'feedback'):
        p = sub.add_parser(name)
        p.add_argument('config', type=Path)
        p.add_argument('--features', type=Path, help='JSON/YAML feature object; conflicts with explicit canonical configuration are rejected')
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
        if name == 'feedback':
            p.add_argument('--search', type=Path, help='JSON/YAML feedback 搜索参数；显式命令行参数覆盖文件值')
            p.add_argument('--ttft-s', type=float, help='目标首 token 时延 / s；默认使用 config.targets.ttft_s')
            p.add_argument('--tpot-s', '--tpop-s', dest='tpot_s', type=float, action=_ConsistentTPOT,
                           help='目标每输出 token 时延 / s；TPOP 作为 TPOT 的兼容别名')
            p.add_argument('--prefill-lengths', type=int, nargs='+', help='Prefill 输入 token 长度候选')
            p.add_argument('--decode-batch-sizes', type=int, nargs='+', help='固定批次大小候选，联合执行 Prefill 和 Decode')
            p.add_argument('--output-length', type=int, help='输出 token 数（含首 token），至少为 2')
            p.add_argument('--max-candidates', type=int, help='允许评估的最大候选数')
            p.add_argument('--engine', choices=['auto', 'simulate', 'streaming'], help='候选仿真引擎')
    p = sub.add_parser('import-hf')
    p.add_argument('config', type=Path)
    p.add_argument('--overrides', type=Path)
    p.add_argument('--output', '-o', required=True, type=Path)
    p = sub.add_parser('schema')
    p.add_argument('--output', '-o', type=Path)
    p = sub.add_parser('list-features', help='List modeled options, required scopes and unsupported serving features')
    p.add_argument('--output', '-o', type=Path)
    return root


def _main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='backslashreplace')
    args = parser().parse_args(argv)
    try:
        if args.command == 'web':
            from .web import main as web_main
            return web_main(['--host', args.host, '--port', str(args.port), '--output', str(args.output)])
        if args.command in ('schema', 'list-features'):
            payload = Config.model_json_schema() if args.command == 'schema' else feature_catalog()
            data = json.dumps(payload, ensure_ascii=False, indent=2)
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
        config = load_config(args.config, features=args.features)
        if args.command == 'validate':
            print(json.dumps({'status': 'valid', 'schema_version': config.schema_version, 'model': config.model.name,
                              'feature_plan': build_feature_plan(config)}, ensure_ascii=False))
            return 0
        if args.command == 'sweep':
            result = sweep(config, read_data(args.axes), args.mode)
        elif args.command == 'size-su':
            result = size_su(config, args.mode)
        elif args.command == 'feedback':
            search = read_data(args.search) if args.search else {}
            if not isinstance(search, dict):
                raise ValueError('feedback --search must contain a JSON/YAML object')
            for key in ('ttft_s', 'tpot_s', 'prefill_lengths', 'decode_batch_sizes',
                        'output_length', 'max_candidates', 'engine'):
                value = getattr(args, key)
                if value is not None:
                    if key == 'tpot_s':
                        search.pop('tpop_s', None)
                    search[key] = value
            result = feedback(config, FeedbackSpec.model_validate(search))
        else:
            result = run(config, args.command, full_detail=args.full_detail)
        output = write_report(result, args.output, full_detail=getattr(args, 'full_detail', False))
        summary = {'status': result['status'], 'output': str(output), 'metrics': {k: v for k, v in result.get('metrics', {}).items() if k != 'requests'}}
        if args.command == 'size-su':
            summary['searches'] = {k: {'status': v['status'], 'minimum_su_size': v.get('minimum_su_size')} for k, v in result['searches'].items()}
        if args.command == 'feedback':
            best = result.get('best')
            summary.pop('metrics', None)
            summary.update(counts=result['counts'], search_complete=result['search_complete'],
                           best=({'id': best['id'], 'parameters': best['parameters'],
                                  'metrics': {key: best['metrics'].get(key) for key in
                                              ('ttft_s', 'tpot_s', 'tpop_s', 'latency_breakdown',
                                               'decode_throughput_tokens_s', 'decode_window_s')}}
                                 if best else None))
        print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
        return 2 if result['status'] in ('error', 'missing_input') else (3 if result['status'] in ('infeasible', 'slo_failed', 'no_feasible_candidate') else 0)
    except (ValueError, KeyError, OSError) as exc:
        print(json.dumps({'status': 'invalid_input', 'message': str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


def main(argv: list[str] | None = None) -> int:
    # HTTP handlers live for the server lifetime and can retain reference cycles.
    # Numerical web workers manage their own short-lived GC policy.
    arguments = sys.argv[1:] if argv is None else argv
    if arguments and arguments[0] == 'web':
        return _main(argv)
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
