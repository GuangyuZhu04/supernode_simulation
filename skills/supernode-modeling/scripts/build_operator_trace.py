#!/usr/bin/env python3
"""Bind caller-supplied serial operators to a structured component shape.

--list inspects aggregate, unprofiled kernels without executing the simulation.
Build input is JSON {id, source: Source, operations: [KernelTraceOperation]}.
The output is a calibration overlay, not measured evidence or a serving trace.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
if (ROOT / 'supernode_sim').is_dir():
    sys.path.insert(0, str(ROOT))

from supernode_sim.analysis import compile_graph
from supernode_sim.config import Config, KernelTrace, Source, load_config


def aggregate_graph(config: Config):
    data = config.model_dump()
    if data['hardware'].get('kernel_traces') or data['execution']['features']['megakernel']['enabled']:
        raise ValueError('Use an aggregate base configuration without kernel_traces or megakernel')
    # Profiling changes the timing resources, not the declared aggregate work.
    data['hardware']['kernel_profiles'] = {}
    return compile_graph(data)


def describe_event(event, config: Config) -> dict | None:
    # Layer IDs alone do not distinguish same-work, different-shape budget
    # kernels. Only the structured compiler currently exposes both identities.
    if not all(event.metadata.get(key) for key in ('model_shape_signature', 'workload_shape')):
        return None
    keys = [key for key in event.work if key.startswith('npu') and '.compute' in key]
    if not event.metadata.get('kernel') or not event.metadata.get('phase') or len(keys) != 1:
        return None
    key = keys[0]
    node = key.split('.')[0]
    if set(event.work) - {key, node + '.hbm'} or set(event.slots) - {node + '.compute_stream'}:
        return None
    metadata = {key: event.metadata[key] for key in (
        'component', 'layer', 'model_shape_signature', 'tokens', 'workload_shape',
        'tensor_parallel_degree', 'expert_parallel_degree', 'sequence_parallel_degree',
    ) if key in event.metadata}
    if not metadata:
        return None
    return {'event_id': event.id, 'kernel': event.metadata['kernel'],
        'phase': event.metadata['phase'],
        'dtype': 'fp32' if key.endswith('.fp32') else config.execution.precision.compute,
        'flops': event.work[key], 'memory_bytes': event.work.get(node + '.hbm', 0),
        'original_startup_s': event.startup_s, 'metadata_match': metadata}


def build_calibration(config: Config, event_id: str, document: dict) -> dict:
    if not isinstance(document, dict) or set(document) != {'id', 'source', 'operations'}:
        raise ValueError('Operator document must contain exactly id, source, operations')
    source = Source.model_validate(document['source'])
    event = next((event for event in aggregate_graph(config).events if event.id == event_id), None)
    if event is None:
        raise ValueError(f'Unknown aggregate event: {event_id}')
    descriptor = describe_event(event, config)
    if descriptor is None:
        raise ValueError('Selected event lacks a supported local compute/HBM shape identity; '
                         'model_shape_signature and workload_shape are required')
    trace = KernelTrace.model_validate({
        'id': document['id'], 'source': source.description + (f' ({source.uri})' if source.uri else ''),
        'hardware_name': config.hardware.name,
        'calibration_context': config.hardware.calibration_context,
        **{key: descriptor[key] for key in ('phase', 'dtype', 'flops', 'memory_bytes', 'metadata_match')},
        'operations': document['operations'],
    })
    return {'hardware': {'kernel_traces': {descriptor['kernel']: [trace.model_dump(exclude_none=True)]}},
            'source': source.model_dump(exclude_none=True)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--list', action='store_true', help='Inspect supported original kernel events')
    parser.add_argument('--kernel')
    parser.add_argument('--phase')
    parser.add_argument('--limit', type=int, default=20)
    parser.add_argument('--event-id')
    parser.add_argument('--operators', type=Path)
    parser.add_argument('-o', '--output', type=Path)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.list:
            if args.limit < 1 or args.event_id or args.operators:
                raise ValueError('--list requires a positive limit and no --event-id/--operators')
            rows = [row for event in aggregate_graph(config).events
                    if (row := describe_event(event, config)) is not None
                    and (args.kernel is None or row['kernel'] == args.kernel)
                    and (args.phase is None or row['phase'] == args.phase)]
            result = {'total_matching_events': len(rows), 'events': rows[:args.limit],
                      'truncated': len(rows) > args.limit, 'timing_profiles_applied': False}
        else:
            if not args.event_id or not args.operators or args.kernel or args.phase:
                raise ValueError('Build requires --event-id and --operators; --kernel/--phase filter only --list')
            result = build_calibration(config, args.event_id,
                json.loads(args.operators.read_text(encoding='utf-8-sig')))
        content = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(content, encoding='utf-8')
        else:
            print(content, end='')
        return 0
    except (OSError, ValueError) as error:
        parser.exit(2, f'operator trace: {error}\n')


if __name__ == '__main__':
    raise SystemExit(main())
