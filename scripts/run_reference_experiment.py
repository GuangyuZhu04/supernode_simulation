"""Run/resume a large reference with bounded report memory and progress output.

Checkpoints are gzip JSON Lines, never executable pickle. Physics checkpoints
are committed only after both record files close and are tied to config/code.
"""
from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STREAM_THRESHOLD = 50_000
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from supernode_sim import analysis
from supernode_sim.config import load_config
from supernode_sim.report import write_report


class JSONLRecords:
    def __init__(self, path: Path, count: int):
        self.path, self.count = path, count

    def __len__(self):
        return self.count

    def __iter__(self):
        count = 0
        with gzip.open(self.path, 'rt', encoding='utf-8') as stream:
            for line in stream:
                count += 1
                yield json.loads(line)
        if count != self.count:
            raise ValueError(f'Checkpoint record count mismatch: {self.path}')


def save_records(path: Path, records, *, compresslevel: int = 6):
    temporary = path.with_suffix(path.suffix + '.tmp')
    count = 0
    try:
        with gzip.open(temporary, 'wt', encoding='utf-8', compresslevel=compresslevel) as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')
                count += 1
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return count


def fingerprint(config):
    snapshot = config.model_dump()
    snapshot['calibration'] = None
    digest = hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode())
    for name in ('core', 'config', 'features', 'components', 'sparse_attention', 'speculative', 'megakernel', 'disaggregated',
                 'structured', 'training', 'inference', 'topology', 'engine', 'calibration', 'cache'):
        digest.update((ROOT / 'supernode_sim' / (name + '.py')).read_bytes())
    return digest.hexdigest()


def load_checkpoint(directory: Path, expected: str):
    metadata = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    if metadata['fingerprint'] != expected:
        raise ValueError('Checkpoint configuration or numerical engine differs; choose a new output directory')
    result = metadata['simulation']
    streamed = max(metadata['trace_count'], metadata['allocation_count'], metadata['memory_log_count']) > STREAM_THRESHOLD
    for key in ('trace', 'memory_log'):
        path = directory / (key + '.jsonl.gz')
        if path.stat().st_size != metadata['file_sizes'][key]:
            raise ValueError(f'Checkpoint file size mismatch: {path}')
        records = JSONLRecords(path, metadata[key + '_count'])
        result[key] = records if streamed else list(records)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--output', '-o', type=Path, required=True)
    parser.add_argument('--full-detail', action='store_true',
                        help='Retain all payloads and tensor rows; use a separate full-detail checkpoint')
    parser.add_argument('--checkpoint-compression', type=int, choices=range(1, 10), default=6,
                        help='gzip compression level for new checkpoints (default: 6; use 1 for faster writes)')
    args = parser.parse_args(argv)
    config = load_config(args.config)
    key = fingerprint(config)
    # A lean checkpoint cannot restore omitted event payloads. Keep full-detail
    # checkpoints separate while still loading legacy/default paths unchanged.
    directory = args.output / 'checkpoint' / (key[:20] + ('-full' if args.full_detail else ''))
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    def progress(stage, **values):
        print(json.dumps({'stage': stage, 'elapsed_s': round(time.perf_counter() - started, 2), **values}, ensure_ascii=False), flush=True)

    original = {name: getattr(analysis, name) for name in ('compile_graph', 'graph_estimate', 'simulate_graph', 'build_memory_breakdown')}

    def compile_graph(spec):
        progress('compile_started')
        graph = original['compile_graph'](spec)
        progress('compile_complete', events=len(graph.events), allocations=len(graph.initial_allocations) + sum(len(e.allocs) for e in graph.events))
        return graph

    def graph_estimate(graph):
        result = original['graph_estimate'](graph)
        progress('analytical_bounds_complete')
        return result

    def simulate_graph(graph, **kwargs):
        if (directory / 'manifest.json').exists():
            progress('loading_completed_simulation_checkpoint')
            return load_checkpoint(directory, key)
        progress('simulation_started')
        result = original['simulate_graph'](graph, **kwargs)
        progress('simulation_complete', status=result['status'], duration_s=result['duration_s'])
        metadata = {'fingerprint': key, 'allocation_count': len(graph.initial_allocations) + sum(len(e.allocs) for e in graph.events),
                    'simulation': {k: v for k, v in result.items() if k not in ('trace', 'memory_log')}, 'file_sizes': {}}
        for field in ('trace', 'memory_log'):
            path = directory / (field + '.jsonl.gz')
            metadata[field + '_count'] = save_records(path, result.pop(field), compresslevel=args.checkpoint_compression)
            metadata['file_sizes'][field] = path.stat().st_size
            progress('checkpoint_records_saved', records=field, count=metadata[field + '_count'])
        temporary = directory / 'manifest.json.tmp'
        with temporary.open('w', encoding='utf-8') as stream:
            json.dump(metadata, stream, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
        temporary.replace(directory / 'manifest.json')
        progress('checkpoint_committed')
        return load_checkpoint(directory, key)

    def breakdown(graph, simulation=None, estimate=None, **kwargs):
        progress('memory_breakdown_started')
        if not args.full_detail and simulation and isinstance(simulation.get('memory_log'), JSONLRecords):
            kwargs['include_ephemeral_details'] = False
        result = original['build_memory_breakdown'](graph, simulation, estimate, **kwargs)
        progress('memory_breakdown_complete', retained_tensors=len(result['tensors']))
        return result

    analysis.compile_graph, analysis.graph_estimate = compile_graph, graph_estimate
    analysis.simulate_graph, analysis.build_memory_breakdown = simulate_graph, breakdown
    enabled = gc.isenabled()
    gc.disable()
    try:
        result = analysis.run(config, 'simulate', full_detail=args.full_detail)
        progress('export_started', status=result['status'])
        output = write_report(result, args.output, full_detail=args.full_detail)
        progress('complete', status=result['status'], output=str(output), metrics=result['metrics'])
        return 0 if result['status'] == 'feasible' else 3
    finally:
        for name, value in original.items():
            setattr(analysis, name, value)
        if enabled:
            gc.enable()


if __name__ == '__main__':
    raise SystemExit(main())
