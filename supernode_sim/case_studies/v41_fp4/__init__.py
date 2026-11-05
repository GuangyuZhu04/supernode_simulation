"""The 2026-11-02 V4.1 FP4 POD/DFlash case, using the native scheduler.

This input contract is deliberately separate from the generic Config: the
original study uses CED, shared CSA2 state, and separate prefill/decode batches.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator


CASE_ID = 'deepseek_v41_fp4_512_ep128'
MODE = 'pod_dflash'
NAME = 'DeepSeek V4.1 Pro 10T · FP4 · 512 卡 / EP128'
CONTEXTS = {'1K': 1024, '4K': 4096, '16K': 16384, '32K': 32768,
            '64K': 65536, '128K': 131072, '256K': 262144, '512K': 524288,
            '1M': 1048576, '4M': 4194304}
NOTES = [
    '条件仿真：synthetic 64 层 / 7168 hidden / 1792 experts / top-24；未做实机或模型质量校准。',
    '仅目标路由专家 gate/up/down 权重为 FP4；每 32 个末轴元素含 1 byte scale。其他权重、FP8 计算、KV、通信和 DFlash 保持原实验设置。',
    '显式计入非融合 FP4→FP8 转换的读取、缓冲写入/读取、标量工作及临时容量。',
    'TTFT 含完整目标 prefill 与 DFlash KV 初始化；TPOT 为固定上下文的一轮 draft + verify + commit 耗时除以净接受 token 数。',
    'Prefill batch 固定为 1；Decode batch 是整个 512 卡部署的总量，DP 活跃副本和本地 batch 显式换算。',
    '单个逻辑 Full mesh SU POD；各方向独立，无 SO 流量。阶段串行，不含排队、continuous batching 或 P99。',
    '950DT 96 GB / 4 TB/s，950PR 128 GB / 1.6 TB/s；容量来自用户输入，带宽和效率沿用原实验假设。',
    '4M 是容量/流量外推，不代表模型上下文质量有效或可部署。',
]


class CaseConfig(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    case_id: Literal['deepseek_v41_fp4_512_ep128'] = CASE_ID
    chip: Literal['950dt', '950pr'] = '950dt'
    cards: Literal[512] = 512
    ep: Literal[128] = 128
    tp: Literal[1, 2, 4] = 4
    sp: Literal[1, 2, 4] = 1
    context_tokens: Literal[1024, 4096, 16384, 32768, 65536, 131072,
                            262144, 524288, 1048576, 4194304] = 131072
    prefill_batch: Literal[1] = 1
    decode_batch: Literal[1, 4, 8, 16, 32, 64] = 32
    accepted: Literal[3, 6, 9, 12] = 6
    block: Literal[4, 8, 12, 16] = 8

    @model_validator(mode='before')
    @classmethod
    def reject_boolean_numbers(cls, value):
        if isinstance(value, dict) and any(isinstance(v, bool) for v in value.values()):
            raise ValueError('案例数值字段不能是布尔值')
        return value

    @model_validator(mode='after')
    def supported_layout(self):
        if self.sp not in (1, self.tp):
            raise ValueError('SP 必须为 1 或 TP，不额外乘入总卡数')
        if (self.accepted, self.block) not in ((3, 4), (6, 8), (9, 12), (12, 16)):
            raise ValueError('接受数/块长必须是原实验的 3/4、6/8、9/12 或 12/16')
        return self

    @property
    def dp(self):
        return self.cards // (self.tp * self.ep)


def catalog():
    return {'id': CASE_ID, 'name': NAME, 'mode': MODE,
            'presets': [{'id': chip, 'name': f'{chip.upper()} · {capacity} GB/卡',
                         'config': CaseConfig(chip=chip).model_dump()}
                        for chip, capacity in [('950dt', 96), ('950pr', 128)]],
            'contexts': CONTEXTS, 'notes': NOTES, 'schema': CaseConfig.model_json_schema()}


def provenance():
    here = Path(__file__).resolve().parent
    engine = here.parents[1]
    files = [*here.glob('*.py'), *here.glob('*.json'), *engine.glob('*.py')]
    hashes = {str(p.relative_to(engine)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(files)}
    return {'source_experiment': 'deepseek-v41-pro-10t-pod-dflash-fp4-20261102',
            'source_fingerprint': '08d738522909d6dbf41d3f4e7abc6880ea5da35caa3aee29adcd1e8e9a1259bb',
            'files_sha256': hashes,
            'numeric_fingerprint': hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()}


def simulate_case(value, progress=None):
    from . import model
    spec = CaseConfig.model_validate(value)
    config = spec.model_dump()
    active_dp = min(spec.decode_batch, spec.dp)
    local_batch = spec.decode_batch // active_dp
    phases = {}
    for phase, batch, accepted in [('prefill', 1, 1), ('draft', local_batch, 1),
                                   ('verify', local_batch, 1), ('commit', local_batch, spec.accepted)]:
        if progress:
            progress('simulating', f'正在仿真 {phase} · {spec.chip.upper()} · TP{spec.tp} EP128 SP{spec.sp}')
        phases[phase] = model.run_phase(spec.chip, spec.tp, spec.ep, spec.sp, batch,
                                       spec.context_tokens, spec.block, phase, accepted, keep_stages=True)
    pre = phases['prefill']
    dec = model.aggregate([phases[p] for p in ('draft', 'verify', 'commit')])
    ttft = pre['duration_s']
    tpot = dec['duration_s'] / spec.accepted if dec['duration_s'] is not None else None
    breakdown = {}
    for metric, phase, divisor in [('ttft', pre, 1), ('tpot', dec, spec.accepted)]:
        breakdown[metric] = ({k: v / divisor for k, v in phase['breakdown_s'].items()}
                             if phase['breakdown_s'] is not None else None)
    return {'mode': MODE, 'status': 'feasible' if pre['status'] == dec['status'] == 'feasible' else 'infeasible',
            'config': config, 'model_name': NAME, 'hardware': model.hardware(spec.chip, spec.cards),
            'model': model.MODEL, 'draft': model.DRAFT, 'notes': NOTES,
            'precision': model.PRECISION_NOTE, 'provenance': provenance(),
            'layout': {'tp': spec.tp, 'ep': spec.ep, 'pp': 1, 'dp': spec.dp, 'sp': spec.sp,
                       'active_decode_replicas': active_dp, 'local_decode_batch': local_batch},
            'metrics': {'ttft_s': ttft, 'tpot_s': tpot, 'tpop_s': tpot,
                        'round_s': dec['duration_s'],
                        'decode_tokens_s': spec.decode_batch / tpot if tpot else None,
                        'prefill_hbm_GB': pre['memory']['hbm_peak_bytes'] / 1e9,
                        'decode_hbm_GB': dec['memory']['hbm_peak_bytes'] / 1e9,
                        'prefill_status': pre['status'], 'decode_status': dec['status'],
                        'latency_breakdown': breakdown},
            'phases': phases,
            'decode_physical_work': ({k: v * active_dp for k, v in dec['physical_work'].items()}
                                     if dec['physical_work'] is not None else None)}


def write_report(result, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    def save(name, value):
        (directory / name).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
    save('result.json', result)
    save('config.resolved.json', result['config'])
    save('provenance.json', result['provenance'])
    with (directory / 'latency_breakdown.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['metric', 'component', 'seconds'])
        for metric, parts in result['metrics']['latency_breakdown'].items():
            for component, seconds in (parts or {'unavailable': None}).items():
                writer.writerow([metric, component, seconds])
    with (directory / 'phases.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['phase', 'status', 'duration_s', 'completed_events', 'hbm_peak_bytes', 'host_peak_bytes', 'capacity_check'])
        for name, phase in result['phases'].items():
            writer.writerow([name, phase['status'], phase['duration_s'], phase['native_completed_events'],
                             phase['memory']['hbm_peak_bytes'], phase['memory']['host_peak_bytes'], phase['capacity_check']])
    metrics = result['metrics']
    lines = ['# ' + NAME, '', f"状态：{result['status']}", '',
             f"配置：{json.dumps(result['config'], ensure_ascii=False)}", '',
             f"TTFT：{metrics['ttft_s']} s；固定上下文 TPOT：{metrics['tpot_s']} s。", '',
             *['- ' + note for note in NOTES], '',
             '逐阶段资源工作量、峰值构成、原生事件完成数与全部算子记录见 result.json。',
             '阶段时钟相互独立；prefill batch 与 decode batch 不同，不拼接为同一请求的端到端时间线。', '']
    (directory / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    # Keep the complete per-operator records in result.json, bounded UI summary.
    compact = {**result, 'phases': {name: {k: v for k, v in phase.items() if k != 'stages'}
                                   for name, phase in result['phases'].items()}}
    save('ui.result.json', compact)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=NAME)
    parser.add_argument('config', type=Path)
    parser.add_argument('-o', '--output', type=Path, required=True)
    args = parser.parse_args(argv)
    result = simulate_case(json.loads(args.config.read_text(encoding='utf-8')))
    write_report(result, args.output)
    print(json.dumps(result['metrics'], ensure_ascii=False))
    return 0 if result['status'] == 'feasible' else 3
