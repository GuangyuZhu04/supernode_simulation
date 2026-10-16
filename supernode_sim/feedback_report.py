"""Reports for finite feedback searches over latency-constrained inference."""
from __future__ import annotations

import csv
import json
from pathlib import Path


def _number(value) -> str:
    if value is None:
        return '—'
    if isinstance(value, bool):
        return '是' if value else '否'
    return f'{value:,.6g}' if isinstance(value, (int, float)) else str(value)


def _cell(value) -> str:
    return str(value).replace('|', '\\|').replace('\n', ' ')


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def feedback_markdown(result: dict) -> str:
    from .report import latency_markdown
    targets = result['targets']
    search = result['search_space']
    counts = result['counts']
    best = result.get('best')
    model = result.get('config', {}).get('model', {})
    lines = [
        '# Feedback：时延约束下的并行与负载搜索', '',
        f"状态：**{result['status']}**；模型：`{_cell(model.get('name', '—'))}`。", '',
        f"目标 TTFT ≤ {_number(targets.get('ttft_s'))} s；TPOT（兼容输入名 TPOP）≤ {_number(targets.get('tpot_s'))} s。",
        '在满足时延和资源约束的候选中，优先最大化 Decode 吞吐量（token/s）。', '',
        '模型、拓扑、硬件及其余执行参数沿用基线配置；搜索 TP、SP、EP、Prefill 长度和 Decode batch size。',
        '每个候选采用同一固定 batch 联合执行 Prefill 和 Decode，batch size 同时作用于两个阶段。',
        'Decode 吞吐量只统计首 token 之后的输出 token；首 token 计入 TTFT，不计入 Decode 吞吐量。',
        '计算公式：所有请求的（输出 token 数 − 1）之和 ÷（最晚末 token 完成时间 − 最早首 token 完成时间）。',
        '当 DP 或 PP 大于 1 时，此全局 Decode 窗口可能与其他请求尚未完成的 Prefill 重叠。',
        f"每请求输出长度为 {_number(search.get('output_length'))} token（含首 token）。", '',
        '## 搜索范围与完成情况', '',
        f"搜索完成：{'是' if result.get('search_complete') else '否'}；候选上限：{_number(search.get('max_candidates'))}。",
        f"Prefill 长度候选：`{_cell(_json(search.get('prefill_lengths', [])))}`。",
        f"Decode batch size 候选：`{_cell(_json(search.get('decode_batch_sizes', [])))}`。",
        f"并行布局数：{len(search.get('layouts', []))}；完整布局和默认值来源见 result.json 的 search_space。", '',
        '| 总候选 | 可行 | 配置无效 | SLO 未达标 | 资源不可行 | 错误 |',
        '|---:|---:|---:|---:|---:|---:|',
        '| ' + ' | '.join(_number(counts.get(key, 0)) for key in
                         ('total', 'feasible', 'invalid_config', 'slo_failed', 'infeasible', 'error')) + ' |', '',
        ('结果仅表示已声明有限网格和当前仿真模型下的最优值，不是连续参数空间或真实部署的全局最优值。'
         if result.get('search_complete') else
         '当前结果只覆盖成功评估的候选，尚不能确认整个有限网格的最优值。'),
    ]
    if not result.get('search_complete'):
        lines += ['', '搜索未完整完成；现有候选和失败原因保留在 candidates.csv 与 result.json，不能据此确认整个网格的最优值。']
    if best:
        params = best['parameters']
        metrics = best['metrics']
        heading = '最优可行候选' if result.get('search_complete') else '当前已验证的最佳候选（搜索未完成）'
        lines += ['', f'## {heading}', '',
                  f"候选：`{_cell(best['id'])}`；引擎：`{_cell(best.get('engine', '—'))}`。", '',
                  '| TP | SP | EP | Sequence parallel | Prefill 长度 / token | Decode batch size |',
                  '|---:|---:|---:|---|---:|---:|',
                  '| ' + ' | '.join(_number(params.get(key)) for key in
                                   ('tp', 'sp', 'ep', 'sequence_parallel', 'prefill_length', 'decode_batch_size')) + ' |', '',
                  '| 指标 | 实际值 |', '|---|---:|']
        for label, key in (
            ('TTFT / s', 'ttft_s'), ('TPOT（TPOP）/ s', 'tpot_s'),
            ('Decode 吞吐量 / token/s', 'decode_throughput_tokens_s'),
            ('Decode 时间窗口 / s', 'decode_window_s'),
            ('全程吞吐量 / token/s', 'throughput_tokens_s'),
        ):
            lines.append(f'| {label} | {_number(metrics.get(key))} |')
        lines += ['', '最佳完整配置已导出为 best.config.json，可使用 simulate 命令复现该候选。']
        lines += latency_markdown(metrics, heading='### 最佳候选 TTFT / TPOT（TPOP）时延拆解')
    else:
        lines += ['', '## 未找到可行候选', '',
                  '当前候选中没有同时满足约束的配置；具体时延、SLO 判断和失败诊断见 candidates.csv 与 result.json。']
    boundaries = result.get('feasible_boundaries', [])
    lines += ['', '## 可行长度与 batch 边界', '',
              '以下最大值分别在同一布局的可行候选中取值，最大长度和最大 batch 可能来自不同候选，不能直接组合。']
    if boundaries:
        lines += ['', '| TP | SP | EP | Sequence parallel | 最大 Prefill 长度 | 最大 Decode batch | 可行候选 ID |',
                  '|---:|---:|---:|---|---:|---:|---|']
        for row in boundaries:
            values = [_number(row.get(key)) for key in
                      ('tp', 'sp', 'ep', 'sequence_parallel', 'max_prefill_length', 'max_decode_batch_size')]
            values.append(_cell(', '.join(str(item) for item in row.get('candidate_ids', []))))
            lines.append('| ' + ' | '.join(values) + ' |')
    else:
        lines += ['', '当前没有可行边界。']
    frontier = result.get('pareto_frontier', [])
    lines += ['', 'Pareto 候选 ID：' + (', '.join(f'`{_cell(item)}`' for item in frontier) if frontier else '无') + '。',
              '', '## 产物与复现', '',
              '- result.json：完整搜索参数、候选指标、SLO、失败诊断、可行边界与最佳配置。',
              '- candidates.csv：逐候选参数、指标、状态、SLO 与失败诊断，可筛选比较。',
              '- latency_breakdown.csv：所有候选的 TTFT / TPOT（TPOP）及计算、通信、同步、调度、内存准入与资源争用分项；单位秒，空值表示不可用。',
              '- config.resolved.json：应用校准后的基线快照，避免重复加载校准。']
    if best:
        lines += ['- best.config.json：最佳候选的完整可复现配置。']
    if result.get('assumptions'):
        lines += ['', '## 假设', '']
        lines.extend(f'- {_cell(item)}' for item in result['assumptions'])
    return '\n'.join(lines) + '\n'


def write_feedback_report(result: dict, directory: str | Path) -> Path:
    from .report import write_latency_csv
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    snapshot = dict(result['config'])
    snapshot['calibration'] = None
    payloads = {'result.json': result, 'config.resolved.json': snapshot}
    best = result.get('best')
    if best:
        best_snapshot = dict(best['config'])
        best_snapshot['calibration'] = None
        payloads['best.config.json'] = best_snapshot
    else:
        # Reusing an output directory must not leave an old winner behind.
        (directory / 'best.config.json').unlink(missing_ok=True)
    for name, payload in payloads.items():
        with (directory / name).open('w', encoding='utf-8') as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
    fields = [
        'id', 'tp', 'sp', 'ep', 'sequence_parallel', 'prefill_length', 'decode_batch_size',
        'status', 'ttft_s', 'tpot_s', 'tpop_s', 'decode_throughput_tokens_s', 'decode_window_s',
        'throughput_tokens_s', 'engine', 'pareto_optimal', 'slo', 'diagnostics', 'memory_peaks_bytes',
    ]
    frontier = set(result.get('pareto_frontier', []))
    with (directory / 'candidates.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for candidate in result.get('candidates', []):
            row = {**candidate.get('parameters', {}), **candidate.get('metrics', {}),
                   'id': candidate['id'], 'status': candidate['status'],
                   'engine': candidate.get('engine'), 'pareto_optimal': candidate['id'] in frontier,
                   'slo': _json(candidate.get('slo', {})),
                   'diagnostics': _json(candidate.get('diagnostics', [])),
                   'memory_peaks_bytes': _json(candidate.get('memory_peaks_bytes', {}))}
            writer.writerow(row)
    write_latency_csv(result, directory)
    (directory / 'report.md').write_text(feedback_markdown(result), encoding='utf-8')
    return directory.resolve()
