"""Artifacts for the explicitly scoped request-arrival batching simulator."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path


def token_trace(result: dict) -> dict | None:
    """Export the documented token-analysis contract, without claiming measurement."""
    window = result.get('time_window_relative_s', result['time_window_s'])
    if window['end'] <= window['start']:
        return None
    end = math.nextafter(window['end'], math.inf)
    if not math.isfinite(end):
        raise ValueError('Cannot export a finite half-open trace window')
    trace = {
        'metadata': {
            'source': 'supernode_sim FCFS batch simulation; not observed hardware timestamps',
            'status': 'assumed', 'evidence_kind': 'simulation', 'timer_boundary': 'client',
            'model': result['config']['model']['name'],
            'hardware': result['config']['hardware']['name'],
            'runtime': 'supernode_sim non-preemptive FCFS batches', 'timestamp_granularity': 'token',
            'arrival_boundary': 'explicit input arrival_s; all subsequent queueing is included',
            'clock_origin_s': result.get('clock_origin_s', 0.0),
            'clock_basis': 'relative to clock_origin_s; avoids losing microsecond intervals at epoch-sized origins',
            'window_end_policy': 'next representable float after final completion to include it in [start,end)',
        },
        'measurement_window_s': [window['start'], end],
        'requests': [{
            'id': row['id'], 'arrival_s': row.get('arrival_relative_s', row['arrival_s']),
            'admission_s': row.get('admission_relative_s', row['admission_s']),
            'token_times_s': row.get('token_times_relative_s', row['token_times_s']),
            'finish_s': row.get('finish_relative_s', row['finish_s']),
            'input_tokens': row['input_tokens'], 'output_tokens': row['output_tokens'],
            'status': 'success' if row['status'] == 'completed' else 'failed',
        } for row in result['requests']],
    }
    slo = result['serving'].get('slo') or {}
    trace['metadata']['serving_slo'] = slo
    trace['metadata']['slo_exported'] = all(slo.get(key) is not None for key in ('ttft_s', 'tpot_s'))
    trace['metadata']['slo_export_policy'] = (
        'The standalone analyzer requires both thresholds; single-threshold goodput remains in serving result.json.')
    if all(slo.get(key) is not None for key in ('ttft_s', 'tpot_s')):
        trace['slo'] = {key: slo[key] for key in ('ttft_s', 'tpot_s')}
    return trace


def write_serving_report(result: dict, output: str | Path) -> Path:
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name, value in (
        ('result.json', result), ('config.resolved.json', result['config']),
        ('serving.resolved.json', result['serving']), ('token-trace.json', token_trace(result)),
    ):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2,
            allow_nan=False) + '\n', encoding='utf-8')
    fields = ['id', 'status', 'instance_id', 'batch_id', 'arrival_s', 'admission_s', 'model_start_s',
              'first_token_s', 'last_token_s', 'finish_s', 'input_tokens', 'requested_output_tokens',
              'output_tokens', 'admission_wait_s', 'model_ttft_s', 'server_ttft_s',
              'client_ttft_s', 'tpot_s', 'e2e_s', 'slo_passed', 'arrival_relative_s',
              'admission_relative_s', 'model_start_relative_s', 'first_token_relative_s',
              'last_token_relative_s', 'finish_relative_s']
    with (output / 'requests.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(result['requests'])
    hardware = result['hardware']
    lines = ['# 请求级成批服务仿真', '',
        f"状态：{result['status']}。实例数：{hardware['instance_count']}；总设备数：{hardware['total_npus']}。", '',
        '显式到达请求按 FCFS 成批；每批由原算子/通信/内存引擎执行，整批结束后实例才接纳下一批。',
        '结果是指定输入与调度假设下的预测；分位数不代表生产服务保障。', '',
        '| 指标（秒） | 样本数 | 平均 | P50 | P95 | P99 |',
        '|---|---:|---:|---:|---:|---:|']
    for metric, values in result['metrics']['distributions'].items():
        def value(key):
            number = values.get(key)
            return '—' if number is None else f'{number:.8g}'
        lines.append(f"| {metric} | {values.get('sample_count', 0)} | {value('mean_s')} | {value('p50_s')} | {value('p95_s')} | {value('p99_s')} |")
    lines.extend(['', '吞吐：', '', '```json',
        json.dumps(result['throughput'], ensure_ascii=False, indent=2), '```', '',
        '时钟边界：`client_ttft = admission_wait + server_ttft`；server TTFT 包含本批 scheduler overhead。单 token TPOT 为空。',
        '分位数 cohort：接纳等待含失败请求，其余仅成功且指标可定义的请求。绝对时间仅展示；指标和 token trace 使用相对时钟，原点见 clock_origin_s。',
        '失败批不产生成功 token，不自动缩批；完整失败原因、执行条件与来源见 result.json。', '',
        '适用范围：', '', *['- ' + str(item) for item in result['limitations']], '',
        ('逐请求明细见 requests.csv；token-trace.json 可由 skill 的 analyze_inference_trace.py 复核。'
         if result['time_window_s']['duration'] > 0 else
         '无正时长统计窗口；token-trace.json 为 null，不能用于吞吐或分位数分析。'),
        '模型配置与服务调度配置分别保存在 config.resolved.json 和 serving.resolved.json。', ''])
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return output
