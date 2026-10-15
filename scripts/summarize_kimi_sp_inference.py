"""Summarize SP16/SP32 experiments and compare with the archived TP8 baseline.

This reporter never rewrites the baseline directory. New scheduled traffic is
cross-checked against independently generated padded-SP algebra. Run with the
repository .venv Python; the HTML report has no external assets or libraries.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from summarize_kimi_10t_inference import (
    case_finished, compact_diagnostics, failure_required_bytes, fmt, md_table,
    network_nodes, network_racks, peak_info, read_json, write_csv,
)


def runtime_provenance(directory):
    """Reuse a fully checked proof only while artifacts and frozen inputs agree."""
    path = directory / 'audit/runtime-provenance.json'
    proof = None
    if path.exists():
        proof = json.loads(path.read_text(encoding='utf-8-sig'))
        valid = proof.get('module_hashes_verified') and proof.get('normalizer_import_verified')
        valid = valid and len(proof.get('anchor_proofs', [])) == 3 and all(row['verified'] for row in proof['anchor_proofs'])
        for name, row in proof.get('cases', {}).items():
            artifact = directory / 'cases' / (name + '.json')
            valid = valid and artifact.exists() and hashlib.sha256(artifact.read_bytes()).hexdigest() == row['artifact_sha256']
        inputs = proof.get('input_file_sha256')
        if inputs is None:
            # The first full proof predates input-guard export. Its unchanged
            # original config/profile files all predate the proof subprocess.
            manifest = read_json(directory / 'manifest.json')
            originals = ['manifest.json', 'hardware.profile.json'] + [row['config'] for row in manifest['cases']]
            valid = valid and all((directory / name).stat().st_mtime <= path.stat().st_mtime for name in originals)
            inputs = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in originals}
            inputs.update({name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                           for name in ('runtime-frozen/manifest.json', 'audit/worker-provenance.json')})
            proof['input_file_sha256'] = inputs
            proof['input_guard_note'] = 'First full normalization proof: original config/profile mtimes predate verification; raw guards captured immediately after verification.'
        valid = valid and all((directory / name).exists() and hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest
                              for name, digest in inputs.items())
        frozen = read_json(directory / 'runtime-frozen/manifest.json')
        valid = valid and frozen['runtime_snapshot_id'] == proof.get('runtime_snapshot_id')
        valid = valid and all(hashlib.sha256((directory / 'runtime-frozen' / row['path']).read_bytes()).hexdigest() == row['sha256']
                              for row in frozen['full_package_modules'])
        if not valid:
            proof = None
    if proof is None:
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/audit_kimi_sp_runtime.py'), str(directory)],
                                check=True, capture_output=True, text=True, encoding='utf-8')
        proof = json.loads(result.stdout)
    if 'verified_at_Asia_Shanghai' not in proof:
        proof['verified_at_Asia_Shanghai'] = datetime.fromtimestamp(path.stat().st_mtime, timezone(timedelta(hours=8))).isoformat(timespec='seconds')
    path.write_text(json.dumps(proof, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return proof


def metadata(case):
    return {key: case[key] for key in ('top_k', 'context', 'input_length', 'batch_size', 'tp', 'sp', 'ep')}


def phase_row(case, result, phase, source):
    data = result.get('phase_metrics', {}).get(phase)
    if not data:
        return None
    comm = data.get('communication', {})
    semantic = comm.get('payload_by_semantic_bytes', {})
    collective = comm.get('payload_by_collective_bytes', {})
    physical = comm.get('physical_bytes_by_domain', {})
    hbm = {key: value for key, value in data.get('resource_work', {}).items()
           if key.startswith('npu') and key.endswith('.hbm')}
    card, maximum = max(hbm.items(), key=lambda item: item[1], default=(None, None))
    duration = data.get('duration_s')
    return {'case': case['name'], **metadata(case), 'phase': phase, 'source': source,
            'status': data.get('status'), 'duration_s': duration,
            'communication_scope': 'complete_phase' if data.get('status') == 'feasible' else 'completed_components_only',
            'hbm_work_scope': 'complete_phase' if data.get('status') == 'feasible' else 'executed_events_including_partial_failed_component',
            'transport_payload_bytes': comm.get('transport_payload_bytes', 0),
            'dispatch_payload_bytes': semantic.get('dispatch', 0),
            'combine_payload_bytes': semantic.get('combine', 0),
            'tp_all_gather_payload_bytes': collective.get('all_gather', 0),
            'tp_reduce_scatter_payload_bytes': collective.get('reduce_scatter', 0),
            'tp_all_reduce_payload_bytes': collective.get('all_reduce', 0),
            'physical_su_hopbytes': physical.get('SU', 0),
            'physical_so_hopbytes': physical.get('SO', 0),
            'su_payload_bytes': physical.get('SU', 0),
            'so_payload_bytes': physical.get('SO', 0) / 4,
            'hbm_work_total_bytes': sum(hbm.values()), 'hbm_work_max_per_card_bytes': maximum,
            'hbm_work_max_card': card, 'peak_hbm_bytes': data.get('peak_hbm_bytes'),
            'aggregate_network_payload_Bps': comm.get('transport_payload_bytes', 0) / duration if duration else None}


def check_traffic(actual, expected):
    pairs = {'transport_payload_bytes': 'total_network_payload_bytes',
             'dispatch_payload_bytes': 'dispatch_network_bytes', 'combine_payload_bytes': 'combine_network_bytes',
             'tp_all_gather_payload_bytes': 'tp_all_gather_payload_bytes',
             'tp_reduce_scatter_payload_bytes': 'tp_reduce_scatter_payload_bytes',
             'tp_all_reduce_payload_bytes': 'tp_all_reduce_payload_bytes',
             'su_payload_bytes': 'same_su_payload_bytes', 'so_payload_bytes': 'cross_su_payload_bytes',
             'physical_su_hopbytes': 'physical_su_hopbytes', 'physical_so_hopbytes': 'physical_so_hopbytes'}
    for actual_key, expected_key in pairs.items():
        if not math.isclose(actual[actual_key], expected[expected_key], rel_tol=1e-9, abs_tol=1):
            raise AssertionError(f"{actual['case']}:{actual['phase']} {actual_key}: {actual[actual_key]} != {expected[expected_key]}")
    total = sum(actual[key] for key in ('dispatch_payload_bytes', 'combine_payload_bytes',
                                       'tp_all_gather_payload_bytes', 'tp_reduce_scatter_payload_bytes',
                                       'tp_all_reduce_payload_bytes'))
    if not math.isclose(total, actual['transport_payload_bytes'], rel_tol=1e-9, abs_tol=1):
        raise AssertionError(f"{actual['case']}:{actual['phase']} semantic/collective partition does not sum to total")


def planned_rows(manifest, expected):
    rows = []
    for case in manifest['cases']:
        audit = expected[case['tp'], case['top_k'], case['input_length'], case['batch_size']]
        for phase, key in (('prefill', 'prefill'), ('decode', 'one_decode_step')):
            data = audit[key]
            rows.append({'case': case['name'], **metadata(case), 'phase': phase,
                         'scope': 'analytical_planned_full_phase', **data,
                         'tokens': data['valid_input_rows'],
                         'transport_payload_bytes': data['total_network_payload_bytes'],
                         'dispatch_payload_bytes': data['dispatch_network_bytes'],
                         'combine_payload_bytes': data['combine_network_bytes'],
                         'su_payload_bytes': data['same_su_payload_bytes'],
                         'so_payload_bytes': data['cross_su_payload_bytes']})
    return rows


def collect(directory):
    manifest = read_json(directory / 'manifest.json')
    audit = read_json(directory / 'audit/traffic-formulas.json')
    expected = {(row['tp'], row['top_k'], row['input_length'], row['batch_size']): row for row in audit['matrix']}
    provenance = runtime_provenance(directory)
    capacity = read_json(directory / 'hardware.profile.json')['hardware']['npu']['hbm_bytes']
    rows, phases, nodes, peaks, failures, issues = [], [], [], [], [], []
    fingerprints = Counter()
    for case in manifest['cases']:
        planned = expected[case['tp'], case['top_k'], case['input_length'], case['batch_size']]
        row = {'case': case['name'], **metadata(case), 'case_status': 'pending',
               'prefill_status': 'pending', 'decode_status': 'pending', 'decode_source': None,
               'ttft_s': None, 'tpot_s': None, 'standalone_decode_step_s': None,
               'decode_step_s': None, 'runtime_provenance_verified': None,
               'prefill_planned_payload_bytes': planned['prefill']['total_network_payload_bytes'],
               'prefill_planned_su_payload_bytes': planned['prefill']['same_su_payload_bytes'],
               'prefill_planned_so_payload_bytes': planned['prefill']['cross_su_payload_bytes'],
               'decode_planned_payload_bytes': planned['one_decode_step']['total_network_payload_bytes'],
               'decode_planned_so_payload_bytes': planned['one_decode_step']['cross_su_payload_bytes'],
               'decode_planned_ag_payload_bytes': planned['one_decode_step']['tp_all_gather_payload_bytes'],
               'decode_planned_rs_payload_bytes': planned['one_decode_step']['tp_reduce_scatter_payload_bytes']}
        path = directory / 'cases' / f"{case['name']}.json"
        if not path.exists():
            rows.append(row)
            continue
        try:
            artifact = read_json(path)
        except (OSError, ValueError) as exc:
            row['case_status'] = 'unreadable'
            issues.append(f"{case['name']}: {exc}")
            rows.append(row)
            continue
        stored = artifact.get('fingerprint', 'unknown')
        fingerprints[stored] += 1
        runtime = provenance['cases'].get(case['name'])
        if not runtime or not runtime['runtime_provenance_verified']:
            issues.append(f"{case['name']}: frozen runtime/config/worker provenance not verified")
        if runtime:
            row.update({key: runtime[key] for key in ('observed_disk_fingerprint', 'runtime_snapshot_fingerprint',
                        'recorded_config_sha256', 'runtime_config_sha256', 'runtime_config_sha256_matches',
                        'observed_disk_matches_runtime_snapshot', 'runtime_snapshot_id', 'runtime_provenance_verified',
                        'producer_shard', 'producer_pid', 'producer_log', 'artifact_sha256')})
            row['ordered_disk_observation_9modules_fingerprint_at_audit'] = runtime['current_disk_fingerprint']
            row['ordered_disk_observation_9modules_matches_disk_at_audit'] = runtime['observed_disk_matches_current_disk']
        all_result = artifact.get('all') or {}
        standalone = artifact.get('standalone_decode')
        metrics = all_result.get('metrics') or {}
        row['case_status'] = all_result.get('status', 'unknown')
        row['full_run_output_throughput_tokens_s'] = metrics.get('throughput_tokens_s')
        prefill = phase_row(case, all_result, 'prefill', 'end_to_end')
        decode_result = standalone or all_result
        decode_source = 'standalone_cache_resident' if standalone else 'end_to_end'
        decode = phase_row(case, decode_result, 'decode', decode_source)
        for phase, phase_result, data, source_key in (('prefill', all_result, prefill, 'prefill'),
                                                    ('decode', decode_result, decode, 'one_decode_step')):
            if not data:
                row[phase + '_status'] = 'not_run'
                continue
            row[phase + '_status'] = data['status']
            phases.append(data)
            if data['status'] == 'feasible':
                check_traffic(data, planned[source_key])
            phase_nodes = network_nodes(case, phase_result, phase, data['source'], data)
            for node in phase_nodes:
                node.update(tp=case['tp'], sp=case['sp'], ep=case['ep'])
            nodes.extend(phase_nodes)
            row[phase + '_actual_payload_bytes'] = data['transport_payload_bytes']
            row[phase + '_peak_hbm_bytes'] = data['peak_hbm_bytes']
        if prefill:
            row['prefill_observed_duration_s'] = prefill['duration_s']
            if prefill['status'] == 'feasible':
                row['ttft_s'] = metrics.get('ttft_s') or prefill['duration_s']
        if decode:
            row.update(decode_source=decode_source, decode_actual_so_payload_bytes=decode['so_payload_bytes'],
                       decode_actual_ag_payload_bytes=decode['tp_all_gather_payload_bytes'],
                       decode_actual_rs_payload_bytes=decode['tp_reduce_scatter_payload_bytes'],
                       decode_hbm_work_total_bytes=decode['hbm_work_total_bytes'],
                       decode_hbm_work_max_per_card_bytes=decode['hbm_work_max_per_card_bytes'])
            if decode['status'] == 'feasible':
                duration = decode['duration_s'] if standalone else metrics.get('tpot_s', decode['duration_s'])
                row['decode_step_s'] = duration
                row['standalone_decode_step_s' if standalone else 'tpot_s'] = duration
                row['decode_step_output_throughput_tokens_s'] = case['batch_size'] / duration if duration else None
        pool, peak, categories, failed, layer, diagnostics = peak_info(all_result)
        required = failure_required_bytes(diagnostics, capacity)
        row.update(peak_hbm_pool=pool, peak_hbm_bytes=peak, failure_layer_1based=layer,
                   failure_component=failed.get('component'), failure_event=failed.get('id'),
                   failure_required_hbm_bytes=required,
                   failure_component_admitted_peak_hbm_bytes=failed.get('peak_hbm_bytes'),
                   diagnostics_json=json.dumps(compact_diagnostics(diagnostics), ensure_ascii=False))
        for category, amount in categories.items():
            peaks.append({'case': case['name'], **metadata(case), 'pool': pool,
                          'peak_hbm_bytes': peak, 'category': category, 'bytes_at_same_pool_peak': amount})
        if diagnostics or failed:
            failures.append({'case': case['name'], **metadata(case), 'layer_1based': layer,
                             'event': failed.get('id'), 'component': failed.get('component'),
                             'failure_required_hbm_bytes': required, 'diagnostics': diagnostics})
        rows.append(row)
    finished = sum(case_finished(row) for row in rows)
    replays = []
    for name in ('sp16-top32-1M-b1', 'sp32-top64-1M-b32'):
        path = directory / 'audit/frozen-reference-replays' / (name + '.json')
        record = read_json(path) if path.exists() else {'passed': False, 'status': 'pending'}
        if record.get('passed') is True:
            original = provenance['cases'][name]
            assert record['runtime_snapshot_id'] == provenance['runtime_snapshot_id']
            assert record['config_sha256'] == original['runtime_config_sha256']
            assert record['runtime_snapshot_fingerprint'] == original['runtime_snapshot_fingerprint']
            assert record['numeric_values_checked'] > 0
        replays.append({'case': name, **record, 'evidence_file': str(path.relative_to(directory))})
    replays_passed = all(row.get('passed') is True for row in replays)
    return {'manifest': manifest, 'audit': audit, 'rows': rows, 'phases': phases, 'nodes': nodes,
            'peaks': peaks, 'failures': failures, 'issues': issues, 'fingerprints': dict(fingerprints),
            'complete': finished == len(rows) and not issues and replays_passed, 'finished': finished,
            'runtime_provenance': provenance, 'reference_replays': replays,
            'planned': planned_rows(manifest, expected)}


def number(value):
    return float(value) if value not in (None, '') else None


def archive_baseline(path):
    with (path / 'summary.csv').open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    baseline = {}
    for row in rows:
        key = int(row['top_k']), int(row['input_length']), int(row['batch_size'])
        artifact_path = path / 'cases' / (row['case'] + '.json')
        if artifact_path.exists():
            artifact = read_json(artifact_path)
            row['archived_fingerprint'] = artifact.get('fingerprint')
            row['archived_config_sha256'] = artifact.get('config_sha256')
        baseline[key] = row
    return baseline


def comparisons(rows, baseline):
    output = []
    for row in rows:
        old = baseline[row['top_k'], row['input_length'], row['batch_size']]
        old_ttft = number(old.get('ttft_s'))
        old_decode = number(old.get('tpot_s')) or number(old.get('standalone_decode_step_s'))
        new_decode = row['decode_step_s']
        old_payload = number(old.get('decode_planned_payload_bytes'))
        output.append({'case': row['case'], **metadata(row), 'baseline_case': old['case'],
                       'baseline_status': 'archived_prior_run_not_reexecuted',
                       'baseline_recorded_fingerprint': old.get('archived_fingerprint'),
                       'baseline_recorded_config_sha256': old.get('archived_config_sha256'),
                       'baseline_prefill_status': old['prefill_status'], 'sp_prefill_status': row['prefill_status'],
                       'baseline_ttft_s': old_ttft, 'sp_ttft_s': row['ttft_s'],
                       'ttft_ratio_sp_over_tp8': row['ttft_s'] / old_ttft if row['ttft_s'] is not None and old_ttft else None,
                       'prefill_newly_feasible': row['prefill_status'] == 'feasible' and old['prefill_status'] != 'feasible',
                       'baseline_decode_source': old.get('decode_source'), 'sp_decode_source': row['decode_source'],
                       'baseline_decode_step_s': old_decode, 'sp_decode_step_s': new_decode,
                       'decode_ratio_sp_over_tp8': new_decode / old_decode if new_decode is not None and old_decode else None,
                       'baseline_decode_payload_bytes': old_payload,
                       'sp_decode_payload_bytes': row['decode_planned_payload_bytes'],
                       'decode_payload_scope': 'complete_phase_planned_payload',
                       'decode_payload_ratio_sp_over_tp8': row['decode_planned_payload_bytes'] / old_payload if old_payload else None})
    return output


def md_report(data, comparison):
    rows, audit = data['rows'], data['audit']
    status = '已完成' if data['complete'] else '中间结果，尚未完成'
    lines = ['# Kimi K3 约10T：Megatron SP16 / SP32 推理仿真', '',
             f"**{status}：{data['finished']}/{len(rows)}组。** 生成时间：{datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')}（Asia/Shanghai）。", '',
             '保持10.171385035872T、3328专家、93层（69 KDA、24 MLA）、top32/top64和原硬件/精度基线。每组包含prefill及首个decode步：128k/256k/512k/1M（分别131072/262144/524288/1048576 tokens），batch 1/2/4/8/16/32。', '',
             '并行采用Megatron语义：TP16/SP16/EP64、TP32/SP32/EP32，SP复用TP通信组。权重仍按TP切分；norm与AttnRes按序列切分，attention/FFN前all-gather、之后reduce-scatter。每组分别跨2/4个8卡节点，完全处于64卡scale-up机柜内；共16柜、1024卡。', '',
             '每卡144GB HBM，FP8权重/计算、FP16通信/activation/KV、FP32 KDA状态；柜内单向1TB/s/卡，80%效率；柜间400Gb/s/节点，80%效率。模型与性能参数仍未实测校准。', '',
             '实验期间工作区源码发生并发修改。三个持续运行的worker仍使用启动时已加载的runtime；本报告以恢复的冻结源码、三个首案例SHA锚点、全部配置SHA及worker/log归属校验实际runtime，保留每份原始artifact的磁盘观测指纹。磁盘观测与冻结runtime的差异单列，不声称当前引擎指纹匹配。', '',
             '本次结果保存在独立目录outputs/kimi-k3-10t-sp；可执行冻结版本在runtime-frozen/original。两组完整重放显式从该冻结package导入，不读取当前工作区engine；其配置、runtime指纹与原始结果逐项核对。', '',
             f"冻结runtime：`{data['runtime_provenance']['runtime_snapshot_id']}`。已验证{data['runtime_provenance']['verified_case_count']}/{len(rows)}份配置和producer证据；{data['runtime_provenance']['observed_disk_fingerprints_drifting_from_runtime']}份原始磁盘观测指纹与已加载runtime不同。九文件磁盘观测只对应旧observer的有序9模块，不等同现已增加cache模块的当前引擎整体指纹。详见audit/runtime-provenance.json、runtime-frozen/manifest.json。", '',
             '静态HBM（含权重、FP8 scales、8GB runtime、256MB workspace）由TP8的32.596955GB/卡降至SP16的29.310190GB、SP32的27.666807GB。Compressed MLA KV仍在TP间复制，不能再除以SP；KDA按TP切分heads，无跨SP前缀状态通信。', '',
             '短decode采用等长collective padding：每个非空EP lane的1个query补到16/32行，norm/残差使用对应分片缓冲；attention、MoE实际计算与KV缓存只计有效token。因此decode通信量受padding明显影响。Embedding加入reduce-scatter；输出只all-gather所选最后位置，随后按vocab切分投影。', '',
             'TTFT为预加载权重后到首token的批内最大时间；TPOT只测首token后的一个decode间隔。Prefill失败时TTFT为空，预置KV的独立decode时间单列，不证明端到端服务可行。', '',
             '## 上下文与batch端点', '',
             'Decode时间含端到端TPOT或条件性的独立decode，完整区分见summary.csv。']
    overview = []
    for sp in (16, 32):
        for top_k in (32, 64):
            for context in ('128k', '256k', '512k', '1M'):
                group = sorted((row for row in rows if row['sp'] == sp and row['top_k'] == top_k and row['context'] == context), key=lambda row: row['batch_size'])
                first, last = group[0], group[-1]
                overview.append([sp, top_k, context, ', '.join(f'{key}:{count}' for key, count in Counter(row['prefill_status'] for row in group).items()),
                                 fmt(first['ttft_s']), fmt(last['ttft_s']),
                                 fmt(first['decode_step_s'] * 1000 if first['decode_step_s'] is not None else None),
                                 fmt(last['decode_step_s'] * 1000 if last['decode_step_s'] is not None else None)])
    lines += ['', md_table(['SP/TP', 'Top-k', '上下文', 'Prefill组数', 'TTFT B1 s', 'TTFT B32 s', 'Decode B1 ms', 'Decode B32 ms'], overview), '',
              '## 流量', '',
              '以下为全1024卡聚合逻辑transport payload，每个有向发送计一次；dispatch/combine与all_to_all_v是同一数据的两种视图。AG/RS是另两类互斥payload。SO物理hopbytes为柜间逻辑payload的4倍；所有TP/SP AG/RS都留在SU内。', '']
    lines += [md_table(['SP/TP', 'Top-k', 'Decode总 GB/query', 'AG GB/query', 'RS GB/query', '柜间 GB/query'],
                       [[row['sp'], row['top_k'], fmt(row['total_network_payload_bytes'] / 1e9, 6),
                         fmt(row['tp_all_gather_payload_bytes'] / 1e9, 6), fmt(row['tp_reduce_scatter_payload_bytes'] / 1e9, 6),
                         fmt(row['cross_su_payload_bytes'] / 1e9, 6)] for row in audit['one_decode_step_per_query']]), '',
              '上述decode流量乘batch，当前布局下不随上下文长度变化；上下文长度影响本地KV读与attention计算。增加TP会按当前MoE实现复制各TP slice的完整latent dispatch，跨柜MoE流量相对TP8在SP16/SP32分别变为2倍/4倍；decode AG/RS另受等长padding放大。未分块prefill仍保留原kernel和生命周期假设。', '',
              '容量/流量单位采用GB=10^9B、TB=10^12B；CSV的字节和速率为byte与byte/s。完整192个阶段计划在planned_phase_traffic.csv。实际phase_traffic.csv只对已完成阶段逐项与独立公式核对总payload、dispatch、combine、AG、RS、AR、SU/SO；失败prefill流量仅为已完成组件，不能作为完整计划。节点/机柜表按实际resource work导出，失败组件可能含部分事件。HBM工作量是读写流量，不能当容量。', '',
              '## 与归档TP8基线比较', '',
              '基线读取上一轮48组summary.csv，不改写或用新schema重解释旧结果。比较维持相同top-k、上下文和batch；decode列注明端到端/独立状态。边界通信方面，旧引擎未计embedding/output通信，新SP明确加入RS/AG，分项在audit中保留。', '']
    cmpview = [row for row in comparison if row['batch_size'] in (1, 32)]
    lines += [md_table(['SP', 'Top-k', '上下文', 'B', '旧TTFT s', '新TTFT s', 'TTFT 新/旧', '旧decode ms', '新decode ms', 'decode 新/旧'],
                       [[row['sp'], row['top_k'], row['context'], row['batch_size'], fmt(row['baseline_ttft_s']), fmt(row['sp_ttft_s']),
                         fmt(row['ttft_ratio_sp_over_tp8']), fmt(row['baseline_decode_step_s'] * 1000 if row['baseline_decode_step_s'] is not None else None),
                         fmt(row['sp_decode_step_s'] * 1000 if row['sp_decode_step_s'] is not None else None), fmt(row['decode_ratio_sp_over_tp8'])]
                        for row in cmpview]), '',
              '## 全部96组', '', md_table(['案例', 'Prefill', 'TTFT s', 'TPOT ms', '独立decode ms', 'Prefill已准入峰值 GB', '失败申请所需 GB'],
                       [[row['case'], row['prefill_status'], fmt(row['ttft_s']), fmt(row['tpot_s'] * 1000 if row['tpot_s'] is not None else None),
                         fmt(row['standalone_decode_step_s'] * 1000 if row['standalone_decode_step_s'] is not None else None),
                         fmt(row.get('prefill_peak_hbm_bytes') / 1e9 if row.get('prefill_peak_hbm_bytes') is not None else None),
                         fmt(row.get('failure_required_hbm_bytes') / 1e9 if row.get('failure_required_hbm_bytes') is not None else None)] for row in rows]), '',
              '已准入峰值不含被拒绝的申请；失败所需量取144GB+shortage_bytes，仅表示当时该申请的容量下界。peak_components.csv取同卡同一峰值时刻构成，可以相加；不混加各组件不同时刻峰值。', '',
              '## 复现与限制', '',
              '预测沿用均匀专家路由、分数化的均衡expert assignments、合并的routed/shared MoE kernel和粗粒度roofline。未模拟生产请求队列、continuous batching、稀疏路由通信优化或P99。Megatron SP不等同context parallelism，也不意味着长上下文KV按SP切分。真实kernel/collective与路由trace校准后，绝对时间可能变化。', '',
              '文件：report.html（离线筛选/图表）、summary.csv（96组）、comparison.csv（与旧48组匹配）、phase_traffic.csv（实际）、planned_phase_traffic.csv（计划）、node_traffic.csv、rack_traffic.csv、peak_components.csv、failure_diagnostics.json、report_status.json；原始配置与结果在configs/、cases/。', '',
              '独立公式：`python outputs/kimi-k3-10t-sp/audit/derive_traffic.py`。报告：`python scripts/summarize_kimi_sp_inference.py`。', '',
              f"冻结runtime与配置/producer证据校验{data['runtime_provenance']['verified_case_count']}份。两组隔离完整重放状态：" + '；'.join('[' + row['case'] + '](' + row['evidence_file'].replace('\\', '/') + ')' + ('=passed，核对' + str(row['numeric_values_checked']) + '个数值' if row.get('passed') is True else '=pending/failed') for row in data['reference_replays']) + '。完整性和读取问题见report_status.json；完整性未通过时报告只表示中间结果。']
    if data['issues']:
        lines += ['', '读取/一致性问题：', ''] + ['- ' + issue for issue in data['issues']]
    return '\n'.join(lines) + '\n'


HTML = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Kimi K3 SP16/SP32推理仿真</title>
<style>body{margin:0;background:#f3f6f9;color:#173047;font:15px/1.6 system-ui,"Microsoft Yahei",sans-serif}main{max-width:1450px;margin:auto;padding:25px}h1{font-size:28px}h2{font-size:21px}.panel{background:#fff;border:1px solid #dae3eb;border-radius:10px;margin:18px 0;padding:20px}.notice{padding:13px;background:#e2f2ec;border-left:4px solid #238569}select{padding:6px;font:inherit;margin:0 18px 10px 6px}.plots{display:grid;grid-template-columns:1fr 1fr;gap:20px}svg{width:100%;height:auto}.scroll{overflow:auto;max-height:660px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:8px;text-align:right;border-bottom:1px solid #e1e8ee;white-space:nowrap}th{background:#edf2f6;position:sticky;top:0}td:first-child,th:first-child{text-align:left}.bad{color:#a63434}.good{color:#13765a}small{color:#536a7d}a{color:#166fa0}@media(max-width:800px){.plots{grid-template-columns:1fr}main{padding:12px}}</style>
<main><h1>Kimi K3 约10T · Megatron SP16 / SP32</h1><p>3328 experts · Top32/Top64 · 1024卡 / 16柜 · 保留TP权重切分，SP复用TP ranks</p><div class="notice">__STATUS__</div>
<section class="panel"><p>TP16/SP16/EP64 与 TP32/SP32/EP32，分别跨2/4个8卡节点，均留在64卡scale-up机柜内。每卡144GB HBM；FP8权重/计算、FP16通信/KV。静态HBM：SP16 29.310GB、SP32 27.667GB；归档TP8 32.597GB。</p><p>Norm与AttnRes按序列切分；attention/FFN前AG、之后RS。Compressed MLA KV仍按TP复制；SP不切分attention context。Decode每个非空EP lane只有1个query，等长collective补到16/32行，attention/FFN计算与KV只计有效token；Norm/AttnRes处理补齐后的本地分片行。</p><p>TTFT为空表示prefill未完成；独立decode从驻留KV开始，条件性时间不证明端到端可行。预测沿用仓库未实测校准的kernel、路由与硬件基线。</p></section>
<section class="panel"><label>SP/TP<select id="sp"><option value="all">全部</option><option>16</option><option>32</option></select></label><label>Top-k<select id="top"><option value="all">全部</option><option>32</option><option>64</option></select></label><label>上下文<select id="context"><option value="all">全部</option><option>128k</option><option>256k</option><option>512k</option><option>1M</option></select></label><label>Batch<select id="batch"><option value="all">全部</option><option>1</option><option>2</option><option>4</option><option>8</option><option>16</option><option>32</option></select></label><div class="plots"><div id="ttft"></div><div id="decode"></div></div><small>横轴batch；点悬浮显示案例与数值。独立decode曲线使用虚线。长上下文曲线可能重合；筛选单一上下文便于比较SP。</small></section>
<section class="panel"><h2>96组结果</h2><div class="scroll"><table><thead><tr><th>案例</th><th>Prefill</th><th>TTFT s</th><th>TPOT ms</th><th>独立decode ms</th><th>Prefill峰值 GB</th><th>失败所需 GB</th><th>完整prefill TB</th><th>已完成prefill TB</th><th>计划decode GB/步</th><th>计划AG+RS GB/步</th></tr></thead><tbody id="results"></tbody></table></div></section>
<section class="panel"><h2>相同负载对比归档TP8</h2><p>旧48组仅作为归档读取，未用新版schema重算。Decode比值对比单步时间；各侧的端到端/独立来源在CSV中保留。载荷比值使用完整阶段的计划字节量。旧引擎缺失的embedding/output边界通信，在新SP中明确计入。</p><div class="scroll"><table><thead><tr><th>案例</th><th>旧TTFT s</th><th>新TTFT s</th><th>TTFT 新/旧</th><th>旧decode ms</th><th>新decode ms</th><th>decode 新/旧</th><th>计划payload 新/旧</th></tr></thead><tbody id="compare"></tbody></table></div></section>
<section class="panel"><h2>流量口径与文件</h2><p>GB=10^9B，TB=10^12B；CSV以byte、byte/s、second为单位。全超节点逻辑transport payload按有向发送计一次。Dispatch/combine与all_to_all_v是相同数据的两种视图；AG/RS另计。全部TP/SP通信留在SU内；柜间MoE的四跳物理hopbytes是柜间逻辑payload的4倍。失败阶段实际流量只包括已完成组件，完整计划为公式需求。</p><p>TP/SP变大使当前MoE模型复制更多TP slice的完整latent dispatch；SP16/SP32跨柜流量相对TP8为2倍/4倍。Decode等长padding还会增加AG/RS。单步网络量随batch线性增长，不随上下文变化；KV/HBM读与attention算时会随上下文增长。</p><p><a href="report.md">完整报告</a> · <a href="summary.csv">96组CSV</a> · <a href="comparison.csv">对比CSV</a> · <a href="phase_traffic.csv">实际阶段流量</a> · <a href="planned_phase_traffic.csv">计划阶段流量</a> · <a href="node_traffic.csv">节点流量</a> · <a href="rack_traffic.csv">机柜流量</a> · <a href="peak_components.csv">同峰值构成</a> · <a href="audit/parallelism-review.md">独立语义审计</a></p></section></main>
<script>const rows=__ROWS__,comparisons=__COMPARISONS__;
const f=(v,d=3)=>v==null?'—':Number(v).toLocaleString('zh-CN',{minimumFractionDigits:d,maximumFractionDigits:d});
const controls=['sp','top','context','batch'].map(id=>document.getElementById(id));
function selected(r){return controls.every((el,i)=>el.value==='all'||String(r[['sp','top_k','context','batch_size'][i]])===el.value)}
function filtered(){return rows.filter(selected)}
function plot(key,scale,title,unit){const rs=filtered().filter(r=>r[key]!=null),W=650,H=425,L=74,R=18,T=42,B=152;const max=Math.max(1,...rs.map(r=>r[key]*scale))*1.1,x=b=>L+Math.log2(b)/5*(W-L-R),y=v=>H-B-v/max*(H-T-B);let s=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${title}"><text x="${L}" y="23" font-size="17">${title} (${unit})</text>`;for(let i=0;i<=4;i++){let v=max*i/4;s+=`<line x1="${L}" y1="${y(v)}" x2="${W-R}" y2="${y(v)}" stroke="#e2e9ef"/><text x="${L-8}" y="${y(v)+4}" font-size="11" text-anchor="end">${f(v,v>10?0:2)}</text>`}for(let b of [1,2,4,8,16,32])s+=`<text x="${x(b)}" y="${H-B+22}" font-size="12" text-anchor="middle">${b}</text>`;s+=`<text x="${W-R}" y="${H-B+40}" text-anchor="end" font-size="11">Batch</text>`;const groups={};for(let r of rs){let g=`SP${r.sp}/Top${r.top_k}/${r.context}`+(key==='decode_step_s'&&r.decode_source==='standalone_cache_resident'?'*':'');(groups[g]??=[]).push(r)}const colors=['#166e9c','#d2742b','#248768','#8659a4','#b33758','#617d35','#9c6d21','#397fbb','#176565','#b36a81','#7a6234','#6d71b4','#3b8c4b','#bc543d','#7d5498','#247b96'];Object.entries(groups).forEach(([g,points],i)=>{points.sort((a,b)=>a.batch_size-b.batch_size);let color=colors[i%colors.length];s+=`<polyline fill="none" stroke="${color}" stroke-width="2" stroke-dasharray="${g.endsWith('*')?'6 4':'none'}" points="${points.map(r=>x(r.batch_size)+','+y(r[key]*scale)).join(' ')}"/>`;for(let r of points)s+=`<circle cx="${x(r.batch_size)}" cy="${y(r[key]*scale)}" r="3.5" fill="${color}"><title>${r.case}: ${f(r[key]*scale)} ${unit}${g.endsWith('*')?' (独立decode)':''}</title></circle>`;s+=`<text x="${L+i%4*145}" y="${H-B+62+Math.floor(i/4)*18}" fill="${color}" font-size="10">${g}</text>`});if(!rs.length)s+=`<text x="${L}" y="100" font-size="14" fill="#97612b">所选场景暂无已完成数值</text>`;return s+'</svg>'}
function render(){document.getElementById('results').innerHTML=filtered().map(r=>`<tr><td>${r.case}</td><td class="${r.prefill_status==='feasible'?'good':'bad'}">${r.prefill_status}</td><td>${f(r.ttft_s)}</td><td>${f(r.tpot_s==null?null:r.tpot_s*1000)}</td><td>${f(r.standalone_decode_step_s==null?null:r.standalone_decode_step_s*1000)}</td><td>${f(r.prefill_peak_hbm_bytes==null?null:r.prefill_peak_hbm_bytes/1e9)}</td><td>${f(r.failure_required_hbm_bytes==null?null:r.failure_required_hbm_bytes/1e9)}</td><td>${f(r.prefill_planned_payload_bytes/1e12)}</td><td>${f(r.prefill_actual_payload_bytes==null?null:r.prefill_actual_payload_bytes/1e12)}</td><td>${f(r.decode_planned_payload_bytes/1e9)}</td><td>${f((r.decode_planned_ag_payload_bytes+r.decode_planned_rs_payload_bytes)/1e9)}</td></tr>`).join('');document.getElementById('compare').innerHTML=comparisons.filter(selected).map(r=>`<tr><td>${r.case}</td><td>${f(r.baseline_ttft_s)}</td><td>${f(r.sp_ttft_s)}</td><td>${f(r.ttft_ratio_sp_over_tp8)}</td><td>${f(r.baseline_decode_step_s==null?null:r.baseline_decode_step_s*1000)}</td><td>${f(r.sp_decode_step_s==null?null:r.sp_decode_step_s*1000)}</td><td>${f(r.decode_ratio_sp_over_tp8)}</td><td>${f(r.decode_payload_ratio_sp_over_tp8)}</td></tr>`).join('');document.getElementById('ttft').innerHTML=plot('ttft_s',1,'TTFT','s');document.getElementById('decode').innerHTML=plot('decode_step_s',1000,'单步decode','ms')}
controls.forEach(el=>el.onchange=render);render();</script></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', nargs='?', type=Path, default=ROOT / 'outputs/kimi-k3-10t-sp')
    parser.add_argument('--baseline', type=Path, default=ROOT / 'outputs/kimi-k3-10t-1024')
    args = parser.parse_args()
    directory = args.directory.resolve()
    data = collect(directory)
    baseline = archive_baseline(args.baseline.resolve())
    comparison = comparisons(data['rows'], baseline)
    racks = network_racks(data['nodes'])
    by_name = {row['case']: row for row in data['rows']}
    for rack in racks:
        rack.update({key: by_name[rack['case']][key] for key in ('sp', 'tp', 'ep')})
    for name, values in [('summary', data['rows']), ('comparison', comparison), ('phase_traffic', data['phases']),
                         ('planned_phase_traffic', data['planned']), ('node_traffic', data['nodes']),
                         ('rack_traffic', racks), ('peak_components', data['peaks'])]:
        write_csv(directory / (name + '.csv'), values)
    (directory / 'failure_diagnostics.json').write_text(json.dumps(data['failures'], ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (directory / 'report.md').write_text(md_report(data, comparison), encoding='utf-8')
    status = ('已完成' if data['complete'] else '中间结果') + f"：{data['finished']}/{len(data['rows'])}组；prefill失败时TTFT为空，独立decode单列。"
    if not all(row.get('passed') is True for row in data['reference_replays']):
        status += ' 两组隔离完整重放尚未全部通过，最终验证待完成。'
    document = HTML.replace('__STATUS__', html.escape(status)).replace('__ROWS__', json.dumps(data['rows'], ensure_ascii=False).replace('</', '<\\/'))
    document = document.replace('__COMPARISONS__', json.dumps(comparison, ensure_ascii=False).replace('</', '<\\/'))
    provenance_panel = ('<section class="panel"><p>实验期间源码有并发修改；报告按恢复的冻结runtime校验三个首案例SHA锚点、96份配置SHA和producer日志，原始磁盘观测指纹全部保留。磁盘观测与已加载runtime的差异单列，不声称与当前引擎代码相同。'
                        '本实验与可执行冻结版本位于独立目录outputs/kimi-k3-10t-sp及其runtime-frozen/original。'
                        '<a href="audit/runtime-provenance.json">Runtime证据</a> · <a href="runtime-frozen/manifest.json">冻结源码清单</a> · '
                        '<a href="audit/frozen-reference-replays/sp16-top32-1M-b1.json">SP16完整重放</a> · '
                        '<a href="audit/frozen-reference-replays/sp32-top64-1M-b32.json">SP32完整重放</a></p></section>')
    document = document.replace('<section class="panel">', provenance_panel + '<section class="panel">', 1)
    (directory / 'report.html').write_text(document, encoding='utf-8')
    status_data = {'complete': data['complete'], 'finished_cases': data['finished'], 'case_count': len(data['rows']),
                   'actual_phase_rows': len(data['phases']), 'planned_phase_rows': len(data['planned']),
                   'node_rows': len(data['nodes']), 'rack_rows': len(racks), 'comparison_rows': len(comparison),
                   'fingerprints': data['fingerprints'], 'read_issues': data['issues'], 'traffic_checks_passed': True,
                   'runtime_snapshot_id': data['runtime_provenance']['runtime_snapshot_id'],
                   'runtime_verified_case_count': data['runtime_provenance']['verified_case_count'],
                   'observed_disk_fingerprints_drifting_from_runtime': data['runtime_provenance']['observed_disk_fingerprints_drifting_from_runtime'],
                   'runtime_verification': 'Frozen original nine-module source, three initial-worker anchors, original-normalized config SHA256, original process lifetime and matching case-completion logs; original disk observation headers preserved.',
                   'ordered_disk_observer_scope': 'Ordered nine-module disk observation; not the full current engine fingerprint (current runner now includes cache as tenth module).',
                   'reference_replays': [{key: row.get(key) for key in ('case', 'passed', 'evidence_file', 'runtime_snapshot_id', 'numeric_values_checked', 'maximum_absolute_numeric_difference')} for row in data['reference_replays']],
                   'reference_replay_numeric_values_checked': sum(row.get('numeric_values_checked', 0) for row in data['reference_replays']),
                   'baseline': 'archived prior TP8 run; existing baseline files not modified',
                   'baseline_recorded_fingerprints': {row['case']: row.get('archived_fingerprint') for row in baseline.values()},
                   'generated_at_Asia_Shanghai': datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')}
    (directory / 'report_status.json').write_text(json.dumps(status_data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in status_data.items()
                      if key not in ('fingerprints', 'baseline_recorded_fingerprints')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
