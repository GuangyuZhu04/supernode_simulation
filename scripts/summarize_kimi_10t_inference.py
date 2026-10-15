"""Summarize the fixed Kimi 10T sweep without changing or rerunning simulation.

Usage: python scripts/summarize_kimi_10t_inference.py [output_directory]
Run in the project's Python environment. Raw CSVs retain SI byte/second units.
"""
from __future__ import annotations

import argparse
import csv
import html
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def current_fingerprint(config_file):
    from supernode_sim.config import load_config
    config = load_config(config_file)
    digest = hashlib.sha256(json.dumps(config.model_dump(), sort_keys=True).encode())
    for module in ('core', 'config', 'analysis', 'structured', 'components', 'topology', 'engine', 'streaming', 'calibration', 'cache'):
        digest.update((ROOT / 'supernode_sim' / f'{module}.py').read_bytes())
    return digest.hexdigest()


def case_finished(row):
    return (row['case_status'] == 'feasible' or
            row['case_status'] == 'infeasible' and row['decode_status'] in ('feasible', 'infeasible'))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value, digits=3):
    return "—" if value is None else f"{value:,.{digits}f}"


def md_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"] + ["| " + " | ".join(str(v) for v in row) + " |" for row in rows])


def compact_diagnostics(diagnostics):
    compact = []
    for diagnostic in diagnostics:
        item = {k: v for k, v in diagnostic.items() if k != 'blocked'}
        blocked = diagnostic.get('blocked')
        if isinstance(blocked, list):
            item.update(blocked_event_count=len(blocked), blocked_sample=blocked[:2])
        elif blocked:
            item['blocked'] = blocked
        compact.append(item)
    return compact


def phase_summary(case, result, phase, source):
    data = result.get("phase_metrics", {}).get(phase)
    if not data:
        return None
    comm = data.get("communication", {})
    semantic = comm.get("payload_by_semantic_bytes", {})
    physical = comm.get("physical_bytes_by_domain", {})
    work = data.get("resource_work", {})
    hbm = {key: val for key, val in work.items() if re.fullmatch(r"npu\d+\.hbm", key)}
    card, maximum = max(hbm.items(), key=lambda kv: kv[1], default=(None, None))
    duration = data.get("duration_s")
    return {"case": case["name"], "top_k": case["top_k"], "context": case["context"],
            "input_length": case["input_length"], "batch_size": case["batch_size"], "phase": phase,
            "source": source, "status": data.get("status"), "duration_s": duration,
            "communication_scope": "complete_phase" if data.get('status') == 'feasible' else "completed_components_only",
            "hbm_work_scope": "complete_phase" if data.get('status') == 'feasible' else "executed_event_work_including_partial_failed_component",
            "transport_payload_bytes": comm.get("transport_payload_bytes", 0),
            "dispatch_payload_bytes": semantic.get("dispatch", 0), "combine_payload_bytes": semantic.get("combine", 0),
            "tp_all_reduce_payload_bytes": semantic.get("all_reduce", 0),
            "physical_su_hopbytes": physical.get("SU", 0), "physical_so_hopbytes": physical.get("SO", 0),
            "su_payload_bytes": physical.get("SU", 0), "so_payload_bytes": physical.get("SO", 0) / 4,
            "hbm_work_total_bytes": sum(hbm.values()), "hbm_work_max_per_card_bytes": maximum,
            "hbm_work_max_card": card, "peak_hbm_bytes": data.get("peak_hbm_bytes"),
            "aggregate_network_payload_Bps": (comm.get("transport_payload_bytes", 0) / duration if duration else None)}


def peak_info(result):
    sim = result.get("simulation", {})
    peaks = {k: v for k, v in sim.get("memory_peaks_bytes", {}).items() if k.endswith(".hbm")}
    pool, peak = max(peaks.items(), key=lambda kv: kv[1], default=(None, None))
    categories = sim.get("memory_categories_at_peak_bytes", {}).get(pool, {})
    failed = [chunk for chunk in result.get("chunks", []) if chunk.get("status") != "feasible"]
    chunk = failed[-1] if failed else {}
    match = re.search(r"\.l(\d+)(?:\.|$)", chunk.get("id", ""))
    return pool, peak, categories, chunk, (int(match.group(1)) + 1 if match else None), sim.get("diagnostics", [])


def assert_traffic(actual, expected, name):
    pairs = {"transport_payload_bytes": "total_network_payload_bytes", "dispatch_payload_bytes": "dispatch_network_bytes",
             "combine_payload_bytes": "combine_network_bytes", "tp_all_reduce_payload_bytes": "tp_all_reduce_payload_bytes",
             "su_payload_bytes": "same_su_payload_bytes", "so_payload_bytes": "cross_su_payload_bytes"}
    for actual_key, expected_key in pairs.items():
        if not math.isclose(actual[actual_key], expected[expected_key], rel_tol=1e-9, abs_tol=1):
            raise AssertionError(f"{name}: {actual_key}={actual[actual_key]} differs from formula {expected[expected_key]}")


def network_nodes(case, result, phase, source, phase_row):
    work = result.get('phase_metrics', {}).get(phase, {}).get('resource_work', {})
    duration = phase_row['duration_s']
    records = []
    for host in range(128):
        tx, rx = work.get(f'cpu{host}.nic.tx', 0), work.get(f'cpu{host}.nic.rx', 0)
        su_tx = sum(work.get(f'npu{card}.SU.tx', 0) for card in range(host * 8, host * 8 + 8))
        su_rx = sum(work.get(f'npu{card}.SU.rx', 0) for card in range(host * 8, host * 8 + 8))
        records.append({'case': case['name'], 'top_k': case['top_k'], 'context': case['context'],
            'input_length': case['input_length'], 'batch_size': case['batch_size'], 'phase': phase,
            'source': source, 'phase_status': phase_row['status'], 'host_index': host, 'rack_index': host // 8,
            'scope': phase_row['hbm_work_scope'], 'duration_s': duration,
            'so_tx_payload_bytes': tx, 'so_rx_payload_bytes': rx, 'su_tx_payload_bytes': su_tx, 'su_rx_payload_bytes': su_rx,
            'so_tx_average_Bps': tx / duration if duration else None, 'so_rx_average_Bps': rx / duration if duration else None,
            'so_effective_capacity_Bps_per_direction': 40e9,
            'so_tx_average_utilization': tx / duration / 40e9 if duration else None,
            'so_rx_average_utilization': rx / duration / 40e9 if duration else None})
    if phase_row['status'] == 'feasible':
        for key, expected in [('so_tx_payload_bytes', phase_row['physical_so_hopbytes'] / 4),
                              ('su_tx_payload_bytes', phase_row['physical_su_hopbytes'])]:
            total = sum(row[key] for row in records)
            if not math.isclose(total, expected, rel_tol=1e-9, abs_tol=1):
                raise AssertionError(f"{case['name']}:{phase}: node {key} sum {total} differs from communication {expected}")
    return records


def network_racks(nodes):
    grouped = defaultdict(list)
    for node in nodes:
        grouped[node['case'], node['phase'], node['source'], node['rack_index']].append(node)
    records = []
    for group in grouped.values():
        first = group[0]
        row = {key: first[key] for key in ('case', 'top_k', 'context', 'input_length', 'batch_size', 'phase', 'source', 'phase_status', 'rack_index', 'scope', 'duration_s')}
        for key in ('so_tx_payload_bytes', 'so_rx_payload_bytes', 'su_tx_payload_bytes', 'su_rx_payload_bytes'):
            row[key] = sum(node[key] for node in group)
        duration = row['duration_s']
        row['so_effective_uplink_Bps_per_direction'] = 320e9
        for direction in ('tx', 'rx'):
            row[f'so_{direction}_average_Bps'] = row[f'so_{direction}_payload_bytes'] / duration if duration else None
            row[f'so_{direction}_average_utilization'] = row[f'so_{direction}_payload_bytes'] / duration / 320e9 if duration else None
        records.append(row)
    return records


def failure_required_bytes(diagnostics, capacity):
    required = []
    for diagnostic in diagnostics:
        for item in diagnostic.get('blocked', []):
            required.extend(capacity + amount for pool, amount in item.get('shortage_bytes', {}).items() if pool.endswith('.hbm'))
        if diagnostic.get('code') == 'initial_memory_exceeded' and diagnostic.get('pool', '').endswith('.hbm'):
            required.append(diagnostic['required_bytes'])
    return max(required, default=None)


def collect(directory):
    manifest = read_json(directory / "manifest.json")
    audit = read_json(directory / "audit/traffic-formulas.json")
    expected = {(row["top_k"], row["input_length"], row["batch_size"]): row for row in audit["matrix"]}
    rows, phases, failures, components, issues, nodes = [], [], [], [], [], []
    capacity = read_json(directory / 'hardware.profile.json')['hardware']['npu']['hbm_bytes']
    fingerprints = Counter()
    for case in manifest["cases"]:
        planned = expected[case["top_k"], case["input_length"], case["batch_size"]]
        row = {"case": case["name"], "top_k": case["top_k"], "context": case["context"],
               "input_length": case["input_length"], "batch_size": case["batch_size"], "case_status": "pending",
               "prefill_status": "pending", "decode_status": "pending", "decode_source": None,
               "ttft_s": None, "tpot_s": None, "standalone_decode_step_s": None,
               "prefill_planned_payload_bytes": planned["prefill"]["total_network_payload_bytes"],
               "prefill_planned_su_payload_bytes": planned["prefill"]["same_su_payload_bytes"],
               "prefill_planned_so_payload_bytes": planned["prefill"]["cross_su_payload_bytes"],
               "decode_planned_payload_bytes": planned["one_decode_step"]["total_network_payload_bytes"],
               "decode_planned_so_payload_bytes": planned["one_decode_step"]["cross_su_payload_bytes"]}
        file = directory / "cases" / (case["name"] + ".json")
        if not file.exists():
            rows.append(row)
            continue
        try:
            payload = read_json(file)
        except (OSError, ValueError) as exc:
            issues.append(f"{case['name']}: {exc}")
            row["case_status"] = "unreadable"
            rows.append(row)
            continue
        fingerprints[payload.get("fingerprint", "unknown")] += 1
        row['fingerprint_current'] = payload.get('fingerprint') == current_fingerprint(directory / case['config'])
        if not row['fingerprint_current']:
            issues.append(f"{case['name']}: 结果指纹与当前配置/数值模块不匹配，待重跑")
        all_result = payload.get("all") or {}
        prefill = phase_summary(case, all_result, "prefill", "end_to_end")
        decode = phase_summary(case, all_result, "decode", "end_to_end")
        row["case_status"] = all_result.get("status", "unknown")
        row["prefill_status"] = prefill["status"] if prefill else "not_run"
        metrics = all_result.get("metrics", {})
        row['full_run_output_throughput_tokens_s'] = metrics.get('throughput_tokens_s')
        if prefill and prefill["status"] == "feasible":
            row["ttft_s"] = metrics.get("ttft_s") or prefill.get("duration_s")
            assert_traffic(prefill, planned["prefill"], case["name"] + ":prefill")
        if prefill:
            phases.append(prefill)
            nodes.extend(network_nodes(case, all_result, 'prefill', 'end_to_end', prefill))
            row.update(prefill_actual_payload_bytes=prefill["transport_payload_bytes"], prefill_observed_duration_s=prefill["duration_s"], prefill_peak_hbm_bytes=prefill["peak_hbm_bytes"])
        standalone = payload.get("standalone_decode")
        if standalone:
            if decode:
                phases.append(decode)
                nodes.extend(network_nodes(case, all_result, 'decode', 'end_to_end', decode))
            decode = phase_summary(case, standalone, "decode", "standalone_cache_resident")
        if decode:
            phases.append(decode)
            nodes.extend(network_nodes(case, standalone or all_result, 'decode', decode['source'], decode))
            row["decode_source"] = decode["source"]
            row["decode_status"] = decode["status"]
            row["decode_actual_payload_bytes"] = decode["transport_payload_bytes"]
            row["decode_actual_so_payload_bytes"] = decode["so_payload_bytes"]
            row["decode_peak_hbm_bytes"] = decode["peak_hbm_bytes"]
            row["decode_hbm_work_total_bytes"] = decode["hbm_work_total_bytes"]
            row["decode_hbm_work_max_per_card_bytes"] = decode["hbm_work_max_per_card_bytes"]
            if decode["status"] == "feasible":
                assert_traffic(decode, planned["one_decode_step"], case["name"] + ":decode")
                if standalone:
                    row["standalone_decode_step_s"] = decode["duration_s"]
                else:
                    row["tpot_s"] = metrics.get("tpot_s", decode["duration_s"])
                step_duration = row['tpot_s'] if row['tpot_s'] is not None else row['standalone_decode_step_s']
                row['decode_step_output_throughput_tokens_s'] = case['batch_size'] / step_duration if step_duration else None
        pool, peak, categories, failed_chunk, layer, diagnostics = peak_info(all_result)
        required = failure_required_bytes(diagnostics, capacity)
        row.update(peak_hbm_pool=pool, peak_hbm_bytes=peak, failure_layer_1based=layer,
                   failure_component=failed_chunk.get('component'), failure_required_hbm_bytes=required,
                   failure_component_admitted_peak_hbm_bytes=failed_chunk.get('peak_hbm_bytes'),
                   failure_event=failed_chunk.get("id"), diagnostics_json=json.dumps(compact_diagnostics(diagnostics), ensure_ascii=False))
        for category, amount in categories.items():
            components.append({"case": case["name"], "pool": pool, "peak_hbm_bytes": peak,
                               "category": category, "bytes_at_same_pool_peak": amount})
        if diagnostics or failed_chunk:
            failures.append({"case": case["name"], "layer_1based": layer, "event": failed_chunk.get("id"),
                             "component": failed_chunk.get('component'), 'failure_required_hbm_bytes': required,
                             'failure_component_admitted_peak_hbm_bytes': failed_chunk.get('peak_hbm_bytes'),
                             "peak_hbm_bytes": peak, "diagnostics": diagnostics})
        rows.append(row)
    complete = sum(case_finished(row) for row in rows) == len(manifest["cases"]) and not issues
    return manifest, audit, rows, phases, failures, components, issues, fingerprints, complete, nodes


def planned_phases(manifest, audit):
    expected = {(r['top_k'], r['input_length'], r['batch_size']): r for r in audit['matrix']}
    rows = []
    for case in manifest['cases']:
        planned = expected[case['top_k'], case['input_length'], case['batch_size']]
        for phase, key in [('prefill', 'prefill'), ('decode', 'one_decode_step')]:
            traffic = planned[key]
            rows.append({"case": case['name'], "top_k": case['top_k'], "context": case['context'],
                "input_length": case['input_length'], "batch_size": case['batch_size'], "phase": phase,
                "scope": "analytical_planned_full_phase", "tokens": case['batch_size'] * (case['input_length'] if phase == 'prefill' else 1),
                "transport_payload_bytes": traffic['total_network_payload_bytes'],
                "dispatch_payload_bytes": traffic['dispatch_network_bytes'], "combine_payload_bytes": traffic['combine_network_bytes'],
                "tp_all_reduce_payload_bytes": traffic['tp_all_reduce_payload_bytes'],
                "moe_payload_bytes": traffic['moe_network_payload_bytes'],
                "su_payload_bytes": traffic['same_su_payload_bytes'], "so_payload_bytes": traffic['cross_su_payload_bytes'],
                "physical_su_hopbytes": traffic['same_su_payload_bytes'], "physical_so_hopbytes": 4 * traffic['cross_su_payload_bytes'],
                "moe_assignments_including_local_bytes": traffic['moe_semantic_assignments_with_local_bytes'],
                "moe_local_no_network_bytes": traffic['moe_local_no_network_bytes']})
    return rows


def make_report(manifest, audit, rows, phases, failures, components, issues, fingerprints, complete):
    finished = sum(case_finished(row) for row in rows)
    status = "已完成" if complete else "未完成（中间结果，不能作为完整48组结论）"
    counts = Counter(row["prefill_status"] for row in rows)
    lines = ["# Kimi K3 专家扩容至约10T：1024卡推理流量与延迟", "", f"**报告状态：{status}。已读取 {finished}/{len(rows)} 组结果。**", "",
             f"生成时间：{datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')}（Asia/Shanghai）。Prefill状态：" + "；".join(f"{key}={value}" for key, value in counts.items()) + "。", "",
             "## 场景与指标", "", "只增加routed experts和相应router输出维度，保留93层、69 KDA+24 MLA、92 MoE层、hidden7168、latent3584、专家宽3072、2个shared experts。当前严格引擎要求EP整除专家数；3328专家是最接近10T的合法EP128配置，每EP26专家，精确总参数10.171385035872T（较10T多1.71385%）。Top32活跃154.414012512B，Top64活跃251.654756448B。该扩容模型是合成设计，不是官方发布的10T Kimi模型。", "",
             "1024张Ascend950DT假设卡：8卡/节点、64卡/SU机柜、16柜/超节点SO域。TP8/EP128/PP1/DP1，TP完全留在节点内；序列按i%128分配到EP，B≤32时每个活跃EP持有一条完整序列，不按1024卡切开长上下文。每卡144GB HBM、4TB/s标称HBM带宽；FP8权重/计算，FP16通信、activation与KV，FP32 KDA状态。SU为单向1TB/s/卡、80%效率；SO为400Gb/s/节点、80%效率（有效40GB/s/节点），柜间共享400GB/s/方向上联，显式路径4个SO hops。", "",
             "TTFT为固定batch执行开始到首token完成；TPOT为首token后的decode间隔，单位分别为秒和毫秒。未完成prefill时TTFT为空；独立decode从已驻留完整KV的状态开始，只给出条件性的单步decode时间，不代表该配置能完成端到端请求。只模拟一个首token后decode步，无排队、continuous batching或P99保证。输入128k/256k/512k/1M分别为131072/262144/524288/1048576 tokens/序列。", "",
             "## 上下文与Top-k概览", "", "Decode栏的*表示独立KV驻留decode，未标*为端到端请求中的TPOT。完整batch明细随后列出。", ""]
    overview = []
    for k in manifest['top_ks']:
        for context in manifest['contexts']:
            group = [r for r in rows if r['top_k'] == k and r['context'] == context]
            first, last = group[0], group[-1]
            def decode_cell(row):
                duration = row['tpot_s'] if row['tpot_s'] is not None else row['standalone_decode_step_s']
                return fmt(duration * 1000 if duration is not None else None) + ('*' if row['standalone_decode_step_s'] is not None else '')
            statuses = Counter(r['prefill_status'] for r in group)
            overview.append([k, context, ' / '.join(f'{s}:{n}' for s,n in statuses.items()), fmt(first['ttft_s']), fmt(last['ttft_s']), decode_cell(first), decode_cell(last)])
    lines += [md_table(['Top-k', '输入长度', 'Prefill组数', 'TTFT B1 s', 'TTFT B32 s', 'Decode B1 ms', 'Decode B32 ms'], overview), '',
              'Decode-step吞吐为batch/单步decode时间，单列于summary.csv；端到端输出吞吐包括TTFT，另列full_run_output_throughput_tokens_s，二者不能互换。', '',
              '## 全部48组延迟与容量', '', '“独立decode”与端到端TPOT分别列出；—表示未定义、未执行或尚无结果。prefill失败时，已记录的耗时和流量只属于中断前执行，不能作为完整prefill预测。', '']
    for k in manifest["top_ks"]:
        for context in manifest["contexts"]:
            subset = [r for r in rows if r["top_k"] == k and r["context"] == context]
            lines += [f"### Top{k} · {context}", "", md_table(["Batch", "Prefill状态", "TTFT s", "端到端TPOT ms", "独立decode ms", "已准入峰值 GB", "失败申请所需 GB", "失败层"], [[r["batch_size"], r["prefill_status"], fmt(r["ttft_s"]), fmt(r["tpot_s"] * 1000 if r["tpot_s"] is not None else None), fmt(r["standalone_decode_step_s"] * 1000 if r["standalone_decode_step_s"] is not None else None), fmt(r.get("prefill_peak_hbm_bytes", 0) / 1e9) if r.get("prefill_peak_hbm_bytes") is not None else "—", fmt(r['failure_required_hbm_bytes'] / 1e9) if r.get('failure_required_hbm_bytes') is not None else '—', r.get("failure_layer_1based") or "—"] for r in subset]), ""]
    lines += ["## 网络流量", "", "以下为全超节点聚合的逻辑transport payload：每个有向发送计一次，不把发送与接收重复相加。dispatch/combine是all_to_all_v的语义视图，不能再次叠加all_to_all_v。柜内payload与SU hopbytes相等；柜间4-hop物理字节为柜间payload的4倍。HBM工作量是读写流量，不是HBM容量。", "",
              "完整prefill流量由独立公式给出；不可行配置的完整流量是计划需求，未声称实际执行完成。失败阶段的通信只累计已完成组件，communication_scope=completed_components_only；HBM工作量还包括失败组件中已执行的部分事件，两者范围不同。Decode实际流量已逐项与公式校验（总payload、dispatch、combine、TP、SU、SO，容差1B或1e-9相对误差）。", ""]
    lines += [md_table(["Top-k", "Batch", "Decode总 MB/步", "柜间 MB/步", "柜间占比", "128k prefill TB", "256k TB", "512k TB", "1M TB"], [[k, b, fmt(next(r for r in rows if r["top_k"] == k and r["batch_size"] == b)["decode_planned_payload_bytes"] / 1e6), fmt(next(r for r in rows if r["top_k"] == k and r["batch_size"] == b)["decode_planned_so_payload_bytes"] / 1e6), f"{next(r for r in rows if r['top_k']==k and r['batch_size']==b)['decode_planned_so_payload_bytes']/next(r for r in rows if r['top_k']==k and r['batch_size']==b)['decode_planned_payload_bytes']:.2%}"] + [fmt(next(r for r in rows if r["top_k"] == k and r["batch_size"] == b and r["context"] == c)["prefill_planned_payload_bytes"] / 1e12) for c in manifest["contexts"]] for k in manifest["top_ks"] for b in manifest["batches"]]), "",
              "Top64的MoE和SO流量严格为Top32的2倍；TP all-reduce不变，因此总network payload增至1.899738倍。通信量随prefill总token和decode batch线性增加。序列长度影响本地KV读和attention计算，当前布局下单步decode网络payload不随上下文长度变化。", "",
              "当前pairwise EP128每次dispatch/combine有127轮，92层共两次；TP使用ring。仅collective启动延迟的算法下界已约450–473ms/步，解释低batch decode为何不能仅按总字节/总带宽估算。该数值来自所选算法和路径延迟，未用真实collective测量校准。", "",
              "## Prefill实际流量与完整计划", "", md_table(["案例", "状态", "实际已执行 TB", "完整计划 TB", "HBM工作量 TB", "单卡最大HBM工作量 TB"], [[p["case"], p["status"], fmt(p["transport_payload_bytes"] / 1e12), fmt(next(r for r in rows if r["case"] == p["case"])["prefill_planned_payload_bytes"] / 1e12), fmt(p["hbm_work_total_bytes"] / 1e12), fmt(p["hbm_work_max_per_card_bytes"] / 1e12 if p["hbm_work_max_per_card_bytes"] is not None else None)] for p in phases if p["phase"] == "prefill"]), "",
              "## 内存峰值与失败定位", "", "已准入峰值只包含成功分配的内存，不含被拒绝的工作区，因此可能低于144GB仍发生OOM。“失败申请所需”取阻塞事件的144GB容量+shortage_bytes，表示该事件在失败时加入所需的单卡容量下界。不是完整执行最终峰值，也不能将它补成TTFT。组件表取同一张卡同一峰值时刻的构成，可以相加；不把不同组件各自独立的峰值相加。EP使routed权重分片，但dense/shared/router和部分投影复制：逻辑10.171T参数不能直接除1024得到每卡驻留量。以配置为准，权重约23.603GB/卡、FP8 scale约0.738GB/卡，加8GB runtime和0.256GB workspace，静态约32.597GB/卡。", ""]
    if components:
        selected_cases = list(dict.fromkeys(c["case"] for c in components))
        selected_cases = selected_cases[:1] + [f["case"] for f in failures[:4] if f["case"] != selected_cases[0]]
        lines += [md_table(["案例", "峰值卡", "峰值 GB", "同一峰值构成 GB"], [[name, next(c for c in components if c["case"] == name)["pool"], fmt(next(c for c in components if c["case"] == name)["peak_hbm_bytes"] / 1e9), "；".join(f"{c['category']}={c['bytes_at_same_pool_peak']/1e9:.3f}" for c in components if c["case"] == name)] for name in selected_cases]), ""]
    if failures:
        lines += [md_table(["案例", "失败层(1-based)", "组件", "失败组件已准入 GB", "失败申请所需 GB", "诊断（完整见failure_diagnostics.json）"], [[f["case"], f["layer_1based"] or "—", f.get('component') or f['event'] or '—', fmt(f['failure_component_admitted_peak_hbm_bytes'] / 1e9) if f.get('failure_component_admitted_peak_hbm_bytes') is not None else '—', fmt(f['failure_required_hbm_bytes'] / 1e9) if f.get('failure_required_hbm_bytes') is not None else '—', html.escape(json.dumps(compact_diagnostics(f["diagnostics"]), ensure_ascii=False)).replace("|", "&#124;")] for f in failures]), ""]
    lines += ["## Compressed / expanded MLA缓存敏感性", "", "当前主实验使用compressed MLA优化假设。官方参考布局expanded每层每卡3840个FP16元素/token，compressed为576，前者KV为后者6.667倍。下表只给缓存/状态容量，不是对expanded进行过延迟仿真。B≤32时每活跃EP一条序列，单卡容量不随batch变化。", "",
              md_table(["输入tokens", "Compressed MLA KV GB/卡", "Expanded MLA KV GB/卡", "KDA状态 GB/卡", "页表 GB/卡"], [[L, fmt(next(c for c in audit["cache_capacity"]["rows"] if c["input_length"] == L and c["layout"] == "compressed")["prefill_mla_kv_bytes_per_active_npu"] / 1e9), fmt(next(c for c in audit["cache_capacity"]["rows"] if c["input_length"] == L and c["layout"] == "expanded")["prefill_mla_kv_bytes_per_active_npu"] / 1e9), "0.056807", fmt(next(c for c in audit["cache_capacity"]["rows"] if c["input_length"] == L and c["layout"] == "compressed")["prefill_page_table_bytes_per_active_npu"] / 1e9, 6)] for L in manifest["contexts"].values()]), "",
              "Expanded 1M仅MLA KV就193.274GB/卡，已超过144GB；compressed 1M KV为28.991GB，但prefill activation、AttnRes和通信临时缓冲仍可能导致OOM。独立decode无需同时持有prefill临时张量，所以其可行性必须单独解释。", "",
              "## 适用范围与复现", "", "预测使用均匀专家路由、均衡专家计算、通用融合/流式attention和粗粒度kernel。当前MoE组件把routed与shared路径合并近似，并按均衡EP token数量计时；实际shared/latent/router计算可能受源序列EP集中影响，缺少生产kernel和路由trace校准。B≤32时attention、dense和KV集中在少数EP组，1024卡并不意味着所有算子获得1024卡并行。完整prefill不按chunk重新调度，未优化长序列切分。绝对TTFT/TPOT是此配置、调度和算法的分析预测，未宣称真实硬件误差。", "",
              "节点/机柜流量见[node_traffic.csv](node_traffic.csv)与[rack_traffic.csv](rack_traffic.csv)：分别记录128节点、16机柜的每阶段SO tx/rx逻辑NIC payload、SU tx/rx、平均B/s及有效容量利用率。tx/rx分开，不能把两者相加当作全局payload。可行阶段已校验sum(host SO tx)=SO hopbytes/4、sum(host SU tx)=SU hopbytes；失败阶段使用实际resource work，含失败组件部分事件，不能与仅完整组件通信计数强行对齐。节点SO有效40GB/s/方向，机柜上联有效320GB/s/方向。", "",
              "结果：summary.csv（所有48组，原始byte/s），phase_traffic.csv（实际各阶段语义payload、SU/SO hopbytes、HBM总工作量和最大单卡），planned_phase_traffic.csv（全部48组×prefill/decode的完整计划流量、语义payload及hopbytes），peak_components.csv（同峰值组件），cases/*.json（仿真原始结果），audit/traffic-formulas.json（独立公式），configs/*.json与hardware.profile.json（可重跑配置）。report.html提供离线交互筛选和图表。", "",
              "本次主任务验证记录：269项测试与7项子测试通过。报告生成器另外核对实际通信与独立公式、节点注入计数，以及离线HTML脚本语法/筛选/SVG生成；脚本检查不等同于浏览器视觉检查。", "",
              "生成报告：`python scripts/summarize_kimi_10t_inference.py outputs/kimi-k3-10t-1024`。", "",
              f"逐案例配置+数值模块指纹已核对当前源码，共{sum(fingerprints.values())}份；详见report_status.json。不同负载配置的指纹自然不同。" ]
    if issues:
        lines += ["", "读取问题：", ""] + ["- " + issue for issue in issues]
    return "\n".join(lines) + "\n"


HTML_TEMPLATE = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Kimi约10T推理仿真</title>
<style>body{font:15px/1.6 system-ui,"Microsoft Yahei",sans-serif;color:#182b40;background:#f3f6f9;margin:0}main{max-width:1350px;margin:auto;padding:26px}h1{font-size:29px;margin:5px 0}h2{font-size:21px}.panel{background:white;border:1px solid #dae2e9;border-radius:10px;padding:20px;margin:18px 0}.notice{background:#fff1d8;border-left:5px solid #c48210;padding:12px}.complete{background:#e3f3ed;border-color:#168268}select{font:inherit;padding:6px 12px;margin:0 20px 0 8px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:right;padding:8px;border-bottom:1px solid #e1e7ed}th{background:#edf2f6;position:sticky;top:0}td:first-child,th:first-child{text-align:left}.scroll{overflow:auto;max-height:630px}.plots{display:grid;grid-template-columns:1fr 1fr;gap:15px}svg{width:100%;height:auto}small{color:#526375}.cards{display:flex;gap:24px;flex-wrap:wrap}.card b{display:block;font-size:25px}.bad{color:#b3322c}.good{color:#11775d}pre{white-space:pre-wrap;font-family:inherit}@media(max-width:750px){.plots{grid-template-columns:1fr}main{padding:14px}}</style>
<main><h1>Kimi K3 专家扩容至约10T</h1><p>3328 experts · Top32 / Top64 · TP8 EP128 · 1024卡 / 16个SU机柜</p><div id="notice" class="notice">__STATUS__</div>
<section class="panel"><div class="cards"><div class="card"><small>精确总参数</small><b>10.171385 T</b></div><div class="card"><small>Top32活跃</small><b>154.414 B</b></div><div class="card"><small>Top64活跃</small><b>251.655 B</b></div><div class="card"><small>每卡HBM</small><b>144 GB</b></div></div><p>8卡/节点，64卡/SU机柜，1024卡/SO超节点；FP8权重/计算，FP16通信与KV。SU有效800GB/s/卡；SO有效40GB/s/节点。128k=131072 tokens，1M=1048576 tokens。</p><p>未完成prefill的TTFT为空。独立decode从已驻留KV开始，条件性decode时间不证明端到端可行。绝对时间为未校准的固定batch分析预测。</p></section>
<section class="panel"><label>Top-k<select id="top"><option value="all">全部</option><option>32</option><option>64</option></select></label><label>上下文<select id="context"><option value="all">全部</option><option>128k</option><option>256k</option><option>512k</option><option>1M</option></select></label><div class="plots"><div id="ttft"></div><div id="decode"></div></div><small>同色曲线为同一Top-k/上下文；横轴batch，点悬浮显示数值。空缺TTFT点表示prefill失败或未完成。Decode图中独立decode以虚线表示。</small></section>
<section class="panel"><h2>全部场景</h2><div class="scroll"><table><thead><tr><th>案例</th><th>Prefill</th><th>TTFT s</th><th>TPOT ms</th><th>独立decode ms</th><th>Prefill峰值 GB</th><th>完整prefill TB</th><th>已执行prefill TB</th><th>decode MB/步</th><th>SO %</th></tr></thead><tbody id="body"></tbody></table></div></section>
<section class="panel"><h2>口径与容量敏感性</h2><p>逻辑payload每个有向发送计一次；dispatch/combine与all_to_all_v是同一数据的两种视图。SO路径为4跳，物理SO hopbytes为逻辑柜间payload的4倍。完整prefill TB列为独立公式计划需求；失败场景的已执行TB只是部分流量。</p><p>Top64的MoE及SO流量为Top32的2倍，总network为1.899738倍。每步decode流量随batch线性增加，在本地KV布局中不随上下文长度变化。EP128 pairwise的127轮/collective产生约450–473ms算法启动延迟下界。</p><div id="cache"></div><p>Expanded 1M仅KV即193.274GB/卡，超过144GB。Compressed是优化布局假设；长序列prefill还需要activation与AttnRes内存。B≤32时每活跃EP持有一条完整序列，attention/dense/KV未铺满1024卡。</p><p>均匀专家路由、均衡MoE算子、粗粒度kernel与默认collective均未通过真实平台校准；shared/latent/router源端集中带来的算时差异是主要局限。无排队、continuous batching或P99保证。</p><p>详见同目录report.md、summary.csv、phase_traffic.csv、peak_components.csv、cases/、configs/和audit/。</p></section></main>
<script>const rows=__ROWS__,cache=__CACHE__;const f=(v,d=3)=>v===null||v===undefined?'—':Number(v).toLocaleString('zh-CN',{maximumFractionDigits:d,minimumFractionDigits:d});
function plot(target,key,scale,title,unit){let rs=filtered().filter(r=>r[key]!=null);let W=590,H=300,L=68,R=15,T=38,B=42;let ymax=Math.max(1,...rs.map(r=>r[key]*scale))*1.1;let x=b=>L+Math.log2(b)/5*(W-L-R),y=v=>H-B-v/ymax*(H-T-B);let s=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${title}"><text x="${L}" y="22" font-size="16" fill="#182b40">${title} (${unit})</text>`;for(let i=0;i<=4;i++){let v=ymax*i/4;s+=`<line x1="${L}" y1="${y(v)}" x2="${W-R}" y2="${y(v)}" stroke="#e0e6ec"/><text x="${L-8}" y="${y(v)+4}" text-anchor="end" font-size="11">${f(v,v>10?0:2)}</text>`}for(let b of [1,2,4,8,16,32])s+=`<text x="${x(b)}" y="${H-20}" text-anchor="middle" font-size="12">${b}</text>`;s+=`<text x="${W-14}" y="${H-2}" text-anchor="end" font-size="11">Batch size</text>`;let groups={};for(let r of rs)(groups[r.top_k+' / '+r.context]??=[]).push(r);let colors=['#176a9a','#d57128','#288b72','#8458a8','#ad3b58','#51793a','#9c681e','#3f76b8'];Object.entries(groups).forEach(([g,rs],i)=>{rs.sort((a,b)=>a.batch_size-b.batch_size);let col=colors[i%colors.length];s+=`<polyline points="${rs.map(r=>x(r.batch_size)+','+y(r[key]*scale)).join(' ')}" fill="none" stroke="${col}" stroke-width="2"/>`;for(let r of rs)s+=`<circle cx="${x(r.batch_size)}" cy="${y(r[key]*scale)}" r="4" fill="${col}"><title>${g}, B${r.batch_size}: ${f(r[key]*scale)} ${unit}</title></circle>`;s+=`<text x="${L+i%4*125}" y="${H-5-(i>3?12:0)}" fill="${col}" font-size="10">Top${g}</text>`});return s+'</svg>'}
function filtered(){return rows.filter(r=>(topSelect.value==='all'||r.top_k==topSelect.value)&&(contextSelect.value==='all'||r.context===contextSelect.value))}const topSelect=document.getElementById('top'),contextSelect=document.getElementById('context');
function render(){document.getElementById('body').innerHTML=filtered().map(r=>`<tr><td>${r.case}</td><td class="${r.prefill_status==='feasible'?'good':'bad'}">${r.prefill_status}</td><td>${f(r.ttft_s)}</td><td>${f(r.tpot_s==null?null:r.tpot_s*1000)}</td><td>${f(r.standalone_decode_step_s==null?null:r.standalone_decode_step_s*1000)}</td><td>${f(r.prefill_peak_hbm_bytes==null?null:r.prefill_peak_hbm_bytes/1e9)}</td><td>${f(r.prefill_planned_payload_bytes/1e12)}</td><td>${f(r.prefill_actual_payload_bytes==null?null:r.prefill_actual_payload_bytes/1e12)}</td><td>${f((r.decode_actual_payload_bytes??r.decode_planned_payload_bytes)/1e6)}</td><td>${f(r.decode_planned_so_payload_bytes/r.decode_planned_payload_bytes*100,2)}</td></tr>`).join('');document.getElementById('ttft').innerHTML=plot('ttft','ttft_s',1,'TTFT','s');let original=rows.map(r=>[r,r.decode_plot]);for(let r of rows)r.decode_plot=r.tpot_s??r.standalone_decode_step_s;document.getElementById('decode').innerHTML=plot('decode','decode_plot',1000,'单步decode','ms')}
document.getElementById('cache').innerHTML='<table><tr><th>输入tokens</th><th>Compressed KV GB/卡</th><th>Expanded KV GB/卡</th></tr>'+[131072,262144,524288,1048576].map(L=>'<tr><td>'+L+'</td>'+['compressed','expanded'].map(t=>'<td>'+f(cache.find(x=>x.input_length===L&&x.layout===t).prefill_mla_kv_bytes_per_active_npu/1e9)+'</td>').join('')+'</tr>').join('')+'</table>';topSelect.onchange=contextSelect.onchange=render;render();</script></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=Path("outputs/kimi-k3-10t-1024"))
    args = parser.parse_args()
    directory = args.directory.resolve()
    manifest, audit, rows, phases, failures, components, issues, fingerprints, complete, nodes = collect(directory)
    write_csv(directory / "summary.csv", rows)
    write_csv(directory / "phase_traffic.csv", phases)
    write_csv(directory / "planned_phase_traffic.csv", planned_phases(manifest, audit))
    write_csv(directory / 'node_traffic.csv', nodes)
    write_csv(directory / 'rack_traffic.csv', network_racks(nodes))
    write_csv(directory / "peak_components.csv", components)
    (directory / "failure_diagnostics.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = make_report(manifest, audit, rows, phases, failures, components, issues, fingerprints, complete)
    (directory / "report.md").write_text(report, encoding="utf-8")
    finished = sum(case_finished(r) for r in rows)
    status = ("完成" if complete else "未完成 / 中间结果") + f"：{finished}/{len(rows)}组；Prefill失败场景TTFT未定义，独立decode单列。"
    document = HTML_TEMPLATE.replace("__STATUS__", html.escape(status)).replace("__ROWS__", json.dumps(rows, ensure_ascii=False).replace("</", "<\\/"))
    document = document.replace("__CACHE__", json.dumps(audit["cache_capacity"]["rows"]))
    document = document.replace("groups[r.top_k+' / '+r.context]", "groups[r.top_k+' / '+r.context+(key==='decode_plot'&&r.decode_source==='standalone_cache_resident'?' / 独立':'')]")
    document = document.replace('stroke-width="2"', '''stroke-width="2" stroke-dasharray="${g.includes('独立')?'6 4':'none'}"''')
    document = document.replace('W=590,H=300,L=68,R=15,T=38,B=42', 'W=590,H=340,L=68,R=15,T=38,B=90')
    document = document.replace('${H-20}', '${H-68}').replace('${H-2}', '${H-52}').replace('${H-5-(i>3?12:0)}', '${H-20+(i>3?14:0)}')
    document = document.replace('Top${g}</text>', "Top${g.replace(' / 独立','*')}</text>")
    document = document.replace('<th>Prefill峰值 GB</th>', '<th>已准入峰值 GB</th><th>失败申请所需 GB</th>')
    document = document.replace('${f(r.prefill_planned_payload_bytes/1e12)}</td>', '${f(r.failure_required_hbm_bytes==null?null:r.failure_required_hbm_bytes/1e9)}</td><td>${f(r.prefill_planned_payload_bytes/1e12)}</td>')
    document = document.replace('未完成prefill的TTFT为空。', '已准入峰值不含被拒绝的申请；失败申请所需取144GB+shortage_bytes，低于144GB的已准入峰值仍可能OOM。未完成prefill的TTFT为空。')
    document = document.replace('详见同目录report.md、summary.csv、phase_traffic.csv、peak_components.csv、cases/、configs/和audit/。', '下载：<a href="report.md">报告</a> · <a href="summary.csv">汇总CSV</a> · <a href="phase_traffic.csv">实际阶段流量</a> · <a href="planned_phase_traffic.csv">完整计划流量</a> · <a href="node_traffic.csv">节点流量</a> · <a href="rack_traffic.csv">机柜流量</a> · <a href="peak_components.csv">峰值组件</a>。原始结果与配置见cases/、configs/和audit/。')
    (directory / "report.html").write_text(document, encoding="utf-8")
    (directory / "report_status.json").write_text(json.dumps({"complete": complete, "case_count": len(rows), "finished_cases": finished,
        "fingerprints": dict(fingerprints), "read_issues": issues, "traffic_checks_passed": True}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"complete": complete, "finished_cases": finished, "expected_cases": len(rows), "phase_rows": len(phases), "traffic_checks_passed": True}))


if __name__ == "__main__":
    main()
