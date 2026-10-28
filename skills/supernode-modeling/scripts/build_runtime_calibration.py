#!/usr/bin/env python3
"""Build environment-locked collective response curves from declared observations.

Input schema_version=1; environments=[{id, hardware, calibration_context}];
observations=[{id, environment_id, kind:'collective', split:'train'|'validation',
status:'measured'|'public'|'synthetic', evidence_kind:'measurement'|'estimate'|
'synthetic', source, duration_s, collective, semantic, phase, dtype, algorithm,
timing_scope, ranks, size_bytes, payload_matrix?, precision, metadata_match?}].
metadata_match optionally locks operator shape/format tags and is also passed
into the real graph metadata. Precision explicitly
declares compute/activation/communication. Hardware is normalized by HardwareSpec
before graph construction; output records those effective settings. Existing
collective_profiles/models are rejected rather than silently overwritten.

Required context keys: device_sku, cann_version, hccl_version, driver_version,
runtime, runtime_version, transport, mode, timer_boundary. Additional context is
also locked. Values must be known nonempty strings; unknown versions must not be
invented. timer_boundary declares device_only/whole_call separately from the
transport/activation_collective simulation scope. This version targets explicit
Ascend software contexts; it does not infer versions for other vendors.

Replicate training samples at one payload use their median, with sample count
and observed range in the report. At least two distinct training payloads are
required per exact environment/family. Validation observations never choose
training anchors. The runtime interpolates only inside the trained interval;
outside it, the actual simulator uses its physical model without extrapolation.
Every input observation is simulated before and after model application. The
full matrix's routing proportions, physical rank order and target environment
stay locked; a different routing distribution requires a separate family.

This tool does not collect measurements, rewrite an existing target hardware
file, or certify real hardware accuracy. All provenance is caller-declared.
Example: python build_runtime_calibration.py dataset.json --output outputs/runtime-fit
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
SELECTORS = ("collective", "semantic", "phase", "dtype", "algorithm", "timing_scope")
DTYPES = {"fp4", "fp8", "fp16", "bf16", "fp32", "int8"}
CONTEXT_KEYS = {"device_sku", "cann_version", "hccl_version", "driver_version", "runtime", "runtime_version", "transport", "mode", "timer_boundary"}
UNKNOWN = {"unknown", "unspecified", "not_reported", "not reported", "unavailable", "tbd", "?", "n/a"}


class CalibrationBuildError(ValueError):
    """Invalid evidence bundle or failed simulator application check."""


def _modules():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from supernode_sim.config import HardwareSpec
    from supernode_sim.core import Graph
    from supernode_sim.engine import simulate
    from supernode_sim.topology import Topology
    return HardwareSpec, Graph, Topology, simulate


def _object(value, path):
    if not isinstance(value, dict):
        raise CalibrationBuildError(f"{path} must be an object")
    return value


def _text(value, path):
    if not isinstance(value, str) or not value.strip() or value.strip().lower() in UNKNOWN:
        raise CalibrationBuildError(f"{path} must be a known nonempty string")
    return value


def _number(value, path, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalibrationBuildError(f"{path} must be a finite {'positive' if positive else 'nonnegative'} number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise CalibrationBuildError(f"{path} must be finite") from exc
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        raise CalibrationBuildError(f"{path} must be a finite {'positive' if positive else 'nonnegative'} number")
    return result


def _json_tree(value, path="dataset"):
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CalibrationBuildError(f"{path} contains a nonfinite number")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                raise CalibrationBuildError(f"{path} keys must be nonempty strings")
            _json_tree(item, f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _json_tree(item, f"{path}[{index}]")
        return
    raise CalibrationBuildError(f"{path} must contain only JSON values")


def _fields(obj, required, optional, path):
    missing, extra = required - set(obj), set(obj) - required - optional
    if missing or extra:
        raise CalibrationBuildError(f"{path}: missing fields {sorted(missing)}; unsupported fields {sorted(extra)}")


def _choice(value, choices, path):
    if not isinstance(value, str) or value not in choices:
        raise CalibrationBuildError(f"{path} must be one of {', '.join(sorted(choices))}")
    return value


def _metadata_numbers(value):
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {key: _metadata_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_metadata_numbers(item) for item in value]
    return value


def _environments(rows):
    HardwareSpec, _, _, _ = _modules()
    if not isinstance(rows, list) or not rows:
        raise CalibrationBuildError("environments must be a nonempty array")
    environments, portable_ids = {}, set()
    for index, raw in enumerate(rows):
        path = f"environments[{index}]"
        raw = _object(raw, path)
        _fields(raw, {"id", "hardware", "calibration_context"}, set(), path)
        identity = _text(raw["id"], f"{path}.id")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", identity):
            raise CalibrationBuildError(f"{path}.id must use letters, digits, dash, dot or underscore")
        if identity.casefold() in portable_ids:
            raise CalibrationBuildError(f"duplicate environment id (case-insensitive artifact name): {identity}")
        portable_ids.add(identity.casefold())
        hardware = copy.deepcopy(_object(raw["hardware"], f"{path}.hardware"))
        _text(hardware.get("name"), f"{path}.hardware.name")
        if hardware.get("collective_profiles") or hardware.get("collective_models"):
            raise CalibrationBuildError(f"{path}: existing collective calibrations must not be overwritten; provide a separate baseline hardware configuration")
        context = _object(raw["calibration_context"], f"{path}.calibration_context")
        if CONTEXT_KEYS - set(context):
            raise CalibrationBuildError(f"{path}.calibration_context missing {sorted(CONTEXT_KEYS - set(context))}")
        for key, value in context.items():
            _text(value, f"{path}.calibration_context.{key}")
        _choice(context["timer_boundary"], {"device_only", "whole_call"}, f"{path}.calibration_context.timer_boundary")
        if "calibration_context" in hardware and hardware["calibration_context"] != context:
            raise CalibrationBuildError(f"{path}: hardware.calibration_context conflicts with the environment context")
        hardware["calibration_context"] = copy.deepcopy(context)
        environments[identity] = HardwareSpec.model_validate(hardware).model_dump()
    return environments


def _observations(rows, environments):
    if not isinstance(rows, list) or not rows:
        raise CalibrationBuildError("observations must be a nonempty array")
    observations, ids = [], set()
    required = {"id", "environment_id", "kind", "split", "status", "evidence_kind", "source", "duration_s", "ranks", "size_bytes", "precision", *SELECTORS}
    for index, raw in enumerate(rows):
        path = f"observations[{index}]"
        raw = _object(raw, path)
        _fields(raw, required, {"payload_matrix", "metadata_match"}, path)
        row = copy.deepcopy(raw)
        row["metadata_match"] = _metadata_numbers(_object(row.get("metadata_match", {}), f"{path}.metadata_match"))
        reserved = {"phase", "operation", "semantic", "wire_dtype", "buffer_dtype", "collective", "algorithm", "communicator", "payload_bytes", "type", "path", "resource_domains", "domains", "round"}
        if set(row["metadata_match"]) & reserved:
            raise CalibrationBuildError(f"{path}.metadata_match must not override reserved collective metadata")
        identity = _text(row["id"], f"{path}.id")
        if identity in ids:
            raise CalibrationBuildError(f"duplicate observation id: {identity}")
        ids.add(identity)
        environment_id = _text(row["environment_id"], f"{path}.environment_id")
        if environment_id not in environments:
            raise CalibrationBuildError(f"{path}: unknown environment_id {environment_id}")
        _choice(row["kind"], {"collective"}, f"{path}.kind")
        _choice(row["split"], {"train", "validation"}, f"{path}.split")
        _choice(row["status"], {"measured", "public", "synthetic"}, f"{path}.status")
        _choice(row["evidence_kind"], {"measurement", "estimate", "synthetic"}, f"{path}.evidence_kind")
        if (row["status"] == "synthetic" and row["evidence_kind"] != "synthetic") or (row["status"] == "measured" and row["evidence_kind"] != "measurement"):
            raise CalibrationBuildError(f"{path}: status and evidence_kind contradict each other")
        for field in ("source", "semantic", "phase"):
            _text(row[field], f"{path}.{field}")
        row["duration_s"] = _number(row["duration_s"], f"{path}.duration_s", positive=True)
        row["size_bytes"] = _number(row["size_bytes"], f"{path}.size_bytes")
        _choice(row["collective"], {"all_to_all_v", "all_gather", "reduce_scatter", "all_reduce"}, f"{path}.collective")
        _choice(row["dtype"], DTYPES, f"{path}.dtype")
        _choice(row["timing_scope"], {"transport", "activation_collective"}, f"{path}.timing_scope")
        if row["timing_scope"] == "activation_collective" and row["semantic"] in {"dispatch", "combine"}:
            shape = row["metadata_match"]
            fields = {"moe_hidden_width", "moe_tensor_bytes_per_token", "moe_metadata_bytes_per_token", "moe_top_k", "moe_num_experts"}
            if fields - set(shape):
                raise CalibrationBuildError(f"{path}.metadata_match requires known MoE shape fields {sorted(fields - set(shape))}")
            for key in fields:
                minimum = 0 if key == "moe_metadata_bytes_per_token" else 1
                if type(shape[key]) is not int or shape[key] < minimum:
                    raise CalibrationBuildError(f"{path}.metadata_match.{key} must be an integer >= {minimum}")
            if shape["moe_top_k"] > shape["moe_num_experts"]:
                raise CalibrationBuildError(f"{path}.metadata_match.moe_top_k must not exceed moe_num_experts")
            dtype_bytes = {"fp4": .5, "fp8": 1, "int8": 1, "fp16": 2, "bf16": 2, "fp32": 4}
            if shape["moe_tensor_bytes_per_token"] != shape["moe_hidden_width"] * dtype_bytes[row["dtype"]]:
                raise CalibrationBuildError(f"{path}.metadata_match tensor bytes must equal hidden width times declared dtype bytes")
        precision = _object(row["precision"], f"{path}.precision")
        _fields(precision, {"compute", "activation", "communication"}, set(), f"{path}.precision")
        for key, dtype in precision.items():
            _choice(dtype, DTYPES if key != "compute" else {"fp8", "fp16", "bf16", "fp32"}, f"{path}.precision.{key}")
        if precision["communication"] != row["dtype"]:
            raise CalibrationBuildError(f"{path}.precision.communication must equal the declared wire dtype")
        ranks = row["ranks"]
        if not isinstance(ranks, list) or len(ranks) < 2 or any(not isinstance(rank, str) or re.fullmatch(r"npu(?:0|[1-9][0-9]*)", rank) is None for rank in ranks):
            raise CalibrationBuildError(f"{path}.ranks must contain at least two distinct physical npu indices")
        if len(set(ranks)) != len(ranks) or any(int(rank[3:]) >= environments[environment_id]["npu_count"] for rank in ranks):
            raise CalibrationBuildError(f"{path}.ranks must be distinct existing physical devices")
        world = len(ranks)
        if row["collective"] == "all_to_all_v":
            _choice(row["algorithm"], {"direct", "pairwise"}, f"{path}.algorithm")
            if row["size_bytes"] != 0:
                raise CalibrationBuildError(f"{path}.size_bytes must be zero for all_to_all_v; actual bytes come from payload_matrix")
            matrix = row.get("payload_matrix")
            if not isinstance(matrix, list) or len(matrix) != world or any(not isinstance(values, list) or len(values) != world for values in matrix):
                raise CalibrationBuildError(f"{path}.payload_matrix must be ranks x ranks actual payload bytes")
            row["payload_matrix"] = [[_number(value, f"{path}.payload_matrix[{i}][{j}]") for j, value in enumerate(values)] for i, values in enumerate(matrix)]
            # Match Topology's row-wise summation exactly at interval endpoints.
            payload = math.fsum(math.fsum(value for j, value in enumerate(values) if i != j)
                                for i, values in enumerate(row["payload_matrix"]))
        else:
            _choice(row["algorithm"], {"ring"}, f"{path}.algorithm")
            if row.get("payload_matrix") is not None:
                raise CalibrationBuildError(f"{path}.payload_matrix is only supported for all_to_all_v")
            row["payload_matrix"] = None
            per_rank = row["size_bytes"] * (world - 1) / world * (2 if row["collective"] == "all_reduce" else 1)
            payload = math.fsum([per_rank] * world)
        row["payload_bytes"] = _number(payload, f"{path}.total_off_diagonal_payload", positive=True)
        observations.append(row)
    return observations


def _simulate(hardware, row):
    _, Graph, Topology, simulate = _modules()
    topology, graph = Topology(hardware, row["precision"]), Graph()
    topology.populate(graph)
    topology.collective(graph, "observed", row["collective"], row["ranks"], row["size_bytes"],
                        payload_matrix=row["payload_matrix"],
                        algorithm=row["algorithm"] if row["collective"] == "all_to_all_v" else "pairwise",
                        timing_scope=row["timing_scope"],
                        metadata={**row["metadata_match"], "phase": row["phase"], "operation": row["semantic"], "wire_dtype": row["dtype"]})
    result = simulate(graph, retain_event_payload=False)
    if result["status"] != "feasible":
        raise CalibrationBuildError(f"observation {row['id']} simulation failed: {result['status']}: {result['diagnostics']}")
    records = [event.metadata["collective_calibration"] for event in graph.events if "collective_calibration" in event.metadata]
    if len(records) != 1:
        raise CalibrationBuildError(f"observation {row['id']} did not produce exactly one collective calibration record")
    return result["duration_s"], records[0]


def build_calibration(document: Any) -> dict[str, Any]:
    _json_tree(document)
    document = _object(document, "dataset")
    _fields(document, {"schema_version", "environments", "observations"}, set(), "dataset")
    if type(document["schema_version"]) is not int or document["schema_version"] != 1:
        raise CalibrationBuildError("schema_version must be integer 1")
    environments = _environments(document["environments"])
    rows = _observations(document["observations"], environments)
    from supernode_sim.collective_calibration import collective_family_signature
    groups = defaultdict(list)
    for row in rows:
        row["family_signature"] = collective_family_signature(environments[row["environment_id"]], row["ranks"], row["collective"], row["size_bytes"], row["payload_matrix"], precision=row["precision"], metadata_match=row["metadata_match"])
        key = (row["environment_id"], row["family_signature"], *(row[field] for field in SELECTORS))
        groups[key].append(row)
    family_reports, models_by_environment = [], defaultdict(list)
    for key, members in sorted(groups.items()):
        train = [row for row in members if row["split"] == "train"]
        if len({row["evidence_kind"] for row in train}) > 1:
            raise CalibrationBuildError(f"environment {key[0]}: training evidence kinds must not be mixed in one response family")
        buckets = defaultdict(list)
        for row in train:
            buckets[row["payload_bytes"]].append(row)
        samples = [{"payload_bytes": payload, "duration_s": statistics.median(row["duration_s"] for row in values),
                    "sample_count": len(values), "min_duration_s": min(row["duration_s"] for row in values),
                    "max_duration_s": max(row["duration_s"] for row in values), "observation_ids": sorted(row["id"] for row in values)}
                   for payload, values in sorted(buckets.items())]
        identity = "curve-" + hashlib.sha256(json.dumps(key, ensure_ascii=True).encode()).hexdigest()[:16]
        report = {"id": identity, "environment_id": key[0], "family_signature": key[1],
                  "selectors": {field: members[0][field] for field in SELECTORS}, "metadata_match": members[0]["metadata_match"], "training_samples": samples,
                  "training_observation_ids": sorted(row["id"] for row in train),
                  "validation_observation_ids": sorted(row["id"] for row in members if row["split"] == "validation"),
                  "provenance": {"evidence_kinds": sorted({row["evidence_kind"] for row in train}),
                                 "status_counts": dict(Counter(row["status"] for row in train)), "sources": sorted({row["source"] for row in train})},
                  "status": "model_built" if len(samples) >= 2 else "insufficient_training_payloads"}
        family_reports.append(report)
        if len(samples) >= 2:
            model = {"id": identity, **report["selectors"], "metadata_match": report["metadata_match"], "world_size": len(members[0]["ranks"]),
                     "hardware_name": environments[key[0]]["name"], "family_signature": key[1],
                     "samples": [{field: sample[field] for field in ("payload_bytes", "duration_s")} for sample in samples],
                     "source": "Dataset training observations only; caller-declared provenance: " + json.dumps(report["provenance"], ensure_ascii=True, sort_keys=True)}
            models_by_environment[key[0]].append(model)
    HardwareSpec, _, _, _ = _modules()
    output_hardware = {}
    for identity, hardware in environments.items():
        calibrated = copy.deepcopy(hardware)
        calibrated["collective_models"] = models_by_environment[identity]
        output_hardware[identity] = HardwareSpec.model_validate(calibrated).model_dump()
    results = []
    for row in rows:
        before, _ = _simulate(environments[row["environment_id"]], row)
        after, record = _simulate(output_hardware[row["environment_id"]], row)
        applied = record["status"] == "matched"
        model = next((model for model in models_by_environment[row["environment_id"]]
                      if model["family_signature"] == row["family_signature"] and all(model[field] == row[field] for field in SELECTORS)), None)
        in_range = model is not None and model["samples"][0]["payload_bytes"] <= row["payload_bytes"] <= model["samples"][-1]["payload_bytes"]
        if applied != in_range:
            raise CalibrationBuildError(f"observation {row['id']}: runtime applicability differs from the declared trained interval: {record}")
        if applied:
            if record.get("model_id") != model["id"] or record.get("method") not in {"exact_model", "interpolated"}:
                raise CalibrationBuildError(f"observation {row['id']}: the expected response model was not applied: {record}")
            expected = max(before, record["service_duration_s"])
            if not math.isclose(after, expected, rel_tol=1e-9, abs_tol=1e-12):
                raise CalibrationBuildError(f"observation {row['id']}: response time was not max(physical simulation, model timer)")
        if not applied and not math.isclose(after, before, rel_tol=1e-10, abs_tol=1e-15):
            raise CalibrationBuildError(f"observation {row['id']}: unmatched model did not fall back to the physical simulation")
        result = {key: row[key] for key in ("id", "environment_id", "split", "status", "evidence_kind", "source", "family_signature", "payload_bytes", "duration_s")}
        result.update(model_applied=applied, model_id=model["id"] if model else None,
                      trained_interval_bytes=[model["samples"][0]["payload_bytes"], model["samples"][-1]["payload_bytes"]] if model else None,
                      baseline_duration_s=before, calibrated_duration_s=after, runtime_record=record,
                      baseline_absolute_relative_error=abs(before - row["duration_s"]) / row["duration_s"],
                      calibrated_absolute_relative_error=abs(after - row["duration_s"]) / row["duration_s"],
                      fallback_reason=None if applied else ("out_of_range" if model else "no_trained_family"))
        results.append(result)
    coverage = {}
    for identity in environments:
        validation = [row for row in results if row["environment_id"] == identity and row["split"] == "validation"]
        applied = [row for row in validation if row["model_applied"]]
        coverage[identity] = {"validation_count": len(validation), "in_range_applied_count": len(applied),
                              "fallback_count": len(validation) - len(applied),
                              "range_coverage_fraction": len(applied) / len(validation) if validation else None,
                              "in_range_mean_absolute_relative_error": statistics.mean(row["calibrated_absolute_relative_error"] for row in applied) if applied else None}
    return {"schema_version": 1, "status": "built" if any(models_by_environment.values()) else "insufficient_training_data",
            "real_hardware_accuracy_claimed": False, "hardware_by_environment": output_hardware,
            "families": family_reports, "observations": results, "validation_coverage": coverage,
            "notes": [
                "All source/status/evidence_kind values are caller declarations; synthetic or public estimates are not independently verified hardware measurements.",
                "Validation rows never enter training medians, range selection, or model parameter selection. Reported range coverage includes rejected/out-of-range rows.",
                "Only same-environment, same-family payloads interpolate. Full routing proportions, ordered physical ranks, software context and physical hardware remain locked; normalized ratios use the runtime's documented 1e-12 resolution.",
                "The observed payload matrix must describe actual transferred bytes, not a requested HCCL size or an inferred algorithm-bandwidth conversion.",
                "algorithm selects a simulator template, not a claim about the observed HCCL implementation. Record known on-device algorithm and FP8/HIF8 format details as additional calibration_context identities; no low-precision format aliases are inferred.",
                "Graphs retain physical network/resource limits. activation_collective observations add a concurrent whole-call timer; this builder does not independently reconstruct unknown pack/unpack kernels.",
                "Output hardware contains normalized schema settings, including defaults. Review this separate artifact before explicitly adopting it for a target configuration.",
                "Per-environment output never modifies the input or an existing target hardware file. This is a scoped collective calibration, not end-to-end serving validation.",
            ]}


def render_markdown(report):
    synthetic = all(row["evidence_kind"] == "synthetic" for row in report["observations"])
    lines = ["# Ascend 运行时校准数据处理结果", "",
             "本次输入全部为 synthetic；没有进行 Ascend 实机测量，不能将这些结果当作 A2/A3 的实机准确率。" if synthetic else
             "来源与测量类型由输入声明；本工具不认证 Ascend 实机准确率。", "",
             "| 环境 | 独立训练 payload 点 | 训练采样次数 | 验证样本 | 范围内插值 | 超界回退 | 其他未覆盖 | 验证覆盖率 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for identity, coverage in report["validation_coverage"].items():
        families = [family for family in report["families"] if family["environment_id"] == identity]
        validation = [row for row in report["observations"] if row["environment_id"] == identity and row["split"] == "validation"]
        interpolated = sum(row["runtime_record"].get("method") == "interpolated" for row in validation)
        outside = sum(row["fallback_reason"] == "out_of_range" for row in validation)
        other = sum(row["fallback_reason"] not in {None, "out_of_range"} for row in validation)
        fraction = coverage["range_coverage_fraction"]
        lines.append(f"| {identity} | {sum(len(family['training_samples']) for family in families)} | {sum(len(family['training_observation_ids']) for family in families)} | {len(validation)} | {interpolated} | {outside} | {other} | {fraction:.1%} |" if fraction is not None else
                     f"| {identity} | {sum(len(family['training_samples']) for family in families)} | {sum(len(family['training_observation_ids']) for family in families)} | 0 | 0 | 0 | 0 | 无验证样本 |")
    lines += ["", "重复训练采样取中位数，报告保留样本范围和次数；验证数据不参与曲线生成。每条观测均经过实际事件图仿真，范围外使用物理模型回退，不外推。覆盖率表示模型适用范围，不表示精度达标。", "",
              "各环境生成独立硬件片段。片段包含显式软件环境、完整硬件设置及新曲线；采用前需审查与目标环境的一致性，工具不会修改现场配置。", "",
              "algorithm 是仿真算法模板，不代表现场 HCCL 算法已被识别；FP8/HIF8 等实际格式和 HCCL 算法信息应另存于环境身份字段。", ""]
    for family in report["families"]:
        if family["metadata_match"]:
            lines.append(f"- {family['environment_id']} / {family['selectors']['semantic']} 的形状约束：`{json.dumps(family['metadata_match'], ensure_ascii=False, sort_keys=True)}`。这些值来自输入声明。")
    if "command" in report:
        lines += ["", "复现命令：", "", "```text", report["command"], "```"]
    return "\n".join(lines) + "\n"


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise CalibrationBuildError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise CalibrationBuildError(f"nonfinite JSON number: {value}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", "-o", type=Path, default=Path("outputs/runtime-calibration"), help="new calibration artifact directory")
    args = parser.parse_args(argv)
    try:
        if args.output.resolve() == args.input.resolve():
            raise CalibrationBuildError("output must not overwrite the input dataset")
        raw_input = args.input.read_bytes()
        document = json.loads(raw_input.decode("utf-8-sig"), object_pairs_hook=_unique_keys, parse_constant=_reject_constant)
        report = build_calibration(document)
        report["input_dataset"] = {"path": str(args.input.resolve()), "sha256": hashlib.sha256(raw_input).hexdigest()}
        report["command"] = subprocess.list2cmdline(["python", "skills/supernode-modeling/scripts/build_runtime_calibration.py", str(args.input), "--output", str(args.output)])
        targets = {"report.json": report, **{f"environment.{identity}.json": {"hardware": hardware} for identity, hardware in report["hardware_by_environment"].items()}}
        for name in (*targets, "report.md"):
            target = args.output / name
            if target.resolve() == args.input.resolve() or (target.exists() and target.samefile(args.input)):
                raise CalibrationBuildError("output must not overwrite the input dataset")
        encoded = {name: json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n" for name, value in targets.items()}
        encoded["report.md"] = render_markdown(report)
        args.output.mkdir(parents=True, exist_ok=True)
        for name, value in encoded.items():
            (args.output / name).write_text(value, encoding="utf-8")
        print(f"Wrote {args.output / 'report.json'} and {len(report['hardware_by_environment'])} separate environment artifacts; real hardware accuracy is not certified.")
    except (CalibrationBuildError, OSError, ValueError, TypeError, OverflowError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
