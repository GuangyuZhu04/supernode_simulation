#!/usr/bin/env python3
"""Reproduce a bounded DeepEP V1 public-reference collective timing fit.

Only EP16/EP32 observations train startup + slope * physical_floor_s, subject
to startup >= 0 and slope >= 1. EP64/EP128 observations are held out. The floor
is recomputed from the actual graph's accumulated resource work/capacity, not
from the published latency. The resulting exact-shape collective profiles are
then applied to fresh graphs and simulated, with a concurrent completion timer
so transport and the fitted total are combined with max, not added.

This is a public-reference fit with uniform unmasked routing assumptions, not
an end-to-end inference calibration or an independently reproduced hardware
measurement. The published runs do not bind their masks and routing graph to
the test-source snapshot. Only stdlib plus this repository's simulator is used.

Example: python skills/supernode-modeling/scripts/fit_deepep_collectives.py
         --output outputs/deepep-fit --world-sizes 64 128
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
REVISION = "a56d6156febcd9976e55adc85b5155bfac9f28f8"
DOCUMENTATION_URL = f"https://github.com/deepseek-ai/DeepEP/blob/{REVISION}/docs/legacy.md"
TEST_SOURCE_URL = f"https://github.com/deepseek-ai/DeepEP/blob/{REVISION}/tests/legacy/test_low_latency.py"
TRAINING_WORLD_SIZES = (16, 32)
HOLDOUT_WORLD_SIZES = (64, 128)
PUBLIC_LATENCIES_US = {
    16: {"dispatch": 118.0, "combine": 195.0},
    32: {"dispatch": 155.0, "combine": 273.0},
    64: {"dispatch": 173.0, "combine": 314.0},
    128: {"dispatch": 192.0, "combine": 369.0},
}
TOKENS_PER_RANK = 128
HIDDEN = 7168
TOP_K = 8
BYTES_PER_ROUTED_TOKEN = {"dispatch": HIDDEN + (HIDDEN // 128) * 4 + 16, "combine": HIDDEN * 2}
WIRE_DTYPES = {"dispatch": "fp8", "combine": "bf16"}
SCOPE = "public-reference fit with uniform unmasked routing assumptions"


class FitError(ValueError):
    """Invalid fitting data or an experiment that did not apply its profiles."""


def _nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FitError(f"{label} must be a finite nonnegative number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise FitError(f"{label} must be a finite nonnegative number") from exc
    if not math.isfinite(number) or number < 0:
        raise FitError(f"{label} must be a finite nonnegative number")
    return number


def fit_nonnegative_affine(points: list[tuple[float, float]]) -> dict[str, Any]:
    """Least squares with intercept >= 0, slope >= 1; enumerate both boundaries.

    Points are (physical_floor_s, observed_s). Equal x values are supported;
    the slope=1 boundary gives a deterministic solution in that degenerate case.
    Neither fit evaluation nor parameter selection reads held-out observations.
    """
    if not isinstance(points, (list, tuple)) or len(points) < 2:
        raise FitError("at least two training points are required")
    pairs = []
    for index, pair in enumerate(points):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise FitError(f"points[{index}] must be (physical_floor_s, observed_s)")
        pairs.append((_nonnegative(pair[0], f"points[{index}].floor"), _nonnegative(pair[1], f"points[{index}].observed")))
    count = len(pairs)
    try:
        mean_x = math.fsum(x / count for x, _ in pairs)
        mean_y = math.fsum(y / count for _, y in pairs)
        variance = math.fsum((x - mean_x) ** 2 for x, _ in pairs)
        covariance = math.fsum((x - mean_x) * (y - mean_y) for x, y in pairs)
        candidates = [(max(0.0, math.fsum((y - x) / count for x, y in pairs)), 1.0, "slope=1")]
        sum_xx = math.fsum(x * x for x, _ in pairs)
        slope_at_zero = max(1.0, math.fsum(x * y for x, y in pairs) / sum_xx) if sum_xx else 1.0
        candidates.append((0.0, slope_at_zero, "startup=0"))
        if variance:
            slope = covariance / variance
            startup = mean_y - slope * mean_x
            if startup >= 0 and slope >= 1:
                candidates.append((startup, slope, "interior"))
        scored = [(math.fsum((startup + slope * x - y) ** 2 for x, y in pairs), slope, startup, boundary)
                  for startup, slope, boundary in candidates]
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise FitError("training data exceeds finite fitting arithmetic") from exc
    if not all(math.isfinite(value) for row in scored for value in row[:3]):
        raise FitError("training data exceeds finite fitting arithmetic")
    sse, slope, startup, boundary = min(scored)
    return {"startup_s": startup, "slope": slope, "training_sse_s2": sse,
            "active_constraint": boundary, "training_sample_count": count,
            "formula": "startup_s + slope * physical_floor_s"}


def fit_reference_models(physical_floors: dict[int, dict[str, float]],
                         observations_us: dict[int, dict[str, float]] | None = None) -> dict[str, dict[str, Any]]:
    """Fit only the fixed EP16/EP32 split, even if the input includes holdouts."""
    observations_us = PUBLIC_LATENCIES_US if observations_us is None else observations_us
    models = {}
    for semantic in WIRE_DTYPES:
        try:
            points = [(physical_floors[world][semantic], _nonnegative(observations_us[world][semantic], f"EP{world}.{semantic}.observed_us") * 1e-6)
                      for world in TRAINING_WORLD_SIZES]
        except (KeyError, TypeError) as exc:
            raise FitError(f"missing EP16/EP32 training data for {semantic}") from exc
        models[semantic] = {**fit_nonnegative_affine(points), "training_world_sizes": list(TRAINING_WORLD_SIZES)}
    return models


def hardware_for(world_size: int, profiles: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 8 or world_size % 8:
        raise FitError("world_size must be an integer multiple of 8, at least 8")
    return {
        "name": f"DeepEP V1 H800 reference abstraction EP{world_size}",
        "npu_count": world_size, "npus_per_host": 8, "su_size": 8,
        "topology": "full_mesh", "routing_policy": "shortest_hop",
        "npu": {"hbm_bytes": 80e9, "hbm_Bps": 3e12, "efficiency": 1.0},
        "scale_up": {"bandwidth_Bps": 160e9, "injection_Bps": 160e9, "latency_s": 0, "efficiency": 1.0},
        "scale_out": {"bandwidth_Bps": 50e9, "injection_Bps": 50e9, "host_injection_Bps": 400e9,
                      "latency_s": 0, "efficiency": 1.0},
        "collective_profiles": list(profiles or []),
    }


def payload_matrix(world_size: int, semantic: str) -> list[list[float]]:
    hardware_for(world_size)
    if semantic not in WIRE_DTYPES:
        raise FitError("semantic must be dispatch or combine")
    cell = TOKENS_PER_RANK * TOP_K * BYTES_PER_ROUTED_TOKEN[semantic] / world_size
    return [[cell] * world_size for _ in range(world_size)]


def _simulator_modules():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from supernode_sim.core import Graph
    from supernode_sim.engine import simulate
    from supernode_sim.topology import Topology
    return Graph, Topology, simulate


def build_collective_graph(world_size: int, semantic: str, profiles: list[dict[str, Any]] | None = None):
    Graph, Topology, _ = _simulator_modules()
    hardware = hardware_for(world_size, profiles)
    topology = Topology(hardware, {"compute": "bf16", "activation": "bf16", "communication": WIRE_DTYPES[semantic]})
    graph = Graph()
    topology.populate(graph)
    topology.collective(
        graph, f"deepep.ep{world_size}.{semantic}", "all_to_all_v", [f"npu{rank}" for rank in range(world_size)], 0,
        payload_matrix=payload_matrix(world_size, semantic), algorithm="direct",
        metadata={"phase": "decode", "operation": semantic, "wire_dtype": WIRE_DTYPES[semantic]},
        timing_scope="activation_collective",
    )
    return graph


def resource_floor(graph) -> dict[str, Any]:
    """Bound elapsed time using accumulated work on every actual graph resource."""
    work = defaultdict(float)
    for event in graph.events:
        for resource, amount in event.work.items():
            work[resource] += amount
    floors = {resource: amount / graph.resources[resource].capacity for resource, amount in work.items()}
    floor = max(floors.values(), default=0.0)
    limiting = sorted(resource for resource, value in floors.items() if math.isclose(value, floor, rel_tol=1e-12, abs_tol=0))
    return {"duration_s": floor, "limiting_resources": limiting,
            "method": "max_r(sum_event(work[event,r]) / resource[r].capacity)"}


def make_profile(world_size: int, semantic: str, model: dict[str, Any], floor_s: float) -> dict[str, Any]:
    _simulator_modules()
    from supernode_sim.collective_calibration import collective_applicability_signature
    matrix = payload_matrix(world_size, semantic)
    sends = [math.fsum(value for column, value in enumerate(row) if row_index != column) for row_index, row in enumerate(matrix)]
    receives = [math.fsum(matrix[row][column] for row in range(world_size) if row != column) for column in range(world_size)]
    return {
        "id": f"deepep-v1-fit-ep{world_size}-{semantic}", "collective": "all_to_all_v",
        "semantic": semantic, "phase": "decode", "world_size": world_size,
        "dtype": WIRE_DTYPES[semantic], "algorithm": "direct", "timing_scope": "activation_collective",
        "total_payload_bytes": math.fsum(sends), "send_payload_bytes": sends, "recv_payload_bytes": receives,
        "duration_s": model["startup_s"] + model["slope"] * floor_s,
        "source": f"{SCOPE}; fit EP16/EP32 only; documentation {DOCUMENTATION_URL}; payload assumptions {TEST_SOURCE_URL}; documentation revision does not identify the measured runtime commit",
        "hardware_name": hardware_for(world_size)["name"],
        "applicability_signature": collective_applicability_signature(
            hardware_for(world_size), [f"npu{rank}" for rank in range(world_size)], "all_to_all_v", 0, matrix,
            precision={"compute": "bf16", "activation": "bf16", "communication": WIRE_DTYPES[semantic]}),
    }


def _simulation_summary(graph) -> tuple[dict[str, Any], dict[str, Any]]:
    _, _, simulate = _simulator_modules()
    result = simulate(graph, retain_event_payload=False)
    if result["status"] != "feasible":
        raise FitError(f"collective simulation failed: {result['status']}: {result['diagnostics']}")
    from supernode_sim.collective_calibration import summarize_collective_calibration
    summary = summarize_collective_calibration(graph, result)
    return {key: result[key] for key in ("status", "duration_s", "total_events", "completed_events")}, summary


def _error(predicted: float, observed: float) -> dict[str, float]:
    difference = predicted - observed
    return {"signed_error_s": difference, "absolute_error_s": abs(difference), "relative_error": abs(difference) / observed}


def run_experiment(world_sizes: tuple[int, ...] = HOLDOUT_WORLD_SIZES) -> dict[str, Any]:
    if not world_sizes or len(set(world_sizes)) != len(world_sizes) or any(world not in HOLDOUT_WORLD_SIZES for world in world_sizes):
        raise FitError("holdout world sizes must be a distinct nonempty subset of 64 and 128; EP16/EP32 are fixed training cases")
    worlds = (*TRAINING_WORLD_SIZES, *sorted(world_sizes))
    floors, before = {}, {}
    for world in worlds:
        floors[world], before[world] = {}, {}
        for semantic in WIRE_DTYPES:
            graph = build_collective_graph(world, semantic)
            floors[world][semantic] = resource_floor(graph)
            before[world][semantic], _ = _simulation_summary(graph)
    models = fit_reference_models({world: {semantic: values["duration_s"] for semantic, values in phases.items()}
                                   for world, phases in floors.items()})
    rows, hardware_profiles = [], {}
    for world in worlds:
        profiles = [make_profile(world, semantic, models[semantic], floors[world][semantic]["duration_s"]) for semantic in WIRE_DTYPES]
        hardware_profiles[str(world)] = hardware_for(world, profiles)
        for semantic, profile in zip(WIRE_DTYPES, profiles):
            graph = build_collective_graph(world, semantic, profiles)
            after, coverage = _simulation_summary(graph)
            if coverage["matched_collectives"] != 1 or profile["id"] not in coverage["matched_profile_ids"]:
                raise FitError(f"fitted profile did not match EP{world} {semantic}: {coverage}")
            expected = max(before[world][semantic]["duration_s"], profile["duration_s"])
            if not math.isclose(after["duration_s"], expected, rel_tol=1e-9, abs_tol=1e-12):
                raise FitError(f"EP{world} {semantic}: profile timing was not max(network, fitted duration)")
            observed = PUBLIC_LATENCIES_US[world][semantic] * 1e-6
            rows.append({
                "world_size": world, "semantic": semantic, "split": "train" if world in TRAINING_WORLD_SIZES else "holdout",
                "reference_s": observed, "physical_floor": floors[world][semantic],
                "remote_host_fraction": (world - 8) / world,
                "bytes_per_routed_token": BYTES_PER_ROUTED_TOKEN[semantic],
                "logical_bytes_per_rank_including_diagonal": TOKENS_PER_RANK * TOP_K * BYTES_PER_ROUTED_TOKEN[semantic],
                "before": {**before[world][semantic], **_error(before[world][semantic]["duration_s"], observed)},
                "after": {**after, **_error(after["duration_s"], observed)},
                "fitted_profile": profile, "profile_application": coverage,
            })
    metrics = {}
    for split in ("train", "holdout"):
        selected = [row for row in rows if row["split"] == split]
        metrics[split] = {"sample_count": len(selected), **{
            f"{stage}_mean_absolute_relative_error": math.fsum(row[stage]["relative_error"] for row in selected) / len(selected)
            for stage in ("before", "after")
        }}
    return {
        "schema_version": 1, "scope": SCOPE, "evidence_kind": "public_reference_fit",
        "source": {"documentation": DOCUMENTATION_URL, "payload_test": TEST_SOURCE_URL, "documentation_revision": REVISION,
                   "measured_runtime_revision": None, "raw_measurements_available": False},
        "training_world_sizes": list(TRAINING_WORLD_SIZES), "holdout_world_sizes": sorted(world_sizes),
        "fitted_models": models, "aggregate_errors": metrics, "cases": rows, "hardware_with_profiles": hardware_profiles,
        "payload_assumptions": {"tokens_per_rank": TOKENS_PER_RANK, "hidden": HIDDEN, "top_k": TOP_K,
            "dispatch_bytes_formula": "7168 + (7168 / 128) * 4 + 16 = 7408",
            "combine_bytes_formula": "7168 * 2 = 14336",
            "matrix": "uniform all-to-all-v; each cell = 128 * 8 * wire_bytes / world_size, including diagonal; diagonal excluded from transport/profile send/receive vectors"},
        "limitations": [
            "Public-reference collective fit only; this is not an end-to-end inference calibration, nor a reproduced GPU experiment.",
            "The published timing table does not identify exact masking, token-to-expert routing, benchmark runtime commit, or a fully specified timer boundary.",
            "Uniform unmasked routing, eight devices per host, full-mesh abstraction and explicit 160/50/400 GB/s resource limits are modeling assumptions; no hidden efficiency was fitted.",
            "EP16/EP32 alone determine parameters. EP64/EP128 are evaluated only after fitting; no holdout errors enter parameter selection.",
            "The activation_collective timer covers the published pack/transport/unpack scope assumption and completes concurrently with physical transport; it is not added again to transport duration.",
            "Profiles additionally guard effective abstract hardware/IO, ordered physical ranks and the complete payload matrix with a versioned signature. This guard is not authentication of physical hardware or benchmark evidence; runtime changes still require independent review.",
            "A two-point fit provides no confidence interval or universal accuracy guarantee. The remaining holdout residual is reported without further fitting.",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# DeepEP V1 公开微基准拟合实验", "", f"范围：{report['scope']}。只拟合集合通信，未完成端到端推理校准。", "",
             "训练：EP16、EP32；留出：" + "、".join(f"EP{world}" for world in report["holdout_world_sizes"]) + "。", "",
             "| 阶段 | 启动项 (μs) | 斜率 | 约束 |", "|---|---:|---:|---|"]
    for semantic, model in report["fitted_models"].items():
        lines.append(f"| {semantic} | {model['startup_s'] * 1e6:.6f} | {model['slope']:.9f} | {model['active_constraint']} |")
    lines += ["", "公式：duration = startup + slope × physical_floor；startup ≥ 0，slope ≥ 1。物理下界从真实事件图累计资源工作量计算。", "",
              "| EP | 阶段 | 集合 | 公开 (μs) | 物理下界 (μs) | 拟合前仿真 (μs) | 拟合后仿真 (μs) | 前绝对相对误差 | 后绝对相对误差 |",
              "|---:|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in report["cases"]:
        lines.append(f"| {row['world_size']} | {row['semantic']} | {row['split']} | {row['reference_s'] * 1e6:.3f} | {row['physical_floor']['duration_s'] * 1e6:.3f} | {row['before']['duration_s'] * 1e6:.3f} | {row['after']['duration_s'] * 1e6:.3f} | {row['before']['relative_error']:.2%} | {row['after']['relative_error']:.2%} |")
    holdout = report["aggregate_errors"]["holdout"]
    lines += ["", f"留出样本平均绝对相对误差：{holdout['before_mean_absolute_relative_error']:.2%} → {holdout['after_mean_absolute_relative_error']:.2%}。", "",
              f"[公开时间表]({DOCUMENTATION_URL}) · [payload 测试源码]({TEST_SOURCE_URL})", ""]
    if "command" in report:
        lines += ["复现命令：", "", "```text", report["command"], "```", ""]
    lines.extend(f"- {note}" for note in report["limitations"])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("outputs/deepep-fit"), help="output directory")
    parser.add_argument("--world-sizes", nargs="+", type=int, choices=HOLDOUT_WORLD_SIZES, default=list(HOLDOUT_WORLD_SIZES), help="held-out cases to simulate; training stays EP16/EP32")
    args = parser.parse_args(argv)
    try:
        report = run_experiment(tuple(args.world_sizes))
        report["command"] = subprocess.list2cmdline(["python", "skills/supernode-modeling/scripts/fit_deepep_collectives.py", "--output", str(args.output), "--world-sizes", *map(str, args.world_sizes)])
        report["artifacts"] = ["report.json", "report.md", *[f"profiles.ep{world}.json" for world in report["hardware_with_profiles"]]]
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        (args.output / "report.md").write_text(render_markdown(report), encoding="utf-8")
        for world, hardware in report["hardware_with_profiles"].items():
            (args.output / f"profiles.ep{world}.json").write_text(json.dumps({"hardware": hardware}, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Wrote {args.output / 'report.json'} and report.md; held-out mean relative error "
              f"{report['aggregate_errors']['holdout']['after_mean_absolute_relative_error']:.2%}. Scope: {SCOPE}.")
    except (FitError, OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
