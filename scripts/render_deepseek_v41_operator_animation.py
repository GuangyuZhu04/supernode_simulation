"""Build an offline, evidence-backed V4.1 operator walkthrough.

The full-model replay uses the existing FP4 case adapter. A separate, explicitly
scoped micrograph exercises the current kernel_traces API on that model's shared
expert GEMM budget. It does not pretend the case adapter calls kernel_traces.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from supernode_sim.case_studies.v41_fp4 import CaseConfig, simulate_case, write_report
from supernode_sim.case_studies.v41_fp4.architecture import inventory, summarize
from supernode_sim.calibration import apply_kernel_profiles, model_shape_signature
from supernode_sim.config import KernelTrace
from supernode_sim.core import Allocation, Event, Graph, Resource
from supernode_sim.engine import simulate
from supernode_sim.operator_trace import apply_kernel_traces

TEMPLATE = ROOT / "scripts/templates/deepseek-v41-operator-animation.html"
DEFAULT_OUTPUT = ROOT / "docs/flow-animation/deepseek-v41-operator-simulation.html"


def close(a, b):
    assert math.isclose(a, b, rel_tol=2e-8, abs_tol=1e-8), (a, b)


def source_ref(path, symbol):
    """Resolve current source locations, rather than copying old line numbers."""
    source = (ROOT / path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = tree
    for part in symbol.split("."):
        node = next(x for x in node.body if isinstance(
            x, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and x.name == part)
    return {"file": path, "symbol": symbol, "line": node.lineno,
            "sha256": hashlib.sha256(source.encode()).hexdigest()}


def resource_rates(result):
    hw = result["hardware"]
    npu, cpu = hw["npu"], hw["cpu"]
    rates = {f"NPU.{d}": npu["flops"][d] * npu["efficiency"] for d in ("fp8", "fp32")}
    rates.update({"NPU.hbm": npu["hbm_Bps"],
                  "SU.tx": min(hw["scale_up"]["bandwidth_Bps"], hw["scale_up"]["injection_Bps"]) * hw["scale_up"]["efficiency"],
                  "HOST": hw["host_link"]["injection_Bps"] * hw["host_link"]["efficiency"],
                  "CPU.dram": cpu["dram_Bps"], "CPU.fp32": cpu["flops"] * cpu["efficiency"],
                  "CPU.iops": cpu["iops"]})
    rates["SU.rx"] = rates["SU.tx"]
    return rates


def trace_example(result, phase):
    """A real single-rank simulation, isolated from the full-model metrics.

    Recover each matrix's work/IO from the same Tensor inventory as the case
    adapter. The extra 1 us startup and 256 KiB workspace per GEMM are declared
    synthetic teaching assumptions. No new HBM traffic is invented: the parent
    already counts IO at each matrix boundary.
    """
    model, cfg = result["model"], result["config"]
    parent_row = next(x for x in result["phases"][phase]["stages"]
                      if x["layer"] == 0 and x["name"] == "MoE_shared.matrix.fp8")
    matrices = [t for t in inventory(model, include_draft=False)
                if t.layer == 0 and ".shared." in t.name]
    params = sum(t.local_params(cfg["tp"], cfg["ep"]) for t in matrices)
    m = int(round(parent_row["work"]["NPU.fp8"] / (2 * params)))
    ops = []
    for name in ("up", "gate", "down"):
        tensor = next(t for t in matrices if t.name.endswith("." + name))
        n, k = tensor.local_shape(cfg["tp"], cfg["ep"])
        ops.append({"kernel": "shared." + name, "flops": 2 * m * n * k,
                    "memory_bytes": sum(tensor.bytes(cfg["tp"], cfg["ep"])) + m * (n + k),
                    "startup_s": 1e-6, "workspace_bytes": 256 * 1024,
                    "matmul": {"m": m, "n": n, "k": k, "batch": 1}})
    close(sum(x["flops"] for x in ops), parent_row["work"]["NPU.fp8"])
    close(sum(x["memory_bytes"] for x in ops), parent_row["work"]["NPU.hbm"])
    scope = {"model_shape_signature": model_shape_signature(model), "layer": 0,
             "tp": cfg["tp"], "ep": cfg["ep"], "sp": cfg["sp"],
             "workload_shape": {"phase": phase, "query_tokens_per_active_rank": m,
                                "context_tokens": cfg["context_tokens"],
                                "batch": cfg["prefill_batch"] if phase == "prefill" else cfg["decode_batch"],
                                "block": 1 if phase == "prefill" else cfg["block"]}}
    trace = KernelTrace.model_validate({
        "id": f"v41-shared-{phase}",
        "source": "Synthetic: V4.1 tensor inventory GEMM budget split; 1 us startup and 256 KiB workspace per operation are illustrative, not measured.",
        "hardware_name": result["hardware"]["name"], "dtype": "fp8", "phase": phase,
        "flops": parent_row["work"]["NPU.fp8"], "memory_bytes": parent_row["work"]["NPU.hbm"],
        "metadata_match": scope, "operations": ops}).model_dump()
    rates = resource_rates(result)
    activation = m * model["dim"]
    graph = Graph(resources={
        "npu0.compute": Resource("npu0.compute", rates["NPU.fp8"], "FLOP/s", "compute"),
        "npu0.hbm": Resource("npu0.hbm", rates["NPU.hbm"], "byte/s")},
        memory_caps={"npu0.hbm": result["hardware"]["npu"]["hbm_bytes"]},
        events=[Event("input.ready"), Event("shared.parent", ("input.ready",),
            {"npu0.compute": trace["flops"], "npu0.hbm": trace["memory_bytes"]},
            slots=("npu0.compute_stream",),
            allocs=(Allocation("shared.input", "npu0.hbm", activation, "activation"),),
            releases=("shared.input",),
            metadata={**scope, "phase": phase, "kernel": "MoE_shared.matrix.fp8", "component": "MoE_shared"}),
            Event("successor.ready", ("shared.parent",))])
    config = {"hardware": {**result["hardware"], "kernel_profiles": {},
                            "kernel_traces": {"MoE_shared.matrix.fp8": [trace]}},
              "execution": {"precision": {"compute": "fp8"}}}
    baseline = simulate(copy.deepcopy(graph))
    # A same-work, wrong-context selector must NOT silently match.
    mismatch_graph = copy.deepcopy(graph)
    mismatch_graph.events[1].metadata["workload_shape"]["context_tokens"] += 1
    apply_kernel_traces(mismatch_graph, config)
    assert mismatch_graph.metadata["operator_trace"]["matched_events"] == 0
    apply_kernel_traces(graph, config)
    apply_kernel_profiles(graph, config)
    expanded = simulate(graph)
    assert baseline["status"] == expanded["status"] == "feasible"
    close(baseline["duration_s"], parent_row["duration_s"])
    close(expanded["duration_s"], baseline["duration_s"] + 3e-6)
    close(expanded["resource_work"]["npu0.compute"], baseline["resource_work"]["npu0.compute"])
    close(expanded["resource_work"]["npu0.hbm"], baseline["resource_work"]["npu0.hbm"])
    close(expanded["memory_peaks_bytes"]["npu0.hbm"], activation + 256 * 1024)
    assert expanded["memory_final_bytes"]["npu0.hbm"] == 0
    assert graph.metadata["operator_trace"]["matched_events"] == 1
    assert graph.metadata["operator_trace"]["generated_operations"] == 3
    events = {e.id: e for e in graph.events}
    assert events["shared.parent"].deps == ("shared.parent.operator_trace.2",)
    assert events["successor.ready"].deps == ("shared.parent",)
    assert events["shared.parent"].releases == ("shared.input",)
    return {"phase": phase, "parent": parent_row, "scope": scope, "trace": trace,
            "operations": ops, "baseline_s": baseline["duration_s"],
            "expanded_s": expanded["duration_s"], "activation_bytes": activation,
            "workspace_peak_bytes": 256 * 1024,
            "micrograph_peak_bytes": expanded["memory_peaks_bytes"]["npu0.hbm"],
            "summary": graph.metadata["operator_trace"],
            "kernel_calibration": graph.metadata.get("kernel_calibration"),
            "mismatch": mismatch_graph.metadata["operator_trace"],
            "events": [{"id": e.id, "deps": e.deps, "work": e.work,
                        "startup_s": e.startup_s, "releases": e.releases} for e in graph.events],
            "execution": expanded["trace"]}


def snapshot(result):
    assert result["status"] == "feasible", "The walkthrough requires a feasible case."
    rates, phases = resource_rates(result), {}
    for phase, value in result["phases"].items():
        clock, records = 0., []
        for i, row in enumerate(value["stages"]):
            service = max((v / rates[k] for k, v in row["work"].items()), default=0.)
            close(row["duration_s"], service + row["startup_s"])
            records.append({k: row[k] for k in ("name", "component", "layer", "work", "totals", "startup_s", "dtype", "duration_s", "bottleneck")})
            records[-1].update(index=i, start_s=clock, end_s=clock + row["duration_s"])
            clock += row["duration_s"]
        close(clock, value["duration_s"])
        close(sum(value["breakdown_s"].values()), clock)
        assert len(records) == value["native_completed_events"]
        phases[phase] = {k: value[k] for k in ("duration_s", "breakdown_s", "memory", "native_completed_events", "component_latency_s")}
        phases[phase]["records"] = records
    close(sum(phases[p]["duration_s"] for p in ("draft", "verify", "commit")), result["metrics"]["round_s"])
    close(result["metrics"]["round_s"] / result["config"]["accepted"], result["metrics"]["tpot_s"])
    refs = [("supernode_sim/case_studies/v41_fp4/architecture.py", "synthetic_config"),
            ("supernode_sim/case_studies/v41_fp4/target.py", "Phase.block"),
            ("supernode_sim/case_studies/v41_fp4/target.py", "Phase.attention"),
            ("supernode_sim/case_studies/v41_fp4/target.py", "Phase.moe"),
            ("supernode_sim/case_studies/v41_fp4/target.py", "Phase.engram"),
            ("supernode_sim/case_studies/v41_fp4/model.py", "Phase.kernel"),
            ("supernode_sim/case_studies/v41_fp4/model.py", "Phase.build"),
            ("supernode_sim/case_studies/v41_fp4/model.py", "Phase.run"),
            ("supernode_sim/operator_trace.py", "apply_kernel_traces"),
            ("supernode_sim/calibration.py", "apply_kernel_profiles"),
            ("supernode_sim/engine.py", "_max_min_rates"),
            ("supernode_sim/engine.py", "simulate"),
            ("supernode_sim/case_studies/v41_fp4/__init__.py", "simulate_case")]
    return {"version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
            "config": result["config"], "model": result["model"],
            "model_totals": summarize(result["model"]), "draft": result["draft"],
            "hardware": result["hardware"], "layout": result["layout"],
            "metrics": result["metrics"], "rates": rates, "phases": phases,
            "trace_examples": {p: trace_example(result, p) for p in ("prefill", "verify")},
            "sources": [source_ref(*ref) for ref in refs], "provenance": result["provenance"],
            "notes": result["notes"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--artifacts", type=Path, default=ROOT / "outputs/deepseek-v41-operator-animation")
    args = parser.parse_args()
    # Fixed reviewed teaching scenario. Model changes must come from the case API.
    config = CaseConfig().model_dump()
    result = simulate_case(config)
    data = snapshot(result)
    args.artifacts.mkdir(parents=True, exist_ok=True)
    write_report(result, args.artifacts / "simulation")
    (args.artifacts / "animation-data.json").write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    template = TEMPLATE.read_text(encoding="utf-8")
    assert template.count("__ANIMATION_DATA__") == 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(template.replace("__ANIMATION_DATA__", payload), encoding="utf-8")
    print(json.dumps({"html": str(args.output), "events": sum(p["native_completed_events"] for p in data["phases"].values()),
                      "ttft_s": data["metrics"]["ttft_s"], "tpot_s": data["metrics"]["tpot_s"],
                      "trace_parents": 2, "generated_suboperators": 6}, ensure_ascii=False))


if __name__ == "__main__":
    main()
