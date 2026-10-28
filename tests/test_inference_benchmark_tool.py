"""Behavioral checks for evidence-scoped standalone inference comparisons."""

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "skills/supernode-modeling/scripts/compare_inference_benchmarks.py"
SPEC = importlib.util.spec_from_file_location("compare_inference_benchmarks", SCRIPT)
benchmark_tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark_tool)


def bundle():
    context = {
        "model": {"name": "test-model", "revision": "sha-test"},
        "hardware": {"sku": "test-device", "count": 8, "topology": "single host; test fabric"},
        "runtime": {"name": "test-runtime", "version": "1.0", "backend_version": "2.0"},
        "precision": {"weights": "BF16", "activations": "BF16", "kv_cache": "BF16", "communication": "BF16"},
        "parallelism": {"tp": 8, "pp": 1, "ep": 8, "dp": 1},
        "workload": {
            "input_tokens": 1024, "output_tokens": 128, "batch_size": 16,
            "batch_scope": "global requests", "concurrency": 16,
            "arrival_process": "closed loop; synthetic prompts", "cache_state": "prefix cache disabled; warm kernels",
        },
        "execution": {"speculative_decoding": False, "chunked_prefill": False},
    }
    metric = {
        "value": 10, "unit": "ms", "phase": "decode", "timer_boundary": "engine",
        "aggregation": "mean of per-request TPOT", "token_basis": "output tokens excluding first",
        "window": {"duration_s": 60, "warmup": "excluded", "cohort": "completed requests"},
    }
    reference = {"id": "test-ref", "status": "measured", "evidence_kind": "measurement", "source": "unit-test fixture; no real measurement", "context": context, "metrics": {"tpot": metric}}
    candidate = copy.deepcopy(reference)
    candidate.update(id="test-candidate", status="simulated", evidence_kind="simulation", source="test simulator revision abc")
    candidate["metrics"]["tpot"]["value"] = 12
    return {"schema_version": 1, "reference": reference, "candidate": candidate}


def test_matched_conditions_report_scoped_errors_without_validation_pass():
    document = bundle()
    original = copy.deepcopy(document)
    result = benchmark_tool.compare_benchmarks(document)
    metric = result["metrics"]["tpot"]
    assert result["status"] == "comparable"
    assert metric["signed_difference"] == 2
    assert metric["absolute_error"] == 2
    assert metric["relative_error"] == pytest.approx(0.2)
    assert metric["definition"]["unit"] == "ms"
    assert result["validation"]["real_measurement_comparison_metrics"] == ["tpot"]
    assert result["validation"]["validation_passed"] is None
    assert not result["validation"]["calibration_performed"]
    assert document == original


@pytest.mark.parametrize("status", ["synthetic", "spec", "assumed", "simulated"])
def test_nonmeasurement_reference_never_becomes_real_validation(status):
    document = bundle()
    document["reference"]["status"] = status
    document["candidate"]["metrics"]["tpot"]["value"] = 10
    result = benchmark_tool.compare_benchmarks(document)
    assert result["metrics"]["tpot"]["absolute_error"] == 0
    assert result["validation"]["real_measurement_comparison_metrics"] == []
    assert not result["validation"]["reference_declared_measurement"]
    assert result["validation"]["validation_passed"] is None


def test_public_hardware_benchmark_with_synthetic_prompts_remains_measurement():
    document = bundle()
    document["reference"]["status"] = "public"
    result = benchmark_tool.compare_benchmarks(document)
    assert result["validation"]["reference_declared_measurement"]
    assert result["validation"]["real_measurement_comparison_metrics"] == ["tpot"]


@pytest.mark.parametrize("evidence_kind", ["estimate", "specification", "simulation", "synthetic"])
def test_publication_is_not_proof_of_hardware_measurement(evidence_kind):
    document = bundle()
    document["reference"].update(status="public", evidence_kind=evidence_kind)
    result = benchmark_tool.compare_benchmarks(document)
    assert result["metrics"]["tpot"]["absolute_error"] == 2
    assert not result["validation"]["real_evidence_eligible"]
    assert not result["validation"]["reference_declared_measurement"]
    assert result["validation"]["real_measurement_comparison_metrics"] == []


def test_measurement_candidate_can_compare_against_real_measurement():
    document = bundle()
    document["candidate"].update(status="measured", evidence_kind="measurement")
    result = benchmark_tool.compare_benchmarks(document)
    assert result["validation"]["real_evidence_eligible"]
    assert result["validation"]["candidate_declared_measurement"]
    assert result["validation"]["validation_passed"] is None


def test_simulated_label_with_synthetic_values_is_not_a_real_simulation_result():
    document = bundle()
    document["candidate"]["evidence_kind"] = "synthetic"
    result = benchmark_tool.compare_benchmarks(document)
    assert not result["validation"]["real_evidence_eligible"]
    assert not result["validation"]["candidate_declared_simulation"]


def test_evidence_kind_is_required_and_must_be_valid():
    for value in (None, "public", []):
        document = bundle()
        document["reference"]["evidence_kind"] = value
        with pytest.raises(benchmark_tool.BenchmarkError, match="evidence_kind"):
            benchmark_tool.compare_benchmarks(document)
    document = bundle()
    del document["reference"]["evidence_kind"]
    with pytest.raises(benchmark_tool.BenchmarkError, match="evidence_kind"):
        benchmark_tool.compare_benchmarks(document)


@pytest.mark.parametrize("unknown", [None, "unknown", " UNKNOWN ", "not_reported", "", {}, []])
def test_even_identical_unknown_context_cannot_be_compared(unknown):
    document = bundle()
    for label in ("reference", "candidate"):
        document[label]["context"]["hardware"]["topology"] = unknown
    result = benchmark_tool.compare_benchmarks(document)
    assert result["status"] == "not_comparable"
    assert any(reason["path"] == "reference.context.hardware.topology" and reason["code"] == "unknown" for reason in result["context_reasons"])
    assert "absolute_error" not in result["metrics"]["tpot"]


def test_required_groups_and_subfields_missing_on_both_sides_are_not_inferred():
    document = bundle()
    for label in ("reference", "candidate"):
        del document[label]["context"]["runtime"]
        del document[label]["context"]["precision"]["communication"]
    result = benchmark_tool.compare_benchmarks(document)
    assert result["status"] == "not_comparable"
    paths = {reason["path"] for reason in result["context_reasons"]}
    assert "reference.context.runtime" in paths
    assert "candidate.context.precision.communication" in paths


@pytest.mark.parametrize("group,field,value", [
    ("model", "revision", "another-checkpoint"),
    ("hardware", "count", 16),
    ("precision", "communication", "FP8"),
    ("workload", "batch_scope", "per GPU requests"),
    ("workload", "cache_state", "warm prefix cache"),
    ("execution", "speculative_decoding", True),
])
def test_context_differences_block_all_metrics_with_a_precise_path(group, field, value):
    document = bundle()
    document["candidate"]["context"][group][field] = value
    result = benchmark_tool.compare_benchmarks(document)
    assert result["status"] == "not_comparable"
    assert any(reason["path"] == f"context.{group}.{field}" and reason["code"] == "different" for reason in result["context_reasons"])
    assert result["validation"]["real_measurement_comparison_metrics"] == []


def test_additional_rank_maps_are_compared_without_multiplying_parallelism_degrees():
    document = bundle()
    for label in ("reference", "candidate"):
        document[label]["context"]["parallelism"]["tp"] = {"attention_tp": 8, "ranks": list(range(8))}
        document[label]["context"]["parallelism"]["moe_ep"] = {"ranks": list(range(8)), "reuse_attention_ranks": True}
    assert benchmark_tool.compare_benchmarks(document)["status"] == "comparable"
    document["candidate"]["context"]["parallelism"]["moe_ep"]["ranks"][2] = 9
    result = benchmark_tool.compare_benchmarks(document)
    assert any(reason["path"] == "context.parallelism.moe_ep.ranks[2]" for reason in result["context_reasons"])


@pytest.mark.parametrize("field", ["input_tokens", "output_tokens"])
@pytest.mark.parametrize("value", [[-1], [True], [1.5], ["1"], [None], [[1]], {"mean": 128}, {}])
def test_invalid_token_count_structures_are_rejected_on_both_sides(field, value):
    document = bundle()
    for label in ("reference", "candidate"):
        document[label]["context"]["workload"][field] = copy.deepcopy(value)
    with pytest.raises(benchmark_tool.BenchmarkError, match=f"workload.{field}"):
        benchmark_tool.compare_benchmarks(document)


def test_exact_per_request_token_arrays_are_compared_in_order():
    document = bundle()
    for label in ("reference", "candidate"):
        document[label]["context"]["workload"]["input_tokens"] = [0, 512, 1024]
        document[label]["context"]["workload"]["output_tokens"] = [128, 64, 0]
    assert benchmark_tool.compare_benchmarks(document)["status"] == "comparable"
    document["candidate"]["context"]["workload"]["input_tokens"] = [512, 0, 1024]
    result = benchmark_tool.compare_benchmarks(document)
    assert result["status"] == "not_comparable"
    assert any(reason["path"] == "context.workload.input_tokens[0]" for reason in result["context_reasons"])


def test_paired_per_request_token_arrays_require_equal_lengths():
    document = bundle()
    document["reference"]["context"]["workload"].update(input_tokens=[1024, 512], output_tokens=[128])
    with pytest.raises(benchmark_tool.BenchmarkError, match="equal per-request lengths"):
        benchmark_tool.compare_benchmarks(document)


@pytest.mark.parametrize("field", ["batch_size", "concurrency"])
@pytest.mark.parametrize("value", [[0], [True], [1], [], {"degree": 1}, {}, 0, False, 1.0])
def test_batch_size_and_concurrency_require_positive_scalar_integers(field, value):
    document = bundle()
    for label in ("reference", "candidate"):
        document[label]["context"]["workload"][field] = copy.deepcopy(value)
    with pytest.raises(benchmark_tool.BenchmarkError, match=f"workload.{field}"):
        benchmark_tool.compare_benchmarks(document)


@pytest.mark.parametrize("value", [
    [], [8], {"degree": 0}, {"degree": True}, {"attention_tp": -1},
    {"ranks": [True]}, {"ranks": [-1]}, {"ranks": [0, 0]},
    {"ranks": [0, 1.0]}, {"ranks": "0,1"}, {"group": {"degree": 0}},
])
def test_invalid_parallelism_arrays_degrees_and_rank_ids_are_rejected(value):
    document = bundle()
    for label in ("reference", "candidate"):
        document[label]["context"]["parallelism"]["tp"] = copy.deepcopy(value)
    with pytest.raises(benchmark_tool.BenchmarkError, match="parallelism.tp"):
        benchmark_tool.compare_benchmarks(document)


def test_invalid_rank_ids_in_extra_parallelism_maps_are_rejected():
    document = bundle()
    for label in ("reference", "candidate"):
        document[label]["context"]["parallelism"]["moe_ep"] = {"ranks": [0, -1]}
    with pytest.raises(benchmark_tool.BenchmarkError, match="parallelism.moe_ep.ranks"):
        benchmark_tool.compare_benchmarks(document)


def test_unknown_exact_counts_remain_incomparable_with_distribution_metadata():
    document = bundle()
    for label in ("reference", "candidate"):
        document[label]["context"]["workload"].update(
            input_tokens=None, batch_size="unknown", input_distribution={"mean": 1024, "stddev": 256},
        )
    result = benchmark_tool.compare_benchmarks(document)
    assert result["status"] == "not_comparable"
    assert any(reason["path"] == "reference.context.workload.input_tokens" for reason in result["context_reasons"])


def test_nested_unknown_in_extra_context_blocks_comparison():
    document = bundle()
    for label in ("reference", "candidate"):
        document[label]["context"]["scheduler"] = {"limits": [{"max_tokens": None}]}
    result = benchmark_tool.compare_benchmarks(document)
    assert result["status"] == "not_comparable"
    assert any(reason["path"] == "candidate.context.scheduler.limits[0].max_tokens" for reason in result["context_reasons"])


@pytest.mark.parametrize("field,changed", [
    ("unit", "s"), ("phase", "prefill"), ("timer_boundary", "client"),
    ("aggregation", "p99"), ("token_basis", "input plus output"),
    ("window", {"duration_s": 30, "warmup": "excluded", "cohort": "completed requests"}),
])
def test_metric_definition_mismatch_does_not_poison_other_metrics(field, changed):
    document = bundle()
    for label in ("reference", "candidate"):
        document[label]["metrics"]["unaffected"] = copy.deepcopy(document[label]["metrics"]["tpot"])
    document["candidate"]["metrics"]["tpot"][field] = changed
    result = benchmark_tool.compare_benchmarks(document)
    assert result["status"] == "partially_comparable"
    assert result["metrics"]["unaffected"]["status"] == "comparable"
    assert result["metrics"]["tpot"]["status"] == "not_comparable"
    assert any(reason["path"].startswith(f"metrics.tpot.{field}") for reason in result["metrics"]["tpot"]["reasons"])


def test_missing_metric_value_or_definition_never_defaults_to_zero():
    for field in benchmark_tool.METRIC_FIELDS:
        document = bundle()
        for label in ("reference", "candidate"):
            del document[label]["metrics"]["tpot"][field]
        result = benchmark_tool.compare_benchmarks(document)
        assert result["status"] == "not_comparable"
        assert "absolute_error" not in result["metrics"]["tpot"]
    document = bundle()
    document["candidate"]["metrics"] = {}
    result = benchmark_tool.compare_benchmarks(document)
    assert result["metrics"]["tpot"]["reasons"] == [{"path": "candidate.metrics.tpot", "code": "missing"}]


def test_zero_reference_and_single_token_missing_tpot_are_explicit():
    document = bundle()
    document["reference"]["metrics"]["tpot"]["value"] = 0
    result = benchmark_tool.compare_benchmarks(document)
    assert result["metrics"]["tpot"]["absolute_error"] == 12
    assert result["metrics"]["tpot"]["relative_error"] is None
    json.dumps(result, allow_nan=False)
    document["reference"]["metrics"]["tpot"]["value"] = None
    result = benchmark_tool.compare_benchmarks(document)
    assert result["status"] == "not_comparable"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, True, "12", [], {}])
def test_invalid_metric_values_are_rejected(value):
    document = bundle()
    document["candidate"]["metrics"]["tpot"]["value"] = value
    with pytest.raises(benchmark_tool.BenchmarkError):
        benchmark_tool.compare_benchmarks(document)


@pytest.mark.parametrize("change", [
    {"schema_version": True}, {"schema_version": 2}, {"extra": True},
    {"candidate": []},
])
def test_invalid_document_types_and_fields_are_rejected(change):
    document = bundle()
    document.update(change)
    with pytest.raises(benchmark_tool.BenchmarkError):
        benchmark_tool.compare_benchmarks(document)


def test_invalid_context_types_nonfinite_extras_and_duplicate_record_ids_are_rejected():
    document = bundle()
    document["candidate"]["context"]["hardware"]["count"] = True
    with pytest.raises(benchmark_tool.BenchmarkError, match="hardware.count"):
        benchmark_tool.compare_benchmarks(document)
    document = bundle()
    document["candidate"]["context"]["extra"] = float("nan")
    with pytest.raises(benchmark_tool.BenchmarkError, match="nonfinite"):
        benchmark_tool.compare_benchmarks(document)
    document = bundle()
    document["candidate"]["id"] = document["reference"]["id"]
    with pytest.raises(benchmark_tool.BenchmarkError, match="distinct"):
        benchmark_tool.compare_benchmarks(document)


def test_cli_outputs_report_and_valid_not_comparable_is_exit_zero(tmp_path):
    source = tmp_path / "bundle.json"
    output = tmp_path / "nested" / "report.json"
    document = bundle()
    document["reference"]["context"]["model"]["revision"] = None
    source.write_text(json.dumps(document), encoding="utf-8")
    run = subprocess.run([sys.executable, str(SCRIPT), str(source), "-o", str(output)], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "not_comparable"
    assert not run.stdout


@pytest.mark.parametrize("invalid", [
    '{"schema_version": NaN}', '{"schema_version":1,"schema_version":1}',
    '{"reference":{"id":"a","id":"b"}}', '{broken',
])
def test_cli_invalid_json_nonfinite_and_duplicate_keys_exit_two(tmp_path, invalid):
    source = tmp_path / "bundle.json"
    source.write_text(invalid, encoding="utf-8")
    run = subprocess.run([sys.executable, str(SCRIPT), str(source)], capture_output=True, text=True)
    assert run.returncode == 2
    assert "error:" in run.stderr
    assert not run.stdout


def test_cli_refuses_to_overwrite_input(tmp_path):
    source = tmp_path / "bundle.json"
    original = json.dumps(bundle())
    source.write_text(original, encoding="utf-8")
    run = subprocess.run([sys.executable, str(SCRIPT), str(source), "-o", "bundle.json"], cwd=tmp_path, capture_output=True, text=True)
    assert run.returncode == 2
    assert "must not overwrite the input" in run.stderr
    assert source.read_text(encoding="utf-8") == original


def test_synthetic_example_runs_without_claiming_real_validation():
    example = SCRIPT.parents[1] / "assets/benchmark-comparison.synthetic.json"
    result = benchmark_tool.compare_benchmarks(json.loads(example.read_text(encoding="utf-8")))
    assert result["status"] == "comparable"
    assert result["validation"]["real_measurement_comparison_metrics"] == []
    assert result["validation"]["validation_passed"] is None
