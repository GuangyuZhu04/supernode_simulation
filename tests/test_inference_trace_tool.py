"""Behavioral checks for the standalone token trace analysis skill resource."""

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "skills/supernode-modeling/scripts/analyze_inference_trace.py"
SPEC = importlib.util.spec_from_file_location("analyze_inference_trace", SCRIPT)
trace_tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trace_tool)


def trace(requests=None, window=None):
    return {
        "metadata": {
            "timer_boundary": "client", "status": "synthetic",
            "source": "unit test; not a hardware measurement",
            "model": "synthetic", "hardware": "none", "runtime": "none",
        },
        "measurement_window_s": window or [0, 10],
        "requests": requests if requests is not None else [
            {"id": "r1", "arrival_s": 1, "token_times_s": [2, 3, 5], "finish_s": 6},
        ],
    }


def test_formula_and_finish_endpoint_are_explicit():
    result = trace_tool.analyze_trace(trace())
    row = result["requests"][0]
    assert row["ttft_s"] == 1
    assert row["tpot_s"] == 1.5
    assert row["e2e_s"] == 5
    assert row["completion_endpoint"] == "finish_s"
    assert result["metrics"]["pooled_token_itl"]["mean_s"] == 1.5
    assert result["throughput"] == {"completed_requests_per_s": 0.1, "output_tokens_per_s": 0.3}


def test_observed_admission_separates_queue_and_service_without_changing_client_ttft():
    document = trace([{'id': 'queued', 'arrival_s': 0, 'admission_s': 2,
                       'token_times_s': [3, 4]}])
    result = trace_tool.analyze_trace(document)
    row = result['requests'][0]
    assert row['ttft_s'] == 3
    assert row['admission_wait_s'] == 2
    assert row['server_ttft_s'] == 1
    assert row['ttft_s'] == row['admission_wait_s'] + row['server_ttft_s']
    assert result['metrics']['admission_wait']['mean_s'] == 2
    assert result['metrics']['server_ttft']['mean_s'] == 1
    assert result['counts']['cohort_admission_observed_requests'] == 1
    old = trace_tool.analyze_trace(trace())
    assert old['requests'][0]['admission_wait_s'] is None
    assert old['metrics']['admission_wait']['sample_count'] == 0


@pytest.mark.parametrize('admission', [-1, 0.5, 2.5, True, float('nan')])
def test_invalid_admission_timestamp_is_rejected(admission):
    with pytest.raises(trace_tool.TraceError):
        trace_tool.analyze_trace(trace([{'id': 'a', 'arrival_s': 1,
            'admission_s': admission, 'token_times_s': [2, 3]}]))


def test_single_and_zero_token_requests_have_null_tpot():
    document = trace([
        {"id": "single", "arrival_s": 0, "token_times_s": [1]},
        {"id": "empty", "arrival_s": 0, "token_times_s": [], "finish_s": 2},
    ])
    document["slo"] = {"ttft_s": 1, "tpot_s": 0}
    result = trace_tool.analyze_trace(document)
    assert [row["tpot_s"] for row in result["requests"]] == [None, None]
    assert result["requests"][0]["completion_endpoint"] == "last_token_s"
    assert result["requests"][1]["ttft_s"] is None
    assert result["metrics"]["request_tpot"]["sample_count"] == 0
    assert result["metrics"]["request_tpot"]["p99_s"] is None
    assert result["goodput"]["passed_requests"] == 1
    assert result["goodput"]["tpot_not_applicable_requests"] == 2
    assert result["goodput"]["ttft_unavailable_requests"] == 1
    json.dumps(result, allow_nan=False)


def test_ragged_requests_do_not_conflate_request_tpot_with_pooled_itl():
    document = trace([
        {"id": "short", "arrival_s": 0, "token_times_s": [1, 5]},
        {"id": "long", "arrival_s": 0, "token_times_s": [1, 2, 3, 4]},
    ])
    result = trace_tool.analyze_trace(document)
    assert result["metrics"]["request_tpot"]["mean_s"] == 2.5
    assert result["metrics"]["request_tpot"]["sample_count"] == 2
    assert result["metrics"]["pooled_token_itl"]["mean_s"] == 1.75
    assert result["metrics"]["pooled_token_itl"]["sample_count"] == 4
    assert result["metrics"]["request_tpot"]["p95_s"] == pytest.approx(3.85)


def test_window_completion_cohort_and_exclusions_are_counted():
    document = trace([
        {"id": "cross-start", "arrival_s": 0, "token_times_s": [1, 2, 3]},
        {"id": "warmup", "arrival_s": 0, "token_times_s": [2, 3], "warmup": True},
        {"id": "failed", "arrival_s": 1, "token_times_s": [2], "finish_s": 3, "status": "failed"},
        {"id": "incomplete", "arrival_s": 1, "token_times_s": [3], "status": "incomplete"},
        {"id": "cross-end", "arrival_s": 2, "token_times_s": [3, 4]},
        {"id": "before", "arrival_s": 0, "token_times_s": [1]},
        {"id": "warmup-failed", "arrival_s": 1, "token_times_s": [2], "status": "failed", "warmup": True},
    ], window=[2, 4])
    result = trace_tool.analyze_trace(document)
    counts = result["counts"]
    assert counts["completed_in_window_requests"] == 1
    assert counts["cohort_arrived_before_window_requests"] == 1
    assert counts["completed_at_or_after_window_end_requests"] == 1
    assert counts["completed_before_window_requests"] == 1
    assert counts["failed_requests"] == 2
    assert counts["excluded_warmup_requests"] == 2
    assert counts["excluded_non_success_non_warmup_requests"] == 2
    assert counts["observed_tokens_in_window"] == 8
    assert counts["excluded_warmup_tokens_in_window"] == 3
    assert counts["excluded_non_success_non_warmup_tokens_in_window"] == 2
    assert counts["eligible_tokens_in_window"] == 3
    assert result["metrics"]["ttft"]["mean_s"] == 1
    assert result["metrics"]["pooled_token_itl"]["sample_count"] == 2
    assert result["throughput"] == {"completed_requests_per_s": 0.5, "output_tokens_per_s": 1.5}


def test_joint_goodput_requires_both_thresholds_per_request():
    document = trace([
        {"id": "ttft-fail", "arrival_s": 0, "token_times_s": [2, 2.1]},
        {"id": "tpot-fail", "arrival_s": 0, "token_times_s": [0.5, 2]},
        {"id": "pass", "arrival_s": 0, "token_times_s": [0.5, 1]},
    ])
    document["slo"] = {"ttft_s": 1, "tpot_s": 1}
    result = trace_tool.analyze_trace(document)
    assert result["goodput"]["passed_requests"] == 1
    assert result["goodput"]["joint_requests_per_s"] == 0.1


@pytest.mark.parametrize("tokens", [[1, 0.5], [-1], [float("nan")], [float("inf")], [True], ["1"]])
def test_invalid_token_times_are_rejected(tokens):
    document = trace([{"id": "r1", "arrival_s": 1, "token_times_s": tokens}])
    with pytest.raises(trace_tool.TraceError):
        trace_tool.analyze_trace(document)


@pytest.mark.parametrize("change", [
    {"output_tokens": 5}, {"chunk_sizes": [1, 2, 3]}, {"finish_s": 4},
    {"input_tokens": True}, {"warmup": "false"}, {"status": "ok"},
    {"arrival_s": -1}, {"finish_s": None},
])
def test_ambiguous_or_invalid_request_fields_are_rejected(change):
    document = trace()
    document["requests"][0].update(change)
    with pytest.raises(trace_tool.TraceError):
        trace_tool.analyze_trace(document)


def test_duplicate_ids_missing_tokens_and_success_without_completion_are_rejected():
    document = trace()
    document["requests"].append(copy.deepcopy(document["requests"][0]))
    with pytest.raises(trace_tool.TraceError, match="duplicated"):
        trace_tool.analyze_trace(document)
    for row in ({"id": "x", "arrival_s": 0}, {"id": "x", "arrival_s": 0, "token_times_s": []}):
        with pytest.raises(trace_tool.TraceError):
            trace_tool.analyze_trace(trace([row]))


def test_empty_trace_and_incomplete_request_do_not_generate_nan():
    result = trace_tool.analyze_trace(trace([]))
    assert result["metrics"]["ttft"]["mean_s"] is None
    assert result["throughput"]["completed_requests_per_s"] == 0
    result = trace_tool.analyze_trace(trace([{"id": "waiting", "arrival_s": 0, "token_times_s": [], "status": "incomplete"}]))
    assert result["requests"][0]["e2e_s"] is None
    assert result["counts"]["incomplete_requests"] == 1
    json.dumps(result, allow_nan=False)


def test_partial_tokens_do_not_invent_a_completion_endpoint():
    for status in ("incomplete", "failed", "cancelled"):
        result = trace_tool.analyze_trace(trace([
            {"id": "partial", "arrival_s": 0, "token_times_s": [1, 2], "status": status},
        ]))
        assert result["requests"][0]["completion_s"] is None
        assert result["requests"][0]["e2e_s"] is None
        assert result["metrics"]["e2e"]["sample_count"] == 0
        assert result["throughput"]["output_tokens_per_s"] == 0


def test_declared_chunk_granularity_and_nonfinite_metadata_are_rejected():
    document = trace()
    document["metadata"]["timestamp_granularity"] = "chunk"
    with pytest.raises(trace_tool.TraceError, match="timestamp_granularity"):
        trace_tool.analyze_trace(document)
    document = trace()
    document["metadata"]["extra"] = float("nan")
    with pytest.raises(trace_tool.TraceError, match="nonfinite"):
        trace_tool.analyze_trace(document)


@pytest.mark.parametrize("window", [[1, 1], [2, 1], [-1, 1], [0, float("inf")], [0]])
def test_invalid_windows_are_rejected(window):
    with pytest.raises(trace_tool.TraceError):
        trace_tool.analyze_trace(trace(window=window))


def test_cli_writes_json_and_invalid_input_exits_two(tmp_path):
    source = tmp_path / "trace.json"
    output = tmp_path / "metrics.json"
    source.write_text(json.dumps(trace()), encoding="utf-8")
    run = subprocess.run([sys.executable, str(SCRIPT), str(source), "-o", str(output)], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert json.loads(output.read_text(encoding="utf-8"))["metrics"]["ttft"]["mean_s"] == 1
    for invalid in ('{"metadata": NaN}', '{"metadata": {}, "metadata": {}}', '{broken'):
        source.write_text(invalid, encoding="utf-8")
        run = subprocess.run([sys.executable, str(SCRIPT), str(source)], capture_output=True, text=True)
        assert run.returncode == 2
        assert "error:" in run.stderr
        assert not run.stdout


def test_large_finite_values_do_not_overflow_mean():
    summary = trace_tool.summarize([1e308, 1e308])
    assert summary["mean_s"] == 1e308
    json.dumps(summary, allow_nan=False)


def test_cli_creates_missing_nested_output_parent(tmp_path):
    source = tmp_path / "trace.json"
    output = tmp_path / "outputs" / "nested" / "metrics.json"
    source.write_text(json.dumps(trace()), encoding="utf-8")
    run = subprocess.run([sys.executable, str(SCRIPT), str(source), "-o", str(output)], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert json.loads(output.read_text(encoding="utf-8"))["metrics"]["ttft"]["mean_s"] == 1


def test_cli_refuses_to_overwrite_its_input_trace(tmp_path):
    source = tmp_path / "trace.json"
    original = json.dumps(trace())
    source.write_text(original, encoding="utf-8")
    run = subprocess.run(
        [sys.executable, str(SCRIPT), str(source), "-o", "trace.json"],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert run.returncode == 2
    assert "must not overwrite the input trace" in run.stderr
    assert source.read_text(encoding="utf-8") == original
    assert not run.stdout
