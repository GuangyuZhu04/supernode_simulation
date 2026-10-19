#!/usr/bin/env python3
"""Analyze measured or synthetic token-level traces without external packages.

Input JSON:
  metadata: required nonempty strings source, model, hardware, runtime;
    timer_boundary: client | engine;
    status: synthetic | public | measured | user | assumed.
    Optional timestamp_granularity must be "token". Extra metadata is preserved.
  measurement_window_s: [start, end], finite nonnegative seconds, end > start.
  requests: [{id: nonempty string, arrival_s: seconds, token_times_s: [seconds],
    finish_s?: seconds, status?: success | failed | cancelled | incomplete,
    warmup?: boolean, input_tokens?: integer, output_tokens?: integer}].
    status defaults to success; output_tokens must equal len(token_times_s).
    Each timestamp is an individual output token observed at the declared timer
    boundary, in one monotonic clock domain. Chunk-arrival times are unsupported.
  slo?: {ttft_s: nonnegative seconds, tpot_s: nonnegative seconds}.

TTFT = first token - arrival; TPOT = (last - first) / (output_tokens - 1).
TPOT is null for <=1 output token. E2E uses finish_s when supplied, otherwise
last token for a successful request; the output records the endpoint. A successful empty
response requires finish_s; its TTFT and TPOT are null.

The latency cohort consists of successful, non-warmup requests whose completion
is in [start, end), including requests that arrived before start. Token
throughput counts in-window token observations from all successful non-warmup
requests, even those completing outside the window. Explicit incomplete,
failed and cancelled requests are excluded and counted. Full-request ITLs from
the latency cohort are pooled separately from the mean of request TPOTs.

Short windows or small samples do not establish production p99 guarantees.
No fitting, queue inference, clock synchronization or chunk-to-token expansion
is performed. Example: python analyze_inference_trace.py trace.json -o metrics.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any


class TraceError(ValueError):
    """Invalid or ambiguous input trace."""


PROVENANCE_STATUSES = {"synthetic", "public", "measured", "user", "assumed"}
REQUEST_STATUSES = {"success", "failed", "cancelled", "incomplete"}
QUANTILE_METHOD = "linear interpolation at (n - 1) * p (Hyndman-Fan type 7)"
REQUEST_FIELDS = {
    "id", "arrival_s", "token_times_s", "finish_s", "status", "warmup",
    "input_tokens", "output_tokens",
}


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TraceError(f"{label} must be a finite nonnegative number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise TraceError(f"{label} must be finite") from exc
    if not math.isfinite(result) or result < 0:
        raise TraceError(f"{label} must be a finite nonnegative number")
    return result


def _count(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TraceError(f"{label} must be a nonnegative integer")
    return value


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TraceError(f"{label} must be an object")
    return value


def _required(obj: dict[str, Any], key: str, label: str) -> Any:
    if key not in obj:
        raise TraceError(f"{label}.{key} is required")
    return obj[key]


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TraceError(f"{label} must be a nonempty string")
    return value


def _choice(value: Any, choices: set[str], label: str) -> str:
    if not isinstance(value, str) or value not in choices:
        raise TraceError(f"{label} must be one of {', '.join(sorted(choices))}")
    return value


def summarize(values: list[float]) -> dict[str, Any]:
    """Summarize unweighted samples; empty percentiles are null, never NaN."""
    ordered = sorted(values)
    size = len(ordered)

    def quantile(p: float) -> float | None:
        if not size:
            return None
        position = (size - 1) * p
        lower = math.floor(position)
        upper = math.ceil(position)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    # Scale before summing so large finite durations cannot overflow their mean.
    maximum = ordered[-1] if size else 0.0
    mean = maximum * (math.fsum(value / maximum for value in ordered) / size) if maximum else (0.0 if size else None)
    return {
        "sample_count": size,
        "mean_s": mean,
        "p50_s": quantile(0.50),
        "p95_s": quantile(0.95),
        "p99_s": quantile(0.99),
        "quantile_method": QUANTILE_METHOD,
    }


def _rate(count: int, duration: float) -> float:
    rate = count / duration
    if not math.isfinite(rate):
        raise TraceError("measurement window is too short to represent a finite throughput")
    return rate


def analyze_trace(document: Any) -> dict[str, Any]:
    """Validate a trace and return JSON-compatible, explicitly scoped metrics."""
    document = _object(document, "trace")
    unknown = set(document) - {"metadata", "measurement_window_s", "requests", "slo"}
    if unknown:
        raise TraceError(f"unsupported trace fields: {', '.join(sorted(unknown))}")
    metadata = _object(_required(document, "metadata", "trace"), "metadata")
    for key in ("source", "model", "hardware", "runtime"):
        _text(_required(metadata, key, "metadata"), f"metadata.{key}")
    _choice(metadata.get("timer_boundary"), {"client", "engine"}, "metadata.timer_boundary")
    _choice(metadata.get("status"), PROVENANCE_STATUSES, "metadata.status")
    _choice(metadata.get("timestamp_granularity", "token"), {"token"}, "metadata.timestamp_granularity")
    try:
        json.dumps(metadata, allow_nan=False)
    except (ValueError, TypeError, OverflowError) as exc:
        raise TraceError("metadata must be valid JSON without nonfinite numbers") from exc

    window = _required(document, "measurement_window_s", "trace")
    if not isinstance(window, list) or len(window) != 2:
        raise TraceError("measurement_window_s must be [start, end]")
    start, end = [_number(value, f"measurement_window_s[{index}]") for index, value in enumerate(window)]
    if end <= start:
        raise TraceError("measurement_window_s end must be greater than start")
    duration = end - start

    slo = document.get("slo")
    if "slo" in document:
        slo = _object(slo, "slo")
        if set(slo) != {"ttft_s", "tpot_s"}:
            raise TraceError("slo must contain exactly ttft_s and tpot_s")
        slo = {key: _number(value, f"slo.{key}") for key, value in slo.items()}

    requests = _required(document, "requests", "trace")
    if not isinstance(requests, list):
        raise TraceError("requests must be a list")
    ids: set[str] = set()
    rows: list[dict[str, Any]] = []
    ttfts: list[float] = []
    tpots: list[float] = []
    e2es: list[float] = []
    itls: list[float] = []
    counts = {
        "total_requests": len(requests),
        "warmup_requests": 0,
        "failed_requests": 0,
        "cancelled_requests": 0,
        "incomplete_requests": 0,
        "excluded_warmup_requests": 0,
        "excluded_non_success_non_warmup_requests": 0,
        "eligible_success_requests": 0,
        "completed_before_window_requests": 0,
        "completed_in_window_requests": 0,
        "completed_at_or_after_window_end_requests": 0,
        "cohort_arrived_before_window_requests": 0,
        "cohort_zero_token_requests": 0,
        "cohort_single_token_requests": 0,
        "observed_tokens_in_window": 0,
        "excluded_warmup_tokens_in_window": 0,
        "excluded_non_success_non_warmup_tokens_in_window": 0,
        "eligible_tokens_in_window": 0,
    }
    goodput_passed = 0
    goodput_tpot_na = 0
    goodput_ttft_missing = 0

    for index, raw in enumerate(requests):
        label = f"requests[{index}]"
        request = _object(raw, label)
        unknown = set(request) - REQUEST_FIELDS
        if unknown:
            raise TraceError(f"{label}: unsupported fields {', '.join(sorted(unknown))}; chunk traces are unsupported")
        request_id = _text(_required(request, "id", label), f"{label}.id")
        if request_id in ids:
            raise TraceError(f"{label}.id is duplicated: {request_id}")
        ids.add(request_id)
        arrival = _number(_required(request, "arrival_s", label), f"{label}.arrival_s")
        status = _choice(request.get("status", "success"), REQUEST_STATUSES, f"{label}.status")
        warmup = request.get("warmup", False)
        if not isinstance(warmup, bool):
            raise TraceError(f"{label}.warmup must be boolean")
        raw_tokens = _required(request, "token_times_s", label)
        if not isinstance(raw_tokens, list):
            raise TraceError(f"{label}.token_times_s must be a list of individual token times")
        tokens = [_number(value, f"{label}.token_times_s[{j}]") for j, value in enumerate(raw_tokens)]
        previous = arrival
        for token in tokens:
            if token < previous:
                raise TraceError(f"{label}.token_times_s must be nondecreasing and no earlier than arrival_s")
            previous = token
        output_tokens = len(tokens)
        if "output_tokens" in request and _count(request["output_tokens"], f"{label}.output_tokens") != output_tokens:
            raise TraceError(f"{label}.output_tokens must match the number of token_times_s; chunk timestamps cannot substitute for token timestamps")
        if "input_tokens" in request:
            _count(request["input_tokens"], f"{label}.input_tokens")
        if "finish_s" in request:
            completion = _number(request["finish_s"], f"{label}.finish_s")
            if completion < previous:
                raise TraceError(f"{label}.finish_s must be no earlier than arrival and the last token")
            endpoint = "finish_s"
        else:
            completion = tokens[-1] if tokens and status == "success" else None
            endpoint = "last_token_s" if completion is not None else None
        if status == "success" and completion is None:
            raise TraceError(f"{label}: a successful zero-token response requires finish_s")

        ttft = tokens[0] - arrival if tokens else None
        tpot = (tokens[-1] - tokens[0]) / (output_tokens - 1) if output_tokens > 1 else None
        e2e = completion - arrival if completion is not None else None
        in_window_tokens = sum(start <= token < end for token in tokens)
        counts["observed_tokens_in_window"] += in_window_tokens
        counts["warmup_requests"] += int(warmup)
        if status != "success":
            counts[f"{status}_requests"] += 1
        eligible = status == "success" and not warmup
        in_cohort = eligible and completion is not None and start <= completion < end
        if warmup:
            exclusion = "warmup"
            counts["excluded_warmup_requests"] += 1
            counts["excluded_warmup_tokens_in_window"] += in_window_tokens
        elif status != "success":
            exclusion = status
            counts["excluded_non_success_non_warmup_requests"] += 1
            counts["excluded_non_success_non_warmup_tokens_in_window"] += in_window_tokens
        else:
            counts["eligible_success_requests"] += 1
            counts["eligible_tokens_in_window"] += in_window_tokens
            if completion < start:
                exclusion = "completed_before_window"
                counts["completed_before_window_requests"] += 1
            elif completion >= end:
                exclusion = "completed_at_or_after_window_end"
                counts["completed_at_or_after_window_end_requests"] += 1
            else:
                exclusion = None
                counts["completed_in_window_requests"] += 1
                counts["cohort_arrived_before_window_requests"] += int(arrival < start)
                counts["cohort_zero_token_requests"] += int(output_tokens == 0)
                counts["cohort_single_token_requests"] += int(output_tokens == 1)
                if ttft is not None:
                    ttfts.append(ttft)
                if tpot is not None:
                    tpots.append(tpot)
                e2es.append(e2e)
                itls.extend(later - earlier for earlier, later in zip(tokens, tokens[1:]))
        meets_slo = None
        if slo is not None and in_cohort:
            goodput_tpot_na += int(tpot is None)
            goodput_ttft_missing += int(ttft is None)
            meets_slo = ttft is not None and ttft <= slo["ttft_s"] and (tpot is None or tpot <= slo["tpot_s"])
            goodput_passed += int(meets_slo)
        rows.append({
            "id": request_id, "status": status, "warmup": warmup,
            "input_tokens": request.get("input_tokens"), "output_tokens": output_tokens,
            "ttft_s": ttft, "tpot_s": tpot, "e2e_s": e2e,
            "tpot_applicable": output_tokens > 1,
            "completion_s": completion, "completion_endpoint": endpoint,
            "in_latency_cohort": in_cohort, "latency_exclusion_reason": exclusion,
            "observed_tokens_in_window": in_window_tokens, "meets_joint_slo": meets_slo,
        })

    completed = counts["completed_in_window_requests"]
    return {
        "schema_version": 1,
        "metadata": metadata,
        "measurement_window_s": [start, end],
        "measurement_duration_s": duration,
        "definitions": {
            "window": "half-open [start, end)",
            "latency_cohort": "success, non-warmup, completion in window; arrivals before window are included",
            "ttft": "first output token time - arrival_s",
            "request_tpot": "(last token time - first token time) / (output_tokens - 1); null for <=1 token",
            "pooled_token_itl": "all adjacent token intervals of latency-cohort requests; each interval has equal weight, including intervals before window start",
            "e2e": "finish_s - arrival_s when supplied, otherwise last token time - arrival_s for success only; see each request's completion_endpoint",
            "token_throughput": "in-window token observations from success non-warmup requests, regardless of their completion window",
            "joint_slo": "cohort request satisfies TTFT and applicable TPOT thresholds; <=1 token has no applicable TPOT, missing TTFT fails",
        },
        "counts": counts,
        "metrics": {
            "ttft": summarize(ttfts), "request_tpot": summarize(tpots),
            "e2e": summarize(e2es), "pooled_token_itl": summarize(itls),
        },
        "throughput": {
            "completed_requests_per_s": _rate(completed, duration),
            "output_tokens_per_s": _rate(counts["eligible_tokens_in_window"], duration),
        },
        "goodput": None if slo is None else {
            "thresholds_s": slo, "evaluated_requests": completed,
            "passed_requests": goodput_passed,
            "failed_requests": completed - goodput_passed,
            "tpot_not_applicable_requests": goodput_tpot_na,
            "ttft_unavailable_requests": goodput_ttft_missing,
            "joint_requests_per_s": _rate(goodput_passed, duration),
        },
        "requests": rows,
        "notes": [
            "Short measurement windows or small samples cannot establish production p99 tail-latency guarantees; report duration and sample counts.",
            "Client and engine timer boundaries are not interchangeable; this tool does not infer queue, network, tokenization, or detokenization costs.",
            "Completion-based latency cohorts can censor slow in-flight requests; report excluded/incomplete counts and use a representative steady-state window.",
            "Failed, cancelled, incomplete and warmup tokens are excluded from output throughput and explicitly counted; observed_tokens_in_window includes them.",
            "A token timestamp must represent an individually observed token; do not replicate a chunk timestamp across its tokens.",
        ],
    }


def _reject_constant(value: str) -> None:
    raise TraceError(f"invalid nonfinite JSON number: {value}")


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TraceError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="JSON token trace")
    parser.add_argument("-o", "--output", type=Path, help="write JSON to this path instead of stdout")
    args = parser.parse_args(argv)
    try:
        if args.output and args.output.resolve() == args.input.resolve():
            raise TraceError("output must not overwrite the input trace")
        document = json.loads(args.input.read_text(encoding="utf-8-sig"), parse_constant=_reject_constant, object_pairs_hook=_unique_keys)
        result = analyze_trace(document)
        encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded, encoding="utf-8")
        else:
            print(encoded, end="")
    except (TraceError, OSError, UnicodeError, ValueError, OverflowError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
