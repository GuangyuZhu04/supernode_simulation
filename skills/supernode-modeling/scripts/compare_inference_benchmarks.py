#!/usr/bin/env python3
"""Compare explicitly matched inference benchmarks using only the standard library.

Input: {schema_version: 1, reference: record, candidate: record}.
Each record contains id, status, evidence_kind, source, context and a metrics
object. Status is public, measured, simulated, synthetic, spec or assumed;
evidence_kind is measurement, estimate, specification, simulation or synthetic.
Source is a nonempty citation/description. Public availability does not imply a
measurement: published estimates must use evidence_kind=estimate. Synthetic
input prompts do not make a measured hardware run synthetic evidence. This tool trusts declarations; it
does not authenticate a URL or infer measurements from a product specification.

Required context groups and fields are listed in CONTEXT_FIELDS below. Runtime
means the target serving runtime, not the simulator implementation; put simulator
version/configuration in candidate.source. Additional context is also compared.
Missing fields, null, unknown markers or empty containers prevent comparison.
Use explicit values such as false or "disabled" for known disabled features.
Parallelism may use positive integers or structured objects, not arrays. Within
objects, degree and *_degree are positive integers; component *_tp/pp/ep/dp
fields are positive degrees or nested objects. Ranks is an array of distinct
nonnegative integer rank IDs. Other declarations are compared
exactly, but rank coverage, reuse and group semantics are not authenticated. No
product of TP/PP/EP/DP degrees is inferred. Input/output token counts must be
nonnegative integers or ordered arrays of per-request nonnegative integers;
paired input/output arrays must have equal length.
Batch size, concurrency and hardware count must be positive integers. Unknown
scalar counts may be null/unknown; do not substitute a distribution object for
exact token counts. Put distribution metadata in extra context fields, with
unknown exact counts set to null. No implicit unit conversion,
hardware scaling, batch scaling, cache assumption or condition interpolation occurs.

Each metric is {value, unit, phase, timer_boundary, aggregation, token_basis,
window}. Value is a finite nonnegative number, or null/unknown/missing when not
available. All definition fields must be known and equal for that metric. Window
is a string or object describing duration, cohort, warmup and sample scope; use
explicit "not_applicable" where appropriate. Different metric definitions only
block the affected metric; different execution context blocks every metric.

Comparable metrics report candidate-reference, absolute error and relative error
as a fraction of |reference|. Relative error is null for a zero reference. These
are descriptive differences, never an automatic fit or a validation pass. Only
public/measured references with evidence_kind=measurement, compared with
simulated/simulation or public-or-measured/measurement candidates, are eligible
for real-measurement comparison. Synthetic/spec/assumed examples remain arithmetic
comparisons. No efficiency or model parameter is changed.

CLI: python compare_inference_benchmarks.py comparison.json -o report.json
Exit 0: valid report, including not_comparable. Exit 2: invalid input or I/O.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any


class BenchmarkError(ValueError):
    """Invalid benchmark input, distinct from valid but incomparable evidence."""


PROVENANCE_STATUSES = {"public", "measured", "simulated", "synthetic", "spec", "assumed"}
EVIDENCE_KINDS = {"measurement", "estimate", "specification", "simulation", "synthetic"}
UNKNOWN_MARKERS = {"unknown", "unspecified", "not_reported", "not reported", "unavailable", "tbd", "?", "n/a"}
CONTEXT_FIELDS = {
    "model": ("name", "revision"),
    "hardware": ("sku", "count", "topology"),
    "runtime": ("name", "version", "backend_version"),
    "precision": ("weights", "activations", "kv_cache", "communication"),
    "parallelism": ("tp", "pp", "ep", "dp"),
    "workload": ("input_tokens", "output_tokens", "batch_size", "batch_scope", "concurrency", "arrival_process", "cache_state"),
    "execution": ("speculative_decoding", "chunked_prefill"),
}
METRIC_FIELDS = ("value", "unit", "phase", "timer_boundary", "aggregation", "token_basis", "window")
_MISSING = object()


def _object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BenchmarkError(f"{path} must be an object")
    return value


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkError(f"{path} must be a nonempty string")
    return value


def _unknown(value: Any) -> bool:
    return value is None or (isinstance(value, str) and (
        not value.strip() or value.strip().lower() in UNKNOWN_MARKERS
    )) or (isinstance(value, (dict, list)) and not value)


def _validate_json(value: Any, path: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise BenchmarkError(f"{path} contains a nonfinite number")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                raise BenchmarkError(f"{path} keys must be nonempty strings")
            _validate_json(item, f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json(item, f"{path}[{index}]")
        return
    raise BenchmarkError(f"{path} must contain only JSON values")


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BenchmarkError(f"{path} must be a finite nonnegative number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise BenchmarkError(f"{path} must be a finite nonnegative number") from exc
    if not math.isfinite(number) or number < 0:
        raise BenchmarkError(f"{path} must be a finite nonnegative number")
    return number


def _integer(value: Any, path: str, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise BenchmarkError(f"{path} must be an integer >= {minimum}")


def _unknown_scalar(value: Any) -> bool:
    return value is None or (isinstance(value, str) and _unknown(value))


def _parallel_mapping(mapping: dict[str, Any], path: str) -> None:
    """Check declared degrees/rank IDs without inferring rank-group semantics."""
    for key, value in mapping.items():
        item_path = f"{path}.{key}"
        if _unknown_scalar(value):
            continue
        if key == "degree" or key.endswith("_degree"):
            _integer(value, item_path, 1)
        elif key == "ranks":
            if not isinstance(value, list):
                raise BenchmarkError(f"{item_path} must be an array of distinct nonnegative integer rank IDs")
            for index, rank in enumerate(value):
                _integer(rank, f"{item_path}[{index}]", 0)
            if len(set(value)) != len(value):
                raise BenchmarkError(f"{item_path} must not contain duplicate rank IDs")
        elif isinstance(value, dict):
            _parallel_mapping(value, item_path)
        elif key in {"tp", "pp", "ep", "dp"} or key.endswith(("_tp", "_pp", "_ep", "_dp")):
            _integer(value, item_path, 1)


def _context_types(context: dict[str, Any], path: str) -> None:
    for group in CONTEXT_FIELDS:
        value = context.get(group)
        if not _unknown(value):
            _object(value, f"{path}.{group}")
    for group, fields in {
        "model": ("name", "revision"), "runtime": ("name", "version", "backend_version"),
        "hardware": ("sku",), "workload": ("batch_scope",),
    }.items():
        value = context.get(group)
        if isinstance(value, dict):
            for field in fields:
                item = value.get(field)
                if not _unknown(item):
                    _text(item, f"{path}.{group}.{field}")
    for group, fields, allowed in (
        ("hardware", ("topology",), (str, dict, list)),
        ("precision", CONTEXT_FIELDS["precision"], (str, dict)),
        ("workload", ("arrival_process", "cache_state"), (str, dict)),
        ("execution", CONTEXT_FIELDS["execution"], (bool, str, dict)),
    ):
        value = context.get(group)
        if isinstance(value, dict):
            for field in fields:
                item = value.get(field)
                if not _unknown(item) and not isinstance(item, allowed):
                    raise BenchmarkError(f"{path}.{group}.{field} has an invalid type")
    for group, fields in {
        "hardware": ("count",), "parallelism": ("tp", "pp", "ep", "dp"),
        "workload": ("input_tokens", "output_tokens", "batch_size", "concurrency"),
    }.items():
        value = context.get(group)
        if not isinstance(value, dict):
            continue
        if group == "parallelism":
            _parallel_mapping(value, f"{path}.{group}")
        for field in fields:
            item = value.get(field)
            item_path = f"{path}.{group}.{field}"
            if _unknown_scalar(item):
                continue
            if group == "parallelism" and isinstance(item, dict):
                _parallel_mapping(item, item_path)
                continue
            if field in {"input_tokens", "output_tokens"}:
                if isinstance(item, list):
                    for index, count in enumerate(item):
                        _integer(count, f"{item_path}[{index}]", 0)
                else:
                    _integer(item, item_path, 0)
            else:
                _integer(item, item_path, 1)
    workload = context.get("workload")
    if isinstance(workload, dict):
        inputs, outputs = workload.get("input_tokens"), workload.get("output_tokens")
        if isinstance(inputs, list) and isinstance(outputs, list) and len(inputs) != len(outputs):
            raise BenchmarkError(f"{path}.workload input_tokens/output_tokens arrays must have equal per-request lengths")


def _record(record: Any, label: str) -> dict[str, Any]:
    record = _object(record, label)
    required = {"id", "status", "evidence_kind", "source", "context", "metrics"}
    missing = required - set(record)
    unsupported = set(record) - required
    if missing or unsupported:
        raise BenchmarkError(f"{label}: missing fields {sorted(missing)}; unsupported fields {sorted(unsupported)}")
    _text(record["id"], f"{label}.id")
    _text(record["source"], f"{label}.source")
    if not isinstance(record["status"], str) or record["status"] not in PROVENANCE_STATUSES:
        raise BenchmarkError(f"{label}.status must be one of {', '.join(sorted(PROVENANCE_STATUSES))}")
    if not isinstance(record["evidence_kind"], str) or record["evidence_kind"] not in EVIDENCE_KINDS:
        raise BenchmarkError(f"{label}.evidence_kind must be one of {', '.join(sorted(EVIDENCE_KINDS))}")
    _context_types(_object(record["context"], f"{label}.context"), f"{label}.context")
    metrics = _object(record["metrics"], f"{label}.metrics")
    for name, metric in metrics.items():
        path = f"{label}.metrics.{name}"
        metric = _object(metric, path)
        extra = set(metric) - set(METRIC_FIELDS)
        if extra:
            raise BenchmarkError(f"{path}: unsupported fields {', '.join(sorted(extra))}")
        value = metric.get("value")
        if value is not None and not (isinstance(value, str) and _unknown(value)):
            _number(value, f"{path}.value")
        for field in METRIC_FIELDS[1:]:
            value = metric.get(field)
            if value is not None and not isinstance(value, str) and not (field == "window" and isinstance(value, dict)):
                raise BenchmarkError(f"{path}.{field} has an invalid type")
            if not _unknown(value) and not (field == "window" and isinstance(value, dict)):
                _text(value, f"{path}.{field}")
    return record


def _reason(path: str, code: str, **details: Any) -> dict[str, Any]:
    return {"path": path, "code": code, **details}


def _unknown_paths(value: Any, path: str) -> list[dict[str, Any]]:
    if _unknown(value):
        return [_reason(path, "unknown", value=value)]
    if isinstance(value, dict):
        return [reason for key, item in value.items() for reason in _unknown_paths(item, f"{path}.{key}")]
    if isinstance(value, list):
        return [reason for index, item in enumerate(value) for reason in _unknown_paths(item, f"{path}[{index}]")]
    return []


def _differences(reference: Any, candidate: Any, path: str) -> list[dict[str, Any]]:
    if reference is _MISSING or candidate is _MISSING:
        return [_reason(path, "missing", side="reference" if reference is _MISSING else "candidate")]
    if isinstance(reference, dict) and isinstance(candidate, dict):
        return [reason for key in sorted(set(reference) | set(candidate)) for reason in _differences(
            reference.get(key, _MISSING), candidate.get(key, _MISSING), f"{path}.{key}"
        )]
    if isinstance(reference, list) and isinstance(candidate, list):
        return [reason for index in range(max(len(reference), len(candidate))) for reason in _differences(
            reference[index] if index < len(reference) else _MISSING,
            candidate[index] if index < len(candidate) else _MISSING, f"{path}[{index}]"
        )]
    # JSON booleans must not compare equal to numeric 0/1.
    if reference != candidate or isinstance(reference, bool) != isinstance(candidate, bool):
        return [_reason(path, "different", reference=reference, candidate=candidate)]
    return []


def _context_reasons(reference: dict[str, Any], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    reasons = []
    for label, context in (("reference", reference), ("candidate", candidate)):
        reasons.extend(_unknown_paths(context, f"{label}.context"))
        for group, fields in CONTEXT_FIELDS.items():
            if group not in context:
                reasons.append(_reason(f"{label}.context.{group}", "missing"))
            elif isinstance(context[group], dict):
                for field in fields:
                    if field not in context[group]:
                        reasons.append(_reason(f"{label}.context.{group}.{field}", "missing"))
    reasons.extend(_differences(reference, candidate, "context"))
    return reasons


def compare_benchmarks(document: Any) -> dict[str, Any]:
    """Return explicit comparability and error reports, without fitting parameters."""
    _validate_json(document, "comparison")
    document = _object(document, "comparison")
    if set(document) != {"schema_version", "reference", "candidate"}:
        raise BenchmarkError("comparison must contain exactly schema_version, reference and candidate")
    if type(document["schema_version"]) is not int or document["schema_version"] != 1:
        raise BenchmarkError("schema_version must be integer 1")
    reference, candidate = (_record(document[label], label) for label in ("reference", "candidate"))
    if reference["id"] == candidate["id"]:
        raise BenchmarkError("reference.id and candidate.id must be distinct")
    context_reasons = _context_reasons(reference["context"], candidate["context"])
    for label, record in (("reference", reference), ("candidate", candidate)):
        if _unknown(record["source"]):
            context_reasons.append(_reason(f"{label}.source", "unknown", value=record["source"]))
    metrics = {}
    for name in sorted(set(reference["metrics"]) | set(candidate["metrics"])):
        reasons = list(context_reasons)
        left, right = (record["metrics"].get(name, _MISSING) for record in (reference, candidate))
        for label, metric in (("reference", left), ("candidate", right)):
            path = f"{label}.metrics.{name}"
            if metric is _MISSING:
                reasons.append(_reason(path, "missing"))
                continue
            for field in METRIC_FIELDS:
                if field not in metric:
                    reasons.append(_reason(f"{path}.{field}", "missing"))
                else:
                    reasons.extend(_unknown_paths(metric[field], f"{path}.{field}"))
        if left is not _MISSING and right is not _MISSING:
            for field in METRIC_FIELDS[1:]:
                reasons.extend(_differences(left.get(field, _MISSING), right.get(field, _MISSING), f"metrics.{name}.{field}"))
        result = {"status": "not_comparable" if reasons else "comparable", "reasons": reasons}
        if not reasons:
            actual, predicted = _number(left["value"], name), _number(right["value"], name)
            difference = predicted - actual
            relative = abs(difference) / abs(actual) if actual else None
            if not math.isfinite(difference) or (relative is not None and not math.isfinite(relative)):
                raise BenchmarkError(f"metrics.{name}: error exceeds finite numeric range")
            result.update({
                "reference_value": actual, "candidate_value": predicted,
                "definition": {field: left[field] for field in METRIC_FIELDS[1:]},
                "signed_difference": difference, "absolute_error": abs(difference),
                "relative_error": relative,
                "relative_error_note": "undefined for zero reference" if actual == 0 else "fraction of absolute reference value",
            })
        metrics[name] = result
    comparable = [name for name, result in metrics.items() if result["status"] == "comparable"]
    real_reference = reference["status"] in {"public", "measured"} and reference["evidence_kind"] == "measurement"
    simulation_candidate = candidate["status"] == "simulated" and candidate["evidence_kind"] == "simulation"
    measurement_candidate = candidate["status"] in {"public", "measured"} and candidate["evidence_kind"] == "measurement"
    eligible = real_reference and (simulation_candidate or measurement_candidate)
    return {
        "schema_version": 1,
        "status": ("comparable" if len(comparable) == len(metrics) else "partially_comparable") if comparable else "not_comparable",
        "reference": {key: reference[key] for key in ("id", "status", "evidence_kind", "source")},
        "candidate": {key: candidate[key] for key in ("id", "status", "evidence_kind", "source")},
        "context_reasons": context_reasons,
        "metric_counts": {"total": len(metrics), "comparable": len(comparable), "not_comparable": len(metrics) - len(comparable)},
        "metrics": metrics,
        "validation": {
            "reference_declared_measurement": real_reference,
            "candidate_declared_simulation": simulation_candidate,
            "candidate_declared_measurement": measurement_candidate,
            "real_evidence_eligible": bool(comparable) and eligible,
            "real_measurement_comparison_metrics": comparable if eligible else [],
            "validation_passed": None, "calibration_performed": False,
            "note": "Differences alone do not establish validation; no acceptance thresholds, uncertainty assessment or parameter fitting were applied. Provenance declarations are not independently verified.",
        },
    }


def _reject_constant(value: str) -> None:
    raise BenchmarkError(f"invalid nonfinite JSON number: {value}")


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise BenchmarkError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="JSON reference/candidate comparison bundle")
    parser.add_argument("-o", "--output", type=Path, help="write JSON instead of stdout")
    args = parser.parse_args(argv)
    try:
        if args.output and (args.output.resolve() == args.input.resolve() or (
            args.output.exists() and args.input.exists() and args.output.samefile(args.input)
        )):
            raise BenchmarkError("output must not overwrite the input comparison")
        document = json.loads(args.input.read_text(encoding="utf-8-sig"), parse_constant=_reject_constant, object_pairs_hook=_unique_keys)
        encoded = json.dumps(compare_benchmarks(document), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded, encoding="utf-8")
        else:
            print(encoded, end="")
    except (BenchmarkError, OSError, UnicodeError, ValueError, OverflowError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
