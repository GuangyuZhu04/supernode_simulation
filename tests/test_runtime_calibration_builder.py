"""Generic Ascend-context response curves keep training and validation separate."""

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "skills/supernode-modeling/scripts/build_runtime_calibration.py"
ASSET = SCRIPT.parents[1] / "assets/runtime-calibration.synthetic.json"
SPEC = importlib.util.spec_from_file_location("build_runtime_calibration", SCRIPT)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def dataset():
    return json.loads(ASSET.read_text(encoding="utf-8"))


def test_replicates_use_median_and_range_without_changing_input():
    document = dataset()
    original = copy.deepcopy(document)
    report = builder.build_calibration(document)
    assert document == original
    assert report["status"] == "built"
    assert not report["real_hardware_accuracy_claimed"]
    family = next(row for row in report["families"] if row["environment_id"] == "synthetic-a2")
    assert len(family["training_samples"]) == 3
    middle = family["training_samples"][1]
    assert middle["sample_count"] == 3
    assert middle["duration_s"] == pytest.approx(47e-6)
    assert middle["min_duration_s"] == pytest.approx(45e-6)
    assert middle["max_duration_s"] == pytest.approx(49e-6)
    assert not set(family["training_observation_ids"]) & set(family["validation_observation_ids"])
    assert family["provenance"]["evidence_kinds"] == ["synthetic"]


def test_validation_values_never_change_training_anchors_or_interval():
    original = dataset()
    expected = builder.build_calibration(original)
    changed = copy.deepcopy(original)
    for row in changed["observations"]:
        if row["split"] == "validation":
            row["duration_s"] = 12345
    result = builder.build_calibration(changed)
    assert result["hardware_by_environment"] == expected["hardware_by_environment"]
    assert result["families"] == expected["families"]


def test_actual_engine_interpolates_inside_range_and_falls_back_outside():
    report = builder.build_calibration(dataset())
    for environment, coverage in report["validation_coverage"].items():
        assert coverage["validation_count"] == 3
        assert coverage["in_range_applied_count"] == 2
        assert coverage["fallback_count"] == 1
        assert coverage["range_coverage_fraction"] == pytest.approx(2 / 3)
        validation = [row for row in report["observations"] if row["environment_id"] == environment and row["split"] == "validation"]
        for row in validation:
            if row["model_applied"]:
                assert row["runtime_record"]["method"] == "interpolated"
                assert row["calibrated_duration_s"] == pytest.approx(max(row["baseline_duration_s"], row["runtime_record"]["service_duration_s"]))
            else:
                assert row["fallback_reason"] == "out_of_range"
                assert row["runtime_record"]["status"] == "out_of_range"
                assert row["calibrated_duration_s"] == row["baseline_duration_s"]
                assert row["calibrated_absolute_relative_error"] > .1


def test_only_software_context_differs_but_models_still_remain_separate():
    document = dataset()
    document["environments"][1]["hardware"] = copy.deepcopy(document["environments"][0]["hardware"])
    second_context = copy.deepcopy(document["environments"][0]["calibration_context"])
    second_context["runtime_version"] = "another-explicit-runtime-version"
    document["environments"][1]["calibration_context"] = second_context
    sample = document["observations"][0]
    for row in document["observations"]:
        if row["environment_id"] == "synthetic-a3":
            for field in (*builder.SELECTORS, "ranks", "precision", "metadata_match"):
                row[field] = copy.deepcopy(sample[field])
    report = builder.build_calibration(document)
    models = [hardware["collective_models"] for hardware in report["hardware_by_environment"].values()]
    assert all(len(items) == 1 for items in models)
    assert models[0][0]["family_signature"] != models[1][0]["family_signature"]
    assert models[0][0]["samples"] != models[1][0]["samples"]


def test_environment_fragments_replay_after_complete_config_normalization():
    from supernode_sim.config import Config
    from tests.test_moe_wire_precision import configuration

    document = dataset()
    report = builder.build_calibration(document)
    observations = builder._observations(document["observations"], report["hardware_by_environment"])
    for identity, hardware in report["hardware_by_environment"].items():
        full = configuration().model_dump()
        full["hardware"] = hardware
        full["execution"]["ep"] = hardware["npu_count"]
        normalized = Config.model_validate(full).model_dump()["hardware"]
        row = next(row for row in observations if row["environment_id"] == identity and row["split"] == "validation")
        _, record = builder._simulate(normalized, row)
        assert record["status"] == "matched"
        assert record["method"] == "interpolated"
        assert record["model_id"] == hardware["collective_models"][0]["id"]


def test_different_routing_distribution_is_not_covered_by_same_total_bytes():
    document = dataset()
    row = next(row for row in document["observations"] if row["split"] == "validation")
    row["payload_matrix"] = [list(values) for values in zip(*row["payload_matrix"])]
    report = builder.build_calibration(document)
    result = next(item for item in report["observations"] if item["id"] == row["id"])
    assert not result["model_applied"]
    assert result["fallback_reason"] == "no_trained_family"
    assert result["calibrated_duration_s"] == result["baseline_duration_s"]


def test_one_unique_training_payload_reports_insufficient_data_without_fitting():
    document = dataset()
    document["observations"] = [row for row in document["observations"] if row["split"] == "validation" or row["id"] in {"a2-train-1", "a3-train-1"}]
    report = builder.build_calibration(document)
    assert report["status"] == "insufficient_training_data"
    assert all(not hardware["collective_models"] for hardware in report["hardware_by_environment"].values())
    assert all(not row["model_applied"] for row in report["observations"])


@pytest.mark.parametrize("collective", ["all_gather", "reduce_scatter", "all_reduce"])
def test_ring_collectives_use_real_engine_payload_and_interval_semantics(collective):
    document = dataset()
    document["environments"] = document["environments"][:1]
    document["observations"] = [row for row in document["observations"] if row["environment_id"] == "synthetic-a2"]
    for row in document["observations"]:
        row.update(collective=collective, semantic="tensor-sync", algorithm="ring", size_bytes=row["payload_matrix"][0][0] * 10)
        del row["payload_matrix"]
    report = builder.build_calibration(document)
    assert report["validation_coverage"]["synthetic-a2"]["in_range_applied_count"] == 2
    assert report["validation_coverage"]["synthetic-a2"]["fallback_count"] == 1
    for row in report["observations"]:
        if row["split"] == "train":
            assert row["runtime_record"]["method"] == "exact_model"
            assert row["runtime_record"]["signature"]["total_payload_bytes"] == row["payload_bytes"]


def test_operator_shape_metadata_change_requires_a_new_training_family():
    document = dataset()
    row = next(row for row in document["observations"] if row["environment_id"] == "synthetic-a3" and row["split"] == "validation")
    row["metadata_match"]["moe_hidden_width"] = 4096
    row["metadata_match"]["moe_tensor_bytes_per_token"] = 8192
    report = builder.build_calibration(document)
    result = next(item for item in report["observations"] if item["id"] == row["id"])
    assert not result["model_applied"]
    assert result["fallback_reason"] == "no_trained_family"
    assert result["runtime_record"]["status"] in {"family_mismatch", "metadata_mismatch"}


def test_operator_tags_cannot_override_reserved_collective_fields():
    document = dataset()
    document["observations"][0]["metadata_match"] = {"phase": "different-phase"}
    with pytest.raises(ValueError, match="reserved collective metadata"):
        builder.build_calibration(document)


def test_whole_moe_calls_require_shape_and_explicit_timer_boundary():
    document = dataset()
    row = next(row for row in document["observations"] if row["environment_id"] == "synthetic-a3")
    del row["metadata_match"]["moe_top_k"]
    with pytest.raises(ValueError, match="known MoE shape fields"):
        builder.build_calibration(document)
    document = dataset()
    document["environments"][0]["calibration_context"]["timer_boundary"] = "unspecified-timer"
    with pytest.raises(ValueError, match="timer_boundary"):
        builder.build_calibration(document)


@pytest.mark.parametrize("field,value", [
    ("duration_s", 0), ("duration_s", -1), ("duration_s", True),
    ("duration_s", float("nan")), ("duration_s", float("inf")),
    ("ranks", ["npu0", "npu0"]), ("ranks", ["npu0", "npu99"]),
    ("payload_matrix", [[0, True], [1, 0]]), ("kind", "kernel"),
    ("split", "auto"), ("environment_id", "missing"), ("source", "unknown"),
    ("evidence_kind", "measurement"), ("size_bytes", 123),
])
def test_invalid_observations_are_rejected(field, value):
    document = dataset()
    document["observations"][0][field] = value
    with pytest.raises((builder.CalibrationBuildError, ValueError)):
        builder.build_calibration(document)


def test_unknown_context_duplicate_ids_and_existing_calibrations_are_rejected():
    document = dataset()
    document["environments"][0]["calibration_context"]["hccl_version"] = "unknown"
    with pytest.raises(ValueError, match="hccl_version"):
        builder.build_calibration(document)
    document = dataset()
    document["observations"][1]["id"] = document["observations"][0]["id"]
    with pytest.raises(ValueError, match="duplicate observation"):
        builder.build_calibration(document)
    document = dataset()
    document["environments"].append(copy.deepcopy(document["environments"][0]))
    with pytest.raises(ValueError, match="duplicate environment"):
        builder.build_calibration(document)
    document = dataset()
    document["environments"][0]["hardware"]["collective_models"] = [{"id": "existing"}]
    with pytest.raises(ValueError, match="must not be overwritten"):
        builder.build_calibration(document)


def test_environment_ids_cannot_collide_on_case_insensitive_filesystems():
    document = dataset()
    duplicate = copy.deepcopy(document["environments"][0])
    duplicate["id"] = duplicate["id"].upper()
    document["environments"].append(duplicate)
    with pytest.raises(ValueError, match="case-insensitive artifact name"):
        builder.build_calibration(document)


def test_estimates_and_synthetic_training_points_cannot_be_blended():
    document = dataset()
    document["observations"][0].update(status="public", evidence_kind="estimate")
    with pytest.raises(ValueError, match="evidence kinds must not be mixed"):
        builder.build_calibration(document)


def test_cli_writes_separate_validated_environment_artifacts(tmp_path):
    output = tmp_path / "artifacts"
    run = subprocess.run([sys.executable, str(SCRIPT), str(ASSET), "--output", str(output)], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert len(report["hardware_by_environment"]) == 2
    from supernode_sim.config import HardwareSpec
    for identity in report["hardware_by_environment"]:
        artifact = json.loads((output / f"environment.{identity}.json").read_text(encoding="utf-8"))
        assert len(HardwareSpec.model_validate(artifact["hardware"]).collective_models) == 1


@pytest.mark.parametrize("invalid", ['{"schema_version": NaN}', '{"schema_version":1,"schema_version":1}', '{broken'])
def test_cli_rejects_nonfinite_duplicate_keys_and_invalid_json(tmp_path, invalid):
    source = tmp_path / "input.json"
    source.write_text(invalid, encoding="utf-8")
    run = subprocess.run([sys.executable, str(SCRIPT), str(source), "--output", str(tmp_path / "output")], capture_output=True, text=True)
    assert run.returncode == 2
    assert not run.stdout
    assert "error:" in run.stderr


def test_cli_never_overwrites_input_even_when_named_report_json(tmp_path):
    source = tmp_path / "report.json"
    original = json.dumps(dataset())
    source.write_text(original, encoding="utf-8")
    for output in (source, tmp_path):
        run = subprocess.run([sys.executable, str(SCRIPT), str(source), "--output", str(output)], capture_output=True, text=True)
        assert run.returncode == 2
        assert "must not overwrite the input" in run.stderr
        assert source.read_text(encoding="utf-8") == original
