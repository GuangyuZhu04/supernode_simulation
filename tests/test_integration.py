"""Public API, CLI, configuration, search and artifact integration checks."""
import contextlib
import csv
import io
import json
from pathlib import Path
import shutil
import unittest
import uuid

import yaml

from supernode_sim.analysis import compile_graph, run
from supernode_sim.cli import main
from supernode_sim.config import Config, deep_merge, import_hf_config, load_config
from supernode_sim.report import write_report
from supernode_sim.search import size_su, sweep


@contextlib.contextmanager
def temporary_directory(prefix):
    # Python 3.14's mode-0700 mkdtemp DACL excludes the restricted Windows
    # runner token. Normal inherited permissions work and remain workspace-local.
    base = Path(__file__).resolve().parent / ".integration_tmp"
    base.mkdir(exist_ok=True)
    path = base / (prefix + uuid.uuid4().hex)
    path.mkdir()
    try:
        yield path
    finally:
        resolved = path.resolve()
        if resolved.parent != base.resolve() or not resolved.name.startswith(prefix):
            raise RuntimeError("Unsafe integration temporary directory cleanup")
        shutil.rmtree(resolved)
        try:
            base.rmdir()
        except OSError:
            pass


def raw_config(scenario="inference", patch=None):
    workload = ({"scenario": "inference", "batch_size": 2, "input_lengths": [3, 5],
                 "output_lengths": [2, 1]} if scenario == "inference" else
                {"scenario": "training", "batch_size": 2, "sequence_length": 4,
                 "gradient_accumulation": 2, "training_steps": 2})
    value = {
        "model": {"name": "synthetic-integration", "total_params": 1000, "layers": 2,
                  "hidden_dim": 8, "intermediate_dim": 32, "num_heads": 2,
                  "kv_heads": 1, "head_dim": 4},
        "hardware": {"npu_count": 1, "su_size": 1, "npus_per_host": 1,
                     "npu": {"hbm_bytes": 1000000, "runtime_bytes": 0, "workspace_bytes": 0}},
        "workload": workload,
    }
    return deep_merge(value, patch or {})


def specification(scenario="inference", patch=None):
    return Config.model_validate(raw_config(scenario, patch))


class ConfigIntegrationTests(unittest.TestCase):
    def test_yaml_and_json_resolve_to_identical_specs(self):
        with temporary_directory(prefix="snmodel_config_") as folder:
            directory = Path(folder)
            raw = raw_config()
            json_path, yaml_path = directory / "case.json", directory / "case.yaml"
            json_path.write_text(json.dumps(raw), encoding="utf-8")
            yaml_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
            self.assertEqual(load_config(json_path).model_dump(), load_config(yaml_path).model_dump())

    def test_unknown_fields_invalid_dtype_and_parallelism_are_rejected(self):
        patches = [
            {"unexpected": True}, {"model": {"total_paramz": 1000}},
            {"execution": {"precision": {"weight": "fp4"}}},
            {"execution": {"tp": 2}}, {"execution": {"rank_order": [8]}},
        ]
        for patch in patches:
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                specification(patch=patch)

    def test_hf_import_reports_missing_values_and_accepts_explicit_overrides(self):
        with temporary_directory(prefix="snmodel_hf_") as folder:
            path = Path(folder) / "config.json"
            path.write_text(json.dumps({"model_type": "synthetic", "num_hidden_layers": 2,
                "hidden_size": 64, "num_attention_heads": 4, "num_key_value_heads": 2}), encoding="utf-8")
            partial = import_hf_config(path)
            self.assertEqual(partial["model"]["head_dim"], 16)
            self.assertIn("total_params", partial["missing_fields"])
            self.assertIn("active_params", partial["missing_fields"])
            complete = import_hf_config(path, {"total_params": 1000, "active_params": 1000})
            self.assertEqual(complete["missing_fields"], [])
            self.assertIn("sources", complete)


class APIIntegrationTests(unittest.TestCase):
    def test_both_modes_share_config_and_conservative_time_bound(self):
        for scenario in ("training", "inference"):
            with self.subTest(scenario=scenario):
                spec = specification(scenario)
                estimate = run(spec, "estimate")
                simulated = run(spec, "simulate")
                self.assertEqual(estimate["status"], "estimated")
                self.assertEqual(simulated["status"], "feasible", simulated.get("simulation"))
                self.assertEqual(estimate["config_sha256"], simulated["config_sha256"])
                self.assertLessEqual(estimate["estimate"]["duration_lower_bound_s"],
                                     simulated["metrics"]["duration_s"] * (1 + 1e-9))
                self.assertEqual(estimate["metrics"], {})
                self.assertIsNone(estimate["slo"]["satisfied"])

    def test_training_steps_and_inference_request_metrics(self):
        training = run(specification("training"))
        metrics = training["metrics"]
        self.assertEqual(metrics["training_tokens"], 2 * 4 * 2 * 2)
        self.assertEqual(len(metrics["step_times_s"]), 2)
        self.assertAlmostEqual(sum(metrics["step_times_s"]), metrics["duration_s"])
        self.assertAlmostEqual(metrics["step_time_s"], metrics["duration_s"] / 2)
        inference = run(specification())
        metrics = inference["metrics"]
        self.assertEqual(metrics["input_tokens"], 8)
        self.assertEqual(metrics["output_tokens"], 3)
        self.assertEqual(len(metrics["requests"]), 2)
        self.assertGreater(metrics["ttft_s"], 0)
        self.assertGreater(metrics["tpot_s"], 0)
        one_token = next(row for row in metrics["requests"] if len(row["token_completion_s"]) == 1)
        self.assertIsNone(one_token["tpot_s"])

    def test_single_output_has_undefined_tpot_and_cannot_target_it(self):
        spec = specification(patch={"workload": {"output_lengths": [1, 1]}})
        metrics = run(spec)["metrics"]
        self.assertIsNone(metrics["tpot_s"])
        self.assertTrue(all(not row["itl_s"] for row in metrics["requests"]))
        with self.assertRaisesRegex(ValueError, "TPOT"):
            specification(patch={"workload": {"output_lengths": [1, 1]}, "targets": {"tpot_s": 1}})

    def test_capacity_and_slo_failure_are_distinct(self):
        impossible_memory = run(specification(patch={"hardware": {"npu": {"hbm_bytes": 100}}}))
        self.assertEqual(impossible_memory["status"], "infeasible")
        self.assertEqual(impossible_memory["metrics"], {})
        impossible_slo = run(specification(patch={"targets": {"ttft_s": 1e-30}}))
        self.assertEqual(impossible_slo["status"], "slo_failed")
        self.assertEqual(impossible_slo["simulation"]["status"], "feasible")
        self.assertFalse(impossible_slo["slo"]["satisfied"])


class SearchIntegrationTests(unittest.TestCase):
    def test_sweep_cartesian_grid_and_invalid_combinations_are_retained(self):
        spec = specification()
        result = sweep(spec, {"execution.precision.weight": ["fp8", "fp16"],
                              "workload.output_length": [1, 3]})
        self.assertEqual(len(result["candidates"]), 4)
        self.assertTrue(all(row["result"]["status"] == "feasible" for row in result["candidates"]))
        outputs = [row["result"]["metrics"]["output_tokens"] for row in result["candidates"]]
        self.assertEqual(outputs, [2, 6, 2, 6])
        invalid = sweep(spec, {"execution.tp": [1, 2]})
        self.assertEqual([row["result"]["status"] for row in invalid["candidates"]], ["feasible", "invalid_config"])
        with self.assertRaises(ValueError):
            sweep(spec, {"execution.tp": []})

    def search_spec(self, multi=False):
        target = {"candidate_npu_counts": [1, 2, 4], "candidate_su_sizes": [1, 2, 4],
                  "parallel_candidates": [{"tp": 1}, {"tp": 2}, {"tp": 4}]}
        if multi:
            target["total_npus"] = 4
        return specification(patch={"model": {"total_params": 1000000},
             "hardware": {"npu": {"hbm_bytes": 1200000}}, "targets": target})

    def test_single_su_minimum_checks_every_candidate_and_prunes_capacity(self):
        result = size_su(self.search_spec(), "single")["searches"]["single"]
        self.assertEqual(result["minimum_su_size"], 2)
        self.assertEqual(result["candidate_count"], 3)
        by_size = {row["su_size"]: row for row in result["candidates"]}
        self.assertEqual(by_size[1]["result"]["status"], "infeasible")
        self.assertEqual(by_size[1]["result"]["pruned"], "resident_memory_lower_bound")
        self.assertEqual(by_size[4]["result"]["status"], "feasible")
        for row in result["candidates"]:
            self.assertEqual(run(row["result"]["config"])["status"], row["result"]["status"])

    def test_multi_su_bound_and_minimum_over_all_declared_sizes(self):
        missing = size_su(self.search_spec(), "both")
        self.assertEqual(missing["status"], "missing_input")
        self.assertEqual(missing["searches"]["multi"]["status"], "missing_input")
        self.assertEqual(missing["searches"]["single"]["status"], "feasible")
        result = size_su(self.search_spec(multi=True), "multi")["searches"]["multi"]
        self.assertEqual(result["minimum_su_size"], 1)
        self.assertEqual(result["candidate_count"], 3)
        self.assertEqual({row["su_size"] for row in result["candidates"]}, {1, 2, 4})
        self.assertTrue(all(row["npu_count"] == 4 for row in result["candidates"]))
        for row in result["candidates"]:
            self.assertEqual(run(row["result"]["config"])["status"], row["result"]["status"])


class CalibrationIntegrationTests(unittest.TestCase):
    def calibration_case(self):
        spec = specification()
        graph = compile_graph(spec)
        event = next(e for e in graph.events if e.metadata.get("kernel") == "attention_dense")
        profile = {"flops": event.work["npu0.compute"], "memory_bytes": event.work["npu0.hbm"],
                   "duration_s": 0.01, "dtype": "fp16", "phase": "prefill"}
        data = spec.model_dump()
        data["hardware"]["kernel_profiles"] = {"attention_dense": [profile]}
        return event, data

    def test_exact_profile_matches_and_preserves_traffic(self):
        original, data = self.calibration_case()
        graph = compile_graph(data)
        matching = next(e for e in graph.events if e.id == original.id)
        self.assertEqual(matching.work["npu0.hbm"], original.work["npu0.hbm"])
        self.assertEqual(matching.metadata["calibration"]["match"], "exact_work")
        self.assertGreater(graph.metadata["kernel_calibration"]["matched_events"], 0)
        result = run(data)
        traced = next(row for row in result["simulation"]["trace"] if row["id"] == original.id)
        self.assertAlmostEqual(traced["duration_s"], 0.01)

    def test_mismatched_profile_uses_analytic_kernel(self):
        original, data = self.calibration_case()
        data["hardware"]["kernel_profiles"]["attention_dense"][0]["flops"] += 0.25
        graph = compile_graph(data)
        matching = next(e for e in graph.events if e.id == original.id)
        self.assertNotIn("calibration", matching.metadata)
        self.assertEqual(graph.metadata["kernel_calibration"]["matched_events"], 0)
        self.assertIn("attention_dense", graph.metadata["kernel_calibration"]["unmatched_kernels"])

    def test_impossible_measured_duration_is_rejected(self):
        _, data = self.calibration_case()
        data["hardware"]["kernel_profiles"]["attention_dense"][0]["duration_s"] = 1e-30
        with self.assertRaisesRegex(ValueError, "physical"):
            compile_graph(data)


class ArtifactAndCLIIntegrationTests(unittest.TestCase):
    def test_reports_include_replayable_config_csv_and_valid_timeline(self):
        result = run(specification())
        with temporary_directory(prefix="snmodel_report_") as folder:
            directory = write_report(result, folder)
            payload = json.loads((directory / "result.json").read_text(encoding="utf-8"))
            replay = run(load_config(directory / "config.resolved.json"))
            self.assertEqual(payload["config_sha256"], replay["config_sha256"])
            self.assertEqual(payload["metrics"], replay["metrics"])
            self.assertIn("超节点", (directory / "report.md").read_text(encoding="utf-8"))
            for filename in ("resources.csv", "memory.csv", "events.csv"):
                with (directory / filename).open(encoding="utf-8-sig", newline="") as handle:
                    self.assertTrue(list(csv.DictReader(handle)), filename)
            timeline = json.loads((directory / "timeline.json").read_text(encoding="utf-8"))
            self.assertEqual(len(timeline["traceEvents"]), result["simulation"]["completed_events"])
            self.assertTrue(all(row["ph"] == "X" and row["dur"] >= 0 for row in timeline["traceEvents"]))

    def invoke(self, argv):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            status = main(argv)
        return status, stdout.getvalue(), stderr.getvalue()

    def test_cli_success_invalid_infeasible_and_slo_exit_statuses(self):
        with temporary_directory(prefix="snmodel_cli_") as folder:
            directory = Path(folder)
            config_path = directory / "case.json"
            config_path.write_text(json.dumps(raw_config()), encoding="utf-8")
            code, stdout, _ = self.invoke(["validate", str(config_path)])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(stdout)["status"], "valid")
            code, stdout, _ = self.invoke(["simulate", str(config_path), "-o", str(directory / "out")])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(stdout)["status"], "feasible")
            for patch, expected_code, expected_status in [
                ({"hardware": {"npu": {"hbm_bytes": 100}}}, 3, "infeasible"),
                ({"targets": {"ttft_s": 1e-30}}, 3, "slo_failed"),
                ({"unknown": True}, 2, "invalid_input"),
            ]:
                config_path.write_text(json.dumps(raw_config(patch=patch)), encoding="utf-8")
                code, stdout, stderr = self.invoke(["simulate", str(config_path), "-o", str(directory / "out")])
                self.assertEqual(code, expected_code)
                self.assertEqual(json.loads(stderr or stdout)["status"], expected_status)

    def test_cli_missing_search_bound_and_hf_fields_exit_two(self):
        with temporary_directory(prefix="snmodel_cli_missing_") as folder:
            directory = Path(folder)
            path = directory / "case.json"
            path.write_text(json.dumps(raw_config()), encoding="utf-8")
            code, stdout, _ = self.invoke(["size-su", str(path), "--mode", "multi", "-o", str(directory / "out")])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(stdout)["status"], "missing_input")
            path.write_text(json.dumps({"hidden_size": 32}), encoding="utf-8")
            code, stdout, _ = self.invoke(["import-hf", str(path), "-o", str(directory / "import.json")])
            self.assertEqual(code, 2)
            self.assertTrue(json.loads(stdout)["missing_fields"])

    def test_malformed_yaml_is_a_cli_input_error(self):
        with temporary_directory(prefix="snmodel_cli_yaml_") as folder:
            path = Path(folder) / "broken.yaml"
            path.write_text("model: [", encoding="utf-8")
            code, _, stderr = self.invoke(["validate", str(path)])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(stderr)["status"], "invalid_input")


if __name__ == "__main__":
    unittest.main()
