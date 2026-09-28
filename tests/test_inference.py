"""Structural and behavioral checks for the inference execution ledger."""
import copy
import unittest

from supernode_sim.config import Config, deep_merge
from supernode_sim.engine import simulate
from supernode_sim.inference import build_inference, kv_layout_bytes
from supernode_sim.topology import Topology


def configuration(patch=None):
    raw = {
        "model": {"name": "synthetic-test", "total_params": 1000000, "layers": 2,
                  "hidden_dim": 64, "num_heads": 4, "kv_heads": 2, "head_dim": 16},
        "hardware": {"npu_count": 1, "su_size": 1, "npus_per_host": 1,
                     "npu": {"runtime_bytes": 0, "workspace_bytes": 0}},
        "workload": {"scenario": "inference", "batch_size": 2,
                     "input_lengths": [16, 5], "output_lengths": [3, 1]},
    }
    return Config.model_validate(deep_merge(raw, patch or {})).model_dump()


def compiled(patch=None):
    c = configuration(patch)
    return c, build_inference(c, Topology(c["hardware"], c["execution"]["precision"]))


class KVLayoutTests(unittest.TestCase):
    def test_standard_page_padding_and_quantization_are_separate(self):
        m = configuration()["model"]
        sizes = kv_layout_bytes(m, "fp8", [17, 5], layers=2,
                                page_tokens=16, scale_block=128, scale_bytes=4)
        self.assertEqual(sizes["data_bytes"], 3 * 16 * 2 * 16 * 2 * 2)
        self.assertEqual(sizes["scale_bytes"], 3 * 2 * 8 * 4)
        self.assertEqual(sizes["page_table_bytes"], 3 * 2 * 8)
        self.assertEqual(sizes["total_bytes"], sum(sizes[k] for k in
                          ("data_bytes", "scale_bytes", "page_table_bytes")))

    def test_latent_stores_latent_and_rope_once_per_rank(self):
        m = configuration()["model"]
        m.update(attention_kind="latent", kv_latent_dim=32, kv_rope_dim=8)
        one = kv_layout_bytes(m, "fp16", [16], tp=1)
        four = kv_layout_bytes(m, "fp16", [16], tp=4)
        self.assertEqual(one["data_bytes"], 16 * 40 * 2)
        self.assertEqual(one, four)

    def test_kv_heads_replicated_if_tp_exceeds_head_count(self):
        m = configuration()["model"]
        self.assertEqual(kv_layout_bytes(m, "fp16", [16], tp=8)["elements_per_token"], 32)


class InferenceGraphTests(unittest.TestCase):
    def test_ragged_batch_token_completions_and_dynamic_cache(self):
        c, graph = compiled()
        result = simulate(graph)
        self.assertEqual(result["status"], "feasible", result["diagnostics"])
        metadata = graph.metadata
        self.assertEqual(metadata["input_tokens"], 21)
        self.assertEqual(metadata["output_tokens"], 4)
        self.assertEqual(len(metadata["token_completion_events"]["0"]), 3)
        self.assertEqual(len(metadata["sequence_token_completion_events"]["0"]["1"]), 1)
        expected = kv_layout_bytes(c["model"], "fp16", [18, 5], layers=2)["total_bytes"]
        actual = sum(a.size_bytes for e in graph.events for a in e.allocs if a.category == "kv")
        self.assertEqual(actual, expected)
        resident = sum(a.size_bytes for a in graph.initial_allocations)
        self.assertAlmostEqual(result["memory_final_bytes"]["npu0.hbm"], resident + expected)

    def test_weight_precision_independent_of_compute_and_kv(self):
        _, fp16 = compiled()
        _, fp8 = compiled({"execution": {"precision": {"weight": "fp8"}}})
        self.assertEqual(fp8.metadata["logical_memory_bytes"]["weight"] * 2,
                         fp16.metadata["logical_memory_bytes"]["weight"])
        self.assertEqual(fp8.metadata["logical_memory_bytes"]["kv_data"],
                         fp16.metadata["logical_memory_bytes"]["kv_data"])
        self.assertTrue(any(e.metadata.get("operation") == "precision_conversion" for e in fp8.events))
        self.assertTrue(any(a.category == "conversion_buffer" for e in fp8.events for a in e.allocs))
        result = simulate(fp8)
        self.assertEqual(result["status"], "feasible", result["diagnostics"])

    def test_extra_compute_copy_avoids_per_layer_conversion(self):
        _, graph = compiled({"execution": {"precision": {"weight": "fp8"},
                             "features": {"extra_weight_copy_dtype": "fp16"}}})
        self.assertFalse(any(e.metadata.get("operation") == "precision_conversion" for e in graph.events))
        self.assertTrue(any(a.category == "extra_weight_copy" for a in graph.initial_allocations))

    def test_kv_offload_and_engram_share_real_resources_and_overlap(self):
        _, graph = compiled({"execution": {"features": {
            "kv_offload": "host", "engram": {"enabled": True, "table_bytes": 10000,
                "queries_per_token": 1, "record_bytes": 2048, "latency_s": 0}}}})
        kv = next(e for e in graph.events if e.id == "infer.d0.s1.l0.npu0.kv_read")
        engram = next(e for e in graph.events if e.id == "infer.d0.s1.l0.npu0.engram_read")
        common = set(kv.work) & set(engram.work)
        self.assertIn("cpu0.dram", common)
        self.assertIn("npu0.hbm", common)
        self.assertTrue(any(k.startswith("link:") for k in common), common)
        result = simulate(graph)
        self.assertEqual(result["status"], "feasible", result["diagnostics"])
        trace = {row["id"]: row for row in result["trace"]}
        self.assertLess(max(trace[kv.id]["start_s"], trace[engram.id]["start_s"]),
                        min(trace[kv.id]["end_s"], trace[engram.id]["end_s"]))
        self.assertGreater(result["memory_final_bytes"]["cpu0.dram"], 10000)

    def test_moe_dispatch_combine_are_one_set_of_all_to_all_events(self):
        _, graph = compiled({
            "model": {"num_experts": 4, "top_k": 2, "moe_layer_fraction": 1,
                      "expert_parameter_fraction": 0.8},
            "hardware": {"npu_count": 2, "su_size": 2}, "execution": {"ep": 2},
        })
        dispatch = [e for e in graph.events if e.metadata.get("operation") == "dispatch"]
        combine = [e for e in graph.events if e.metadata.get("operation") == "combine"]
        self.assertTrue(dispatch)
        self.assertTrue(all(e.metadata["collective"] == "all_to_all_v" for e in dispatch + combine))
        self.assertEqual(sum(e.metadata["payload_bytes"] for e in dispatch),
                         sum(e.metadata["payload_bytes"] for e in combine))
        self.assertTrue(all(e.metadata["physical_bytes_by_domain"].get("SO", 0) == 0 for e in dispatch + combine))
        result = simulate(graph)
        self.assertEqual(result["status"], "feasible", result["diagnostics"])

    def test_pipeline_and_tp_can_run_without_buffer_leaks(self):
        _, graph = compiled({"hardware": {"npu_count": 4, "su_size": 4},
                             "execution": {"tp": 2, "pp": 2, "rank_order": [3, 1, 2, 0]}})
        result = simulate(graph)
        self.assertEqual(result["status"], "feasible", result["diagnostics"])
        self.assertTrue(any(e.metadata.get("operation") == "pipeline" for e in graph.events))
        final_total = sum(result["memory_final_bytes"].values())
        expected = sum(a.size_bytes for a in graph.initial_allocations)
        expected += sum(a.size_bytes for e in graph.events for a in e.allocs if a.category == "kv")
        self.assertAlmostEqual(final_total, expected)

    def test_small_decode_batch_reads_only_expected_touched_experts(self):
        c, graph = compiled({
            "model": {"num_experts": 16, "top_k": 2, "moe_layer_fraction": 1,
                      "expert_parameter_fraction": 0.8},
            "workload": {"batch_size": 1, "input_lengths": [16], "output_lengths": [2]},
        })
        event = next(e for e in graph.events if e.id == "infer.d0.s1.l0.npu0.experts")
        expected = c["model"]["total_params"] * 0.8 / 2 * (2 / 16) * 2
        self.assertAlmostEqual(event.work["npu0.hbm"], expected)

    def test_wire_precision_adds_conversions_and_releases_staging(self):
        _, graph = compiled({"hardware": {"npu_count": 4, "su_size": 4},
            "execution": {"tp": 2, "pp": 2, "precision": {"communication": "fp32"}}})
        conversions = {e.metadata.get("conversion") for e in graph.events}
        self.assertTrue({"communication_pack", "communication_unpack", "pipeline_pack", "pipeline_unpack"} <= conversions)
        result = simulate(graph)
        self.assertEqual(result["status"], "feasible", result["diagnostics"])
        expected = sum(a.size_bytes for a in graph.initial_allocations)
        expected += sum(a.size_bytes for e in graph.events for a in e.allocs if a.category == "kv")
        self.assertAlmostEqual(sum(result["memory_final_bytes"].values()), expected)

    def test_fp8_kv_conversion_has_explicit_cost_without_full_decompression(self):
        _, graph = compiled({"execution": {"precision": {"kv": "fp8"},
            "features": {"kv_offload": "host"}}})
        conversions = [e for e in graph.events if e.metadata.get("kernel") == "kv_conversion"]
        self.assertTrue(conversions)
        self.assertTrue(all(e.work["npu0.compute"] > 0 for e in conversions))
        result = simulate(graph)
        self.assertEqual(result["status"], "feasible", result["diagnostics"])
        expected = sum(a.size_bytes for a in graph.initial_allocations)
        expected += sum(a.size_bytes for e in graph.events for a in e.allocs if a.category == "kv")
        self.assertAlmostEqual(sum(result["memory_final_bytes"].values()), expected)

    def test_remote_features_and_reproducibility(self):
        _, graph = compiled({"hardware": {"remote": {"enabled": True}},
            "execution": {"features": {"kv_offload": "remote", "engram": {
                "enabled": True, "location": "remote", "table_bytes": 10000}}}})
        first = simulate(graph)
        second = simulate(copy.deepcopy(graph))
        self.assertEqual(first["status"], "feasible", first["diagnostics"])
        self.assertEqual(first, second)
        self.assertGreater(first["memory_final_bytes"]["remote0.memory"], 10000)


if __name__ == "__main__":
    unittest.main()
