"""Independent arithmetic checks for compressed sparse attention boundaries."""
import unittest

from supernode_sim.components import (attention_state, attention_training_state,
    component_activations, inventory_summary, parameter_inventory)
from supernode_sim.sparse_attention import sparse_attention_state, sparse_attention_work


def sparse_model(kind="CSA"):
    spec = {"q_lora_rank": 4, "head_dim": 8, "rope_head_dim": 2,
            "o_groups": 2, "o_lora_rank": 3, "window_size": 5,
            "compress_ratio": 4 if kind == "CSA" else 8}
    if kind == "CSA":
        spec.update(index_num_heads=4, index_head_dim=4, index_topk=2)
    return {"name": "synthetic-sparse", "layers": 1, "hidden_dim": 16,
            "num_heads": 4, "head_dim": 8, "kv_heads": 2,
            "architecture": {"vocab_size": 32, "attention_by_layer": [kind],
                             "ffn_by_layer": ["Dense"], "dense_intermediate_dim": 32,
                             kind.lower(): spec}}


class SparseAttentionTests(unittest.TestCase):
    def test_parameters_are_shapes_not_divided_by_compression_ratio(self):
        for kind in ("CSA", "HCA"):
            model = sparse_model(kind)
            spec = model["architecture"][kind.lower()]
            # Shared latent Q/KV, grouped low-rank output, sink.
            base = 16*4 + 4 + 4*4*8 + 16*8 + 8 + 4*8*3 + 2*3*16 + 4
            c, r = (2 if kind == "CSA" else 1), spec["compress_ratio"]
            main_compressor = 2*16*c*8 + r*c*8 + 8
            indexer = 4*4*4 + 16*4 + 2*16*c*4 + r*c*4 + 4 if kind == "CSA" else 0
            self.assertEqual(inventory_summary(model)["components"][kind]["params"], base + main_compressor + indexer)
            tensors = {t.name: t for t in parameter_inventory(model)}
            self.assertEqual(tensors[f"layers.0.{kind.lower()}.output_down.weight"].shape, (2, 3, 16))
            compressor = tensors[f"layers.0.{kind.lower()}.compressor.kv.weight"]
            self.assertEqual(compressor.local_elements(tp=2), compressor.elements)

    def test_cache_pool_floor_and_overlap_state_are_explicit(self):
        model = sparse_model()
        before = sparse_attention_state(model, 0, [3, 7])
        after = sparse_attention_state(model, 0, [4, 8])
        self.assertEqual(after["compressed_rows"] - before["compressed_rows"], 2)
        self.assertEqual(after["cache_elements"] - before["cache_elements"], 2*(8+4))
        self.assertEqual(after["compression_state_elements"], 2*2*2*2*4*8)
        self.assertEqual(after["indexer_compression_state_elements"], 2*2*2*2*4*4)
        self.assertEqual(before["recurrent_elements"], after["recurrent_elements"])
        self.assertEqual(after["window_elements"], 2*5*8)
        self.assertEqual(after, sparse_attention_state(model, 0, [4, 8], tp=2))
        uniform = attention_state(model, 0, 2, 8)
        self.assertEqual(uniform, sparse_attention_state(model, 0, [8, 8]))

    def test_prefill_and_decode_causal_pairs_match_explicit_enumeration(self):
        for kind in ("CSA", "HCA"):
            model = sparse_model(kind)
            spec = model["architecture"][kind.lower()]
            for length in (1, 3, 4, 7, 8, 9, 17, 100):
                work = sparse_attention_work(model, 0, [length])
                expected_window = sum(min(t, 5) for t in range(1, length+1))
                expected_pool = sum(min(t//spec["compress_ratio"], 2) if kind == "CSA" else t//8
                                    for t in range(1, length+1))
                self.assertEqual(work["window_pairs"], expected_window)
                self.assertEqual(work["compressed_pairs"], expected_pool)
                decode = sparse_attention_work(model, 0, [length], phase="decode")
                self.assertEqual(decode["compressed_rows_emitted"], int(length % spec["compress_ratio"] == 0))
                self.assertEqual(decode["window_pairs"], min(length, 5))

    def test_indexer_scans_full_pool_even_when_topk_saturates(self):
        model = sparse_model()
        a = sparse_attention_work(model, 0, [16], phase="decode")
        b = sparse_attention_work(model, 0, [32], phase="decode")
        self.assertEqual(a["compressed_pairs"], b["compressed_pairs"])
        self.assertEqual(b["indexer_pairs"], 2*a["indexer_pairs"])
        self.assertGreater(b["extra_flops"], a["extra_flops"])
        prefill = sparse_attention_work(model, 0, [8])
        self.assertEqual(prefill["indexer_pairs"], 8*2)
        parallel = sparse_attention_work(model, 0, [8], tp=2)
        self.assertEqual(parallel["indexer_reduce_elements"], 16)
        self.assertEqual(parallel["indexer_score_workspace_elements"], 16*(2+1))

    def test_bulk_prefill_only_retains_window_and_compression_tail(self):
        model = sparse_model()
        work = sparse_attention_work(model, 0, [1024])
        self.assertEqual(work["state_read_elements"], 0)
        self.assertEqual(work["state_write_elements"], 4*2*2*(8+4))
        self.assertEqual(work["cache_write_elements"], 5*8 + 256*(8+4))
        self.assertGreater(work["compression_read_elements"], work["state_write_elements"])

    def test_verify_counts_queries_only_and_preserves_ragged_lengths(self):
        model = sparse_model()
        lengths, queries = [11, 18], [3, 2]
        result = sparse_attention_work(model, 0, lengths, phase="verify", query_lengths=queries)
        individual = [sparse_attention_work(model, 0, [t], phase="decode")
                      for length, q in zip(lengths, queries) for t in range(length-q+1, length+1)]
        for field in ("window_pairs", "compressed_pairs", "indexer_pairs", "compressed_rows_emitted", "extra_flops"):
            self.assertEqual(result[field], sum(item[field] for item in individual))
        self.assertEqual(result["query_tokens"], 5)
        with self.assertRaises(ValueError):
            sparse_attention_work(model, 0, [8], phase="verify")
        with self.assertRaises(ValueError):
            sparse_attention_work(model, 0, [8], query_lengths=[9])

    def test_training_storage_keeps_compression_and_selection(self):
        model = sparse_model()
        saved = component_activations(model, 0, 16, training=True)
        gate = next(row for row in saved if row["name"] == "compression_values_and_gates")
        self.assertEqual(gate["dtype"], "fp32")
        state = attention_training_state(model, 0, 2, 8)
        self.assertEqual(state["compressed_pool_elements"], 2*2*(8+4))
        self.assertGreater(state["selection_index_elements"], 0)
        checkpointed = component_activations(model, 0, 16, training=True, checkpointing=True)
        self.assertEqual([row["name"] for row in checkpointed], ["checkpoint_input"])


if __name__ == "__main__":
    unittest.main()
