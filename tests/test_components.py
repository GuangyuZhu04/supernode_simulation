"""Conservation, shape arithmetic, optimizer grouping and hybrid state checks."""
import unittest

from supernode_sim.components import (attention_state, attention_training_state, attnres_state,
    component_activations, inventory_summary, optimizer_states,
    optimizer_state_inventory, parameter_inventory)


def hybrid_model():
    return {"name": "synthetic-hybrid", "layers": 2, "hidden_dim": 16,
            "num_heads": 4, "head_dim": 4, "kv_heads": 2,
            "num_experts": 4, "top_k": 2,
            "architecture": {"vocab_size": 32, "tie_embeddings": False,
                "attention_by_layer": ["MLA", "KDA"], "ffn_by_layer": ["Dense", "MoE"],
                "dense_intermediate_dim": 32,
                "mla": {"q_lora_rank": 4, "kv_lora_rank": 4,
                        "qk_nope_head_dim": 2, "qk_rope_head_dim": 2, "v_head_dim": 4},
                "kda": {"num_heads": 4, "head_k_dim": 4, "head_v_dim": 4, "conv_kernel": 4},
                "moe": {"num_experts": 4, "top_k": 2, "intermediate_dim": 8,
                        "num_shared_experts": 1}}}


class InventoryTests(unittest.TestCase):
    def test_mla_shapes_and_all_requested_components(self):
        model = hybrid_model()
        summary = inventory_summary(model)
        self.assertEqual(set(summary["components"]), {
            "WordEmbedding", "PreNorm", "PostNorm", "MLA", "KDA", "Dense", "MoE", "Output"})
        self.assertEqual(summary["components"]["MLA"]["params"],
                         16*4 + 4 + 4*4*4 + 16*6 + 4 + 4*4*6 + 16*16)
        self.assertEqual(summary["components"]["Dense"]["params"], 3*16*32)
        self.assertEqual(summary["components"]["MoE"]["params"], 16*4 + 3*4*16*8 + 3*16*8)
        self.assertEqual(summary["total_params"], sum(c["params"] for c in summary["components"].values()))
        self.assertEqual(summary["total_params"] - summary["active_params"], 3*2*16*8 + (32-1)*16)

    def test_tied_weight_alias_counted_once(self):
        model = hybrid_model()
        untied = inventory_summary(model)
        model["architecture"]["tie_embeddings"] = True
        tied = inventory_summary(model)
        self.assertEqual(untied["total_params"] - tied["total_params"], 32*16)
        self.assertEqual(untied["active_params"] - tied["active_params"], 16)
        self.assertEqual(tied["components"]["Output"]["params"], 0)
        alias = parameter_inventory(model)[-1]
        self.assertEqual(alias.elements, 32*16)
        self.assertEqual(alias.accounted_elements, 0)
        self.assertEqual(optimizer_states(alias), [])

    def test_latent_moe_uses_latent_expert_shape_and_hidden_shared_experts(self):
        model = hybrid_model()
        model["architecture"]["moe"].update(latent_dim=8, latent_norm=True)
        tensors = {t.name: t for t in parameter_inventory(model)}
        self.assertEqual(tensors["layers.1.moe.routed.gate.weight"].shape, (4, 8, 8))
        self.assertEqual(tensors["layers.1.moe.shared.gate.weight"].shape, (1, 8, 16))
        self.assertEqual(tensors["layers.1.moe.latent_down.weight"].elements, 16*8)
        self.assertEqual(tensors["layers.1.moe.latent_norm.weight"].elements, 8)

    def test_source_declared_gating_attnres_and_router_tensors(self):
        model = hybrid_model()
        base = inventory_summary(model)["total_params"]
        model["architecture"]["mla"]["use_output_gate"] = True
        model["architecture"]["moe"]["router_bias"] = True
        model["architecture"]["attnres_block_size"] = 1
        summary = inventory_summary(model)
        self.assertEqual(summary["components"]["AttnRes"]["params"], 2*4*16 + 2*16)
        self.assertEqual(summary["total_params"] - base, 16*4*4 + 4 + 2*4*16 + 2*16)
        self.assertEqual(attnres_state(model, 1, 10)["stored_elements"], 10*16*2)
        self.assertEqual(attnres_state(model, 1, 10)["new_elements"], 10*16)
        rows = component_activations(model, 1, 10)
        scratch = next(r for r in rows if r["component"] == "AttnRes")
        self.assertEqual(scratch["dtype"], "fp32")
        self.assertEqual(scratch["lifetime"], "operator_working_set")

    def test_sharding_respects_norm_latent_and_expert_replication(self):
        tensors = {t.name: t for t in parameter_inventory(hybrid_model())}
        self.assertEqual(tensors["layers.0.pre_norm.weight"].local_elements(tp=4, ep=4), 16)
        self.assertEqual(tensors["layers.0.mla.kv_down.weight"].local_elements(tp=4, ep=4), 96)
        self.assertEqual(tensors["layers.1.moe.routed.gate.weight"].local_elements(tp=2, ep=4, fsdp=2), 32)
        self.assertEqual(tensors["layers.1.moe.router.weight"].local_elements(tp=2, ep=4), 64)

    def test_budget_mode_never_fabricates_architecture(self):
        summary = inventory_summary({"total_params": 1e13, "active_params": 4e11})
        self.assertEqual(summary["mode"], "parameter_budget")
        self.assertEqual(set(summary["components"]), {"Unassigned"})
        self.assertEqual(summary["total_params"], 1e13)
        self.assertEqual(summary["active_params"], 4e11)
        with self.assertRaisesRegex(ValueError, "grouping requires"):
            optimizer_states(parameter_inventory({"total_params": 100})[0], "muon")

    def test_muon_has_one_momentum_and_adam_fallback_two_states(self):
        tensors = {t.name: t for t in parameter_inventory(hybrid_model())}
        self.assertEqual(optimizer_states(tensors["layers.0.dense.up.weight"], "muon"), ["master_weight", "muon_momentum"])
        self.assertEqual(optimizer_states(tensors["layers.1.moe.routed.gate.weight"], "muon"), ["master_weight", "muon_momentum"])
        for name in ("word_embeddings.weight", "output.weight", "layers.0.pre_norm.weight"):
            self.assertEqual(optimizer_states(tensors[name], "muon"), ["master_weight", "optimizer_momentum", "optimizer_variance"])
        self.assertEqual(optimizer_states(tensors["layers.0.dense.up.weight"], "adam", False), ["optimizer_momentum", "optimizer_variance"])
        states = optimizer_state_inventory(tensors["layers.0.dense.up.weight"], "muon", tp=2)
        self.assertEqual(sum(s["size_bytes"] for s in states), (16*32/2) * 4 * 2)

    def test_invalid_layer_map_rejected(self):
        model = hybrid_model()
        model["architecture"]["attention_by_layer"] = ["MLA"]
        with self.assertRaisesRegex(ValueError, "must match"):
            parameter_inventory(model)

    def test_kv_tp_replication_for_gqa(self):
        model = hybrid_model()
        model["architecture"]["attention_by_layer"] = ["GQA", "GQA"]
        t = next(t for t in parameter_inventory(model) if t.name == "layers.0.attention.k.weight")
        self.assertEqual(t.local_elements(tp=4), t.elements / 2)

    def test_public_k3_text_shape_reconciles_rounded_parameter_counts(self):
        # Public config text_config (source snapshot documented in docs/sources):
        # 2.8T total / 104B per-token active, without the vision tower.
        mla_layers = set(range(3, 92, 4)) | {92}
        model = {"name": "Kimi-K3-text", "layers": 93, "hidden_dim": 7168,
                 "num_heads": 96, "head_dim": 128, "kv_heads": 96,
                 "architecture": {"vocab_size": 163840, "tie_embeddings": False,
                    "attention_by_layer": ["MLA" if i in mla_layers else "KDA" for i in range(93)],
                    "ffn_by_layer": ["Dense"] + ["MoE"] * 92,
                    "dense_intermediate_dim": 33792, "attnres_block_size": 12,
                    "mla": {"q_lora_rank": 1536, "kv_lora_rank": 512,
                            "qk_nope_head_dim": 128, "qk_rope_head_dim": 64,
                            "v_head_dim": 128, "use_output_gate": True},
                    "kda": {"num_heads": 96, "head_k_dim": 128, "head_v_dim": 128,
                            "conv_kernel": 4, "use_full_rank_gate": True},
                    "moe": {"num_experts": 896, "top_k": 16, "intermediate_dim": 3072,
                            "num_shared_experts": 2, "latent_dim": 3584,
                            "latent_norm": True, "router_bias": True}}}
        summary = inventory_summary(model)
        self.assertEqual(len(summary["components"]["MLA"]["layers"]), 24)
        self.assertEqual(len(summary["components"]["KDA"]["layers"]), 69)
        self.assertLess(abs(summary["total_params"] / 2.8e12 - 1), 0.01)
        self.assertLess(abs(summary["active_params"] / 104e9 - 1), 0.01)
        self.assertEqual(summary["total_params"], 2779484476000)
        self.assertEqual(summary["active_params"], 104189619808)


class ActivationAndStateTests(unittest.TestCase):
    def test_kda_state_constant_vs_mla_linear_context(self):
        model = hybrid_model()
        short = attention_state(model, 1, 2, 16, tp=2)
        long = attention_state(model, 1, 2, 16000, tp=2)
        self.assertEqual(short, long)
        self.assertEqual(short["kind"], "recurrent")
        self.assertEqual(short["recurrent_elements"], 2*4*4*4/2)
        self.assertEqual(short["convolution_elements"], 2*4*12*4/2)
        self.assertEqual(short["recurrent_dtype"], "fp32")
        self.assertEqual(attention_state(model, 0, 2, 32)["elements"],
                         2 * attention_state(model, 0, 2, 16)["elements"])
        self.assertEqual(attention_state(model, 0, 2, 16, tp=1), attention_state(model, 0, 2, 16, tp=4))

    def test_checkpoint_saved_state_only_layer_input(self):
        model = hybrid_model()
        ordinary = component_activations(model, 0, 10, training=True)
        checkpoint = component_activations(model, 0, 10, training=True, checkpointing=True)
        self.assertEqual(sum(r["elements"] for r in checkpoint), 10*16)
        self.assertLess(sum(r["elements"] for r in checkpoint), sum(r["elements"] for r in ordinary))

    def test_training_kda_saved_chunk_states_follow_sequence_chunks(self):
        model = hybrid_model()
        saved = attention_training_state(model, 1, 2, 129, tp=2)
        self.assertEqual(saved["chunks"], 3)
        self.assertEqual(saved["recurrent_chunk_elements"], 3 * 2*4*4*4/2)
        self.assertEqual(attention_training_state(model, 0, 2, 129)["recurrent_chunk_elements"], 0)

    def test_balanced_ep_does_not_divide_source_tokens_twice(self):
        model = hybrid_model()
        one = component_activations(model, 1, 10, tp=2, ep=1)
        four = component_activations(model, 1, 10, tp=2, ep=4)
        self.assertEqual(one, four)
        routed = next(r for r in one if r["name"] == "routed_gate_up")
        self.assertEqual(routed["elements"], 2*10*2*8/2)


if __name__ == "__main__":
    unittest.main()
