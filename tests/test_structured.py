"""Numerical invariants of the shape-derived hybrid execution path."""
import copy
from collections import defaultdict
import unittest

from supernode_sim.analysis import compile_graph
from supernode_sim.config import Config, deep_merge
from supernode_sim.core import DTYPE_BYTES
from supernode_sim.engine import simulate


def configuration(patch=None):
    base = {
        'model': {'name': 'hybrid-regression', 'layers': 2, 'hidden_dim': 16,
                  'num_heads': 4, 'kv_heads': 2, 'head_dim': 4,
                  'architecture': {'vocab_size': 32, 'attention_by_layer': ['MLA', 'KDA'],
                    'ffn_by_layer': ['Dense', 'MoE'], 'dense_intermediate_dim': 32,
                    'mla': {'q_lora_rank': 4, 'kv_lora_rank': 4, 'qk_nope_head_dim': 2,
                            'qk_rope_head_dim': 2, 'v_head_dim': 4, 'use_output_gate': True},
                    'kda': {'num_heads': 4, 'head_k_dim': 4, 'head_v_dim': 4, 'gate_low_rank_dim': 4,
                            'conv_kernel': 4, 'use_full_rank_gate': True},
                    'moe': {'num_experts': 4, 'top_k': 2, 'intermediate_dim': 8,
                            'num_shared_experts': 1, 'router_bias': True}}},
        'hardware': {'npu_count': 1, 'su_size': 1, 'npus_per_host': 1,
                     'npu': {'runtime_bytes': 0, 'workspace_bytes': 0}},
        'workload': {'scenario': 'training', 'batch_size': 2, 'sequence_length': 8,
                     'input_length': 8, 'output_length': 3},
    }
    return Config.model_validate(deep_merge(base, patch or {}))


def compiled(patch=None):
    config = configuration(patch)
    graph = compile_graph(config)
    return config, graph, simulate(graph)


class StructuredTrainingTests(unittest.TestCase):
    def test_bounded_optimizer_pipeline_conserves_work_and_limits_live_buckets(self):
        patch = {'execution': {'features': {'optimizer_offload': True}, 'optimizer': {'kind': 'muon', 'inflight_components_per_rank': 2}}}
        config, bounded, result = compiled(patch)
        self.assert_no_training_leak(bounded, result)
        _, unrestricted, unbounded_result = compiled({'execution': {'features': {'optimizer_offload': True}, 'optimizer': {'kind': 'muon'}}})
        self.assert_no_training_leak(unrestricted, unbounded_result)
        self.assertEqual(bounded.metadata['logical_memory_bytes'], unrestricted.metadata['logical_memory_bytes'])
        self.assertEqual(set(result['resource_work']), set(unbounded_result['resource_work']))
        for resource, amount in result['resource_work'].items():
            self.assertAlmostEqual(amount, unbounded_result['resource_work'][resource], delta=max(1e-6, abs(amount) * 1e-10))
        ends = {row['id']: row['end_s'] for row in result['trace']}
        changes = []
        for row in result['trace']:
            if row['id'].endswith('.D2H'):
                changes.extend([(row['start_s'], 1), (ends[row['id'][:-4] + '.refresh'], -1)])
        active = peak = 0
        for _, change in sorted(changes):
            active += change
            peak = max(peak, active)
        self.assertLessEqual(peak, 2)
        self.assertEqual(active, 0)
        self.assertEqual(Config.model_validate(config.model_dump()).model_dump(), config.model_dump())
        self.assertNotIn('inflight_components_per_rank', configuration().model_dump()['execution']['optimizer'])

    def assert_no_training_leak(self, graph, result):
        self.assertEqual(result['status'], 'feasible', result['diagnostics'])
        resident = defaultdict(float)
        for a in graph.initial_allocations:
            resident[a.pool] += a.size_bytes
        for pool, amount in result['memory_final_bytes'].items():
            self.assertAlmostEqual(amount, resident[pool], places=5, msg=pool)

    def test_adam_and_muon_state_breakdown_conserves_storage(self):
        config, adam, result = compiled()
        self.assert_no_training_leak(adam, result)
        _, muon, muon_result = compiled({'execution': {'optimizer': {'kind': 'muon'}}})
        self.assert_no_training_leak(muon, muon_result)
        logical = adam.metadata['logical_component_state_bytes']
        self.assertEqual(sum(v['weight'] for v in logical.values()), config.model.total_params * 2)
        self.assertEqual(sum(sum(v.values()) for v in logical.values()), adam.metadata['logical_memory_bytes']['total'])
        self.assertLess(muon.metadata['logical_memory_bytes']['total'], adam.metadata['logical_memory_bytes']['total'])
        self.assertIn('muon_momentum', muon.metadata['logical_component_state_bytes']['MLA'])
        self.assertIn('optimizer_variance', muon.metadata['logical_component_state_bytes']['WordEmbedding'])

    def test_attnres_storage_work_and_optimizer_are_included(self):
        _, graph, result = compiled({'model': {'architecture': {'attnres_block_size': 1}}})
        self.assert_no_training_leak(graph, result)
        self.assertTrue(any(e.metadata.get('component') == 'AttnRes' and e.metadata.get('phase') == 'forward' for e in graph.events))
        self.assertTrue(any(e.metadata.get('component') == 'AttnRes' and e.metadata.get('phase') == 'optimizer' for e in graph.events))
        residuals = [a for e in graph.events for a in e.allocs if a.category == 'attnres_residual']
        self.assertEqual(len(residuals), 2)
        self.assertTrue(all(a.metadata['component'] == 'AttnRes' for a in residuals))

    def test_checkpoint_replays_all_kernels_and_collectives_and_releases_temps(self):
        patch = {'hardware': {'npu_count': 4, 'su_size': 4},
                 'execution': {'tp': 2, 'ep': 2, 'features': {'activation_checkpointing': True}},
                 'workload': {'gradient_accumulation': 2}}
        _, graph, result = compiled(patch)
        self.assert_no_training_leak(graph, result)
        components = {e.metadata.get('component') for e in graph.events if e.metadata.get('phase') == 'recompute'}
        self.assertTrue({'WordEmbedding', 'PreNorm', 'PostNorm', 'MLA', 'KDA', 'Dense', 'MoE', 'Output'} <= components)
        self.assertTrue(any('.recompute.dispatch' in e.id for e in graph.events))
        self.assertTrue(any('.recompute.combine' in e.id for e in graph.events))
        self.assertTrue(any('.recompute.MLA.tp' in e.id for e in graph.events))
        no_checkpoint = copy.deepcopy(patch)
        no_checkpoint['execution']['features']['activation_checkpointing'] = False
        _, ordinary, ordinary_result = compiled(no_checkpoint)
        compute = lambda g: sum(e.metadata.get('flops', 0) for e in g.events)
        self.assertGreater(compute(graph), compute(ordinary))

    def test_fsdp_gradient_buffer_is_live_before_backward_and_accumulates(self):
        _, graph, result = compiled({'hardware': {'npu_count': 2, 'su_size': 2},
            'execution': {'dp': 2, 'fsdp': 2}, 'workload': {'gradient_accumulation': 2}})
        self.assert_no_training_leak(graph, result)
        traces = {t['id']: t for t in result['trace']}
        allocate = traces['train.s0.p0.backward.m0.l1.gradient_buffers']
        backward = traces['train.s0.p0.backward.m0.l1.npu0.MoE']
        self.assertLessEqual(allocate['end_s'], backward['start_s'])
        self.assertTrue(allocate['allocations'])
        retained = allocate['allocations'][0]['id']
        release = next(t for t in result['trace'] if retained in t['releases'])
        self.assertIn('backward.m1', release['id'])

    def test_fsdp_sync_each_microbatch_uses_unique_buffers(self):
        _, graph, result = compiled({'hardware': {'npu_count': 2, 'su_size': 2},
            'execution': {'dp': 2, 'fsdp': 2, 'sync_each_microbatch': True},
            'workload': {'gradient_accumulation': 2}})
        self.assert_no_training_leak(graph, result)

    def test_pipeline_1f1b_with_casts_has_no_leaks(self):
        _, graph, result = compiled({'hardware': {'npu_count': 4, 'su_size': 4},
            'execution': {'pp': 2, 'ep': 2, 'precision': {'communication': 'fp32'}},
            'workload': {'gradient_accumulation': 3, 'training_steps': 2}})
        self.assert_no_training_leak(graph, result)
        kernels = {e.metadata.get('kernel') for e in graph.events}
        self.assertTrue({'pipeline_pack', 'pipeline_unpack'} <= kernels)
        self.assertTrue(any(a.category == 'pipeline_buffer' for e in graph.events for a in e.allocs))

    def test_ep_attention_arithmetic_is_conserved(self):
        _, single, _ = compiled()
        _, ep, result = compiled({'hardware': {'npu_count': 2, 'su_size': 2}, 'execution': {'ep': 2}})
        self.assert_no_training_leak(ep, result)
        def flops(g, phase):
            return sum(e.metadata.get('flops', 0) for e in g.events if e.metadata.get('component') == 'MLA' and e.metadata.get('phase') == phase)
        self.assertAlmostEqual(flops(single, 'forward'), flops(ep, 'forward'))
        self.assertAlmostEqual(flops(single, 'backward'), 2 * flops(single, 'forward'))

    def test_low_capacity_is_infeasible_instead_of_clipped(self):
        _, _, result = compiled({'hardware': {'npu': {'hbm_bytes': 100}}})
        self.assertEqual(result['status'], 'infeasible')
        self.assertEqual(result['diagnostics'][0]['code'], 'initial_memory_exceeded')


class StructuredInferenceTests(unittest.TestCase):
    def test_prefill_logits_only_final_position_per_live_sequence(self):
        _, graph, result = compiled({'workload': {'scenario': 'inference', 'input_length': 64, 'output_length': 1}})
        self.assertEqual(result['status'], 'feasible', result['diagnostics'])
        logits = [a for e in graph.events for a in e.allocs
                  if a.metadata.get('component') == 'Output' and a.category == 'activation']
        self.assertEqual(sum(a.size_bytes for a in logits), 2 * 32 * 2)

    def test_mla_cache_grows_kda_state_does_not(self):
        base = {'workload': {'scenario': 'inference', 'output_length': 1}}
        _, short, result = compiled(base)
        self.assertEqual(result['status'], 'feasible', result['diagnostics'])
        base['workload']['input_length'] = 64
        _, long, result = compiled(base)
        self.assertEqual(result['status'], 'feasible', result['diagnostics'])
        amount = lambda g, category: sum(a.size_bytes for e in g.events for a in e.allocs if a.category == category)
        self.assertGreater(amount(long, 'kv_cache'), amount(short, 'kv_cache'))
        self.assertEqual(amount(long, 'recurrent_state'), amount(short, 'recurrent_state'))
        self.assertEqual(amount(long, 'convolution_state'), amount(short, 'convolution_state'))

    def test_quantized_kv_offload_charges_scales_casts_and_no_transient_leaks(self):
        _, graph, result = compiled({'workload': {'scenario': 'inference'},
            'execution': {'precision': {'kv': 'fp8'}, 'features': {'kv_offload': 'host'}}})
        self.assertEqual(result['status'], 'feasible', result['diagnostics'])
        kernels = {e.metadata.get('kernel') for e in graph.events}
        self.assertTrue({'kv_decode_conversion', 'kv_encode_conversion'} <= kernels)
        writes = [e for e in graph.events if e.metadata.get('operation') == 'kv_write']
        self.assertTrue(writes)
        # Two sequences, 8 input tokens, latent4+rope2, FP8 plus scale blocks.
        first = next(e for e in writes if 's0.' in e.id)
        self.assertEqual(first.metadata['payload_bytes'], 2*8*6 + 4)
        allocated = {a.id: a for a in graph.initial_allocations}
        allocated.update({a.id: a for e in graph.events for a in e.allocs})
        released = {aid for e in graph.events for aid in e.releases}
        live_categories = {a.category for aid, a in allocated.items() if aid not in released}
        self.assertFalse({'activation', 'kv_fetch_buffer', 'kv_write_buffer', 'kv_conversion_buffer', 'conversion_buffer'} & live_categories)

    def test_ragged_sequences_stop_compute_and_reads(self):
        _, graph, result = compiled({'workload': {'scenario': 'inference', 'input_lengths': [8, 16], 'output_lengths': [3, 1]},
            'hardware': {'npu_count': 2, 'su_size': 2}, 'execution': {'ep': 2}})
        self.assertEqual(result['status'], 'feasible', result['diagnostics'])
        self.assertFalse(any(e.id == 'infer.d0.s1.l0.npu1.MLA' for e in graph.events))
        self.assertEqual(len(graph.metadata['sequence_token_completion_events']['0']['1']), 1)

    def test_inference_pipeline_casts_release_buffers(self):
        _, graph, result = compiled({'workload': {'scenario': 'inference'},
            'hardware': {'npu_count': 2, 'su_size': 2},
            'execution': {'pp': 2, 'precision': {'communication': 'fp32'}}})
        self.assertEqual(result['status'], 'feasible', result['diagnostics'])
        pipeline = {a.id for e in graph.events for a in e.allocs if a.category == 'pipeline_buffer'}
        released = {aid for e in graph.events for aid in e.releases}
        self.assertTrue(pipeline)
        self.assertTrue(pipeline <= released)


if __name__ == '__main__':
    unittest.main()
