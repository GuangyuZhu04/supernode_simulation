"""Streaming must conserve the original inference graph's numerical result."""
import unittest

from supernode_sim.analysis import compile_graph, execution_metrics
from supernode_sim.engine import simulate
from supernode_sim.streaming import run_streaming
from tests.test_structured import configuration


class StreamingInferenceTests(unittest.TestCase):
    def compare(self, patch):
        config = configuration(patch)
        graph = compile_graph(config)
        expected = simulate(graph)
        actual = run_streaming(config)
        self.assertEqual(actual['status'], expected['status'])
        self.assertAlmostEqual(actual['simulation']['duration_s'], expected['duration_s'], places=12)
        for key in ('resource_work', 'memory_peaks_bytes', 'memory_final_bytes'):
            for resource in set(actual['simulation'][key]) | set(expected[key]):
                self.assertAlmostEqual(actual['simulation'][key].get(resource, 0), expected[key].get(resource, 0),
                    delta=max(1e-7, abs(expected[key].get(resource, 0)) * 1e-10), msg=(key, resource))
        if expected['status'] == 'feasible':
            metrics = execution_metrics(config, graph, expected)
            for metric in ('ttft_s', 'tpot_s'):
                self.assertAlmostEqual(actual['metrics'][metric], metrics[metric], places=12)
            traffic = sum(e.metadata.get('payload_bytes', 0) for e in graph.events)
            self.assertAlmostEqual(sum(p['communication']['transport_payload_bytes'] for p in actual['phase_metrics'].values()), traffic)
        return config, expected, actual

    def test_hybrid_ep_tp_matches_full_graph(self):
        self.compare({'workload': {'scenario': 'inference', 'output_length': 3},
            'hardware': {'npu_count': 4, 'su_size': 2}, 'execution': {'tp': 2, 'ep': 2}})

    def test_attnres_and_quantized_offload_match_full_graph(self):
        self.compare({'workload': {'scenario': 'inference', 'output_length': 3},
            'model': {'architecture': {'attnres_block_size': 1}},
            'hardware': {'npu_count': 4, 'su_size': 2},
            'execution': {'tp': 2, 'ep': 2, 'precision': {'kv': 'fp8', 'communication': 'fp32'},
                'features': {'kv_offload': 'host'}}})

    def test_decode_only_reuses_complete_prefill_persistent_state(self):
        config, expected, all_result = self.compare({'workload': {'scenario': 'inference', 'output_length': 3},
            'model': {'architecture': {'attnres_block_size': 1}},
            'hardware': {'npu_count': 4, 'su_size': 2}, 'execution': {'tp': 2, 'ep': 2}})
        decoded = run_streaming(config, phase='decode')
        self.assertEqual(decoded['status'], 'feasible')
        self.assertIsNone(decoded['metrics']['ttft_s'])
        self.assertAlmostEqual(decoded['metrics']['tpot_s'], all_result['metrics']['tpot_s'], places=12)
        self.assertAlmostEqual(decoded['metrics']['duration_s'], all_result['phase_metrics']['decode']['duration_s'], places=12)
        for pool, size in expected['memory_final_bytes'].items():
            self.assertAlmostEqual(decoded['simulation']['memory_final_bytes'].get(pool, 0), size, places=7)

    def test_shared_cache_replays_only_fitting_schedule(self):
        config = configuration({'workload': {'scenario': 'inference'}})
        cache = {}
        first = run_streaming(config, cache=cache)
        second = run_streaming(config, cache=cache)
        self.assertTrue(any(row['timing_cache_hit'] for row in second['chunks']))
        self.assertEqual(first['metrics'], second['metrics'])
        config.hardware.npu.hbm_bytes = 100
        rejected = run_streaming(config, cache=cache)
        self.assertEqual(rejected['status'], 'infeasible')
        self.assertEqual(rejected['simulation']['diagnostics'][0]['code'], 'initial_memory_exceeded')

    def test_ragged_decode_preserves_maximum_sequence_mean(self):
        config, _, all_result = self.compare({'workload': {'scenario': 'inference',
                'input_lengths': [8, 16], 'output_lengths': [4, 2]},
            'hardware': {'npu_count': 4, 'su_size': 2}, 'execution': {'tp': 2, 'ep': 2}})
        decoded = run_streaming(config, phase='decode')
        self.assertAlmostEqual(decoded['metrics']['tpot_s'], all_result['metrics']['tpot_s'], places=12)

    def test_repeated_moe_components_skip_generation_and_conserve_physics(self):
        _, _, actual = self.compare({'workload': {'scenario': 'inference', 'output_length': 3},
            'model': {'layers': 6, 'architecture': {
                'attention_by_layer': ['MLA', 'KDA', 'MLA', 'KDA', 'MLA', 'KDA'],
                'ffn_by_layer': ['Dense', 'MoE', 'MoE', 'MoE', 'MoE', 'MoE'],
                'attnres_block_size': 2}},
            'hardware': {'npu_count': 4, 'su_size': 2},
            'execution': {'tp': 2, 'ep': 2, 'precision': {'kv': 'fp8', 'communication': 'fp32'},
                'features': {'kv_offload': 'host'}}})
        moe_replays = [row for row in actual['chunks'] if row['component'] == 'MoE' and row['component_macro_hit']]
        self.assertGreaterEqual(len(moe_replays), 9)

    def test_layer_specific_engram_lookup_is_not_skipped_by_norm_macro(self):
        _, _, actual = self.compare({'workload': {'scenario': 'inference', 'output_length': 3},
            'model': {'layers': 4, 'architecture': {
                'attention_by_layer': ['MLA', 'KDA', 'MLA', 'KDA'],
                'ffn_by_layer': ['Dense', 'MoE', 'MoE', 'MoE']}},
            'hardware': {'npu_count': 4, 'su_size': 2},
            'execution': {'tp': 2, 'ep': 2, 'features': {'engram': {
                'enabled': True, 'layers': [1, 3], 'table_bytes': 4096,
                'record_bytes': 64, 'queries_per_token': 2, 'latency_s': 0.00001}}}})
        self.assertTrue(any(row['component'] == 'PreNorm' and row['component_macro_hit']
                            and '.l3.' in row['id'] for row in actual['chunks']))

    def test_shared_cache_preserves_changed_context_and_resident_memory(self):
        patch = {'workload': {'scenario': 'inference', 'output_length': 3},
                 'hardware': {'npu_count': 4, 'su_size': 2},
                 'execution': {'tp': 2, 'ep': 2}}
        cache = {}
        run_streaming(configuration(patch), cache=cache)
        patch['workload']['input_length'] = 16
        patch['hardware']['npu'] = {'runtime_bytes': 1234}
        config = configuration(patch)
        graph = compile_graph(config)
        expected = simulate(graph)
        actual = run_streaming(config, cache=cache)
        self.assertEqual(actual['status'], expected['status'])
        self.assertAlmostEqual(actual['metrics']['duration_s'], expected['duration_s'], places=12)
        self.assertTrue(any(row['timing_cache_hit'] for row in actual['chunks']))
        for key in ('memory_peaks_bytes', 'memory_final_bytes', 'resource_work'):
            for resource in set(expected[key]) | set(actual['simulation'][key]):
                value = expected[key].get(resource, 0)
                self.assertAlmostEqual(actual['simulation'][key].get(resource, 0), value,
                                       delta=max(1e-7, abs(value) * 1e-10), msg=(key, resource))

    def test_prefill_only_has_no_tpot(self):
        config = configuration({'workload': {'scenario': 'inference'}})
        expected = run_streaming(config)
        prefill = run_streaming(config, phase='prefill')
        self.assertAlmostEqual(prefill['metrics']['ttft_s'], expected['metrics']['ttft_s'], places=12)
        self.assertIsNone(prefill['metrics']['tpot_s'])

    def test_sequence_parallel_components_and_seeded_decode_match_full_graph(self):
        config, _, all_result = self.compare({'workload': {'scenario': 'inference', 'output_length': 3},
            'model': {'layers': 6, 'architecture': {
                'attention_by_layer': ['MLA', 'KDA', 'MLA', 'KDA', 'MLA', 'KDA'],
                'ffn_by_layer': ['Dense', 'MoE', 'MoE', 'MoE', 'MoE', 'MoE'],
                'attnres_block_size': 2}},
            'hardware': {'npu_count': 4, 'su_size': 2},
            'execution': {'tp': 2, 'ep': 2, 'sequence_parallel': True,
                'precision': {'kv': 'fp8', 'communication': 'fp32'}, 'features': {'kv_offload': 'host'}}})
        decoded = run_streaming(config, phase='decode')
        self.assertEqual(decoded['status'], 'feasible')
        self.assertAlmostEqual(decoded['metrics']['tpot_s'], all_result['metrics']['tpot_s'], places=12)
        self.assertGreater(sum(row['component_macro_hit'] for row in all_result['chunks']), 0)


if __name__ == '__main__':
    unittest.main()
