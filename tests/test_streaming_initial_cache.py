"""Cached immutable inference residents must preserve placement and capacity."""
import unittest

from supernode_sim.analysis import compile_graph, execution_metrics
from supernode_sim.engine import simulate
from supernode_sim.streaming import run_streaming
from tests.test_structured import configuration


class StreamingInitialCacheTests(unittest.TestCase):
    def config(self, patch=None):
        from supernode_sim.config import deep_merge
        return configuration(deep_merge({
            'workload': {'scenario': 'inference', 'input_length': 8, 'output_length': 2},
            'hardware': {'npu_count': 4, 'su_size': 2, 'npus_per_host': 2},
            'execution': {'tp': 2, 'ep': 2},
        }, patch or {}))

    def assert_equivalent(self, config, cached):
        graph = compile_graph(config)
        normal = simulate(graph)
        self.assertEqual(cached['status'], normal['status'])
        self.assertAlmostEqual(cached['simulation']['duration_s'], normal['duration_s'], places=12)
        for key in ('memory_peaks_bytes', 'memory_final_bytes', 'resource_work'):
            for resource in set(normal[key]) | set(cached['simulation'][key]):
                expected = normal[key].get(resource, 0)
                self.assertAlmostEqual(cached['simulation'][key].get(resource, 0), expected,
                    delta=max(1e-7, abs(expected) * 1e-10), msg=(key, resource))
        if normal['status'] == 'feasible':
            metrics = execution_metrics(config, graph, normal)
            self.assertAlmostEqual(cached['metrics']['ttft_s'], metrics['ttft_s'], places=12)
            self.assertAlmostEqual(cached['metrics']['tpot_s'], metrics['tpot_s'], places=12)
        for key in ('logical_component_state_bytes', 'logical_memory_bytes'):
            self.assertEqual(cached['model_metadata'][key], graph.metadata[key])

    def test_workloads_reuse_resident_cache_for_baseline_and_sp(self):
        for sp in (False, True):
            with self.subTest(sequence_parallel=sp):
                cache = {}
                first = run_streaming(self.config({'execution': {'sequence_parallel': sp}}), cache=cache)
                self.assertFalse(first['model_metadata']['initial_resident_cache']['hit'])
                changed = self.config({'execution': {'sequence_parallel': sp},
                    'workload': {'batch_size': 4, 'input_length': 16, 'output_length': 3}})
                second = run_streaming(changed, cache=cache)
                self.assertTrue(second['model_metadata']['initial_resident_cache']['hit'])
                self.assert_equivalent(changed, second)

    def test_resident_key_guards_placement_dtype_and_reservations(self):
        cache = {}
        run_streaming(self.config(), cache=cache)
        patches = [
            {'execution': {'tp': 1, 'ep': 4}},
            {'execution': {'rank_order': [1, 0, 3, 2]}},
            {'execution': {'precision': {'weight': 'fp8'}}},
            {'hardware': {'npu': {'runtime_bytes': 4096}}},
            {'hardware': {'npu': {'workspace_bytes': 2048}}},
            {'hardware': {'npus_per_host': 1}},
            {'model': {'architecture': {'moe': {'intermediate_dim': 12}}}},
        ]
        for patch in patches:
            with self.subTest(patch=patch):
                config = self.config(patch)
                actual = run_streaming(config, cache=cache)
                self.assertFalse(actual['model_metadata']['initial_resident_cache']['hit'])
                self.assert_equivalent(config, actual)
                replay = run_streaming(config, cache=cache)
                self.assertTrue(replay['model_metadata']['initial_resident_cache']['hit'])

    def test_cached_residents_still_obey_current_memory_capacity(self):
        cache = {}
        run_streaming(self.config(), cache=cache)
        rejected = run_streaming(self.config({'hardware': {'npu': {'hbm_bytes': 100}}}), cache=cache)
        self.assertTrue(rejected['model_metadata']['initial_resident_cache']['hit'])
        self.assertEqual(rejected['status'], 'infeasible')
        self.assertEqual(rejected['simulation']['diagnostics'][0]['code'], 'initial_memory_exceeded')

    def test_engram_initial_resources_bypass_resident_cache(self):
        config = self.config({'execution': {'features': {'engram': {
            'enabled': True, 'table_bytes': 1024, 'record_bytes': 16, 'layers': [0]}}}})
        cache = {}
        first = run_streaming(config, cache=cache)
        second = run_streaming(config, cache=cache)
        self.assertFalse(first['model_metadata']['initial_resident_cache']['enabled'])
        self.assertFalse(second['model_metadata']['initial_resident_cache']['enabled'])
        self.assert_equivalent(config, second)


if __name__ == '__main__':
    unittest.main()
