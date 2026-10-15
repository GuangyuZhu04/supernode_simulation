"""Memory budgets may evict acceleration data, never change numerical results."""
import gc
import weakref

import pytest

from supernode_sim.cache import BoundedCache
from supernode_sim import streaming
from tests.test_structured import configuration


def test_lru_eviction_and_oversized_entries_obey_both_limits():
    cache = BoundedCache(max_bytes=4096, max_entries=2)
    cache['a'] = {'data': [1, 2]}
    cache['b'] = {'data': [3, 4]}
    assert cache['a'] == {'data': [1, 2]}
    cache['c'] = {'data': [5]}
    assert list(cache) == ['a', 'c']
    cache['large'] = 'x' * 4096
    assert 'large' not in cache
    assert cache.current_bytes <= cache.max_bytes
    cache['a'] = 'x' * 4096
    assert 'a' not in cache  # a replaced, oversized value must not leave stale data
    del cache['c']
    assert cache.current_bytes == 0
    cache['small'] = 1
    cache.clear()
    assert cache.current_bytes == 0 and not cache


def test_mapping_views_allow_reads_that_refresh_lru_order():
    cache = BoundedCache()
    cache.update(a=1, b=2, c=3)
    assert list(cache.items()) == [('a', 1), ('b', 2), ('c', 3)]
    assert list(cache.values()) == [1, 2, 3]
    assert dict(cache) == {'a': 1, 'b': 2, 'c': 3}


@pytest.mark.parametrize('limit', [0, 4096, 32768])
def test_evicted_streaming_cache_preserves_results_across_cases(limit):
    cache = BoundedCache(max_bytes=limit, max_entries=3)
    for length in (8, 16, 4):
        config = configuration({'workload': {'scenario': 'inference',
                                'input_length': length, 'output_length': 4}})
        expected = streaming.run_streaming(config, cache={})
        actual = streaming.run_streaming(config, cache=cache)
        assert actual['metrics'] == expected['metrics']
        assert actual['simulation'] == expected['simulation']
        assert len(cache) <= 3 and cache.current_bytes <= limit


@pytest.mark.parametrize('phase', ['all', 'prefill', 'decode'])
@pytest.mark.parametrize('failure', [None, 'capacity', 'callback'])
def test_streaming_releases_builder_without_cyclic_gc(monkeypatch, phase, failure):
    references = []
    original = streaming.Builder

    class TrackingBuilder(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            references.append(weakref.ref(self))

    monkeypatch.setattr(streaming, 'Builder', TrackingBuilder)
    config = configuration({'workload': {'scenario': 'inference', 'output_length': 3}})
    if failure == 'capacity':
        config.hardware.npu.hbm_bytes = 1

    def progress(_):
        if failure == 'callback':
            raise RuntimeError('injected progress failure')

    enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(3):
            # Catch without retaining the traceback, which intentionally owns
            # the failing frame until its exception leaves scope.
            try:
                result = streaming.run_streaming(config, phase=phase, progress=progress)
            except RuntimeError:
                assert failure == 'callback'
            else:
                assert failure != 'callback'
                assert result['status'] == ('infeasible' if failure == 'capacity' else 'feasible')
                del result
            assert all(reference() is None for reference in references)
    finally:
        if enabled:
            gc.enable()
