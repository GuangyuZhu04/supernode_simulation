"""Bounded process execution for independent simulation candidates."""
from __future__ import annotations

from collections import deque
from concurrent.futures import ProcessPoolExecutor
import gc
import itertools
import multiprocessing
import os
import sys


def worker_count(workers: int, count: int) -> int:
    """Resolve the host worker limit; zero is conservative automatic selection."""
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 0:
        raise ValueError('workers must be a nonnegative integer (0=auto, 1=serial)')
    if workers == 0:
        cpu_count = getattr(os, 'process_cpu_count', os.cpu_count)() or 1
        workers = min(cpu_count, 4)
    # ProcessPoolExecutor on Windows supports at most 61 worker processes.
    if sys.platform == 'win32':
        workers = min(workers, 61)
    return min(workers, max(1, count))


def _initialize_worker() -> None:
    # Match the numerical CLI: graphs use IDs, not object back-links. Refcounting
    # releases each candidate without repeated cyclic-GC scans of large graphs.
    gc.disable()


def ordered_map(function, arguments, *, workers: int, count: int):
    """Yield in input order, retaining at most two tasks per worker in flight.

    Serial execution stays in the calling process, preserving ordinary API and
    monkeypatch behavior. Spawn avoids inheriting threads and large parent graphs
    and gives the same import/pickle contract on Windows, Linux and macOS.
    """
    limit = worker_count(workers, count)
    if limit == 1:
        yield from map(function, arguments)
        return
    iterator = iter(arguments)
    executor = ProcessPoolExecutor(max_workers=limit,
        mp_context=multiprocessing.get_context('spawn'), initializer=_initialize_worker)
    pending = deque()
    try:
        for argument in itertools.islice(iterator, 2 * limit):
            pending.append(executor.submit(function, argument))
        while pending:
            yield pending.popleft().result()
            for argument in itertools.islice(iterator, 1):
                pending.append(executor.submit(function, argument))
    finally:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
