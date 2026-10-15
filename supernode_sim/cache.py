"""Size-budgeted caches for optional simulation acceleration."""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import MutableMapping
import sys


def _size_bytes(value) -> int:
    """Count a container's Python heap, counting shared children once per entry.

    Entries are immutable snapshots. Counting shared children again across
    entries deliberately errs towards eviction rather than exceeding a budget.
    This is an accounting limit, not a process RSS limit.
    """
    pending, seen, total = [value], set(), 0
    while pending:
        item = pending.pop()
        identity = id(item)
        if identity in seen:
            continue
        seen.add(identity)
        total += sys.getsizeof(item)
        if isinstance(item, dict):
            pending.extend(item.keys())
            pending.extend(item.values())
        elif isinstance(item, (tuple, list, set, frozenset)):
            pending.extend(item)
    return total


class BoundedCache(MutableMapping):
    """LRU for immutable nested-container values; oversized entries are skipped."""

    def __init__(self, max_bytes: int = 64 * 1024 * 1024, max_entries: int = 256):
        if max_bytes < 0 or max_entries < 0:
            raise ValueError('Cache budgets must be nonnegative')
        self.max_bytes, self.max_entries = max_bytes, max_entries
        self.current_bytes = 0
        self._entries = OrderedDict()

    def __getitem__(self, key):
        value, _ = self._entries[key]
        self._entries.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        if key in self._entries:
            del self[key]
        if not self.max_bytes or not self.max_entries:
            return
        size = _size_bytes((key, value)) + 128  # allowance for the LRU node
        if size > self.max_bytes:
            return
        while self._entries and (self.current_bytes + size > self.max_bytes
                                 or len(self._entries) >= self.max_entries):
            _, (_, old_size) = self._entries.popitem(last=False)
            self.current_bytes -= old_size
        self._entries[key] = (value, size)
        self.current_bytes += size

    def __delitem__(self, key):
        _, size = self._entries.pop(key)
        self.current_bytes -= size

    def __iter__(self):
        # Mapping.items()/values() read through __getitem__, which updates LRU
        # order. Snapshot keys so ordinary mapping inspection stays valid.
        return iter(tuple(self._entries))

    def __len__(self):
        return len(self._entries)

    def clear(self):
        self._entries.clear()
        self.current_bytes = 0
