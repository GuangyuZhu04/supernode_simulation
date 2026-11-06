"""Host budgets must scale and enforce a real allocation ceiling in a child."""
from pathlib import Path
import subprocess
import sys

import pytest

from supernode_sim import host_memory as memory


@pytest.mark.parametrize('total_gib,available_gib,expected_gib', [(8, 4, 2), (64, 48, 24), (512, 400, 200)])
def test_automatic_budget_scales_with_host(monkeypatch, total_gib, available_gib, expected_gib):
    monkeypatch.delenv(memory.MEMORY_ENV, raising=False)
    monkeypatch.setattr(memory, 'available_memory', lambda: (total_gib << 30, available_gib << 30))
    assert memory.worker_memory_limit() == expected_gib << 30


def test_explicit_budget_still_reserves_available_memory(monkeypatch):
    monkeypatch.setattr(memory, 'available_memory', lambda: (64 << 30, 8 << 30))
    monkeypatch.setenv(memory.MEMORY_ENV, '32768')
    assert memory.worker_memory_limit() == 6 << 30
    monkeypatch.setenv(memory.MEMORY_ENV, '512')
    assert memory.worker_memory_limit() == 512 * memory.MIB


@pytest.mark.parametrize('value', ['0', '-1', 'none', '2.5'])
def test_budget_cannot_disable_protection(monkeypatch, value):
    monkeypatch.setenv(memory.MEMORY_ENV, value)
    with pytest.raises(ValueError):
        memory.worker_memory_limit()


def test_insufficient_memory_fails_before_launch(monkeypatch):
    monkeypatch.delenv(memory.MEMORY_ENV, raising=False)
    monkeypatch.setattr(memory, 'available_memory', lambda: (8 << 30, 128 * memory.MIB))
    with pytest.raises(ValueError, match='可用主机内存不足'):
        memory.worker_memory_limit()


@pytest.mark.parametrize('version', [1, 2])
def test_container_budget_uses_limit_and_current_usage(tmp_path, version):
    root = tmp_path if version == 2 else tmp_path / 'memory'
    root.mkdir(exist_ok=True)
    limit, usage = ('memory.max', 'memory.current') if version == 2 else ('memory.limit_in_bytes', 'memory.usage_in_bytes')
    (root / limit).write_text(str(4 << 30))
    (root / usage).write_text(str(3 << 30))
    assert memory._cgroup_memory(256 << 30, 200 << 30, tmp_path) == (4 << 30, 1 << 30)


def test_native_limit_refuses_large_allocation_in_real_subprocess():
    # Only the child is limited. Even on failure this allocation is at most 256
    # MiB, so the regression cannot reproduce the original unbounded OOM.
    code = '''
from supernode_sim.host_memory import apply_worker_memory_limit, peak_memory_bytes, MIB
guard = apply_worker_memory_limit(128 * MIB)
try:
    block = bytearray(256 * MIB)
except MemoryError:
    print('allocation-blocked', peak_memory_bytes())
else:
    raise SystemExit('memory limit was not enforced')
'''
    result = subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True, timeout=20,
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    assert result.returncode == 0, result.stderr + result.stdout
    assert result.stdout.startswith('allocation-blocked ')
