"""Host-side memory budgets for isolated web workers, independent of NPU HBM."""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
import subprocess
import sys

MIB = 1024 * 1024
MEMORY_ENV = 'SUPERNODE_WEB_MEMORY_MIB'
DEFAULT_LIMIT_BYTES = 1024 * MIB  # Conservative fallback only if discovery fails.


def _cgroup_memory(total: int, available: int, root=Path('/sys/fs/cgroup')) -> tuple[int, int]:
    """Honor v1/v2 container limits, including stricter ancestor groups."""
    groups = [(root, 'memory.max', 'memory.current'),
              (root / 'memory', 'memory.limit_in_bytes', 'memory.usage_in_bytes')]
    try:
        for line in Path('/proc/self/cgroup').read_text().splitlines():
            _, controllers, relative = line.split(':', 2)
            if not controllers or 'memory' in controllers.split(','):
                base = root if not controllers else root / 'memory'
                directory = base / relative.lstrip('/')
                if not directory.resolve().is_relative_to(base.resolve()):
                    continue
                while directory != base:
                    groups.append((directory, 'memory.max' if not controllers else 'memory.limit_in_bytes',
                                   'memory.current' if not controllers else 'memory.usage_in_bytes'))
                    directory = directory.parent
    except (OSError, ValueError):
        pass
    for directory, cap_name, used_name in groups:
        try:
            limit = int((directory / cap_name).read_text().strip())
            used = int((directory / used_name).read_text().strip())
        except (OSError, ValueError):
            continue  # Includes the v2 "max" unlimited value.
        total, available = min(total, limit), min(available, max(0, limit - used))
    return total, available


def available_memory() -> tuple[int, int] | None:
    """Return total physical and currently available host bytes, when known."""
    if sys.platform == 'win32':
        class MemoryStatus(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong),
                        *[(name, ctypes.c_ulonglong) for name in
                          ('total', 'available', 'total_page', 'available_page',
                           'total_virtual', 'available_virtual', 'extended')]]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.total, min(status.available, status.available_page)
    elif sys.platform.startswith('linux'):
        try:
            with open('/proc/meminfo', encoding='ascii') as handle:
                values = {key: int(value.split()[0]) * 1024
                          for key, value in (line.split(':', 1) for line in handle)}
            return _cgroup_memory(values['MemTotal'], values['MemAvailable'])
        except (OSError, ValueError, KeyError):
            pass
    elif sys.platform == 'darwin':
        try:
            total = int(subprocess.check_output(['/usr/sbin/sysctl', '-n', 'hw.memsize'], timeout=3))
            stats = subprocess.check_output(['/usr/bin/vm_stat'], text=True, timeout=3)
            import re
            page_size = int(re.search(r'page size of (\d+) bytes', stats).group(1))
            pages = {key.strip(): int(value.strip().rstrip('.'))
                     for key, value in (line.split(':', 1) for line in stats.splitlines()[1:] if ':' in line)}
            return total, page_size * (pages['Pages free'] + pages['Pages inactive'])
        except (OSError, ValueError, KeyError, AttributeError, subprocess.SubprocessError):
            pass
    return None


def worker_memory_limit() -> int:
    """Reserve host headroom even when an explicit worker budget is requested."""
    raw = os.environ.get(MEMORY_ENV)
    requested = DEFAULT_LIMIT_BYTES
    if raw is not None:
        try:
            requested = int(raw) * MIB
        except ValueError:
            raise ValueError(f'{MEMORY_ENV} 必须是正整数 MiB') from None
        if requested <= 0:
            raise ValueError(f'{MEMORY_ENV} 必须大于 0，网页计算不能关闭内存保护')
    memory = available_memory()
    if memory is not None:
        total, available = memory
        requested = (min(total // 2, available // 2) if raw is None else
                     min(requested, available * 3 // 4))
    limit = requested // MIB * MIB
    if limit < 128 * MIB:
        raise ValueError('可用主机内存不足，无法为网页计算保留至少 128 MiB；请释放内存后重试')
    return limit


def apply_worker_memory_limit(limit: int):
    """Enforce in the worker before graph construction; never silently run unbounded.

    Windows limits committed process memory with a Job Object. POSIX limits
    address space (which also includes mappings). The returned Windows handle
    stays alive for the worker lifetime; the OS closes it on process exit.
    """
    if sys.platform != 'win32':
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        if soft != resource.RLIM_INFINITY:
            limit = min(limit, soft)
        if hard != resource.RLIM_INFINITY:
            limit = min(limit, hard)
        resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
        return None

    from ctypes import wintypes

    class BasicLimit(ctypes.Structure):
        _fields_ = [('process_time', ctypes.c_int64), ('job_time', ctypes.c_int64),
                    ('flags', wintypes.DWORD), ('minimum_working_set', ctypes.c_size_t),
                    ('maximum_working_set', ctypes.c_size_t), ('active_processes', wintypes.DWORD),
                    ('affinity', ctypes.c_size_t), ('priority', wintypes.DWORD),
                    ('scheduling', wintypes.DWORD)]

    class ExtendedLimit(ctypes.Structure):
        _fields_ = [('basic', BasicLimit), ('io_counters', ctypes.c_uint64 * 6),
                    ('process_memory', ctypes.c_size_t), ('job_memory', ctypes.c_size_t),
                    ('peak_process_memory', ctypes.c_size_t), ('peak_job_memory', ctypes.c_size_t)]

    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD)
    kernel.SetInformationJobObject.restype = wintypes.BOOL
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    kernel.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = kernel.CreateJobObjectW(None, None)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    limits = ExtendedLimit()
    limits.basic.flags = 0x100  # JOB_OBJECT_LIMIT_PROCESS_MEMORY
    limits.process_memory = limit
    if not kernel.SetInformationJobObject(handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
        error = ctypes.WinError(ctypes.get_last_error())
        kernel.CloseHandle(handle)
        raise error
    if not kernel.AssignProcessToJobObject(handle, kernel.GetCurrentProcess()):
        error = ctypes.WinError(ctypes.get_last_error())
        kernel.CloseHandle(handle)
        raise error
    return handle


def peak_memory_bytes() -> int:
    """Peak RSS on POSIX; peak private commit on Windows, matching its limit."""
    if sys.platform != 'win32':
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak if sys.platform == 'darwin' else peak * 1024)
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [('size', wintypes.DWORD), ('faults', wintypes.DWORD),
                    *[(name, ctypes.c_size_t) for name in
                      ('peak_working', 'working', 'peak_paged', 'paged', 'peak_nonpaged',
                       'nonpaged', 'pagefile', 'peak_pagefile', 'private')]]

    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = (wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD)
    counters = Counters()
    counters.size = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.size):
        raise ctypes.WinError(ctypes.get_last_error())
    return counters.peak_pagefile
