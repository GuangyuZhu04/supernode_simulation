#!/usr/bin/env python3
"""Find the engine, reuse or start its buildless local UI, and print one JSON result.

This copied skill entrypoint uses only the standard library. It never installs
dependencies, opens a browser, starts a simulation, or stops an existing server.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener
import uuid

APP_ID = 'supernode_sim.web'
ASSETS = ('index.html', 'app.js', 'topology.js', 'execution.js', 'styles.css')
_HTTP = build_opener(ProxyHandler({}))  # Local readiness must not use HTTP proxies.
_PROBE = r'''
import importlib.util, json, pathlib, sys
result = {'ok': False, 'python': sys.executable}
try:
    if sys.version_info < (3, 11):
        raise RuntimeError('Python >= 3.11 is required')
    missing = [name for name in ('pydantic', 'yaml') if importlib.util.find_spec(name) is None]
    if missing:
        result['missing_dependencies'] = missing
        raise RuntimeError('Missing engine dependencies: ' + ', '.join(missing))
    import supernode_sim
    import supernode_sim.web
    directory = pathlib.Path(supernode_sim.web.__file__).resolve().parent
    result.update(root=str(directory.parent), version=supernode_sim.__version__)
    missing_assets = [str(directory / 'web_static' / name) for name in ('index.html', 'app.js', 'topology.js', 'execution.js', 'styles.css')
                      if not (directory / 'web_static' / name).is_file() or not (directory / 'web_static' / name).stat().st_size]
    if missing_assets:
        result['missing_assets'] = missing_assets
        raise RuntimeError('Frontend static assets are missing or empty')
    result['ok'] = True
except Exception as exc:
    result['error'] = type(exc).__name__ + ': ' + str(exc)
print(json.dumps(result, ensure_ascii=False))
'''


class LaunchError(Exception):
    def __init__(self, message: str, **details):
        self.details = details
        super().__init__(message)


def _is_project(path: Path) -> bool:
    return (path / 'pyproject.toml').is_file() and (path / 'supernode_sim' / '__init__.py').is_file()


def discover_project(explicit=None) -> Path | None:
    """A copied skill directory is not itself an engine checkout."""
    for value, source in ((explicit, '--project-root'), (os.environ.get('SUPERNODE_SIM_ROOT'), 'SUPERNODE_SIM_ROOT')):
        if value:
            candidate = Path(value).expanduser().resolve()
            if not _is_project(candidate):
                raise LaunchError(f'{source} is not an engine checkout: {candidate}', root=str(candidate),
                                  required=['pyproject.toml', 'supernode_sim/__init__.py'])
            return candidate
    for start in (Path.cwd().resolve(), Path(__file__).resolve().parent):
        for candidate in (start, *start.parents):
            if _is_project(candidate):
                return candidate
    return None


def select_python(root: Path | None) -> Path:
    if root is not None:
        candidates = (root / '.venv' / 'Scripts' / 'python.exe', root / '.venv' / 'bin' / 'python')
        if os.name != 'nt':
            candidates = tuple(reversed(candidates))
        for candidate in candidates:
            if candidate.is_file():
                # Preserve the venv executable path; resolving a POSIX symlink
                # to the base interpreter would lose its virtual environment.
                return candidate.absolute()
    return Path(sys.executable).absolute()


def _environment(root: Path | None) -> dict:
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    if root is not None:
        env['PYTHONPATH'] = str(root) + os.pathsep + env.get('PYTHONPATH', '')
    return env


def _command(argv: list[str]) -> str:
    if os.name == 'nt':
        return '& ' + ' '.join("'" + arg.replace("'", "''") + "'" for arg in argv)
    return shlex.join(argv)


def _repair(python: Path, root: Path | None) -> dict:
    argv = ([str(python), '-m', 'pip', 'install', '-e', str(root)] if root is not None else
            [str(python), '-m', 'pip', 'install', 'pydantic>=2.7,<3', 'PyYAML>=6'])
    return {'repair_argv': argv, 'repair_command': _command(argv),
            'repair_command_shell': 'powershell' if os.name == 'nt' else 'sh'}


def _preflight(python: Path, root: Path | None) -> dict:
    try:
        completed = subprocess.run([str(python), '-c', _PROBE], cwd=str(root or Path.cwd()), env=_environment(root),
                                   capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=10,
                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LaunchError(f'Could not check the project Python: {exc}', python=str(python), **_repair(python, root)) from exc
    try:
        probe = json.loads(completed.stdout)
    except ValueError:
        raise LaunchError('Python preflight did not return JSON', python=str(python),
                          diagnostic=(completed.stderr or completed.stdout)[-4000:], **_repair(python, root)) from None
    if not isinstance(probe, dict) or not probe.get('ok') or completed.returncode:
        details = {key: probe[key] for key in ('missing_assets', 'missing_dependencies') if isinstance(probe, dict) and key in probe}
        if 'missing_assets' in details:
            details['repair'] = 'Restore supernode_sim/web_static from this engine checkout; these assets need no Node build.'
        else:
            details.update(_repair(python, root))
        message = probe.get('error', 'Engine preflight failed') if isinstance(probe, dict) else 'Invalid preflight result'
        if 'No module named \'supernode_sim\'' in message:
            details['repair'] = 'Pass --project-root pointing to the engine checkout, or use Python with supernode-sim installed.'
        raise LaunchError(message, python=str(python), **details)
    if root is not None and not _same_path(probe.get('root', ''), root):
        raise LaunchError('Python resolved a different engine checkout', root=str(root), actual_root=probe.get('root'))
    return probe


def _same_path(left, right) -> bool:
    try:
        return os.path.normcase(str(Path(left).resolve())) == os.path.normcase(str(Path(right).resolve()))
    except (TypeError, ValueError, OSError):
        return False


def _read_json(url: str, timeout: float = .45):
    try:
        with _HTTP.open(Request(url, headers={'Accept': 'application/json'}), timeout=timeout) as response:
            if response.status != 200:
                return None
            raw = response.read(262_145)
            if len(raw) > 262_144:
                return None
            return json.loads(raw)
    except (OSError, ValueError, HTTPError, URLError):
        return None


def _identity(health, root: Path, output: Path) -> bool:
    return (isinstance(health, dict) and health.get('app') == APP_ID and health.get('static_ready') is True
            and _same_path(health.get('root'), root) and _same_path(health.get('output'), output))


def _ready(url: str, root: Path, output: Path):
    health = _read_json(url + 'api/health')
    if not _identity(health, root, output):
        return None
    presets = _read_json(url + 'api/presets')
    if not isinstance(presets, dict) or not isinstance(presets.get('presets'), list):
        return None
    for suffix in ASSETS:
        try:
            with _HTTP.open(url + suffix, timeout=.45) as response:
                if response.status != 200 or not response.read(256):
                    return None
        except (OSError, HTTPError, URLError):
            return None
    return health


def _available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        try:
            listener.bind(('127.0.0.1', port))
            return True
        except OSError:
            return False


def _result(status: str, url: str, root: Path, python: Path, output: Path, health: dict) -> dict:
    log = output / 'service.log'
    return {'status': status, 'url': url, 'inputs_url': url + '#inputs', 'workspace_url': url + '#workspace',
            'root': str(root), 'python': health.get('python') or str(python), 'output': str(output),
            'pid': health.get('pid'), 'log': str(log) if log.is_file() else None}


def _stop_owned(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)


def _tail(path: Path) -> str:
    try:
        with path.open('rb') as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - 4000))
            return handle.read().decode('utf-8', errors='replace')
    except OSError:
        return ''


def launch(project_root=None, port=8765, port_attempts=10, startup_timeout=12, output_root=None) -> dict:
    """Return a machine-readable result without modifying existing servers."""
    context = {}
    try:
        if not isinstance(port, int) or not 1 <= port <= 65535:
            raise LaunchError('port must be an integer from 1 to 65535')
        if not isinstance(port_attempts, int) or not 1 <= port_attempts <= 20:
            raise LaunchError('port-attempts must be an integer from 1 to 20')
        if not isinstance(startup_timeout, (int, float)) or not 1 <= startup_timeout <= 15:
            raise LaunchError('startup-timeout must be between 1 and 15 seconds')
        project = discover_project(project_root)
        python = select_python(project)
        context.update(python=str(python))
        if project:
            context['root'] = str(project)
        probe = _preflight(python, project)
        root = Path(probe['root']).resolve()
        context['root'] = str(root)
        base = Path(output_root).expanduser().resolve() if output_root else (project or Path.cwd()) / 'outputs' / 'skill-web'
        for candidate in range(port, min(65536, port + port_attempts)):
            url = f'http://127.0.0.1:{candidate}/'
            output = (base / str(candidate)).resolve()
            context.update(url=url, inputs_url=url + '#inputs', output=str(output))
            health = _ready(url, root, output)
            if health:
                return _result('reused', url, root, python, output, health)
            if not _available(candidate):
                continue
            output.mkdir(parents=True, exist_ok=True)
            lock = output / '.launch.lock'
            token = uuid.uuid4().hex
            deadline = time.monotonic() + startup_timeout
            try:
                descriptor = os.open(str(lock), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                while time.monotonic() < deadline:
                    health = _ready(url, root, output)
                    if health:
                        return _result('reused', url, root, python, output, health)
                    time.sleep(.15)
                raise LaunchError('Another launch owns this output directory; inspect the lock before retrying.', lock=str(lock))
            with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
                json.dump({'token': token, 'pid': os.getpid()}, handle)
            process = None
            ready = False
            log = output / 'service.log'
            context['log'] = str(log)
            try:
                # Recheck after obtaining the output lock to handle concurrent launchers.
                health = _ready(url, root, output)
                if health:
                    return _result('reused', url, root, python, output, health)
                if not _available(candidate):
                    continue
                command = [str(python), '-u', '-m', 'supernode_sim.web', '--host', '127.0.0.1',
                           '--port', str(candidate), '--output', str(output)]
                options = {'cwd': str(project or Path.cwd()), 'env': _environment(project), 'stdin': subprocess.DEVNULL,
                           'stderr': subprocess.STDOUT, 'close_fds': True}
                options['env']['SUPERNODE_WEB_LAUNCH_TOKEN'] = token
                if os.name == 'nt':
                    options['creationflags'] = (subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS)
                else:
                    options['start_new_session'] = True
                with log.open('ab') as handle:
                    process = subprocess.Popen(command, stdout=handle, **options)
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise LaunchError('Frontend server exited during startup', exit_code=process.returncode,
                                          diagnostic=_tail(log))
                    health = _ready(url, root, output)
                    # Windows venv python.exe is a redirector: the HTTP process
                    # may have a different PID. The inherited random token
                    # identifies this launch without mistaking an older server.
                    if health and health.get('launch_token') == token:
                        ready = True
                        return _result('started', url, root, python, output, health)
                    time.sleep(.15)
                raise LaunchError('Frontend startup timed out; the process started by this call was stopped.', diagnostic=_tail(log))
            finally:
                if process is not None and not ready:
                    _stop_owned(process)
                try:
                    if json.loads(lock.read_text(encoding='utf-8')).get('token') == token:
                        lock.unlink()
                except (OSError, ValueError):
                    pass
        raise LaunchError(f'No reusable or free loopback port in {port}–{min(65535, port + port_attempts - 1)}; existing servers were left running.')
    except LaunchError as exc:
        return {'status': 'error', **context, 'error': str(exc), **exc.details}
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return {'status': 'error', **context, 'error': f'{type(exc).__name__}: {exc}'}


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise LaunchError(message)


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if callable(getattr(stream, 'reconfigure', None)):
            stream.reconfigure(encoding='utf-8', errors='replace')
    parser = _Parser(description=__doc__)
    parser.add_argument('--project-root')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--port-attempts', type=int, default=10)
    parser.add_argument('--startup-timeout', type=float, default=12)
    parser.add_argument('--output-root', '--output', dest='output_root', help='Base directory; each port gets its own subdirectory')
    try:
        args = parser.parse_args(argv)
        result = launch(**vars(args))
    except LaunchError as exc:
        result = {'status': 'error', 'error': str(exc), **exc.details}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0 if result['status'] in ('started', 'reused') else 1


if __name__ == '__main__':
    raise SystemExit(main())
