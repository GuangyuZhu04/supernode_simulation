"""Launch/reuse the skill's local UI without touching a user's running server."""
from __future__ import annotations

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import uuid
from urllib.parse import urlsplit
from urllib.request import urlopen

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "skills" / "supernode-modeling" / "scripts" / "launch_frontend.py"


@pytest.fixture
def launcher():
    name = "frontend_launcher_test_" + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(name, None)


def project_at(path, *, static=True):
    package = path / "supernode_sim"
    package.mkdir(parents=True)
    (path / "pyproject.toml").write_text('[project]\nname = "supernode-sim"\n', encoding="utf-8")
    (package / "__init__.py").write_text("__version__ = '0.1.0'\n", encoding="utf-8")
    (package / "web.py").write_text("# Discovery fixture, never executed.\n", encoding="utf-8")
    if static:
        assets = package / "web_static"
        assets.mkdir()
        for filename in ("index.html", "app.js", "topology.js", "execution.js", "operators.js", "styles.css", "operators.css"):
            (assets / filename).write_text("fixture", encoding="utf-8")
    return path.resolve()


def available_port():
    while True:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        if port != 8765 and port < 65520:
            return port


def get_json(url):
    with urlopen(url, timeout=5) as response:
        assert response.status == 200
        return json.load(response)


@pytest.fixture
def owned_server_processes(launcher, monkeypatch):
    """Keep exact Popen handles, not names or PIDs belonging to other servers."""
    original_popen = subprocess.Popen
    started = []

    def recording_popen(args, *positional, **kwargs):
        process = original_popen(args, *positional, **kwargs)
        if isinstance(args, (list, tuple)) and "supernode_sim.web" in args:
            process.launcher_environment = dict(kwargs.get("env") or {})
            started.append(process)
        return process

    monkeypatch.setattr(launcher.subprocess, "Popen", recording_popen)
    try:
        yield started
    finally:
        for process in started:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


@contextmanager
def serving(server):
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


def test_project_discovery_honors_explicit_env_and_cwd_precedence(launcher, monkeypatch, tmp_path):
    explicit = project_at(tmp_path / "explicit")
    environment = project_at(tmp_path / "environment")
    current = project_at(tmp_path / "current")
    nested = current / "nested" / "working"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    monkeypatch.setenv("SUPERNODE_SIM_ROOT", str(environment))
    assert launcher.discover_project(explicit) == explicit
    assert launcher.discover_project() == environment
    monkeypatch.delenv("SUPERNODE_SIM_ROOT")
    assert launcher.discover_project() == current


def test_invalid_explicit_or_env_root_is_reported_instead_of_launching_other_project(launcher, monkeypatch, tmp_path):
    current = project_at(tmp_path / "current")
    monkeypatch.chdir(current)
    missing = tmp_path / "missing-project"
    with pytest.raises(launcher.LaunchError):
        launcher.discover_project(missing)
    monkeypatch.setenv("SUPERNODE_SIM_ROOT", str(missing))
    with pytest.raises(launcher.LaunchError):
        launcher.discover_project()


def test_bundled_skill_finds_its_engine_but_copied_skill_does_not_become_project_root(launcher, monkeypatch, tmp_path):
    project = project_at(tmp_path / "project")
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    monkeypatch.delenv("SUPERNODE_SIM_ROOT", raising=False)
    monkeypatch.setattr(launcher, "__file__", str(project / "skills" / "supernode-modeling" / "scripts" / "launch_frontend.py"))
    assert launcher.discover_project() == project
    copied = tmp_path / "global-skills" / "supernode-modeling" / "scripts" / "launch_frontend.py"
    monkeypatch.setattr(launcher, "__file__", str(copied))
    assert launcher.discover_project() is None
    monkeypatch.setenv("SUPERNODE_SIM_ROOT", str(project))
    assert launcher.discover_project() == project


def test_project_venv_interpreter_precedes_current_python(launcher, tmp_path):
    project = project_at(tmp_path / "project")
    windows = project / ".venv" / "Scripts" / "python.exe"
    unix = project / ".venv" / "bin" / "python"
    for interpreter in (windows, unix):
        interpreter.parent.mkdir(parents=True)
        interpreter.write_text("interpreter-selection fixture", encoding="utf-8")
    preferred = windows if os.name == "nt" else unix
    fallback = unix if os.name == "nt" else windows
    assert launcher.select_python(project) == preferred
    preferred.unlink()
    assert launcher.select_python(project) == fallback
    fallback.unlink()
    assert launcher.select_python(project).resolve() == Path(sys.executable).resolve()


def test_real_launch_serves_healthy_assets_and_reuses_the_exact_server(launcher, owned_server_processes, tmp_path):
    port = available_port()
    base = tmp_path / "isolated-ui"
    first = launcher.launch(project_root=ROOT, port=port, output_root=base)
    assert first["status"] == "started", first
    assert len(owned_server_processes) == 1
    process = owned_server_processes[0]
    # Windows venv python.exe is a redirector: HTTP PID may differ from Popen PID.
    assert first["inputs_url"] == first["url"] + "#inputs"
    assert first["workspace_url"] == first["url"] + "#workspace"
    health = get_json(first["url"] + "/api/health")
    assert health["app"] == "supernode_sim.web"
    assert health["static_ready"] is True
    assert health["pid"] == first["pid"]
    assert health["launch_token"] == process.launcher_environment["SUPERNODE_WEB_LAUNCH_TOKEN"]
    assert Path(health["root"]).resolve() == ROOT
    assert Path(health["output"]).resolve() == (base / str(port)).resolve()
    assert Path(health["python"]).resolve() == launcher.select_python(ROOT).resolve()
    for asset, expected_type in (("/", "text/html"), ("/app.js", "javascript"), ("/topology.js", "javascript"), ("/execution.js", "javascript"), ("/operators.js", "javascript"), ("/styles.css", "text/css"), ("/operators.css", "text/css")):
        with urlopen(first["url"] + asset, timeout=5) as response:
            assert response.status == 200
            assert expected_type in response.headers["Content-Type"]
            assert response.read()
    assert get_json(first["url"] + "/api/jobs")["jobs"] == []
    again = launcher.launch(project_root=ROOT, port=port, output_root=base)
    assert again["status"] == "reused", again
    assert again["url"] == first["url"]
    assert again["pid"] == first["pid"]
    assert len(owned_server_processes) == 1
    assert process.poll() is None


class OtherService(BaseHTTPRequestHandler):
    def do_GET(self):
        data = json.dumps({"app": "unrelated-local-service", "still_running": True}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *_args):
        pass


@pytest.mark.parametrize("occupied_kind", ["unrelated-service", "different-output"])
def test_occupied_port_chooses_another_without_stopping_its_owner(launcher, owned_server_processes, tmp_path, occupied_kind):
    from supernode_sim.web import create_server

    port = available_port()
    if occupied_kind == "different-output":
        existing = create_server("127.0.0.1", port, tmp_path / "foreign-output")
    else:
        existing = ThreadingHTTPServer(("127.0.0.1", port), OtherService)
    with serving(existing):
        foreign_url = f"http://127.0.0.1:{port}"
        before = get_json(foreign_url + "/api/health")
        result = launcher.launch(project_root=ROOT, port=port, output_root=tmp_path / "new-ui")
        assert result["status"] == "started", result
        assert urlsplit(result["url"]).port != port
        assert len(owned_server_processes) == 1
        assert get_json(foreign_url + "/api/health") == before
        launched_health = get_json(result["url"] + "/api/health")
        assert launched_health["pid"] == result["pid"]
        assert launched_health["launch_token"] == owned_server_processes[0].launcher_environment["SUPERNODE_WEB_LAUNCH_TOKEN"]


@pytest.mark.parametrize("missing_asset", [None, "topology.js", "execution.js", "operators.js", "operators.css"])
def test_missing_static_assets_report_paths_without_starting_a_server(launcher, owned_server_processes, tmp_path, missing_asset):
    project = project_at(tmp_path / "missing-assets", static=missing_asset is not None)
    if missing_asset:
        (project / "supernode_sim" / "web_static" / missing_asset).unlink()
    result = launcher.launch(project_root=project, port=available_port(), output_root=tmp_path / "unused-output")
    assert result["status"] == "error", result
    missing = {Path(path).name for path in result["missing_assets"]}
    assert missing == ({missing_asset} if missing_asset else {"index.html", "app.js", "topology.js", "execution.js", "operators.js", "styles.css", "operators.css"})
    assert "web_static" in result["repair"]
    assert owned_server_processes == []


def test_missing_dependencies_report_a_repair_command_without_installing_or_starting(launcher, owned_server_processes, monkeypatch, tmp_path):
    project = project_at(tmp_path / "dependency-free-project")
    original_run = subprocess.run
    commands = []

    def without_site_packages(args, **kwargs):
        # A real Python -S preflight supplies a dependency-free environment.
        commands.append(list(args))
        return original_run([args[0], "-S", *args[1:]], **kwargs)

    monkeypatch.setattr(launcher.subprocess, "run", without_site_packages)
    result = launcher.launch(project_root=project, port=available_port(), output_root=tmp_path / "unused-output")
    assert result["status"] == "error", result
    assert set(result["missing_dependencies"]) == {"pydantic", "yaml"}
    assert result["repair_argv"] == [str(launcher.select_python(project)), "-m", "pip", "install", "-e", str(project)]
    assert len(commands) == 1
    assert "pip" not in commands[0]
    assert owned_server_processes == []


def test_copied_skill_cli_runs_from_an_unrelated_directory_and_reuses_its_server(tmp_path):
    from supernode_sim.web import create_server

    copied = tmp_path / "copied skill" / "scripts" / "launch_frontend.py"
    copied.parent.mkdir(parents=True)
    copied.write_bytes(SCRIPT.read_bytes())
    cwd = tmp_path / "unrelated cwd"
    cwd.mkdir()
    port = available_port()
    base = tmp_path / "launch outputs with spaces"
    server = create_server("127.0.0.1", port, base / str(port))
    with serving(server):
        env = {**os.environ, "SUPERNODE_SIM_ROOT": str(ROOT), "PYTHONIOENCODING": "utf-8"}
        completed = subprocess.run([sys.executable, str(copied), "--port", str(port), "--output-root", str(base)],
                                   cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8", timeout=25)
        assert completed.returncode == 0, completed.stderr or completed.stdout
        result = json.loads(completed.stdout)
        assert result["status"] == "reused"
        assert Path(result["root"]).resolve() == ROOT
        assert Path(result["output"]).resolve() == (base / str(port)).resolve()
        assert result["pid"] == os.getpid()
        assert get_json(result["url"] + "api/jobs")["jobs"] == []


def test_startup_timeout_stops_only_the_process_it_created_and_closes_the_port(launcher, owned_server_processes, monkeypatch, tmp_path):
    port = available_port()
    # Withhold readiness even after the real server starts, exercising cleanup.
    monkeypatch.setattr(launcher, "_ready", lambda *_args: None)
    result = launcher.launch(project_root=ROOT, port=port, port_attempts=1, startup_timeout=1,
                             output_root=tmp_path / "timeout-ui")
    assert result["status"] == "error", result
    assert "timed out" in result["error"].lower()
    assert len(owned_server_processes) == 1
    assert owned_server_processes[0].poll() is not None
    deadline = time.monotonic() + 3
    while not launcher._available(port) and time.monotonic() < deadline:
        time.sleep(.05)
    assert launcher._available(port), "The failed launch left its server listening"



@pytest.mark.parametrize("missing_asset", ["topology.js", "execution.js", "operators.js", "operators.css"])
def test_health_reports_missing_visualization_script(launcher, tmp_path, missing_asset):
    from supernode_sim.web import create_server

    assets = project_at(tmp_path / "static-fixture") / "supernode_sim" / "web_static"
    (assets / missing_asset).unlink()
    server = create_server("127.0.0.1", 0, tmp_path / "health-output", static_dir=assets)
    with serving(server):
        url = f"http://127.0.0.1:{server.server_address[1]}/"
        assert get_json(url + "api/health")["static_ready"] is False
        (assets / missing_asset).write_text("fixture", encoding="utf-8")
        assert get_json(url + "api/health")["static_ready"] is True
