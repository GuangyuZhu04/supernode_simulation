"""Local, dependency-free HTTP interface to the existing simulation engine.

Run ``python -m supernode_sim.web`` and open the printed loopback URL. Each
numerical task runs in its own cancellable process; no engine is reimplemented.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
import math
import mimetypes
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
from urllib.parse import unquote, urlsplit
import uuid

from pydantic import ValidationError

from .config import Config, load_config, read_data
from .serving import ServingSpec

MAX_BODY_BYTES = 2 * 1024 * 1024
MAX_WEB_BATCH_SIZE = 100_000
SERVING_WEB_LIMITS = {'request_count': 10_000, 'max_batch_size': 1024,
                      'instance_count': 128, 'body_bytes': MAX_BODY_BYTES}
JOB_ID = re.compile(r'^[0-9a-f]{32}$')
ACTIVE = {'queued', 'running'}


class APIError(Exception):
    def __init__(self, status: int, message: str, details=None):
        self.status, self.message, self.details = status, message, details
        super().__init__(message)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_read(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def _atomic_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def validate_config(value) -> dict:
    if not isinstance(value, dict):
        raise APIError(400, 'config 必须是 JSON 对象')
    if value.get('calibration') is not None:
        raise APIError(400, '网页输入不支持读取 calibration 文件；请使用已解析的硬件参数')
    workload = value.get('workload')
    if isinstance(workload, dict):
        # WorkloadSpec expands default lengths to batch_size during validation,
        # before the cancellable worker exists. Bound that allocation in HTTP.
        try:
            raw_batch_size = workload.get('batch_size', 1)
            batch_size = raw_batch_size if isinstance(raw_batch_size, int) else float(raw_batch_size)
        except (TypeError, ValueError, OverflowError):
            batch_size = 0  # Let the engine schema report invalid value types.
        if batch_size > MAX_WEB_BATCH_SIZE:
            raise APIError(400, '网页交互模式的 batch_size 上限为 100000；更大批次请通过 CLI 运行。配置未被修改。')
    try:
        return Config.model_validate(value).model_dump(mode='json')
    except ValidationError as exc:
        raise APIError(400, '配置校验失败', exc.errors(include_url=False, include_context=False, include_input=False)) from exc
    except (ValueError, TypeError) as exc:
        raise APIError(400, str(exc)) from exc


def validate_inputs(config, mode='simulate', serving=None) -> tuple[dict, dict | None]:
    """Validate HTTP-only budgets before invoking the engine's strict schemas."""
    if mode not in ('simulate', 'estimate', 'serve'):
        raise APIError(400, 'mode 必须是 simulate、estimate 或 serve')
    if mode != 'serve' and serving is not None:
        raise APIError(400, 'serving 只能与 mode=serve 一起使用；服务输入未被忽略或修改')
    resolved = validate_config(config)
    if mode != 'serve':
        return resolved, None
    if not isinstance(serving, dict):
        raise APIError(400, 'serve 模式需要独立的 serving JSON 对象')
    requests = serving.get('requests')
    if isinstance(requests, list) and len(requests) > SERVING_WEB_LIMITS['request_count']:
        raise APIError(400, '网页 serving.requests 上限为 10000 条；更大请求 trace 请使用 CLI。输入未被修改。')
    for key in ('max_batch_size', 'instance_count'):
        value = serving.get(key)
        if isinstance(value, int) and value > SERVING_WEB_LIMITS[key]:
            raise APIError(400, f'网页 serving.{key} 上限为 {SERVING_WEB_LIMITS[key]}；更大实验请使用 CLI。输入未被修改。')
    try:
        parsed = ServingSpec.model_validate(serving).model_dump(mode='json')
    except ValidationError as exc:
        raise APIError(400, '服务调度配置校验失败', exc.errors(include_url=False, include_context=False, include_input=False)) from exc
    execution = resolved['execution']
    if resolved['workload']['scenario'] != 'inference':
        raise APIError(400, 'serve 需要 inference 模型配置；训练场景请使用 simulate')
    if execution['dp'] != 1:
        raise APIError(400, 'serve 需要 DP=1；用 serving.instance_count 明确复制隔离硬件实例')
    features = execution['features']
    if features['speculative_decode']['enabled'] or features['prefill_decode_disaggregation']['enabled']:
        raise APIError(400, 'serve 暂不支持 speculative_decode 或 prefill_decode_disaggregation 调度')
    return resolved, parsed


class JobManager:
    def __init__(self, output: str | Path):
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.jobs: dict[str, dict] = {}
        self.processes: dict[str, subprocess.Popen] = {}
        self.threads: dict[str, threading.Thread] = {}
        self.closed = False
        for path in self.output.glob('*/job.json'):
            if not JOB_ID.fullmatch(path.parent.name) or path.parent.is_symlink():
                continue
            try:
                job = _json_read(path)
                if not isinstance(job, dict) or job.get('id') != path.parent.name:
                    continue
                if job.get('status') not in ACTIVE | {'completed', 'failed', 'cancelled'} or job.get('mode') not in ('simulate', 'estimate', 'serve'):
                    continue
                for field in ('created_at', 'started_at', 'finished_at'):
                    if field == 'created_at' or job.get(field):
                        timestamp = datetime.fromisoformat(job[field])
                        if timestamp.tzinfo is None:
                            raise ValueError('Job timestamps must include a time zone')
                if job.get('status') in ACTIVE:
                    job.update(status='failed', phase='failed', error='上一次服务中断；该任务没有完成', finished_at=_now())
                    _atomic_json(path, job)
                self.jobs[job['id']] = job
            except (OSError, ValueError, TypeError, KeyError):
                continue

    def _save(self, job: dict) -> None:
        _atomic_json(self.output / job['id'] / 'job.json', job)

    def _log(self, job: dict, message: str) -> None:
        job.setdefault('logs', []).append({'time': _now(), 'message': message[:3000]})
        job['logs'] = job['logs'][-100:]

    def directory(self, job_id: str) -> Path:
        if not JOB_ID.fullmatch(job_id) or job_id not in self.jobs:
            raise APIError(404, '未找到任务')
        path = self.output / job_id
        if path.resolve().parent != self.output:
            raise APIError(404, '未找到任务')
        return path

    def _view(self, job: dict) -> dict:
        view = copy.deepcopy(job)
        start = view.get('started_at') or view['created_at']
        end = view.get('finished_at') or _now()
        view['elapsed_s'] = max(0, (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds())
        directory = self.directory(job['id'])
        excluded = {'job.json', 'ui.result.json'}
        view['artifacts'] = [{'name': p.name, 'url': f"/api/jobs/{job['id']}/artifacts/{p.name}", 'size_bytes': p.stat().st_size}
                             for p in sorted(directory.iterdir()) if p.is_file() and p.resolve().parent == directory
                             and p.name not in excluded and not p.name.endswith('.tmp')]
        view['config_url'] = f"/api/jobs/{job['id']}/config"
        if job['mode'] == 'serve':
            view['serving_url'] = f"/api/jobs/{job['id']}/serving"
        return view

    def get(self, job_id: str) -> dict:
        with self.lock:
            self.directory(job_id)
            return self._view(self.jobs[job_id])

    def list(self) -> list[dict]:
        with self.lock:
            return [self._view(job) for job in sorted(self.jobs.values(), key=lambda j: j['created_at'], reverse=True)]

    def submit(self, config: dict, mode: str = 'simulate', serving=None) -> dict:
        config, serving = validate_inputs(config, mode, serving)
        with self.lock:
            if self.closed:
                raise APIError(503, '服务正在关闭')
            if self.processes or any(job['status'] in ACTIVE for job in self.jobs.values()):
                raise APIError(409, '已有任务正在运行，请等待完成或取消当前任务')
            job_id = uuid.uuid4().hex
            directory = self.output / job_id
            directory.mkdir()
            _atomic_json(directory / 'input.config.json', config)
            if serving is not None:
                _atomic_json(directory / 'input.serving.json', serving)
            job = {'id': job_id, 'status': 'queued', 'phase': 'queued', 'mode': mode, 'created_at': _now(),
                   'model_name': config['model']['name'], 'scenario': config['workload']['scenario'], 'logs': []}
            if serving is not None:
                job['serving_summary'] = {'request_count': len(serving['requests']),
                    'instance_count': serving['instance_count'], 'max_batch_size': serving['max_batch_size']}
            self.jobs[job_id] = job
            self._log(job, '配置校验通过，等待启动计算进程')
            self._save(job)
            thread = threading.Thread(target=self._execute, args=(job_id,), name=f'simulation-{job_id[:8]}', daemon=True)
            self.threads[job_id] = thread
            thread.start()
            return self._view(job)

    def _execute(self, job_id: str) -> None:
        process = None
        directory = self.output / job_id
        try:
            with self.lock:
                job = self.jobs[job_id]
                if job['status'] == 'cancelled' or self.closed:
                    return
                job.update(status='running', started_at=_now(), phase='validating')
                env = os.environ.copy()
                env['PYTHONIOENCODING'] = 'utf-8'
                package_root = str(Path(__file__).resolve().parent.parent)
                env['PYTHONPATH'] = package_root + os.pathsep + env.get('PYTHONPATH', '')
                process = subprocess.Popen([sys.executable, '-u', '-m', 'supernode_sim.web_worker', str(directory), job['mode']],
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                           text=True, encoding='utf-8', errors='replace', env=env, cwd=package_root,
                                           creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                self.processes[job_id] = process
                self._log(job, '计算进程已启动')
                self._save(job)
            with (directory / 'worker.log').open('w', encoding='utf-8') as logfile:
                for line in process.stdout:
                    logfile.write(line)
                    logfile.flush()
                    try:
                        progress = json.loads(line)
                    except (ValueError, TypeError):
                        continue
                    if not isinstance(progress, dict) or 'phase' not in progress:
                        continue
                    with self.lock:
                        if job['status'] == 'cancelled':
                            continue
                        if progress['phase'] in ('validating', 'simulating', 'estimating', 'serving', 'exporting', 'summarizing'):
                            job['phase'] = progress['phase']
                        if progress['phase'] == 'failed':
                            job['error'] = progress.get('message', '计算失败')
                        if progress.get('result_status'):
                            job['result_status'] = progress['result_status']
                        self._log(job, str(progress.get('message', progress['phase'])))
                        self._save(job)
            code = process.wait()
            with self.lock:
                # Once a terminal status is visible, the next job may start.
                self.processes.pop(job_id, None)
                if job['status'] != 'cancelled':
                    if code == 0 and (directory / 'ui.result.json').is_file():
                        job.update(status='completed', phase='completed')
                    else:
                        job.update(status='failed', phase='failed')
                        job.setdefault('error', f'计算进程退出（code={code}），详情见 worker.log')
                    job['finished_at'] = _now()
                    self._save(job)
        except Exception as exc:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            with self.lock:
                self.processes.pop(job_id, None)
                job = self.jobs[job_id]
                if job['status'] != 'cancelled':
                    job.update(status='failed', phase='failed', error=f'{type(exc).__name__}: {exc}', finished_at=_now())
                    self._log(job, job['error'])
                    self._save(job)
        finally:
            if process is not None and process.stdout is not None:
                process.stdout.close()
            with self.lock:
                self.processes.pop(job_id, None)
                self.threads.pop(job_id, None)

    def cancel(self, job_id: str) -> dict:
        with self.lock:
            self.directory(job_id)
            job = self.jobs[job_id]
            if job['status'] not in ACTIVE:
                return self._view(job)
            job.update(status='cancelled', phase='cancelled', finished_at=_now())
            self._log(job, '用户取消任务，计算进程已请求终止')
            process = self.processes.get(job_id)
            thread = self.threads.get(job_id)
            if process is not None and process.poll() is None:
                process.terminate()
            self._save(job)
        if process is not None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=6)
        return self.get(job_id)

    def close(self) -> None:
        with self.lock:
            self.closed = True
            active_ids = [key for key, job in self.jobs.items() if job['status'] in ACTIVE]
        for job_id in active_ids:
            self.cancel(job_id)
        with self.lock:
            threads = list(self.threads.values())
        for thread in threads:
            thread.join(timeout=6)


class WebServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, output, examples_dir=None, static_dir=None):
        self.manager = JobManager(output)
        self.static_dir = Path(static_dir or Path(__file__).parent / 'web_static').resolve()
        self.presets = {}
        self.serving_presets = {}
        examples = Path(examples_dir or Path(__file__).resolve().parent.parent / 'examples')
        for path in sorted(examples.glob('*')):
            if path.suffix.lower() not in ('.json', '.yaml', '.yml'):
                continue
            try:
                config = load_config(path).model_dump(mode='json')
                config['calibration'] = None
                self.presets[path.stem] = config
            except (OSError, ValueError, TypeError):
                continue  # Sweep axes and calibration fragments are not runnable scenarios.
        try:
            model = self.presets['serving_synthetic_model']
            model, serving = validate_inputs(model, 'serve', read_data(examples / 'serving_synthetic_requests.yaml'))
            self.serving_presets['serving_synthetic'] = {'config': model, 'serving': serving}
        except (KeyError, OSError, ValueError, TypeError, APIError):
            pass  # Installed packages may omit repository examples.
        super().__init__(address, WebHandler)

    def server_close(self) -> None:
        self.manager.close()
        super().server_close()


class WebHandler(BaseHTTPRequestHandler):
    server: WebServer
    protocol_version = 'HTTP/1.0'

    def setup(self):
        super().setup()
        self.connection.settimeout(30)

    def log_message(self, fmt, *args):
        # Polling requests are routine; the task log records numerical progress.
        pass

    def _json(self, value, status=200) -> None:
        data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self._headers()
        self.end_headers()
        if self.command != 'HEAD':
            self.wfile.write(data)

    def _headers(self) -> None:
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")

    def _check_request(self, mutation=False) -> None:
        try:
            host = urlsplit('http://' + self.headers.get('Host', ''))
            allowed = {'localhost', '127.0.0.1', self.server.server_address[0]}
            if host.hostname not in allowed or (host.port or 80) != self.server.server_address[1] or host.username or host.password or host.path:
                raise ValueError('host')
        except ValueError:
            raise APIError(403, '仅接受当前本机服务地址') from None
        if mutation:
            origin = self.headers.get('Origin')
            if origin is not None:
                try:
                    parsed = urlsplit(origin)
                except ValueError:
                    raise APIError(403, '拒绝无效来源请求') from None
                if parsed.scheme != 'http' or parsed.netloc.lower() != self.headers['Host'].lower() or parsed.path or parsed.query or parsed.fragment:
                    raise APIError(403, '拒绝跨来源请求')
            if self.headers.get('Sec-Fetch-Site', '').lower() == 'cross-site':
                raise APIError(403, '拒绝跨站请求')

    def _body(self) -> dict:
        if self.headers.get('Transfer-Encoding'):
            raise APIError(400, '不支持分块请求')
        if self.headers.get_content_type() != 'application/json':
            raise APIError(415, '请使用 Content-Type: application/json')
        try:
            length = int(self.headers.get('Content-Length', '0'))
        except ValueError:
            raise APIError(400, 'Content-Length 无效') from None
        if length < 0 or length > MAX_BODY_BYTES:
            raise APIError(413, '输入 JSON 不能超过 2 MiB')
        def invalid_constant(value):
            raise ValueError(f'JSON 不支持 {value}')
        def finite_float(value):
            parsed = float(value)
            if not math.isfinite(parsed):
                raise ValueError('JSON numbers must be finite')
            return parsed
        try:
            value = json.loads(self.rfile.read(length).decode('utf-8'), parse_constant=invalid_constant, parse_float=finite_float)
        except (ValueError, UnicodeError):
            raise APIError(400, '请求体必须是有效的 JSON') from None
        if not isinstance(value, dict):
            raise APIError(400, '请求体必须是 JSON 对象')
        return value

    def _file(self, path: Path, download=False) -> None:
        if not path.is_file():
            raise APIError(404, '文件不存在')
        content_type = mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
        if path.suffix == '.js':
            content_type = 'application/javascript'
        if path.suffix == '.css':
            content_type = 'text/css'
        with path.open('rb') as handle:
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(os.fstat(handle.fileno()).st_size))
            if download:
                self.send_header('Content-Disposition', f'attachment; filename="{path.name}"')
            self._headers()
            self.end_headers()
            if self.command != 'HEAD':
                shutil.copyfileobj(handle, self.wfile)

    def _route(self, mutation=False) -> None:
        self._check_request(mutation)
        path = unquote(urlsplit(self.path).path)
        if '\\' in path or '\x00' in path or any(p in ('.', '..') for p in path.split('/')):
            raise APIError(404, '路径无效')
        parts = path.strip('/').split('/')
        manager = self.server.manager
        if mutation:
            body = self._body()
            if path == '/api/validate':
                mode = body.get('mode', 'simulate')
                config, serving = validate_inputs(body.get('config'), mode, body.get('serving'))
                execution = config['execution']
                summary = {
                    'model': config['model']['name'], 'scenario': config['workload']['scenario'],
                    'npu_count': config['hardware']['npu_count'], 'su_size': config['hardware']['su_size'],
                    'parallel_product': execution['tp'] * execution['ep'] * execution['pp'] * execution['dp']}
                if serving is not None:
                    summary.update(request_count=len(serving['requests']), instance_count=serving['instance_count'],
                                   total_npus=config['hardware']['npu_count'] * serving['instance_count'])
                self._json({'valid': True, 'mode': mode, 'config': config, 'serving': serving, 'summary': summary})
                return
            if path == '/api/jobs':
                self._json(manager.submit(body.get('config'), body.get('mode', 'simulate'), body.get('serving')), 202)
                return
            if len(parts) == 4 and parts[:2] == ['api', 'jobs'] and parts[3] == 'cancel':
                self._json(manager.cancel(parts[2]))
                return
            raise APIError(404, '接口不存在')
        if path == '/api/presets':
            priority = {'demo_inference': 0, 'demo_training': 1}
            names = {'demo_inference': 'MoE 推理演示', 'demo_training': '稠密模型训练演示'}
            self._json({'presets': [{'id': key, 'name': names.get(key, key.replace('_', ' ')),
                                    'scenario': config['workload']['scenario'], 'model': config['model']['name'],
                                    'description': next((s.get('description', '') for s in config.get('sources', {}).values()), '')}
                                   for key, config in sorted(self.server.presets.items(), key=lambda pair: (priority.get(pair[0], 2), pair[0]))]})
            return
        if path == '/api/serving-presets':
            self._json({'presets': [{'id': key, 'name': '请求级 FCFS 成批演示',
                'model': value['config']['model']['name'], 'request_count': len(value['serving']['requests']),
                'description': '合成到达请求与隔离硬件；固定成批，不是 continuous batching。'}
                for key, value in self.server.serving_presets.items()], 'limits': SERVING_WEB_LIMITS})
            return
        if len(parts) == 3 and parts[:2] == ['api', 'serving-presets']:
            if parts[2] not in self.server.serving_presets:
                raise APIError(404, '服务预设不存在')
            self._json(self.server.serving_presets[parts[2]])
            return
        if path == '/api/health':
            from . import __version__
            self._json({'app': 'supernode_sim.web', 'version': __version__,
                        'root': str(Path(__file__).resolve().parent.parent), 'python': sys.executable,
                        'output': str(manager.output), 'pid': os.getpid(),
                        'launch_token': os.environ.get('SUPERNODE_WEB_LAUNCH_TOKEN'),
                        'static_ready': all((self.server.static_dir / name).is_file() and (self.server.static_dir / name).stat().st_size > 0
                                            for name in ('index.html', 'app.js', 'topology.js', 'execution.js', 'styles.css'))})
            return
        if len(parts) == 3 and parts[:2] == ['api', 'presets']:
            if parts[2] not in self.server.presets:
                raise APIError(404, '预设不存在')
            self._json({'config': self.server.presets[parts[2]]})
            return
        if path == '/api/schema':
            self._json(Config.model_json_schema())
            return
        if path == '/api/serving-schema':
            self._json({**ServingSpec.model_json_schema(), 'x-web-limits': SERVING_WEB_LIMITS})
            return
        if path == '/api/jobs':
            self._json({'jobs': manager.list()})
            return
        if len(parts) >= 3 and parts[:2] == ['api', 'jobs']:
            job = manager.get(parts[2])
            directory = manager.directory(parts[2])
            if len(parts) == 3:
                self._json(job)
                return
            if len(parts) == 4 and parts[3] == 'config':
                self._json({'config': _json_read(directory / 'input.config.json')})
                return
            if len(parts) == 4 and parts[3] == 'serving':
                if job['mode'] != 'serve' or not (directory / 'input.serving.json').is_file():
                    raise APIError(404, '该任务没有服务调度快照')
                self._json({'serving': _json_read(directory / 'input.serving.json')})
                return
            if len(parts) == 4 and parts[3] == 'result':
                if job['status'] != 'completed':
                    raise APIError(409, '任务尚未完成，当前没有完整结果')
                self._json(_json_read(directory / 'ui.result.json'))
                return
            if len(parts) == 5 and parts[3] == 'artifacts':
                allowed = {item['name'] for item in job['artifacts']}
                if parts[4] not in allowed:
                    raise APIError(404, '文件不存在')
                self._file(directory / parts[4], download=True)
                return
            raise APIError(404, '接口不存在')
        if parts[0] == 'api':
            raise APIError(404, '接口不存在')
        target = (self.server.static_dir / (path.lstrip('/') or 'index.html')).resolve()
        if not target.is_relative_to(self.server.static_dir):
            raise APIError(404, '文件不存在')
        self._file(target)

    def _handle(self, mutation=False) -> None:
        try:
            self._route(mutation)
        except APIError as exc:
            body = {'error': exc.message}
            if exc.details is not None:
                body['details'] = exc.details
            self._json(body, exc.status)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except Exception as exc:
            self._json({'error': f'服务处理失败：{type(exc).__name__}: {exc}'}, 500)

    def do_GET(self):
        self._handle()

    def do_HEAD(self):
        self._handle()

    def do_POST(self):
        self._handle(mutation=True)


def create_server(host='127.0.0.1', port=8765, output='outputs/web', *, examples_dir=None, static_dir=None) -> WebServer:
    if not isinstance(port, int) or not 0 <= port <= 65535:
        raise ValueError('端口必须是 0–65535 范围内的整数')
    if host != 'localhost':
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            raise ValueError('web 服务仅支持 localhost 或 IPv4 loopback 地址') from None
        if address.version != 4 or not address.is_loopback:
            raise ValueError('web 服务仅支持本机 IPv4 loopback 地址（例如 127.0.0.1）')
    return WebServer((host, port), output, examples_dir, static_dir)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='backslashreplace')
    parser = argparse.ArgumentParser(description='启动超节点流量仿真本地网页')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--output', default='outputs/web')
    args = parser.parse_args(argv)
    try:
        server = create_server(args.host, args.port, args.output)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f'超节点仿真工作台：http://{args.host}:{server.server_address[1]}', flush=True)
    print(f'结果目录：{server.manager.output}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
