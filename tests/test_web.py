"""Exercise the local web API through HTTP and the real simulation worker."""
from __future__ import annotations

import csv
import gc
import http.client
import io
import json
from pathlib import Path
import threading
import time

import pytest

from supernode_sim.analysis import run
from supernode_sim.config import load_config


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("collector_enabled", [True, False])
def test_cli_web_preserves_caller_gc_state_and_forwards_arguments(monkeypatch, tmp_path, collector_enabled):
    from supernode_sim.cli import main
    import supernode_sim.web as web

    forwarded = []

    def fake_web_main(arguments):
        forwarded.append((arguments, gc.isenabled()))
        return 29

    monkeypatch.setattr(web, "main", fake_web_main)
    options = ["--host", "127.0.0.1", "--port", "8123", "--output", str(tmp_path / "web jobs")]
    original_state = gc.isenabled()
    try:
        (gc.enable if collector_enabled else gc.disable)()
        assert main(["web", *options]) == 29
        assert forwarded == [(options, collector_enabled)]
        assert gc.isenabled() is collector_enabled
    finally:
        (gc.enable if original_state else gc.disable)()


class WebClient:
    def __init__(self, server):
        self.server = server
        self.host, self.port = server.server_address[:2]
        self.origin = f"http://{self.host}:{self.port}"

    def request(self, method, path, payload=None, *, raw=None, headers=None):
        body = raw
        request_headers = dict(headers or {})
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            request_headers.setdefault("Content-Type", "application/json")
        connection = http.client.HTTPConnection(self.host, self.port, timeout=15)
        try:
            connection.request(method, path, body=body, headers=request_headers)
            response = connection.getresponse()
            content = response.read()
            response_headers = dict(response.getheaders())
            if "application/json" in response.getheader("Content-Type", ""):
                content = json.loads(content)
            return response.status, content, response_headers
        finally:
            connection.close()

    def json(self, method, path, payload=None, *, status=200, **kwargs):
        actual_status, content, _ = self.request(method, path, payload, **kwargs)
        assert actual_status == status, content
        assert isinstance(content, dict), content
        return content


@pytest.fixture
def client(tmp_path):
    from supernode_sim.web import create_server

    server = create_server("127.0.0.1", 0, tmp_path / "runs")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield WebClient(server)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


def demo(scenario="inference"):
    return load_config(ROOT / "examples" / f"demo_{scenario}.yaml").model_dump()


def finish(client, job, *, expected="completed"):
    deadline = time.monotonic() + 30
    while job["status"] in ("queued", "running"):
        assert time.monotonic() < deadline, job
        time.sleep(0.05)
        job = client.json("GET", f"/api/jobs/{job['id']}")
    assert job["status"] == expected, job
    return job


def submit(client, config, mode="simulate"):
    return client.json("POST", "/api/jobs", {"config": config, "mode": mode}, status=202)


def test_catalog_exposes_runnable_presets_and_preserves_sources(client):
    catalog = client.json("GET", "/api/presets")["presets"]
    ids = {entry["id"] for entry in catalog}
    assert {"demo_inference", "demo_training"} <= ids
    assert "sweep_precision" not in ids
    for preset in ("demo_inference", "demo_training"):
        config = client.json("GET", f"/api/presets/{preset}")["config"]
        assert config == demo(config["workload"]["scenario"])
        assert config["sources"]["*"]["status"] == "assumed"
        assert client.json("POST", "/api/validate", {"config": config})["valid"]


@pytest.mark.parametrize("scenario", ["inference", "training"])
def test_real_simulation_metrics_trace_and_downloads_match_engine(client, scenario):
    config = client.json("GET", f"/api/presets/demo_{scenario}")["config"]
    expected = run(config)
    job = finish(client, submit(client, config))
    assert job["result_status"] == expected["status"]
    assert job["elapsed_s"] >= 0
    assert job["logs"]
    assert job["id"] in {entry["id"] for entry in client.json("GET", "/api/jobs")["jobs"]}
    assert client.json("GET", job["config_url"])["config"] == config

    result = client.json("GET", f"/api/jobs/{job['id']}/result")
    assert result["status"] == expected["status"]
    assert result["config_sha256"] == expected["config_sha256"]
    for key, value in result["metrics"].items():
        assert value == expected["metrics"][key], key
    assert result["metrics"]["duration_s"] > 0
    assert result["communication"] == expected["estimate"]["communication"]
    assert result["trace"]["total_events"] == expected["simulation"]["completed_events"]
    assert result["trace"]["events"]
    assert all(0 <= event["start_s"] <= event["end_s"] <= result["metrics"]["duration_s"]
               for event in result["trace"]["events"])
    expected_trace = {row["id"]: row for row in expected["simulation"]["trace"]}
    for event in result["trace"]["events"]:
        assert event["start_s"] == expected_trace[event["id"]]["start_s"]
        assert event["end_s"] == expected_trace[event["id"]]["end_s"]
    for pool in result["memory"]["pools"]:
        assert pool["peak_bytes"] == expected["simulation"]["memory_peaks_bytes"].get(pool["pool"], 0)

    artifacts = {entry["name"]: entry["url"] for entry in job["artifacts"]}
    if scenario == 'inference':
        assert result['metrics']['latency_breakdown'] == expected['metrics']['latency_breakdown']
        assert result['metrics']['tpop_s'] == result['metrics']['tpot_s']
        assert 'latency_breakdown.csv' in artifacts
    assert {"report.md", "result.json", "config.resolved.json", "resources.csv", "events.csv", "timeline.json"} <= artifacts.keys()
    saved_result = client.json("GET", artifacts["result.json"])
    assert saved_result["metrics"] == expected["metrics"]
    saved_config = client.json("GET", artifacts["config.resolved.json"])
    assert saved_config == expected["config"]
    timeline = client.json("GET", artifacts["timeline.json"])["traceEvents"]
    assert len(timeline) == expected["simulation"]["completed_events"]
    assert all(row["ph"] == "X" and row["dur"] >= 0 for row in timeline)
    status, table, headers = client.request("GET", artifacts["resources.csv"])
    assert status == 200
    assert "attachment" in headers["Content-Disposition"]
    assert list(csv.DictReader(io.StringIO(table.decode("utf-8-sig"))))

    for suffix in ("../job.json", "%2e%2e%2fjob.json", "job.json", "ui.result.json"):
        status, _, _ = client.request("GET", f"/api/jobs/{job['id']}/artifacts/{suffix}")
        assert status in (403, 404)


def test_estimate_remains_a_lower_bound_without_simulated_slo_or_timeline(client):
    config = demo()
    actual_simulation = run(config)
    job = finish(client, submit(client, config, "estimate"))
    result = client.json("GET", f"/api/jobs/{job['id']}/result")
    assert job["result_status"] == result["status"] == "estimated"
    assert result['metrics']['ttft_s'] is None
    assert result['metrics']['tpop_s'] is None
    assert result['metrics']['latency_breakdown']['unavailable_reason'] == 'estimate_only'
    assert result["slo"]["satisfied"] is None
    assert result["trace"]["events"] == []
    assert result["trace"]["total_events"] == 0
    assert result["simulation"]["duration_s"] is None
    assert 0 < result["estimate"]["duration_lower_bound_s"] <= actual_simulation["metrics"]["duration_s"]
    assert "timeline.json" not in {artifact["name"] for artifact in job["artifacts"]}


def test_infeasible_capacity_is_a_completed_result_with_diagnostics(client):
    config = demo()
    config["hardware"]["npu"]["hbm_bytes"] = 1024
    job = finish(client, submit(client, config))
    result = client.json("GET", f"/api/jobs/{job['id']}/result")
    assert result["status"] == job["result_status"] == "infeasible"
    assert result['metrics']['ttft_s'] is None
    assert result['metrics']['tpop_s'] is None
    assert result['metrics']['latency_breakdown']['unavailable_reason'] == 'execution_incomplete'
    assert result["config"]["hardware"]["npu"]["hbm_bytes"] == 1024
    assert any(row["code"] == "initial_memory_exceeded" for row in result["diagnostics"])


def test_slo_failure_preserves_a_successful_simulation_and_its_metrics(client):
    config = demo()
    config["targets"]["ttft_s"] = 1e-30
    job = finish(client, submit(client, config))
    result = client.json("GET", f"/api/jobs/{job['id']}/result")
    assert job["result_status"] == result["status"] == "slo_failed"
    assert result["slo"]["satisfied"] is False
    assert result["metrics"]["ttft_s"] > config["targets"]["ttft_s"]
    assert result["metrics"]["duration_s"] > 0
    artifact = next(entry for entry in job["artifacts"] if entry["name"] == "result.json")
    complete = client.json("GET", artifact["url"])
    assert complete["simulation"]["status"] == "feasible"
    assert complete["simulation"]["completed_events"] > 0


def test_completed_job_and_artifacts_survive_server_restart(client):
    from supernode_sim.web import create_server

    config = demo()
    job = finish(client, submit(client, config))
    previous_result = client.json("GET", f"/api/jobs/{job['id']}/result")
    artifact_urls = {entry["name"]: entry["url"] for entry in job["artifacts"]}
    previous_metrics = client.json("GET", artifact_urls["result.json"])["metrics"]
    output = client.server.manager.output
    client.server.shutdown()
    client.server.server_close()

    restarted = create_server("127.0.0.1", 0, output)
    thread = threading.Thread(target=restarted.serve_forever, daemon=True)
    thread.start()
    try:
        fresh = WebClient(restarted)
        history = fresh.json("GET", "/api/jobs")["jobs"]
        assert len(history) == 1
        assert history[0]["id"] == job["id"]
        assert history[0]["status"] == "completed"
        assert history[0]["finished_at"] == job["finished_at"]
        assert fresh.json("GET", f"/api/jobs/{job['id']}/result") == previous_result
        assert fresh.json("GET", job["config_url"])["config"] == config
        assert fresh.json("GET", artifact_urls["config.resolved.json"]) == previous_result["config"]
        assert fresh.json("GET", artifact_urls["result.json"])["metrics"] == previous_metrics
    finally:
        restarted.shutdown()
        restarted.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


def test_worker_failure_reports_the_error_and_releases_the_next_job(client):
    config = demo()
    config["hardware"].update(topology="explicit", routes={}, explicit_links=[{
        "id": "one_way", "src": "npu0", "dst": "npu1", "domain": "SU", "bandwidth_Bps": 1e9,
    }])
    assert client.json("POST", "/api/validate", {"config": config})["valid"]
    failed = finish(client, submit(client, config), expected="failed")
    assert "No directed network route" in failed["error"]
    client.json("GET", f"/api/jobs/{failed['id']}/result", status=409)
    finish(client, submit(client, demo(), "estimate"))


def test_busy_job_can_be_cancelled_and_immediately_replaced(client):
    config = demo("training")
    config["workload"]["training_steps"] = 200
    running = submit(client, config)
    client.json("GET", f"/api/jobs/{running['id']}/result", status=409)
    client.json("POST", "/api/jobs", {"config": demo(), "mode": "estimate"}, status=409)
    cancelled = client.json("POST", f"/api/jobs/{running['id']}/cancel", {})
    assert cancelled["status"] == "cancelled"
    assert cancelled["finished_at"]
    finish(client, submit(client, demo(), "estimate"))


def test_validation_rejects_parallel_mismatch_without_changing_hardware(client):
    config = demo()
    config["execution"]["tp"] = 1
    client.json("POST", "/api/validate", {"config": config}, status=400)
    client.json("POST", "/api/jobs", {"config": config, "mode": "simulate"}, status=400)
    config["execution"]["tp"] = 2
    valid = client.json("POST", "/api/validate", {"config": config})
    assert valid["config"]["hardware"]["npu_count"] == 4
    assert valid["config"]["execution"]["tp"] == 2


def test_validation_rejects_external_calibration_and_unknown_fields(client):
    for patch in ({"calibration": "../private.yaml"}, {"unknown_field": True}):
        config = demo()
        config.update(patch)
        client.json("POST", "/api/validate", {"config": config}, status=400)


@pytest.mark.parametrize("batch_size", [
    pytest.param(100001, id="first-over-limit"),
    pytest.param("100001", id="first-over-limit-string"),
    pytest.param(1e9, id="billion-numeric"),
    pytest.param("1000000000", id="billion-string"),
    pytest.param(10 ** 400, id="arbitrary-precision-integer"),
])
def test_web_batch_cap_rejects_oversized_requests_before_schema_expansion(client, batch_size):
    config = demo()
    config["workload"]["batch_size"] = batch_size
    # Keep tiny explicit lengths so a regression cannot allocate a huge list.
    # The cap-specific response distinguishes this guard from a length error.
    config["workload"]["input_lengths"] = [3]
    config["workload"]["output_lengths"] = [2]
    for endpoint in ("/api/validate", "/api/jobs"):
        response = client.json("POST", endpoint, {"config": config, "mode": "estimate"}, status=400)
        assert "100000" in response["error"]
        assert "CLI" in response["error"]
    assert client.json("GET", "/api/jobs")["jobs"] == []


def test_web_batch_cap_allows_ordinary_default_length_expansion(client):
    config = demo()
    config["workload"].update(batch_size=101, input_length=3, output_length=2,
                              input_lengths=[], output_lengths=[])
    result = client.json("POST", "/api/validate", {"config": config})
    assert result["valid"]
    workload = result["config"]["workload"]
    assert workload["batch_size"] == 101
    assert workload["input_lengths"] == [3] * 101
    assert workload["output_lengths"] == [2] * 101


def test_api_rejects_malformed_requests_and_unknown_jobs(client):
    client.json("POST", "/api/validate", raw=b"{", headers={"Content-Type": "application/json"}, status=400)
    client.json("POST", "/api/validate", raw=b"[]", headers={"Content-Type": "application/json"}, status=400)
    client.json("POST", "/api/jobs", {"config": demo(), "mode": "feedback"}, status=400)
    client.json("GET", "/api/jobs/not-a-job", status=404)
    client.json("GET", "/api/jobs/not-a-job/result", status=404)


@pytest.mark.parametrize("numeric_literal", ["NaN", "Infinity", "1e999"])
def test_nonfinite_json_numbers_are_rejected_as_client_errors(client, numeric_literal):
    config = demo()
    config["model"]["total_params"] = float("inf")
    body = json.dumps({"config": config}).replace("Infinity", numeric_literal).encode()
    client.json("POST", "/api/validate", raw=body,
                headers={"Content-Type": "application/json"}, status=400)


def test_malformed_origin_is_rejected_as_a_client_error(client):
    client.json("POST", "/api/validate", {"config": demo()}, headers={"Origin": "http://["}, status=403)


def test_loopback_http_checks_origin_and_host(client):
    payload = {"config": demo()}
    client.json("POST", "/api/validate", payload, headers={"Origin": client.origin})
    for headers in (
        {"Origin": "https://external.example"},
        {"Origin": "null"},
        {"Sec-Fetch-Site": "cross-site"},
        {"Host": f"external.example:{client.port}"},
    ):
        client.json("POST", "/api/validate", payload, headers=headers, status=403)


@pytest.mark.parametrize("path", [
    "/../pyproject.toml",
    "/%2e%2e/pyproject.toml",
    "/%2e%2e%5cpyproject.toml",
    "/api/presets/%2e%2e%2fpyproject",
])
def test_static_and_preset_routes_do_not_expose_other_files(client, path):
    status, content, _ = client.request("GET", path)
    assert status in (403, 404), content
    assert b"build-system" not in (content if isinstance(content, bytes) else json.dumps(content).encode())

