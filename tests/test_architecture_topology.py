"""Architecture hierarchy construction must preserve paths and shared limits."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from supernode_sim.config import HardwareSpec
from supernode_sim.core import Graph
from supernode_sim.engine import simulate
from supernode_sim.topology import Topology


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "skills/supernode-modeling/scripts/build_architecture_topology.py"
EXAMPLE = ROOT / "skills/supernode-modeling/assets/a3-topology.example.json"
module_spec = importlib.util.spec_from_file_location("architecture_topology", SCRIPT)
builder = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(builder)


def example():
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


def topology(spec=None):
    fragment = builder.build_topology_fragment(spec or example())
    hardware = HardwareSpec.model_validate(fragment["hardware"]).model_dump()
    topo, graph = Topology(hardware, {}), Graph()
    topo.populate(graph)
    return fragment, topo, graph


def test_hiam_l1_l2_are_distinct_paths_with_one_shared_host_uplink():
    fragment, topo, _ = topology()
    assert fragment["hardware"]["npu_count"] == 32
    assert fragment["hardware"]["npus_per_host"] == 16
    assert fragment["hardware"]["su_size"] == 32
    same_hiam = topo.path("npu0", "npu1")
    same_host = topo.path("npu0", "npu2")
    cross_host = topo.path("npu0", "npu16")
    assert len(same_hiam) == 1 and all(e.startswith("sio:") for e in same_hiam)
    assert len(same_host) == 2 and all(e.startswith("hccs_l1:") for e in same_host)
    assert len(cross_host) == 4
    assert [e.split(":")[0] for e in cross_host] == ["hccs_l1", "hccs_l2", "hccs_l2", "hccs_l1"]
    assert all(topo.links[e]["domain"] == "SU" for e in cross_host)
    assert "cpu0.nic.tx" not in topo.transfer("x", "npu0", "npu16", 100).work
    other_cross = topo.path("npu2", "npu18")
    shared = set(cross_host) & set(other_cross)
    assert len(shared) == 2 and all(e.startswith("hccs_l2:") for e in shared)


def test_declared_fast_sio_is_not_clamped_to_slower_l1_injection():
    _, topo, graph = topology()
    graph.events.append(topo.transfer("pair", "npu0", "npu1", 1e9))
    result = simulate(graph)
    assert graph.resources["npu0.SU.tx"].capacity == 200e9
    assert result["duration_s"] == pytest.approx(.4e-6 + 1e9 / 200e9)


def test_cross_host_flows_compete_on_shared_uplink():
    _, topo, graph = topology()
    graph.events.extend([topo.transfer("first", "npu0", "npu16", 1e9),
                         topo.transfer("second", "npu2", "npu18", 1e9)])
    result = simulate(graph)
    assert result["status"] == "feasible"
    assert result["duration_s"] == pytest.approx(4.6e-6 + 2e9 / 160e9)
    assert result["resource_work"]["link:hccs_l2:a3.host0.hccs_l1>a3.supod.hccs_l2"] == 2e9
    assert sum(e.metadata["physical_bytes"] for e in graph.events) == 8e9


def test_opposite_directions_use_separate_uplinks():
    _, topo, graph = topology()
    graph.events.extend([topo.transfer("forward", "npu0", "npu16", 1e9),
                         topo.transfer("reverse", "npu16", "npu0", 1e9)])
    assert simulate(graph)["duration_s"] == pytest.approx(4.6e-6 + 1e9 / 100e9)


def test_no_implicit_host_offload_route_and_optional_host_links():
    fragment, topo, _ = topology()
    with pytest.raises(ValueError, match="No directed"):
        topo.path("cpu0", "npu0")
    assert any("No H2D/D2H" in text for text in fragment["assumptions"])
    spec = example()
    spec["host_link"] = {"bandwidth_Bps": 64e9, "latency_s": 2e-6,
                         "source": {"status": "user", "description": "User supplied host-link model"}}
    fragment, topo, graph = topology(spec)
    assert topo.path("cpu0", "npu0") == ("host_link:cpu0>npu0",)
    assert topo.path("npu0", "cpu0") == ("host_link:npu0>cpu0",)
    graph.events.append(topo.transfer("offload", "npu0", "cpu0", 1e9))
    assert simulate(graph)["duration_s"] == pytest.approx(2e-6 + 1e9 / 64e9)


def test_topology_identity_changes_with_actual_tier_parameters():
    original = builder.build_topology_fragment(example())
    changed = example()
    changed["hccs_l2"]["bandwidth_Bps"] /= 2
    changed = builder.build_topology_fragment(changed)
    assert original["hardware"]["name"] != changed["hardware"]["name"]
    assert original["sources"]["hardware.topology"]["status"] == "assumed"
    assert original["sources"]["hardware.topology.architecture"]["status"] == "public"
    assert original["sources"]["hardware.topology.sio"]["status"] == "assumed"
    assert builder.ARCHITECTURE_REVISION in original["sources"]["hardware.topology.architecture"]["uri"]


@pytest.mark.parametrize("patch", [
    {"hosts": True}, {"hosts": 0}, {"hosts": 1.5}, {"hosts": "2"},
    {"sio": {"bandwidth_Bps": 1e9}}, {"unknown": 1},
])
def test_bad_or_incomplete_spec_is_rejected(patch):
    spec = example()
    spec.update(patch)
    with pytest.raises(ValueError):
        builder.build_topology_fragment(spec)


@pytest.mark.parametrize("field,value", [
    ("bandwidth_Bps", 0), ("bandwidth_Bps", float("inf")),
    ("latency_s", -1), ("latency_s", float("nan")),
    ("efficiency", 1.1), ("efficiency", False), ("source", {}),
])
def test_link_values_and_provenance_are_required(field, value):
    spec = example()
    spec["sio"][field] = value
    with pytest.raises(ValueError):
        builder.build_topology_fragment(spec)


def test_cli_generates_hardware_fragment(tmp_path):
    output = tmp_path / "a3-hardware.json"
    result = subprocess.run([sys.executable, str(SCRIPT), "--spec", str(EXAMPLE),
                             "--output", str(output)], capture_output=True, text=True, check=True)
    assert "architecture abstraction" in result.stdout
    fragment = json.loads(output.read_text(encoding="utf-8"))
    HardwareSpec.model_validate(fragment["hardware"])
    assert set(fragment) == {"hardware", "sources", "assumptions"}


def test_cli_refuses_to_overwrite_input_spec(tmp_path):
    spec = tmp_path / "input.json"
    spec.write_text(EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    before = spec.read_bytes()
    result = subprocess.run([sys.executable, str(SCRIPT), "--spec", str(spec),
                             "--output", str(spec)], capture_output=True, text=True)
    assert result.returncode == 2
    assert "must not overwrite" in result.stderr
    assert spec.read_bytes() == before


def test_cli_refuses_an_output_hard_link_to_input(tmp_path):
    spec = tmp_path / "input.json"
    spec.write_text(EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    output = tmp_path / "alias.json"
    try:
        output.hardlink_to(spec)
    except OSError as exc:
        pytest.skip(f"Hard links unavailable: {exc}")
    before = spec.read_bytes()
    result = subprocess.run([sys.executable, str(SCRIPT), "--spec", str(spec),
                             "--output", str(output)], capture_output=True, text=True)
    assert result.returncode == 2
    assert "must not overwrite" in result.stderr
    assert spec.read_bytes() == before


@pytest.mark.parametrize("invalid,expected", [
    ('{"hosts":1,"hosts":2}', "Duplicate JSON key"),
    ('{"source":{"status":"user","status":"assumed"}}', "Duplicate JSON key"),
    ('{"hosts":NaN}', "Non-finite JSON constant"),
    ('{"hosts":Infinity}', "Non-finite JSON constant"),
])
def test_cli_rejects_duplicate_keys_and_nonfinite_json(tmp_path, invalid, expected):
    spec, output = tmp_path / "bad.json", tmp_path / "output.json"
    spec.write_text(invalid, encoding="utf-8")
    result = subprocess.run([sys.executable, str(SCRIPT), "--spec", str(spec),
                             "--output", str(output)], capture_output=True, text=True)
    assert result.returncode == 2
    assert expected in result.stderr
    assert not output.exists()


def test_cli_output_io_failure_is_an_argument_error(tmp_path):
    output = tmp_path / "directory"
    output.mkdir()
    result = subprocess.run([sys.executable, str(SCRIPT), "--spec", str(EXAMPLE),
                             "--output", str(output)], capture_output=True, text=True)
    assert result.returncode == 2
    assert "error:" in result.stderr
    assert "Traceback" not in result.stderr
