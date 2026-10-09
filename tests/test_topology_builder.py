"""Observable topology compiler behavior, including route-domain invariants."""
import copy
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from supernode_sim.config import Config, HardwareSpec, load_config
from supernode_sim.core import Graph
from supernode_sim.engine import simulate
from supernode_sim.topology import Topology


SCRIPT = (Path(__file__).resolve().parents[1] / "skills" / "supernode-modeling" /
          "subskills" / "supernode-topology" / "scripts" / "build_topology.py")
MODULE_SPEC = importlib.util.spec_from_file_location("build_supernode_topology", SCRIPT)
BUILDER = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = BUILDER
MODULE_SPEC.loader.exec_module(BUILDER)


def spec(su="ring", so="clos", *, npu_count=8, su_size=2, npus_per_host=1):
    rates = {"bandwidth_Bps": 100.0, "latency_s": 0.0, "efficiency": 1.0}
    su_count = npu_count // su_size
    domains = [{"id": "east", "su_domains": list(range(su_count))}]
    interlinks = []
    if su_count > 1:
        split = su_count // 2
        domains = [{"id": "east", "su_domains": list(range(split))},
                   {"id": "west", "su_domains": list(range(split, su_count))}]
        interlinks = [{"id": key, "src_domain": src, "dst_domain": dst, **rates}
                      for key, src, dst in (("east-west", "east", "west"), ("west-east", "west", "east"))]
    return {"schema_version": "1.0", "npu_count": npu_count, "su_size": su_size,
            "npus_per_host": npus_per_host, "scale_up": {"topology": su, **rates},
            "scale_out": {"topology": so, "domains": domains, **rates},
            "host_link": rates.copy(), "inter_domain_links": interlinks}


def base(npu_count=8, su_size=2):
    return {"schema_version": "1.0",
            "model": {"name": "tiny", "total_params": 1000.0, "layers": 2, "hidden_dim": 16},
            "workload": {"scenario": "training", "batch_size": 1, "sequence_length": 8},
            "execution": {"dp": npu_count},
            "hardware": {"npu_count": npu_count, "su_size": su_size, "npus_per_host": 1,
                         "npu": {"hbm_bytes": 2e9}, "cpu": {"dram_bytes": 3e9}},
            "targets": {"step_time_s": 1.0}, "sources": {"model": {"status": "user", "description": "tiny fixture"}},
            "assumptions": ["Keep this workload assumption."]}


KINDS = ("full_mesh", "ring", "mesh", "clos", "fat_tree")


@pytest.mark.parametrize("su", KINDS)
@pytest.mark.parametrize("so", KINDS)
def test_all_generated_fabrics_have_directed_paths_and_preserve_su_so(su, so):
    output = BUILDER.compile_topology(spec(su, so))
    data = HardwareSpec.model_validate(output["hardware"]).model_dump()
    topology = Topology(data, {})
    assert data["routing_policy"] == "su_so"
    assert data["routes"] == {}
    assert not ({"model", "workload", "execution"} & output.keys())
    for src in range(8):
        for dst in range(8):
            if src == dst:
                continue
            edges = [topology.links[key] for key in topology.path(f"npu{src}", f"npu{dst}")]
            expected = "SU" if src // 2 == dst // 2 else "SO"
            assert {edge["domain"] for edge in edges} == {expected}
            assert edges[0]["src"] == f"npu{src}"
            assert edges[-1]["dst"] == f"npu{dst}"
            assert not any(edge["src"].startswith("cpu") or edge["dst"].startswith("cpu") for edge in edges)
    assert topology.transfer("h2d", "cpu0", "npu0", 10).metadata["domains"] == ["H2D"]
    assert topology.transfer("d2h", "npu0", "cpu0", 10).metadata["domains"] == ["D2H"]


def test_ring_closes_and_mesh_uses_requested_width():
    ring = BUILDER.compile_topology(spec("ring", npu_count=4, su_size=4))
    topology = Topology(ring["hardware"], {})
    assert len(topology.path("npu0", "npu3")) == 1
    mesh_spec = spec("mesh", npu_count=4, su_size=4)
    mesh_spec["scale_up"]["mesh_width"] = 2
    mesh = Topology(BUILDER.compile_topology(mesh_spec)["hardware"], {})
    assert len(mesh.path("npu0", "npu3")) == 2
    assert not any(edge["src"] == "npu0" and edge["dst"] == "npu3" and edge["domain"] == "SU"
                   for edge in mesh.links.values())


def test_multiple_spines_share_leaf_capacity_without_half_duplex_coupling():
    data = spec("clos", npu_count=4, su_size=4)
    data["scale_up"].update(leaf_size=2, spine_count=3, uplink_Bps=150.0)
    topology = Topology(BUILDER.compile_topology(data)["hardware"], {})
    uplinks = [edge for edge in topology.links.values() if edge["src"] == "su0.leaf0" and ".spine" in edge["dst"]]
    assert len(uplinks) == 3
    assert len({edge["shared_resource"] for edge in uplinks}) == 1
    shared = uplinks[0]["shared_resource"]
    assert topology.resources[shared].capacity == 150
    assert topology.links["su0.spine0>su0.leaf0:SU"]["shared_resource"] != shared
    assert "su0.leaf0>su0.spine0:SU" in topology.path("npu0", "npu3")


def test_so_gateway_uplink_shares_all_npu_attachments():
    data = spec("ring", "full_mesh", npu_count=4, su_size=2, npus_per_host=2)
    data["scale_out"].update(uplink_Bps=50.0, injection_Bps=1000.0)
    topology = Topology(BUILDER.compile_topology(data)["hardware"], {})
    graph = Graph()
    topology.populate(graph)
    graph.events.extend([topology.transfer("a", "npu0", "npu2", 100),
                         topology.transfer("b", "npu1", "npu3", 100)])
    assert graph.events[0].work["shared:su0.gateway.attachment.tx"] == 100
    assert graph.events[1].work["shared:su0.gateway.attachment.tx"] == 100
    assert topology.resources["cpu0.nic.tx"].capacity == 1000
    assert simulate(graph)["duration_s"] == pytest.approx(4.0)


def test_explicit_domain_half_duplex_capacity_couples_opposite_directions():
    data = spec("ring", "full_mesh", npu_count=4, su_size=2)
    for link in data["inter_domain_links"]:
        link.update(shared_group="one-physical-link", shared_Bps=50.0)
    topology = Topology(BUILDER.compile_topology(data)["hardware"], {})
    graph = Graph()
    topology.populate(graph)
    graph.events.extend([topology.transfer("forward", "npu0", "npu2", 100),
                         topology.transfer("reverse", "npu2", "npu0", 100)])
    assert all(event.work["shared:inter:one-physical-link"] == 100 for event in graph.events)
    assert simulate(graph)["duration_s"] == pytest.approx(4.0)


def test_directed_three_domain_cycle_reaches_both_directions_without_inventing_reverse_edges():
    data = spec(npu_count=6, su_size=2)
    data["scale_out"]["domains"] = [{"id": key, "su_domains": [i]} for i, key in enumerate(("a", "b", "c"))]
    data["inter_domain_links"] = [{"id": key, "src_domain": src, "dst_domain": dst,
                                  "bandwidth_Bps": 100.0, "latency_s": 0.0, "efficiency": 1.0}
                                 for key, src, dst in (("ab", "a", "b"), ("bc", "b", "c"), ("ca", "c", "a"))]
    output = BUILDER.compile_topology(data)
    topology = Topology(output["hardware"], {})
    assert {edge["id"] for edge in output["hardware"]["explicit_links"] if edge["id"].startswith("inter:")} == {"inter:ab", "inter:bc", "inter:ca"}
    assert "inter:ab" in topology.path("npu0", "npu2")
    reverse_path = topology.path("npu2", "npu0")
    assert "inter:bc" in reverse_path and "inter:ca" in reverse_path


@pytest.mark.parametrize("name", ("remote", "ssd"))
def test_base_enabled_external_memory_keeps_capacities_and_paths(name):
    original = base()
    original["hardware"][name] = {"enabled": True, "capacity_bytes": 9e9,
                                  "bandwidth_Bps": 200.0, "latency_s": 0.0, "iops": 10.0}
    output = BUILDER.compile_topology(spec(), original)
    assert output["hardware"][name] == original["hardware"][name]
    topology = Topology(output["hardware"], {})
    event = topology.transfer("fetch", f"{name}0", "npu0", 10)
    assert event.metadata["domains"] == sorted(["H2D", name])
    assert event.work["cpu0.dram"] == 10
    assert event.work[f"{name}0.memory"] == 10


def test_merge_preserves_runtime_and_replaces_stale_topology_and_provenance():
    original = base()
    original["hardware"].update(mesh_width=10, leaf_size=20, routes={"old->route": ["stale"]},
                                explicit_links=[], kernel_profiles={"x": [{"flops": 0, "memory_bytes": 0, "duration_s": 1}]})
    original["sources"]["hardware.scale_up.bandwidth_Bps"] = {"status": "measured", "description": "old network"}
    original["sources"]["hardware.npu.hbm_bytes"] = {"status": "measured", "description": "preserve"}
    untouched = copy.deepcopy(original)
    topology_spec = spec()
    topology_spec["sources"] = {"hardware.scale_up.bandwidth_Bps": {"status": "assumed", "description": "new assumed rate"}}
    output = BUILDER.compile_topology(topology_spec, original)
    assert original == untouched
    for key in ("model", "workload", "execution", "targets"):
        assert output[key] == original[key]
    for key in ("npu", "cpu", "kernel_profiles"):
        assert output["hardware"][key] == original["hardware"][key]
    assert not ({"mesh_width", "leaf_size"} & output["hardware"].keys())
    assert output["hardware"]["routes"] == {}
    assert output["sources"]["hardware.scale_up.bandwidth_Bps"]["status"] == "assumed"
    assert output["sources"]["hardware.npu.hbm_bytes"] == original["sources"]["hardware.npu.hbm_bytes"]
    assert output["sources"]["model"] == original["sources"]["model"]
    assert original["assumptions"][0] in output["assumptions"]
    assert Config.model_validate(output).hardware.npu.hbm_bytes == 2e9


def test_changed_npu_count_does_not_invent_parallel_layout():
    with pytest.raises(ValueError, match=r"tp \* pp \* ep \* dp"):
        BUILDER.compile_topology(spec(npu_count=4), base())


@pytest.mark.parametrize("bad_rate", (0.0, -1.0, float("inf"), float("nan"), "100", True))
def test_rejects_nonfinite_nonpositive_or_untyped_rates(bad_rate):
    data = spec()
    data["scale_up"]["bandwidth_Bps"] = bad_rate
    with pytest.raises(ValueError):
        BUILDER.compile_topology(data)


@pytest.mark.parametrize("change,error", [
    (lambda x: x.update(fabric="ring"), "Extra inputs"),
    (lambda x: x["scale_up"].update(unknown_port_rate=1), "Extra inputs"),
    (lambda x: x.update(su_size=3), "su_size must divide"),
    (lambda x: x.update(npus_per_host=3), "npus_per_host must divide"),
    (lambda x: x["scale_out"]["domains"][0]["su_domains"].append(3), "exactly once"),
    (lambda x: x["scale_out"]["domains"][0].update(id="west"), "IDs must be unique"),
    (lambda x: x["scale_out"]["domains"][0].update(id="rack A"), "pattern"),
    (lambda x: x["scale_out"]["domains"][0]["su_domains"].__setitem__(0, 99), "exactly once"),
    (lambda x: x["inter_domain_links"].pop(), "strongly connected"),
    (lambda x: x["inter_domain_links"][0].update(dst_domain="missing"), "declared domain"),
    (lambda x: x["scale_up"].update(spine_count=2), "require clos"),
    (lambda x: x["scale_up"].update(uplink_Bps=100.0), "requires clos"),
    (lambda x: x["host_link"].update(uplink_Bps=100.0), "Extra inputs"),
    (lambda x: x["inter_domain_links"][0].update(shared_group="x"), "supplied together"),
    (lambda x: x.update(sources={"model": {"status": "user"}}), "hardware paths"),
])
def test_rejects_invalid_structures_instead_of_silently_ignoring(change, error):
    data = spec()
    change(data)
    with pytest.raises(ValueError, match=error):
        BUILDER.compile_topology(data)


def test_guard_rejects_dense_graph_before_allocating_links(monkeypatch):
    data = spec("full_mesh", npu_count=4096, su_size=4096)
    def fail(*_args, **_kwargs):
        pytest.fail("edge expansion occurred before the size guard")
    monkeypatch.setattr(BUILDER.NetworkBuilder, "edge", fail)
    with pytest.raises(ValueError, match="exceeding max_links"):
        BUILDER.compile_topology(data)


def test_sparse_large_graph_does_not_materialize_pair_routes():
    data = spec("ring", "ring", npu_count=2048, su_size=64)
    output = BUILDER.compile_topology(data)
    assert len(output["hardware"]["explicit_links"]) < 20_000
    assert output["hardware"]["routes"] == {}


def run_cli(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True, encoding="utf-8")


def test_cli_schema_and_yaml_json_fragment_outputs(tmp_path):
    schema_file = tmp_path / "schema.json"
    result = run_cli("--schema", "-o", schema_file)
    assert result.returncode == 0, result.stderr
    schema = json.loads(schema_file.read_text(encoding="utf-8"))
    assert schema["additionalProperties"] is False
    assert "scale_out" in schema["required"]
    input_file = tmp_path / "spec.yaml"
    input_file.write_text(yaml.safe_dump(spec()), encoding="utf-8")
    json_file, yaml_file = tmp_path / "result.json", tmp_path / "result.yaml"
    for output in (json_file, yaml_file):
        result = run_cli(input_file, "-o", output)
        assert result.returncode == 0, result.stderr
    from_json = json.loads(json_file.read_text(encoding="utf-8"))
    from_yaml = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
    assert from_json == from_yaml
    assert "hardware fragment" in result.stderr


def test_cli_resolves_calibration_before_replacing_topology(tmp_path):
    topology_file, scenario_file = tmp_path / "spec.yaml", tmp_path / "base.yaml"
    calibration_file, output_file = tmp_path / "calibration.yaml", tmp_path / "scenario.yaml"
    topology_file.write_text(yaml.safe_dump(spec()), encoding="utf-8")
    initial = base()
    initial["calibration"] = "calibration.yaml"
    scenario_file.write_text(yaml.safe_dump(initial), encoding="utf-8")
    calibration_file.write_text(yaml.safe_dump({"hardware": {"topology": "mesh", "mesh_width": 2,
                                                             "scale_up": {"bandwidth_Bps": 999.0},
                                                             "npu": {"hbm_Bps": 12345.0}}}), encoding="utf-8")
    result = run_cli(topology_file, "--base", scenario_file, "-o", output_file)
    assert result.returncode == 0, result.stderr
    compiled = load_config(output_file)
    assert compiled.calibration is None
    assert compiled.hardware.topology == "explicit"
    assert compiled.hardware.scale_up.bandwidth_Bps == 100
    assert compiled.hardware.npu.hbm_Bps == 12345
    assert "hardware.calibration" in compiled.sources


def test_cli_invalid_input_returns_error_without_output(tmp_path):
    input_file, output_file = tmp_path / "bad.json", tmp_path / "output.yaml"
    data = spec()
    data["scale_out"]["domains"][0]["su_domains"].append(2)
    input_file.write_text(json.dumps(data), encoding="utf-8")
    result = run_cli(input_file, "-o", output_file)
    assert result.returncode == 2
    assert "exactly once" in result.stderr
    assert not output_file.exists()


def test_copied_helper_uses_explicit_repository_root_from_unrelated_cwd(tmp_path):
    copied_script = tmp_path / "copied" / "build_topology.py"
    copied_script.parent.mkdir()
    shutil.copyfile(SCRIPT, copied_script)
    unrelated_cwd = tmp_path / "elsewhere"
    unrelated_cwd.mkdir()
    environment = {**os.environ, "SUPERNODE_SIM_ROOT": str(SCRIPT.parents[5])}
    result = subprocess.run([sys.executable, str(copied_script), "--schema"], cwd=unrelated_cwd,
                            env=environment, capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["additionalProperties"] is False


def test_optional_hardware_name_updates_only_its_own_source():
    original = base()
    original["hardware"]["name"] = "old hardware"
    original["sources"]["hardware.name"] = {"status": "user", "description": "old source"}
    output = BUILDER.compile_topology(spec(), original)
    assert output["hardware"]["name"] == "old hardware"
    assert output["sources"]["hardware.name"] == original["sources"]["hardware.name"]
    renamed = spec()
    renamed["name"] = "new topology hardware"
    output = BUILDER.compile_topology(renamed, original)
    assert output["hardware"]["name"] == "new topology hardware"
    assert output["sources"]["hardware.name"]["description"] != "old source"
