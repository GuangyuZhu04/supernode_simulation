"""Verify displayed numbers against the frozen simulation, without scaling."""
from pathlib import Path
from xml.etree import ElementTree as ET
import hashlib, json, re, zipfile

WS = Path(__file__).resolve().parent
data = json.loads((WS / "input/selected-data.json").read_text(encoding="utf-8"))
provenance = json.loads((WS / "qa/provenance.json").read_text(encoding="utf-8"))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
for name, digest in provenance["simulation_files"].items():
    assert sha(Path(data["source"]) / name) == digest
assert sha(WS.parent / "outputs/presentations/kimi-k3-10t-brief-20261019.pptx") == provenance["input_sha256"]
ns = {"s": "http://www.w3.org/2000/svg", "a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
texts = {}
for n in (4, 5):
    root = ET.parse(WS / f"svg/page{n}.svg").getroot()
    texts[n] = ["".join(t.itertext()) for t in root.findall(".//s:text", ns)]
checks = []


def require(page, label, value):
    assert any(value in t for t in texts[page]), (page, label, value)
    checks.append({"page": page, "metric": label, "display": value})


for r in data["selected"]:
    require(4, r["name"] + "/layout", r["layout_display"])
    for key, fmt in (("ttft_s", ".2f"), ("tpot_ms", ".2f"), ("decode_tokens_s", ",.1f"),
                     ("peak_hbm_GB", ".2f")):
        require(4, r["name"] + "/" + key, format(r[key], fmt))
    require(4, r["name"] + "/occupancy", f'{r["hbm_occupancy_pct"]:.1f}%')
for r in data["contexts"]:
    for key, fmt in (("ttft_s", ".2f"), ("tpot_ms", ".2f"), ("decode_tokens_s", ",.1f"),
                     ("peak_hbm_GB", ".2f")):
        require(5, r["name"] + "/" + key, format(r[key], fmt))
for r in data["batches"]:
    require(5, r["name"] + "/decode_tokens_s", f'{r["decode_tokens_s"]:,.1f}')
for r in data["resources"]:
    for key in ("prefill_EFLOP", "decode_TFLOP", "prefill_SU_PB",
                "decode_physical_SU_GB", "decode_HBM_TB", "resident_cluster_TB"):
        require(5, r["name"] + "/" + key, f'{r[key]:.2f}')
for r in data["selected"]:
    assert r["global_batch"] == 32 and r["status"] == "feasible"
    assert r["local_batch"] * r["active_replicas"] == 32
    assert r["tp"] * r["ep"] * r["dp"] == 1024
for n in (4, 5):
    p = json.loads((WS / f"notes/page{n}.json").read_text(encoding="utf-8"))
    assert all(p.get(k) for k in ("speech", "transition", "glossary", "red_notes", "sources"))
if (WS / "candidate.pptx").exists():
    with zipfile.ZipFile(WS / "candidate.pptx") as z:
        for n in (4, 5):
            native = "".join(t.text or "" for t in ET.fromstring(z.read(f"ppt/slides/slide{n}.xml")).findall(".//a:t", ns))
            for check in [c for c in checks if c["page"] == n]:
                assert check["display"] in native, ("PPTX", check)
result = {"passed": True, "checked_display_values": len(checks),
          "all_selected_cases_end_to_end_feasible": True, "fixed_global_bs32_and_1024_cards": True,
          "source_pptx_unchanged": True, "source_simulation_hashes_unchanged": True,
          "no_latency_or_throughput_rescaling": True, "checks": checks}
(WS / "qa/data-verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print({k: v for k, v in result.items() if k != "checks"})
