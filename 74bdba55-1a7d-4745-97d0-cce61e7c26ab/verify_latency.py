from pathlib import Path
from collections import Counter
from xml.etree import ElementTree as E
import hashlib, json, math, re, zipfile

WS = Path(__file__).resolve().parent
data = json.loads((WS / "input/latency-data.json").read_text(encoding="utf-8"))
proof = json.loads((WS / "qa/source-proof.json").read_text(encoding="utf-8"))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(Path(proof["source_pptx"])) == proof["source_sha256"]
sim = WS.parent / "outputs/kimi-k3-10t-1m-acceleration-20261019/parallelism-20261020"
for name, digest in proof["simulation_files"].items():
    assert sha(sim/name) == digest
expected = Counter(v for r in data["rows"] for phase in ("prefill", "decode") for v in r[phase]["display"])
svg = E.parse(WS / "svg/page6.svg").getroot()
svg_text = "\n".join("".join(t.itertext()) for t in svg.findall(".//{http://www.w3.org/2000/svg}text"))
pattern = r"(?<![\d.])\d+\.\d{3}(?![\d.])"
actual = Counter(re.findall(pattern, svg_text))
for value, count in expected.items():
    assert actual[value] >= count, ("SVG missing displayed value", value, count, actual[value])
assert sum(expected.values()) == 60
for r in data["rows"]:
    for phase in ("prefill", "decode"):
        factor = 1 if phase == "prefill" else 1000
        full_sum = sum(r[phase]["raw_s"].values()) * factor
        assert math.isclose(full_sum, r[phase]["total"], rel_tol=1e-10, abs_tol=1e-8)
        assert r[phase]["other_total_s"] < 1e-9
candidate = WS / "candidate.pptx"
native_checked = False
if candidate.exists():
    with zipfile.ZipFile(candidate) as z:
        root = E.fromstring(z.read("ppt/slides/slide6.xml"))
        native = "\n".join(t.text or "" for t in root.findall(".//{http://schemas.openxmlformats.org/drawingml/2006/main}t"))
        native_values = Counter(re.findall(pattern, native))
        for value, count in expected.items():
            assert native_values[value] >= count, ("PPTX missing value", value, count)
        notes = "".join(E.fromstring(z.read("ppt/notesSlides/notesSlide6.xml")).itertext())
        for required in ("过渡", "本页名词表", "标红内容详解"):
            assert required in notes
    native_checked = True
result = {"passed": True, "displayed_component_and_total_values": 60,
          "native_pptx_values_checked": native_checked, "stage_sums_checked": 12,
          "raw_sum_max_absolute_error_in_display_units": max(c["absolute_difference"] for c in data["checks"]),
          "source_pptx_unchanged": True, "simulation_files_unchanged": True,
          "other_categories_retained_and_negligible": True, "units_prefill_s_decode_ms": True}
(WS / "qa/data-verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False))
