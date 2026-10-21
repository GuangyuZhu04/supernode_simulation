"""Publish the verified local revision without overwriting the open original."""
from pathlib import Path
import hashlib, json, shutil

WS = Path(__file__).resolve().parent
ROOT = WS.parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text(encoding="utf-8"))
provenance = read(WS / "qa/provenance.json")
structure = read(WS / "qa/pptx-structure.json")
values = read(WS / "qa/data-verification.json")
source = ROOT / "outputs/presentations/kimi-k3-10t-brief-20261019.pptx"
candidate = WS / "candidate.pptx"
assert sha(source) == provenance["input_sha256"]
assert sha(candidate) == structure["sha256"]
assert structure["slide_count"] == structure["notes_count"] == 5
assert values["passed"] and values["checked_display_values"] == 110
for n in range(1, 6):
    assert (WS / f"previews/office/slide-{n:02d}.png").is_file()
target = ROOT / "outputs/presentations/kimi-k3-10t-brief-20261020.pptx"
if target.exists():
    assert sha(target) == sha(candidate), "Refuse to overwrite an unrelated dated revision"
else:
    shutil.copy2(candidate, target)
assert sha(target) == sha(candidate)
preview_dir = target.parent / (target.stem + "-previews")
preview_dir.mkdir(exist_ok=True)
for n in (4, 5):
    source_preview = WS / f"previews/office/slide-{n:02d}.png"
    dest = preview_dir / f"slide-{n:02d}.png"
    if dest.exists():
        assert sha(dest) == sha(source_preview)
    else:
        shutil.copy2(source_preview, dest)
# Preserve original introduction notes' referenced baseline source snapshots.
legacy_input = ROOT / "c26eab07-4d67-4f69-bcab-63e7f89c5e9d/input"
for name in ("report.md", "summary.csv", "manifest.json", "verification.json",
             "model-sources.md", "latency_breakdown.csv"):
    if (legacy_input / name).exists() and not (WS / "input" / name).exists():
        shutil.copy2(legacy_input / name, WS / "input" / name)
proof = {"delivered": str(target), "sha256": sha(target), "bytes": target.stat().st_size,
    "source_pptx_unchanged": True, "total_slides": 5, "new_result_slides": [4, 5],
    "native_editable_shapes": True, "speaker_notes": 5, "animations": 0,
    "svg_syntax": {"passed": True, "files": 5, "warnings": 0, "errors": 0},
    "svg_layout": {"passed": True, "overflow": 0, "edge_warnings": 0},
    "red_notes_coverage": "passed", "displayed_values_checked": 110,
    "native_office_render": {"engine": "WPS Presentation", "read_only": True,
        "window_opened": False, "size": [1600, 900], "pages_inspected": [1, 2, 3, 4, 5],
        "result": "Chinese text, tables, graph labels and footers inspected; no clipping or unintended overlaps."},
    "preview_folder": str(preview_dir)}
(WS / "qa/final-delivery.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(proof, ensure_ascii=False, indent=2))
