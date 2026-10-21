from pathlib import Path
import hashlib, json, shutil

WS = Path(__file__).resolve().parent
ROOT = WS.parent
read = lambda p: json.loads(p.read_text(encoding="utf-8"))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
source = read(WS / "qa/source-proof.json")
preservation = read(WS / "qa/preservation.json")
data = read(WS / "qa/data-verification.json")
render = read(WS / "qa/original-slide-render-comparison.json")
assert sha(Path(source["source_pptx"])) == source["source_sha256"]
assert preservation["passed"] and preservation["original_five_slide_and_notes_xml_unchanged"]
assert data["passed"] and data["native_pptx_values_checked"]
assert data["displayed_component_and_total_values"] == 60 and data["stage_sums_checked"] == 12
assert render["all_original_pages_pixel_identical"]
candidate = WS / "candidate.pptx"
assert sha(candidate) == preservation["output_sha256"]
dest = ROOT / "outputs/presentations/kimi-k3-10t-brief-20261020-v2.pptx"
if dest.exists():
    assert sha(dest) == sha(candidate), "Unrelated v2 file already exists"
else:
    shutil.copy2(candidate, dest)
preview = dest.parent / "kimi-k3-10t-brief-20261020-v2-latency.png"
source_preview = WS / "previews/office/slide-06.png"
if preview.exists():
    assert sha(preview) == sha(source_preview)
else:
    shutil.copy2(source_preview, preview)
result = {"delivered": str(dest), "sha256": sha(dest), "bytes": dest.stat().st_size,
    "slides": 6, "original_slides_preserved": [1, 2, 3, 4, 5], "new_latency_slide": 6,
    "original_content_parts_byte_identical": preservation["original_content_parts_byte_identical"],
    "original_five_slides_pixel_identical": True, "source_pptx_unmodified": True,
    "latency_values_checked": 60, "full_precision_stage_sums_checked": 12,
    "svg_syntax_layout_red_notes": "passed; zero errors/warnings",
    "native_editable_shapes": True, "new_slide_speaker_notes": True,
    "native_office_render": "WPS Presentation, read-only,1600x900; new slide inspected and original five compared pixelwise",
    "preview": str(preview)}
(WS / "qa/final-delivery.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
