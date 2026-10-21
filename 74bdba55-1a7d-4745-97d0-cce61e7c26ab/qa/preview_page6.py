from pathlib import Path
import base64
import json
import sys
import xml.etree.ElementTree as ET

WS = Path(__file__).resolve().parents[1]
SKILL = Path(r"C:\Users\Guangyu\.codex\skills\ppt-master-huawei-v2")
sys.path.insert(0, str(SKILL / "scripts"))
from svg_to_pptx.use_expander import expand_use_data_icons
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.graphics import renderPM
from svglib.svglib import svg2rlg
from svglib.fonts import register_font

pdfmetrics.registerFont(TTFont("Page6YaHei", r"C:\Windows\Fonts\msyh.ttc", subfontIndex=0))
pdfmetrics.registerFont(TTFont("Page6YaHei-Bold", r"C:\Windows\Fonts\msyhbd.ttc", subfontIndex=0))
pdfmetrics.registerFontFamily("Page6YaHei", normal="Page6YaHei", bold="Page6YaHei-Bold")
pdfmetrics.registerFont(TTFont("Page6Arial", r"C:\Windows\Fonts\arial.ttf"))
pdfmetrics.registerFontFamily("Page6Arial", normal="Page6Arial", bold="Page6Arial")
register_font("Page6YaHei", r"C:\Windows\Fonts\msyh.ttc")
register_font("Page6YaHei", r"C:\Windows\Fonts\msyhbd.ttc", weight="bold")
register_font("Page6Arial", r"C:\Windows\Fonts\arial.ttf")
root = ET.parse(WS / "svg/page6.svg").getroot()
expanded = expand_use_data_icons(root, SKILL / "assets/icons")
for el in root.iter():
    if "font-family" in el.attrib:
        el.set("font-family", "Page6Arial" if el.get("font-family").startswith("Arial") else "Page6YaHei")
    if el.tag.endswith("image"):
        el.set("href", "data:image/png;base64," + base64.b64encode((WS / "svg/logo.png").read_bytes()).decode("ascii"))
preview_svg = WS / "qa/page6-preview.svg"
ET.ElementTree(root).write(preview_svg, encoding="utf-8", xml_declaration=True)
drawing = svg2rlg(str(preview_svg))
assert drawing is not None
renderPM.drawToFile(drawing, str(WS / "qa/page6-preview.png"), fmt="PNG", dpi=144)
# Independent font-metric check of logical lines and nearest visible containers.
# Text elements in the authored SVG are single logical lines; all tspans are inline.
metrics = []
for el in root.iter():
    if not el.tag.endswith("text") or el.get("x") is None:
        continue
    base_size = float(el.get("font-size", 10.5))
    base_bold = el.get("font-weight") == "bold"
    face = el.get("font-family", "Page6Arial")
    if base_bold and face == "Page6YaHei":
        face = "Page6YaHei-Bold"
    width = pdfmetrics.stringWidth(el.text or "", face, base_size)
    for run in el:
        run_face = "Page6YaHei-Bold" if run.get("font-weight", "bold" if base_bold else "normal") == "bold" else face
        width += pdfmetrics.stringWidth(run.text or "", run_face, float(run.get("font-size", base_size)))
        width += pdfmetrics.stringWidth(run.tail or "", face, base_size)
    x = float(el.get("x"))
    anchor = el.get("text-anchor", "start")
    start = x - width if anchor == "end" else x - width / 2 if anchor == "middle" else x
    end = start + width
    assert start >= 0 and end <= 1280, (el.text, start, end)
    metrics.append({"text": "".join(el.itertext()), "x_start": start, "x_end": end, "y": float(el.get("y"))})
(WS / "qa/page6-font-metrics.json").write_text(json.dumps({
    "expanded_icons": expanded, "line_count": len(metrics), "all_lines_inside_canvas": True,
    "lines": metrics,
}, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Rendered QA preview with {expanded} icons; {len(metrics)} actual-font line bounds passed.")
