"""Preview only pages 7–8 with an explicit Microsoft YaHei font binding."""
from pathlib import Path
from xml.etree import ElementTree as ET
import sys
import base64

WS = Path(__file__).resolve().parents[1]
SKILL = Path(r"C:\Users\Guangyu\.codex\skills\ppt-master-huawei-v2")
sys.path.insert(0, str(SKILL / "scripts"))
from svg_to_pptx.use_expander import expand_use_data_icons
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.graphics import renderPM
from svglib.svglib import svg2rlg

pdfmetrics.registerFont(TTFont("MicrosoftYaHei", r"C:\Windows\Fonts\msyh.ttc", subfontIndex=0))
pdfmetrics.registerFont(TTFont("MicrosoftYaHei-Bold", r"C:\Windows\Fonts\msyhbd.ttc", subfontIndex=0))
pdfmetrics.registerFontFamily("MicrosoftYaHei", normal="MicrosoftYaHei", bold="MicrosoftYaHei-Bold")

out_dir = WS / "qa/pages78"
out_dir.mkdir(parents=True, exist_ok=True)
for page in (7, 8):
    tree = ET.parse(WS / f"svg/page{page}.svg")
    root = tree.getroot()
    expand_use_data_icons(root, SKILL / "assets/icons")
    for e in root.iter():
        if e.tag.endswith("text") or e.tag.endswith("tspan"):
            e.set("font-family", "MicrosoftYaHei")
        if e.tag.endswith("image"):
            logo = base64.b64encode((SKILL / "assets/pictures/logo.png").read_bytes()).decode("ascii")
            e.set("href", "data:image/png;base64," + logo)
    out_svg = out_dir / f"page{page}-preview.svg"
    tree.write(out_svg, encoding="utf-8", xml_declaration=True)
    drawing = svg2rlg(str(out_svg))
    # The installed svglib resolves SVG text to Helvetica even for registered
    # aliases. Rebind the reportlab drawing only; never alter the source SVG.
    pending = [drawing]
    while pending:
        item = pending.pop()
        pending.extend(getattr(item, "contents", []))
        if hasattr(item, "text") and hasattr(item, "fontName"):
            item.fontName = "MicrosoftYaHei-Bold" if "bold" in item.fontName.lower() else "MicrosoftYaHei"
    renderPM.drawToFile(drawing, str(out_dir / f"page{page}.png"), fmt="PNG", backend="rlPyCairo")
    print(f"Rendered page {page}")
