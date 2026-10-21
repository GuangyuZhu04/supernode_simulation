from pathlib import Path
import sys
import base64
from xml.etree import ElementTree as ET

WS = Path(__file__).resolve().parents[1]
SKILL = Path(r"C:\Users\Guangyu\.codex\skills\ppt-master-huawei-v2")
sys.path.insert(0, str(SKILL / "scripts"))
from svg_to_pptx.use_expander import expand_use_data_icons
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.graphics import renderPM
from svglib.svglib import svg2rlg
from svglib.fonts import register_font

pdfmetrics.registerFont(TTFont("MicrosoftYaHei", r"C:\Windows\Fonts\msyh.ttc", subfontIndex=0))
pdfmetrics.registerFont(TTFont("MicrosoftYaHei-Bold", r"C:\Windows\Fonts\msyhbd.ttc", subfontIndex=0))
pdfmetrics.registerFontFamily("MicrosoftYaHei", normal="MicrosoftYaHei", bold="MicrosoftYaHei-Bold")
register_font("MicrosoftYaHei", rlgFontName="MicrosoftYaHei")
register_font("MicrosoftYaHei", weight="bold", rlgFontName="MicrosoftYaHei-Bold")

for page in (5, 6):
    tree = ET.parse(WS / f"svg/page{page}.svg")
    root = tree.getroot()
    expand_use_data_icons(root, SKILL / "assets/icons")
    for e in root.iter():
        if e.get("font-family"):
            e.set("font-family", "MicrosoftYaHei")
        if e.tag.endswith("image"):
            logo = base64.b64encode((SKILL / "assets/pictures/logo.png").read_bytes()).decode("ascii")
            e.set("href", "data:image/png;base64," + logo)
    out_svg = WS / f"qa/pages56/page{page}-preview.svg"
    tree.write(out_svg, encoding="utf-8", xml_declaration=True)
    drawing = svg2rlg(str(out_svg))
    renderPM.drawToFile(drawing, str(WS / f"previews/page{page}.png"), fmt="PNG", dpi=96)
    print(f"Rendered page {page}")
