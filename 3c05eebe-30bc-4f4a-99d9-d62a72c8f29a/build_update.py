from pathlib import Path
import sys, json, zipfile, hashlib, base64, re
from xml.etree import ElementTree as ET

WS = Path(__file__).resolve().parent
SKILL = Path(r"C:\Users\Guangyu\.codex\skills\ppt-master-huawei-v2")
sys.path.insert(0, str(SKILL / "scripts"))
from svg_to_pptx import create_pptx_with_native_svg
from svg_to_pptx.pptx_discovery import find_notes_files
from svg_to_pptx.use_expander import expand_use_data_icons


def write_notes():
    pages = []
    for n in range(1, 6):
        p = json.loads((WS / f"notes/page{n}.json").read_text(encoding="utf-8"))
        pages.append(p)
        speech = p["speech"]
        if isinstance(speech, list):
            speech = "\n\n".join(speech)
        text = f"【第{n}页 · {p['title']}】\n\n{speech}\n\n过渡：{p['transition']}\n\n"
        text += "【本页名词表 · 英文简称：英文全称，中文全称——一句话解释】\n"
        for g in p["glossary"]:
            text += f"- {g['term']}：{g['full']}——{g['explain']}\n"
        text += "\n【标红内容详解 · 逐条一句话解释】\n"
        for g in p["red_notes"]:
            text += f"- 「{g['quote']}」：{g['explain']}\n"
        text += "\n【来源与数据口径】\n"
        for s in p.get("sources", []):
            text += f"- {s}\n"
        (WS / f"svg/notes/page{n}.md").write_text(text, encoding="utf-8")
    (WS / "notes.json").write_text(json.dumps({"pages": pages}, ensure_ascii=False, indent=2), encoding="utf-8")


def render_svg():
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.graphics import renderPM
    from svglib.svglib import svg2rlg
    from svglib.fonts import register_font
    for name, file in [("微软雅黑", "msyh.ttc"), ("微软雅黑-Bold", "msyhbd.ttc"),
                       ("Arial", "arial.ttf"), ("Arial-Bold", "arialbd.ttf")]:
        pdfmetrics.registerFont(TTFont(name, str(Path(r"C:\Windows\Fonts") / file)))
    for name in ("微软雅黑", "Arial"):
        pdfmetrics.registerFontFamily(name, normal=name, bold=name + "-Bold", italic=name, boldItalic=name + "-Bold")
        register_font(name, rlgFontName=name)
        register_font(name, weight="bold", rlgFontName=name + "-Bold")
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    for n in range(1, 6):
        root = ET.parse(WS / f"svg/page{n}.svg").getroot()
        count = expand_use_data_icons(root, SKILL / "assets/icons")
        for e in root.iter():
            # svglib's comma-separated fallback chooses Arial even for CJK.
            # Preview only: select the registered primary face; source SVG and
            # native PPTX retain the prescribed family list.
            if e.get("font-family"):
                e.set("font-family", e.get("font-family").split(",")[0].strip())
            if e.tag.endswith("image") and e.get("href") == "logo.png":
                e.set("href", "data:image/png;base64," + base64.b64encode((WS / "svg/logo.png").read_bytes()).decode())
        preview = WS / f"previews/page{n}.svg"
        ET.ElementTree(root).write(preview, encoding="utf-8", xml_declaration=True)
        drawing = svg2rlg(str(preview))
        renderPM.drawToFile(drawing, str(WS / f"previews/page{n}.png"), fmt="PNG", backend="rlPyCairo")
        print(f"Rendered page{n}; expanded icons={count}", flush=True)


def build():
    files = [WS / f"svg/page{n}.svg" for n in range(1, 6)]
    notes = find_notes_files(WS / "svg", files)
    out = WS / "candidate.pptx"
    ok = create_pptx_with_native_svg(files, out, use_native_shapes=True, use_compat_mode=False,
                                    notes=notes, transition=None, animation=None)
    assert ok
    ns = {"p": "http://schemas.openxmlformats.org/presentationml/2006/main",
          "a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
    audit = {"slide_count": 0, "notes_count": 0, "slides": [], "native_editable_shapes": True}
    with zipfile.ZipFile(out) as z:
        for n in range(1, 6):
            root = ET.fromstring(z.read(f"ppt/slides/slide{n}.xml"))
            row = {"page": n, "text_count": len(root.findall(".//a:t", ns)),
                   "shape_count": len(root.findall(".//p:sp", ns)), "groups": len(root.findall(".//p:grpSp", ns)),
                   "animations": len(root.findall(".//p:timing", ns)),
                   "transitions": len(root.findall(".//p:transition", ns))}
            assert row["text_count"] > 20 and row["shape_count"] > 20
            assert row["animations"] == row["transitions"] == 0
            audit["slides"].append(row)
            text = "".join(ET.fromstring(z.read(f"ppt/notesSlides/notesSlide{n}.xml")).itertext())
            for title in ("过渡", "本页名词表", "标红内容详解"):
                assert title in text
            audit["notes_count"] += 1
        audit["slide_count"] = len([n for n in z.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)])
        assert audit["slide_count"] == 5
    audit["sha256"] = hashlib.sha256(out.read_bytes()).hexdigest()
    (WS / "qa/pptx-structure.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    op = sys.argv[1] if len(sys.argv) > 1 else "all"
    if op in ("all", "notes"):
        write_notes()
    if op in ("all", "render"):
        render_svg()
    if op in ("all", "build"):
        build()
