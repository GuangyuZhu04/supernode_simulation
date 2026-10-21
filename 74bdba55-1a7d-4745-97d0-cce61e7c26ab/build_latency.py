from pathlib import Path
import base64, hashlib, json, posixpath, re, sys, zipfile
from lxml import etree as E

WS = Path(__file__).resolve().parent
SKILL = Path(r"C:\Users\Guangyu\.codex\skills\ppt-master-huawei-v2")
sys.path.insert(0, str(SKILL / "scripts"))
from svg_to_pptx import create_pptx_with_native_svg
from svg_to_pptx.pptx_discovery import find_notes_files

P = "http://schemas.openxmlformats.org/presentationml/2006/main"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
REL = "http://schemas.openxmlformats.org/package/2006/relationships"
CT = "http://schemas.openxmlformats.org/package/2006/content-types"
EP = "http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
VT = "http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"
sha = lambda data: hashlib.sha256(data).hexdigest()
serialize = lambda node: E.tostring(node, encoding="UTF-8", xml_declaration=True, standalone=True)


def write_notes():
    p = json.loads((WS / "notes/page6.json").read_text(encoding="utf-8"))
    speech = p["speech"] if isinstance(p["speech"], str) else "\n\n".join(p["speech"])
    s = f"【第6页 · {p['title']}】\n\n{speech}\n\n过渡：{p['transition']}\n\n"
    s += "【本页名词表 · 英文简称：英文全称，中文全称——一句话解释】\n"
    for g in p["glossary"]:
        s += f"- {g['term']}：{g['full']}——{g['explain']}\n"
    s += "\n【标红内容详解 · 逐条一句话解释】\n"
    for g in p["red_notes"]:
        s += f"- 「{g['quote']}」：{g['explain']}\n"
    s += "\n【来源与数据口径】\n" + "\n".join("- " + x for x in p["sources"])
    (WS / "svg/notes/page6.md").write_text(s, encoding="utf-8")
    (WS / "notes.json").write_text(json.dumps({"pages": [p]}, ensure_ascii=False, indent=2), encoding="utf-8")


def render_svg():
    from xml.etree import ElementTree as X
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.graphics import renderPM
    from svglib.svglib import svg2rlg
    from svglib.fonts import register_font
    from svg_to_pptx.use_expander import expand_use_data_icons
    for name, filename in (("微软雅黑", "msyh.ttc"), ("微软雅黑-Bold", "msyhbd.ttc"),
                           ("Arial", "arial.ttf"), ("Arial-Bold", "arialbd.ttf")):
        pdfmetrics.registerFont(TTFont(name, str(Path(r"C:\Windows\Fonts") / filename)))
    for name in ("微软雅黑", "Arial"):
        pdfmetrics.registerFontFamily(name, normal=name, bold=name+"-Bold", italic=name, boldItalic=name+"-Bold")
        register_font(name, rlgFontName=name)
        register_font(name, weight="bold", rlgFontName=name+"-Bold")
    X.register_namespace("", "http://www.w3.org/2000/svg")
    root = X.parse(WS / "svg/page6.svg").getroot()
    expand_use_data_icons(root, SKILL / "assets/icons")
    for node in root.iter():
        if node.get("font-family"):
            node.set("font-family", node.get("font-family").split(",")[0].strip())
        if node.tag.endswith("image") and node.get("href") == "logo.png":
            node.set("href", "data:image/png;base64," + base64.b64encode((WS / "svg/logo.png").read_bytes()).decode())
    preview = WS / "previews/page6.svg"
    X.ElementTree(root).write(preview, encoding="utf-8", xml_declaration=True)
    renderPM.drawToFile(svg2rlg(str(preview)), str(WS / "previews/page6.png"), fmt="PNG", backend="rlPyCairo")


def build_addendum():
    files = [WS / "svg/page6.svg"]
    ok = create_pptx_with_native_svg(files, WS / "addendum.pptx", use_native_shapes=True,
        use_compat_mode=False, notes=find_notes_files(WS / "svg", files), transition=None, animation=None)
    assert ok


def resolve(owner, target):
    return posixpath.normpath(posixpath.join(posixpath.dirname(owner), target))


def append_slide():
    """Append one native slide while preserving every original content part."""
    proof = json.loads((WS / "qa/source-proof.json").read_text(encoding="utf-8"))
    source = Path(proof["source_pptx"])
    assert sha(source.read_bytes()) == proof["source_sha256"]
    with zipfile.ZipFile(source) as src, zipfile.ZipFile(WS / "addendum.pptx") as extra:
        original = {n: src.read(n) for n in src.namelist()}
        parts = dict(original)
        additional_types = {}
        repaired_notes_master = False
        original_infos = {info.filename: info for info in src.infolist()}
        presentation = E.fromstring(parts["ppt/presentation.xml"])
        extra_presentation = E.fromstring(extra.read("ppt/presentation.xml"))
        assert presentation.find(f"{{{P}}}sldSz").attrib == extra_presentation.find(f"{{{P}}}sldSz").attrib
        slide_ids = presentation.find(f"{{{P}}}sldIdLst")
        assert len(slide_ids) == 5
        new_slide = "ppt/slides/slide6.xml"
        new_notes = "ppt/notesSlides/notesSlide6.xml"
        assert new_slide not in parts and new_notes not in parts
        parts[new_slide] = extra.read("ppt/slides/slide1.xml")
        rels = E.fromstring(extra.read("ppt/slides/_rels/slide1.xml.rels"))
        new_media = []
        for rel in rels:
            kind, target = rel.get("Type").rsplit("/", 1)[-1], rel.get("Target")
            part = resolve("ppt/slides/slide1.xml", target)
            if kind == "slideLayout":
                assert part in parts and extra.read(part) == parts[part], "Mismatched slide layout"
            elif kind == "notesSlide":
                rel.set("Target", "../notesSlides/notesSlide6.xml")
            elif kind == "image":
                payload = extra.read(part)
                match = next((n for n, b in parts.items() if n.startswith("ppt/media/") and b == payload), None)
                if match is None:
                    match = "ppt/media/latency_" + posixpath.basename(part)
                    assert match not in parts
                    parts[match] = payload
                    new_media.append(match)
                rel.set("Target", posixpath.relpath(match, "ppt/slides"))
            else:
                raise ValueError(f"Unexpected new slide relationship: {kind}")
        parts["ppt/slides/_rels/slide6.xml.rels"] = serialize(rels)
        notes = E.fromstring(extra.read("ppt/notesSlides/notesSlide1.xml"))
        for field in notes.findall(f".//{{{A}}}fld"):
            if (field.get("type") or "").lower() in ("slidenum", "slidenumber"):
                for text in field.findall(f"{{{A}}}t"):
                    text.text = "6"
        parts[new_notes] = serialize(notes)
        note_rels = E.fromstring(extra.read("ppt/notesSlides/_rels/notesSlide1.xml.rels"))
        for rel in note_rels:
            kind = rel.get("Type").rsplit("/", 1)[-1]
            if kind == "slide":
                rel.set("Target", "../slides/slide6.xml")
            elif kind == "notesMaster":
                part = resolve("ppt/notesSlides/notesSlide1.xml", rel.get("Target"))
                if part not in parts:
                    # The earlier skill exporter wrote notes relationships but
                    # omitted their shared master. Add the missing dependency
                    # without modifying any original notes, slide or theme.
                    from pptx import Presentation
                    template_path = WS / "qa/notes-template.pptx"
                    template = Presentation()
                    template.slides.add_slide(template.slide_layouts[6]).notes_slide
                    template.save(template_path)
                    with zipfile.ZipFile(template_path) as note_template:
                        parts[part] = note_template.read("ppt/notesMasters/notesMaster1.xml")
                        master_rels = E.fromstring(note_template.read("ppt/notesMasters/_rels/notesMaster1.xml.rels"))
                        for master_rel in master_rels:
                            assert master_rel.get("Type").endswith("/theme")
                            theme_source = resolve(part, master_rel.get("Target"))
                            theme_bytes = note_template.read(theme_source)
                            theme_target = next((n for n, b in parts.items()
                                if n.startswith("ppt/theme/") and b == theme_bytes), None)
                            if theme_target is None:
                                theme_target = "ppt/theme/latencyNotesTheme.xml"
                                assert theme_target not in parts
                                parts[theme_target] = theme_bytes
                                additional_types[theme_target] = "application/vnd.openxmlformats-officedocument.theme+xml"
                            master_rel.set("Target", posixpath.relpath(theme_target, "ppt/notesMasters"))
                        parts["ppt/notesMasters/_rels/notesMaster1.xml.rels"] = serialize(master_rels)
                    additional_types[part] = "application/vnd.openxmlformats-officedocument.presentationml.notesMaster+xml"
                    repaired_notes_master = True
            else:
                raise ValueError(kind)
        parts["ppt/notesSlides/_rels/notesSlide6.xml.rels"] = serialize(note_rels)
        pr = E.fromstring(parts["ppt/_rels/presentation.xml.rels"])
        rid = "rId" + str(max(int(r.get("Id")[3:]) for r in pr) + 1)
        E.SubElement(pr, f"{{{REL}}}Relationship", Id=rid, Type=R+"/slide", Target="slides/slide6.xml")
        E.SubElement(slide_ids, f"{{{P}}}sldId", {"id": str(max(int(x.get("id")) for x in slide_ids)+1), f"{{{R}}}id": rid})
        if repaired_notes_master:
            master_rid = "rId" + str(max(int(r.get("Id")[3:]) for r in pr) + 1)
            E.SubElement(pr, f"{{{REL}}}Relationship", Id=master_rid,
                         Type=R+"/notesMaster", Target="notesMasters/notesMaster1.xml")
            master_ids = presentation.find(f"{{{P}}}notesMasterIdLst")
            if master_ids is None:
                master_ids = E.Element(f"{{{P}}}notesMasterIdLst")
                before = presentation.find(f"{{{P}}}handoutMasterIdLst")
                if before is None:
                    before = slide_ids
                presentation.insert(list(presentation).index(before), master_ids)
            E.SubElement(master_ids, f"{{{P}}}notesMasterId", {f"{{{R}}}id": master_rid})
        parts["ppt/_rels/presentation.xml.rels"] = serialize(pr)
        parts["ppt/presentation.xml"] = serialize(presentation)
        types = E.fromstring(parts["[Content_Types].xml"])
        extra_types = E.fromstring(extra.read("[Content_Types].xml"))
        for old_name, new_name in (("/ppt/slides/slide1.xml", "/"+new_slide),
                                   ("/ppt/notesSlides/notesSlide1.xml", "/"+new_notes)):
            content_type = next(e.get("ContentType") for e in extra_types if e.get("PartName") == old_name)
            E.SubElement(types, f"{{{CT}}}Override", PartName=new_name, ContentType=content_type)
        for name in new_media:
            extension = name.rsplit(".", 1)[-1]
            if not any(e.get("Extension") == extension for e in types):
                content_type = next(e.get("ContentType") for e in extra_types if e.get("Extension") == extension)
                E.SubElement(types, f"{{{CT}}}Default", Extension=extension, ContentType=content_type)
        for name, content_type in additional_types.items():
            assert not any(e.get("PartName") == "/" + name for e in types)
            E.SubElement(types, f"{{{CT}}}Override", PartName="/"+name, ContentType=content_type)
        parts["[Content_Types].xml"] = serialize(types)
        app = E.fromstring(parts["docProps/app.xml"])
        count = app.find(f"{{{EP}}}Slides")
        if count is not None:
            count.text = "6"
        # Update optional title-list metadata without touching slide content.
        hp = app.find(f"{{{EP}}}HeadingPairs/{{{VT}}}vector")
        title_vector = app.find(f"{{{EP}}}TitlesOfParts/{{{VT}}}vector")
        if hp is not None and title_vector is not None:
            pairs = list(hp)
            for index in range(0, len(pairs)-1, 2):
                heading = "".join(pairs[index].itertext())
                if heading == "Slide Titles":
                    count_node = list(pairs[index+1])[0]
                    old_count = int(count_node.text)
                    assert old_count in (0, 5)
                    start = 1 if old_count == 0 else 6
                    for n in range(start, 7):
                        title = next(E.fromstring(parts[f"ppt/slides/slide{n}.xml"]).iter(f"{{{A}}}t")).text or ""
                        E.SubElement(title_vector, f"{{{VT}}}lpstr").text = title
                    count_node.text = "6"
                    title_vector.set("size", str(len(title_vector)))
        parts["docProps/app.xml"] = serialize(app)
        allowed_changes = {"ppt/presentation.xml", "ppt/_rels/presentation.xml.rels",
                           "[Content_Types].xml", "docProps/app.xml"}
        preserved = []
        for name, payload in original.items():
            if name not in allowed_changes:
                assert parts[name] == payload, "Original content changed: " + name
                preserved.append(name)
        # All internal relationship targets must resolve to actual package parts.
        for name, payload in parts.items():
            if not name.endswith(".rels"):
                continue
            owner = posixpath.join(posixpath.dirname(posixpath.dirname(name)), posixpath.basename(name)[:-5])
            for rel in E.fromstring(payload):
                if rel.get("TargetMode") != "External":
                    target = resolve(owner, rel.get("Target").split("#", 1)[0])
                    assert target in parts, (name, target)
        output = WS / "candidate.pptx"
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as target:
            for name, payload in parts.items():
                target.writestr(original_infos.get(name, name), payload)
    with zipfile.ZipFile(output) as check:
        assert check.testzip() is None
        assert len([n for n in check.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)]) == 6
        new_root = E.fromstring(check.read(new_slide))
        assert len(new_root.findall(f".//{{{P}}}sp")) > 30
        assert not new_root.findall(f".//{{{P}}}timing") and not new_root.findall(f".//{{{P}}}transition")
    audit = {"passed": True, "source_sha256": proof["source_sha256"],
        "output_sha256": sha(output.read_bytes()), "slides": 6, "appended_slide": 6,
        "original_content_parts_byte_identical": len(preserved), "preserved_parts": preserved,
        "changed_container_parts_only": sorted(allowed_changes),
        "original_five_slide_and_notes_xml_unchanged": True, "all_internal_relationships_resolve": True,
        "missing_original_notes_master_dependency_added": repaired_notes_master,
        "new_slide_native_shapes": len(new_root.findall(f".//{{{P}}}sp")),
        "new_slide_animations": 0}
    (WS / "qa/preservation.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in audit.items() if k != "preserved_parts"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "all"
    if action in ("notes", "all"):
        write_notes()
    if action in ("render", "all"):
        render_svg()
    if action in ("build", "all"):
        build_addendum()
    if action in ("build", "append", "all"):
        append_slide()
