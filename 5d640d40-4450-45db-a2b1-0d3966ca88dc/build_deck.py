from pathlib import Path
import sys,json,zipfile,hashlib,base64,re
from xml.etree import ElementTree as ET
WS=Path(__file__).resolve().parent
SKILL=Path(r"C:\Users\Guangyu\.codex\skills\ppt-master-huawei-v2")
sys.path.insert(0,str(SKILL/"scripts"))
from svg_to_pptx import create_pptx_with_native_svg
from svg_to_pptx.pptx_discovery import find_notes_files
from svg_to_pptx.use_expander import expand_use_data_icons

def numbers():
    return sorted(int(p.stem[4:]) for p in (WS/"svg").glob("page*.svg"))

def write_notes():
    pages=[]
    for n in numbers():
        p=json.loads((WS/f"notes/page{n}.json").read_text(encoding="utf8"))
        pages.append(p)
        speech=p["speech"]
        if isinstance(speech,list): speech="\n\n".join(speech)
        txt=f"【第{n}页 · {p['title']}】\n\n{speech}\n\n过渡：{p['transition']}\n\n"
        txt+="【本页名词表 · 英文简称：英文全称，中文全称——一句话解释】\n"
        for g in p["glossary"]: txt+=f"- {g['term']}：{g['full']}——{g['explain']}\n"
        txt+="\n【标红内容详解 · 逐条一句话解释】\n"
        for g in p["red_notes"]: txt+=f"- 「{g['quote']}」：{g['explain']}\n"
        txt+="\n【来源与数据口径】\n"
        for s in p.get("sources",[]): txt+=f"- {s}\n"
        if "raw_latency_components" in p:
            txt+="\n【原始资源时延六项明细】\n"+json.dumps(p["raw_latency_components"],ensure_ascii=False,indent=2)+"\n"
        (WS/f"svg/notes/page{n}.md").write_text(txt,encoding="utf8")
    (WS/"notes.json").write_text(json.dumps({"pages":pages},ensure_ascii=False,indent=2),encoding="utf8")
    print(f"Notes generated for {len(pages)} slides.",flush=True)

def render_svg():
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.graphics import renderPM
    from svglib.svglib import svg2rlg
    for name,file in [("微软雅黑","msyh.ttc"),("微软雅黑-Bold","msyhbd.ttc"),("Arial","arial.ttf"),("Arial-Bold","arialbd.ttf")]:
        pdfmetrics.registerFont(TTFont(name,str(Path(r"C:\Windows\Fonts")/file)))
    pdfmetrics.registerFontFamily("微软雅黑",normal="微软雅黑",bold="微软雅黑-Bold",italic="微软雅黑",boldItalic="微软雅黑-Bold")
    pdfmetrics.registerFontFamily("Arial",normal="Arial",bold="Arial-Bold",italic="Arial",boldItalic="Arial-Bold")
    from svglib.fonts import register_font
    register_font("微软雅黑", rlgFontName="微软雅黑")
    register_font("微软雅黑", weight="bold", rlgFontName="微软雅黑-Bold")
    register_font("Arial", rlgFontName="Arial")
    register_font("Arial", weight="bold", rlgFontName="Arial-Bold")
    ET.register_namespace("","http://www.w3.org/2000/svg")
    for n in numbers():
        root=ET.parse(WS/f"svg/page{n}.svg").getroot()
        expand_use_data_icons(root,SKILL/"assets/icons")
        for e in root.iter():
            if e.tag.endswith("text") or e.tag.endswith("tspan"):
                e.set("font-family","微软雅黑")
            if e.tag.endswith("image") and e.get("href")=="logo.png":
                e.set("href","data:image/png;base64,"+base64.b64encode((WS/"svg/logo.png").read_bytes()).decode())
        preview=WS/f"previews/page{n}.svg"
        ET.ElementTree(root).write(preview,encoding="utf-8",xml_declaration=True)
        drawing=svg2rlg(str(preview))
        renderPM.drawToFile(drawing,str(WS/f"previews/page{n}.png"),fmt="PNG",backend="rlPyCairo")
        print(f"Rendered SVG page{n}",flush=True)

def build():
    nn=numbers()
    assert nn==list(range(1,9)),nn
    files=[WS/f"svg/page{n}.svg" for n in nn]
    notes=find_notes_files(WS/"svg",files)
    out=WS/"candidate.pptx"
    ok=create_pptx_with_native_svg(files,out,use_native_shapes=True,use_compat_mode=False,notes=notes,transition=None,animation=None)
    assert ok
    ns={"p":"http://schemas.openxmlformats.org/presentationml/2006/main","a":"http://schemas.openxmlformats.org/drawingml/2006/main"}
    audit={"slide_count":0,"notes_count":0,"slides":[]}
    extracts=[]
    with zipfile.ZipFile(out) as z:
        for n in nn:
            x=ET.fromstring(z.read(f"ppt/slides/slide{n}.xml"))
            row={"page":n,"text_count":len(x.findall(".//a:t",ns)),"shape_count":len(x.findall(".//p:sp",ns)),"groups":len(x.findall(".//p:grpSp",ns)),"animations":len(x.findall(".//p:timing",ns)),"transitions":len(x.findall(".//p:transition",ns))}
            assert row["text_count"]>20 and row["shape_count"]>20
            assert row["animations"]==0 and row["transitions"]==0
            audit["slides"].append(row)
            texts=[e.text or "" for e in x.findall(".//a:t",ns)]
            extracts.append(f"# Page {n}\n\n"+"\n".join(texts))
            assert str(n) in texts
            assert "Huawei Proprietary - Restricted Distribution" in texts
            note="".join(ET.fromstring(z.read(f"ppt/notesSlides/notesSlide{n}.xml")).itertext())
            for title in ["过渡","本页名词表","标红内容详解","来源与数据口径"]: assert title in note
            audit["notes_count"]+=1
        names=z.namelist()
        audit["slide_count"]=sum(1 for a in names if re.fullmatch(r"ppt/slides/slide\d+\.xml",a))
        assert audit["slide_count"]==8
    audit["sha256"]=hashlib.sha256(out.read_bytes()).hexdigest()
    (WS/"qa/pptx-structure.json").write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding="utf8")
    (WS/"qa/extracted-pptx.txt").write_text("\n\n".join(extracts),encoding="utf8")
    print(json.dumps(audit,ensure_ascii=False,indent=2))

def contact(source="office"):
    from PIL import Image,ImageDraw,ImageFont
    paths=sorted((WS/"previews/office").glob("slide-*.png")) if source=="office" else sorted((WS/"previews").glob("page*.png"),key=lambda p:int(p.stem[4:]))
    assert len(paths)==8,len(paths)
    canvas=Image.new("RGB",(1920,2240),"#e9e9e9")
    draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf",18)
    for i,p in enumerate(paths):
        x=(i%2)*960+10;y=(i//2)*560+22
        canvas.paste(Image.open(p).convert("RGB").resize((940,529)),(x,y))
        draw.text((x,y-21),f"{i+1:02d}",fill="#444444",font=font)
    canvas.save(WS/"previews/contact-sheet.png")
    print("Contact sheet saved.")

if __name__=="__main__":
    op=sys.argv[1] if len(sys.argv)>1 else "all"
    if op in ("all","notes"): write_notes()
    if op in ("all","render"): render_svg()
    if op in ("all","build"): build()
    if op=="contact":contact()
    if op=="contact-svg":contact("svg")
