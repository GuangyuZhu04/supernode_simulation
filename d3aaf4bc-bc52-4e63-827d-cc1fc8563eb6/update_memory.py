"""Update only page 2 of the supplied deck using audited Kimi simulation data."""
from pathlib import Path
import copy
import csv
import hashlib
import json
import sys
import zipfile
from lxml import etree as E

WS = Path(__file__).resolve().parent
ROOT = WS.parent
SKILL = Path(r"C:\Users\Guangyu\.codex\skills\ppt-master-huawei-v2")
DATA = ROOT / "outputs/kimi-k3-10t-1m-acceleration-20261019"
SOURCE = ROOT / "outputs/presentations/kimi-k3-10t-brief-20261020-v2.pptx"
P = "http://schemas.openxmlformats.org/presentationml/2006/main"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS = {"p": P, "a": A, "r": R}
SVG = {"s": "http://www.w3.org/2000/svg"}
SHA = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
serialize = lambda n: E.tostring(n, encoding="UTF-8", xml_declaration=True, standalone=True)


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def prepare():
    assert SOURCE.read_bytes() == (WS / "input/source.pptx").read_bytes()
    paths = [
        DATA / "parallelism-20261020/summary.csv",
        DATA / "parallelism-20261020/verification.json",
        DATA / "decode-batch-20261020/summary.csv",
        DATA / "decode-batch-20261020/verification.json",
    ]
    assert json.loads(paths[1].read_text())["passed"]
    assert json.loads(paths[3].read_text())["checks_passed"]
    all_rows = read_csv(paths[0])
    decode = read_csv(paths[2])
    rows = [r for r in all_rows if r["context"] == "1M" and r["tp"] == "32"
            and r["ep"] == "32" and r["dp"] == "1" and r["sp"] == "32"]
    assert len(rows) == 12
    selected = []
    for batch in [1, 2, 4, 8, 16, 32]:
        pair = [r for r in rows if int(r["global_batch"]) == batch]
        assert {r["chip"] for r in pair} == {"950dt", "960dt"}
        for r in pair:
            assert r["status"] == r["prefill_status"] == r["decode_status"] == "feasible"
            assert int(r["local_batch"]) == batch
            assert int(r["provisioned_cards"]) == 1024
            match = [d for d in decode if d["context"] == "1M"
                     and d["stage"] == "F_kv" and d["chip"] == r["chip"]
                     and int(d["batch"]) == batch]
            assert len(match) == 1
            d = match[0]
            assert abs(float(r["decode_peak_hbm_GB"]) -
                       float(d["decode_peak_hbm_bytes_max_per_card"]) / 1e9) < 1e-9
            assert abs(float(r["resident_cluster_TB"]) -
                       float(d["resident_hbm_bytes_all_cards_at_end"]) / 1e12) < 1e-9
        fields = ["prefill_peak_hbm_GB", "decode_peak_hbm_GB", "resident_cluster_TB"]
        assert all(pair[0][key] == pair[1][key] for key in fields)
        selected.append({"batch": batch, **{key: float(pair[0][key]) for key in fields},
                         "source_cases": [r["replica_case"] for r in pair]})
    dump(WS / "input/selected-memory.json", selected)
    dump(WS / "qa/source-proof.json", {
        "source": str(SOURCE), "source_sha256": SHA(SOURCE),
        "inputs": [{"path": str(p), "sha256": SHA(p)} for p in paths],
        "points_checked": 12, "independent_decode_points_matched": 12,
        "scope": "F / 1M / 1024 cards / TP32 EP32 SP32 PP1 DP1 / top32 / output2",
    })
    svg = E.parse(str(WS / "svg/page2.svg"))
    group = svg.xpath("//s:g[@id='memory-requirement']", namespaces=SVG)[0]
    values = group.xpath(".//s:text/text()", namespaces=SVG)
    for item in selected:
        assert f'{item["prefill_peak_hbm_GB"]:.2f}' in values
        assert f'{item["decode_peak_hbm_GB"]:.2f}' in values
    assert values.count("61.44") == 5 and values.count("42.64") == 6
    assert "28.81 → 43.67" in values
    table_md = "| batch size | Prefill 单卡峰值 GB | Decode 单卡峰值 GB | 输出末端集群常驻 TB |\n"
    table_md += "|---|---|---|---|\n"
    for item in selected:
        table_md += f'| {item["batch"]} | {item["prefill_peak_hbm_GB"]:.2f} | {item["decode_peak_hbm_GB"]:.2f} | {item["resident_cluster_TB"]:.2f} |\n'
    (WS / "content/content2.md").write_text(
        "<!-- 逻辑套路：问题分析；保留原页三栏及三段式布局 -->\n"
        "# 10T 推理需同时满足显存峰值、HBM 带宽与低时延互联\n\n"
        "1M、F、TP32×EP32、SP32、PP=DP=1、top-32、output=2。\n\n"
        + table_md + "\n两平台显存数值相同；batch 为整个 DP 副本的并发序列数。"
        "batch≤EP32 时每个活跃 EP lane 最多持有一条序列，单卡峰值不按 batch 倍增。"
        "集群常驻为同一输出末端时刻的和，不是各卡独立峰值之和。"
        "计算和互联继续采用原页 batch=1 指标。GB/TB 为十进制单位。\n",
        encoding="utf-8")
    with zipfile.ZipFile(SOURCE) as archive:
        texts = []
        for i in range(1, 7):
            slide = E.fromstring(archive.read(f"ppt/slides/slide{i}.xml"))
            texts.append(f"# Slide {i}\n" + "\n".join(slide.xpath("//a:t/text()", namespaces=NS)))
        (WS / "input/source.md").write_text("\n\n".join(texts), encoding="utf-8")

    note = json.loads((WS / "notes/page2.json").read_text(encoding="utf-8"))
    note["speech"] = (
        "这一页保留显存、计算与访存、互联三个资源维度。统一部署为1024卡，"
        "TP32乘EP32，SP32复用TP ranks，PP和DP均为1；固定1M上下文、top-32、output=2、完整组合F。"
        "本次显存表采用2026年10月20日最新并行扫描中的原布局切片，"
        "并与同日独立decode batch扫描逐点交叉核对；prefill和decode各展示batch为1、2、4、8、16、32的结果。"
        "batch表示整个DP副本并发处理的序列数，不是每卡batch。"
        "950DT与960DT的单卡阶段峰值完全相同：prefill在batch 1至16为61.435794595GB，"
        "batch32为62.673769635GB，增加约2.015%；decode六档均为42.642575567GB。"
        "表中均按两位小数展示。所有这12个完整请求点均完成prefill与decode，"
        "未使用失败前的峰值推断完整请求容量。两个阶段的batch各自变化，"
        "本表不表示测试了prefill batch与decode batch不同的分离部署。"
        "单卡峰值近似稳定是EP32请求分布的结果：batch不超过32时，每个活跃EP lane最多持有一条序列，"
        "新增请求主要启用更多lane。prefill在batch32仍存在额外批量工作集峰值，"
        "因此不能把所有显存简单按batch线性外推，也不能泛化到batch大于32或其他并行布局。"
        "集群输出末端常驻量随batch逐档为28.810、29.289、30.248、32.164、35.998、43.665TB，"
        "从batch1到32增长约51.563%。这是一致时刻的常驻量加和，不是单卡峰值乘1024，也不是各卡独立峰值之和。"
        "原页静态预算仍适用：FP8逻辑权重10.17TB，物理权重含复制与scale为19.88TB，"
        "每卡runtime为8GB，workspace为256MB，均已计入相应容量账本。"
        "计算与互联两栏继续使用batch1基线：每卡HBM容量144/288GB、带宽4/9.6TB每秒，"
        "FP8标称峰值1/2PFLOP每秒，计算效率50%；prefill计算1.611EFLOP，"
        "decode集群累计HBM读写8.34TB每步。互联为单逻辑非阻塞SU，SO为0，"
        "单向带宽1/1.1TB每秒，效率80%，启动及传播时延2.1/1微秒，"
        "decode关键路径通信启动占比约86%/87%。这些流量和时延占比不能套用到更大batch。"
        "GB和TB均为十进制单位。所有数据是未实测校准的模型预测，FP8质量也未验证。"
        "第4至5页继续讨论其他并行布局，不能将本页峰值直接套用到所有候选。"
    )
    for g in note["glossary"]:
        if g["term"] == "batch":
            g["term"] = "batch / batch size"
            g["explain"] = "本页为整个DP副本的并发序列数；DP1时等于1024卡模型实例并发，显存扫描1/2/4/8/16/32，其他资源用batch1。"
    note["glossary"] = [g for g in note["glossary"] if g["term"] != "EP lane"]
    note["glossary"].append({
        "term": "EP lane", "full": "Expert Parallel lane，专家并行请求通道",
        "explain": "请求按本地序号模EP分布；EP32且batch≤32时，每个活跃lane最多持有一条序列，lane内由TP32共同处理。",
    })
    note["red_notes"] = [r for r in note["red_notes"] if r["quote"] not in
                         ["61.44", "SP 与压缩、量化共同控制显存峰值", "62.67", "28.81 → 43.67",
                          "batch≤EP32：每个活跃 EP lane 最多 1 条序列，单卡峰值不按 batch 倍增"]]
    note["red_notes"] += [
        {"quote": "62.67", "explain": "2026-10-20并行扫描，1M、F、TP32/EP32/SP32/DP1、batch32，两平台prefill单卡峰值62.673769635GB，较batch1的61.435794595GB增长约2.015%。"},
        {"quote": "28.81 → 43.67", "explain": "同一配置下batch1到32的输出末端集群常驻量由28.810017063936TB增至43.665410067456TB，约增51.563%；不是各卡峰值之和。"},
        {"quote": "batch≤EP32：每个活跃 EP lane 最多 1 条序列，单卡峰值不按 batch 倍增",
         "explain": "本轮请求按序号模EP映射至lane；batch从1到32只增加活跃lane，单lane最大序列数保持1，不能据此推断其他布局或更大batch。"},
    ]
    note["sources"] = [
        str(DATA / "parallelism-20261020/summary.csv") + "；筛选1M、TP32/EP32/DP1/SP32，batch1/2/4/8/16/32，两平台12点；prefill_peak_hbm_GB、decode_peak_hbm_GB、resident_cluster_TB。",
        str(DATA / "decode-batch-20261020/summary.csv") + "；筛选1M、F_kv及相同batch；decode_peak_hbm_bytes_max_per_card / 1e9、resident_hbm_bytes_all_cards_at_end / 1e12。",
        str(DATA / "parallelism-20261020/README.txt") + "；batch、EP请求分布与阶段计时定义。",
        str(DATA / "report.md") + "；模型与硬件条件，物理权重含scale19.877TB，runtime8GB、workspace256MB。",
        str(DATA / "phase_metrics.csv") + "；F、1M、batch1的prefill计算1.611EFLOP、decode HBM累计读写8.338TB/步。",
        str(DATA / "latency_breakdown.csv") + "；F、1M、batch1的decode通信启动占比。",
    ]
    dump(WS / "notes/page2.json", note)
    dump(WS / "notes.json", {"pages": [note]})
    text = f"【第2页 · {note['title']}】\n\n{note['speech']}\n\n过渡：{note['transition']}\n\n"
    text += "【本页名词表 · 英文简称：英文全称，中文全称——一句话解释】\n"
    text += "\n".join(f"- {g['term']}：{g['full']}——{g['explain']}" for g in note["glossary"])
    text += "\n\n【标红内容详解 · 逐条一句话解释】\n"
    text += "\n".join(f"- 「{r['quote']}」：{r['explain']}" for r in note["red_notes"])
    text += "\n\n【来源与数据口径】\n" + "\n".join("- " + s for s in note["sources"])
    text += "\n\n【六档显存数据】\n" + table_md
    (WS / "svg/notes/page2.md").write_text(text, encoding="utf-8")
    print("Prepared: 12 complete prefill/decode points; all independent decode data match.")


def native_table(rows):
    """Native editable PowerPoint table, styled to the SVG reference."""
    emu = lambda px: str(round(px * 9525))
    q = lambda prefix, tag: f"{{{NS[prefix]}}}{tag}"
    frame = E.Element(q("p", "graphicFrame"))
    nv = E.SubElement(frame, q("p", "nvGraphicFramePr"))
    E.SubElement(nv, q("p", "cNvPr"), id="1", name="显存峰值 batch 对照表")
    E.SubElement(nv, q("p", "cNvGraphicFramePr"))
    E.SubElement(nv, q("p", "nvPr"))
    xfrm = E.SubElement(frame, q("p", "xfrm"))
    E.SubElement(xfrm, q("a", "off"), x=emu(70), y=emu(262))
    E.SubElement(xfrm, q("a", "ext"), cx=emu(340), cy=emu(187))
    graphic = E.SubElement(frame, q("a", "graphic"))
    gd = E.SubElement(graphic, q("a", "graphicData"), uri=A + "/table")
    table = E.SubElement(gd, q("a", "tbl"))
    E.SubElement(table, q("a", "tblPr"), firstRow="0", bandRow="0")
    grid = E.SubElement(table, q("a", "tblGrid"))
    for width in [100, 140, 100]:
        E.SubElement(grid, q("a", "gridCol"), w=emu(width))
    values = [["batch size", "Prefill", "Decode"]] + [
        [str(r["batch"]), f'{r["prefill_peak_hbm_GB"]:.2f}', f'{r["decode_peak_hbm_GB"]:.2f}']
        for r in rows]
    for ridx, row in enumerate(values):
        tr = E.SubElement(table, q("a", "tr"), h=emu(28 if ridx == 0 else 26.5))
        for cidx, value in enumerate(row):
            cell = E.SubElement(tr, q("a", "tc"))
            tx = E.SubElement(cell, q("a", "txBody"))
            E.SubElement(tx, q("a", "bodyPr"))
            E.SubElement(tx, q("a", "lstStyle"))
            para = E.SubElement(tx, q("a", "p"))
            pp = E.SubElement(para, q("a", "pPr"), algn="ctr")
            E.SubElement(pp, q("a", "buNone"))
            run = E.SubElement(para, q("a", "r"))
            rp = E.SubElement(run, q("a", "rPr"), lang="en-US",
                              sz="900" if ridx == 0 else "1125",
                              b="1" if ridx == 0 or (ridx == 6 and cidx < 2) else "0")
            fill = E.SubElement(rp, q("a", "solidFill"))
            E.SubElement(fill, q("a", "srgbClr"), val="C00000" if (ridx, cidx) == (6, 1) else "1D1D1A")
            E.SubElement(rp, q("a", "latin"), typeface="Arial")
            E.SubElement(rp, q("a", "ea"), typeface="微软雅黑")
            E.SubElement(run, q("a", "t")).text = value
            props = E.SubElement(cell, q("a", "tcPr"), marL="0", marR="0",
                                 marT="0", marB="0", anchor="ctr")
            for side in ["L", "R", "T", "B"]:
                line = E.SubElement(props, q("a", "ln" + side), w="6350")
                if side == "B":
                    lf = E.SubElement(line, q("a", "solidFill"))
                    E.SubElement(lf, q("a", "srgbClr"), val="BFBFBF" if ridx == 0 else "D9D9D9")
                    E.SubElement(line, q("a", "prstDash"), val="solid")
                else:
                    E.SubElement(line, q("a", "noFill"))
            fill = E.SubElement(props, q("a", "solidFill"))
            E.SubElement(fill, q("a", "srgbClr"), val="FFF1EF" if ridx == 6 else "FFFFFF")
    return frame


def build():
    sys.path.insert(0, str(SKILL / "scripts"))
    from svg_to_pptx import create_pptx_with_native_svg
    from svg_to_pptx.pptx_discovery import find_notes_files
    files = [WS / "svg/page2.svg"]
    assert create_pptx_with_native_svg(
        files, WS / "page2-native.pptx", use_native_shapes=True, use_compat_mode=False,
        notes=find_notes_files(WS / "svg", files), transition=None, animation=None)
    proof = json.loads((WS / "qa/source-proof.json").read_text(encoding="utf-8"))
    assert SHA(SOURCE) == proof["source_sha256"]
    with zipfile.ZipFile(WS / "input/source.pptx") as original, zipfile.ZipFile(WS / "page2-native.pptx") as donor:
        slide = E.fromstring(original.read("ppt/slides/slide2.xml"))
        incoming = E.fromstring(donor.read("ppt/slides/slide1.xml"))
        src_groups = slide.findall("p:cSld/p:spTree/p:grpSp", NS)
        new_groups = incoming.findall("p:cSld/p:spTree/p:grpSp", NS)
        assert len(src_groups) == len(new_groups) == 8
        untouched_groups = {i: E.tostring(src_groups[i], method="c14n") for i in [3, 4, 7]}
        memory = new_groups[2]
        assert "显存：区分 prefill 与 decode" in memory.xpath(".//a:t/text()", namespaces=NS)
        # Replace the SVG grid with a real editable table.
        removed = 0
        for shape in list(memory):
            pos = shape.find("p:spPr/a:xfrm/a:off", NS)
            if pos is not None and 261 <= int(pos.get("y")) / 9525 <= 450:
                memory.remove(shape)
                removed += 1
        assert removed == 28, removed
        rows = json.loads((WS / "input/selected-memory.json").read_text())
        memory.append(native_table(rows))
        next_id = max(int(n.get("id")) for n in slide.findall(".//p:cNvPr", NS)) + 1
        for index in [0, 1, 2, 5, 6]:
            replacement = copy.deepcopy(new_groups[index])
            assert not replacement.xpath(".//@r:embed", namespaces=NS)
            for prop in replacement.findall(".//p:cNvPr", NS):
                prop.set("id", str(next_id))
                next_id += 1
            src_groups[index].getparent().replace(src_groups[index], replacement)
        final_groups = slide.findall("p:cSld/p:spTree/p:grpSp", NS)
        assert all(E.tostring(final_groups[i], method="c14n") == val
                   for i, val in untouched_groups.items())
        notes = E.fromstring(donor.read("ppt/notesSlides/notesSlide1.xml"))
        for field in notes.findall(".//a:fld", NS):
            if field.get("type", "").lower() in ("slidenum", "slidenumber"):
                for text in field.findall("a:t", NS):
                    text.text = "2"
        modified = {
            "ppt/slides/slide2.xml": serialize(slide),
            "ppt/notesSlides/notesSlide2.xml": serialize(notes),
        }
        target = WS / "candidate.pptx"
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as output:
            for info in original.infolist():
                output.writestr(info, modified.get(info.filename, original.read(info.filename)))
    verify()


def verify():
    proof = json.loads((WS / "qa/source-proof.json").read_text(encoding="utf-8"))
    assert SHA(SOURCE) == proof["source_sha256"]
    for item in proof["inputs"]:
        assert SHA(Path(item["path"])) == item["sha256"]
    with zipfile.ZipFile(WS / "input/source.pptx") as a, zipfile.ZipFile(WS / "candidate.pptx") as b:
        assert a.namelist() == b.namelist()
        changed = [name for name in a.namelist() if a.read(name) != b.read(name)]
        assert set(changed) == {"ppt/slides/slide2.xml", "ppt/notesSlides/notesSlide2.xml"}
        slide = E.fromstring(b.read("ppt/slides/slide2.xml"))
        ids = [p.get("id") for p in slide.findall(".//p:cNvPr", NS)]
        assert len(ids) == len(set(ids))
        assert not slide.findall(".//p:timing", NS)
        tables = slide.findall(".//a:tbl", NS)
        assert len(tables) == 1
        rows = tables[0].findall("a:tr", NS)
        selected = json.loads((WS / "input/selected-memory.json").read_text())
        assert len(rows) == 7
        for row, expected in zip(rows[1:], selected):
            assert row.xpath(".//a:t/text()", namespaces=NS) == [
                str(expected["batch"]), f'{expected["prefill_peak_hbm_GB"]:.2f}',
                f'{expected["decode_peak_hbm_GB"]:.2f}']
        for name in b.namelist():
            if name.endswith((".xml", ".rels")):
                E.fromstring(b.read(name))
        count = len(E.fromstring(b.read("ppt/presentation.xml")).find("p:sldIdLst", NS))
        assert count == 6
        notes = E.fromstring(b.read("ppt/notesSlides/notesSlide2.xml"))
        full_notes = "\n".join(notes.xpath("//a:t/text()", namespaces=NS))
        assert all(s in full_notes for s in ["62.673769635", "42.642575567",
                                           "【本页名词表", "【标红内容详解", "过渡："])
    dump(WS / "qa/preservation.json", {
        "passed": True, "changed_parts": changed, "slides": 6,
        "unchanged_slides": [1, 3, 4, 5, 6],
        "other_parts_byte_identical": len(a.namelist()) - 2,
        "page2_compute_network_footer_groups_preserved": True,
        "native_table_rows": 7, "native_table_columns": 3,
        "native_table_values_match_simulation": True,
        "source_pptx_unchanged": True, "candidate_sha256": SHA(WS / "candidate.pptx"),
    })
    print("Verified: only slide2.xml and its speaker notes changed; 6 slides; native 7x3 table.")


if __name__ == "__main__":
    {"prepare": prepare, "build": build, "verify": verify}[sys.argv[1]]()
