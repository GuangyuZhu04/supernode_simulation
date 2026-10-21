from pathlib import Path
from html import escape
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET

WS = Path(__file__).resolve().parents[1]
data_path = WS / "input/latency-data.json"
data = json.loads(data_path.read_text(encoding="utf-8"))
rows = data["rows"]
FONT = "微软雅黑, Arial, sans-serif"
RED = "#C00000"
BLACK = "#1D1D1A"
GRAY = "#666666"
LINE = "#D9D9D9"
COLORS = ["#8C8C8C", "#D9D9D9", "#F1B4B7", "#C00000"]
LABELS = ["计算主导", "HBM主导", "通信传输", "通信启动"]
TITLE = "时延拆解显示：prefill重计算与传输，decode主要受通信启动约束"
svg = []


def put(s):
    svg.append(s)


def text(x, y, value, size=12, fill=BLACK, bold=False, anchor="start", raw=False, id=None):
    extra = f' id="{id}"' if id else ""
    put(
        f'<text{extra} x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
        f'fill="{fill}" font-weight="{"bold" if bold else "normal"}" '
        f'text-anchor="{anchor}">{value if raw else escape(str(value))}</text>'
    )


def span(s):
    return f'<tspan fill="{RED}" font-weight="bold">{escape(s)}</tspan>'


def rect(x, y, w, h, fill, stroke=None, sw=1, rx=None, id=None):
    extra = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
    if rx is not None:
        extra += f' rx="{rx}"'
    if id:
        extra += f' id="{id}"'
    put(f'<rect x="{x:.9f}" y="{y}" width="{w:.9f}" height="{h}" fill="{fill}"{extra}/>')


def line(x1, y1, x2, y2, fill=LINE):
    put(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{fill}" stroke-width="1"/>')


def icon(name, x, y, wh):
    put(f'<use data-icon="chunk-filled/{name}" x="{x}" y="{y}" width="{wh}" height="{wh}" fill="{RED}"/>')


put('<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720">')
put('<g id="page-background">')
rect(0, 0, 1280, 720, "#FFFFFF")
put("</g>")
put('<g id="page-header">')
text(50, 50, TITLE, 28, RED, True)
text(
    50, 77,
    "06 时延拆解 · 同第4页六组｜1M · 全局bs32 · 每平台1024卡 · F · top32 · output2｜2026-10-20",
    12, GRAY,
)
put("</g>")
put('<g id="core-message">')
rect(50, 96, 1180, 62, "#FFF1EF", RED, 1.5, 4)
icon("target-arrow", 70, 115, 22)
text(104, 132, "核心思想", 15, RED, True)
line(180, 110, 180, 146, "#F1B4B7")
text(
    200, 132,
    span("保TP减EP") + "：prefill收益来自" + span("传输缩短")
    + "，decode收益主要来自" + span("通信启动减少") + "。",
    15, BLACK, True, raw=True,
)
put("</g>")

bar_ys = [237, 258, 279, 311, 332, 353]
table_ys = [424, 446, 468, 490, 512, 534]
bar_geometry = []
for phase, px, plotx, xmax, ticks, title in (
    ("prefill", 50, 180, 180, [0, 60, 120, 180], "prefill / TTFT 时延拆解（s）"),
    ("decode", 652, 782, 45, [0, 15, 30, 45], "decode / TPOT 时延拆解（ms）"),
):
    put(f'<g id="{phase}-latency-panel">')
    rect(px, 172, 578, 380, "#FFFFFF", LINE, 1, 4)
    icon("chart-bar" if phase == "prefill" else "chart-line", px + 15, 176, 20)
    text(px + 40, 191, title, 15, RED, True)
    for i, (label, color) in enumerate(zip(LABELS, COLORS)):
        lx = px + 16 + i * 130
        rect(lx, 207, 9, 9, color)
        text(lx + 15, 215, label, 10.5)
    # A common linear scale is used within each phase.  Zero means zero width.
    for tick in ticks:
        tx = plotx + tick / xmax * 385
        line(tx, 226, tx, 362, "#F2F2F2")
        line(tx, 362, tx, 365, "#BFBFBF")
        text(tx, 377, str(tick), 10.5, GRAY, anchor="middle")
    line(plotx, 362, plotx + 385, 362, "#BFBFBF")
    line(px + 16, 295, px + 562, 295, "#F2F2F2")
    for row_i, (row, cy) in enumerate(zip(rows, bar_ys)):
        vals = row[phase]["primary"]
        assert len(vals) == 4
        assert all(v >= 0 for v in vals)
        text(px + 16, cy + 3.5, row["chart_label"], 10.5)
        cursor = plotx
        for j, (value, color) in enumerate(zip(vals, COLORS)):
            width = value / xmax * 385
            if width > 0:
                rect(cursor, cy - 6, width, 12, color, id=f"{phase}-row{row_i + 1}-component{j + 1}")
            bar_geometry.append({
                "phase": phase, "case": row["name"], "component": data["primary_keys"][j],
                "value": value, "unit": row[phase]["unit"], "x": cursor,
                "width": width, "scale_max": xmax, "plot_width": 385,
            })
            cursor += width
        total = row[phase]["total"]
        text(plotx + total / xmax * 385 + 6, cy + 3.5, row[phase]["display"][4], 10.5)
    # Widths are the blueprint's exact 80/93/83/93/93/104 px.
    tx = px + 16
    widths = [80, 93, 83, 93, 93, 104]
    centers = []
    cursor = tx
    for width in widths:
        centers.append(cursor + width / 2)
        cursor += width
    rect(tx, 389, 546, 20, "#F2F2F2")
    for cx, label in zip(centers, ["平台/策略"] + LABELS + ["合计"]):
        text(cx, 402, label, 12, BLACK, True, "middle")
    for row_i, (row, ty) in enumerate(zip(rows, table_ys)):
        # The two balanced cases receive a subtle row fill; all numerical
        # components still come from the source display strings.
        if row["label"] == "兼顾时延":
            rect(tx, ty - 15, 546, 22, "#FFF1EF")
        text(centers[0], ty, row["short_label"], 12, BLACK, row["label"] == "兼顾时延", "middle")
        for j, (cx, value) in enumerate(zip(centers[1:], row[phase]["display"])):
            startup_emphasis = phase == "decode" and row["label"] == "兼顾时延" and j == 3
            text(
                cx, ty, value, 12, RED if startup_emphasis else BLACK,
                startup_emphasis or j == 4, "middle",
                id=f"{phase}-table-row{row_i + 1}-value{j + 1}",
            )
        if row_i < 5:
            line(tx, ty + 7, tx + 546, ty + 7, LINE)
    put("</g>")

put('<g id="attribution-footnote">')
text(
    50, 565,
    "其他=等待/争用/kernel启动/其他服务，合计<10^-9s；分项独立舍入。主导服务归因不双计计算/HBM重叠；模型预测。",
    10.5, GRAY,
)
put("</g>")
gain950 = f'{data["gains"]["950dt"]["balanced_decode_startup_ms_saved"]:.3f}'
gain960 = f'{data["gains"]["960dt"]["balanced_decode_startup_ms_saved"]:.3f}'
gain_quote = f"{gain950} / {gain960} ms"
put('<g id="page-highlights">')
rect(50, 574, 1180, 68, "#F2F2F2", LINE, 1, 4)
text(72, 601, "本页亮点", 15, RED, True)
text(
    180, 601,
    "兼顾组合：950DT/960DT的decode通信启动分别减少" + span(gain_quote) + "。",
    15, raw=True,
)
text(
    180, 626,
    "计算与HBM按主导项归因；decode计算主导为0，" + span("并不代表没有计算") + "。",
    15, raw=True,
)
put("</g>")
put('<g id="footer-group" font-family="Arial, sans-serif" font-size="10.5" fill="#1D1D1A">')
put('<text x="77" y="683">6</text>')
put('<text x="125" y="683">Huawei Proprietary - Restricted Distribution</text>')
put('<image href="logo.png" x="1070" y="663" width="133" height="28" preserveAspectRatio="xMidYMid meet"/>')
put("</g>")
put("</svg>")
svg_text = "\n".join(svg) + "\n"
ET.fromstring(svg_text)
(WS / "svg/page6.svg").write_text(svg_text, encoding="utf-8")

speech = (
    "这一页在原来五页的基础上，把第4页完全相同的六组组合拆到时延归因。"
    "原、兼、解分别表示原并行、兼顾时延和解码优先；条形旁的三个数字依次是TP与SP、EP、DP。"
    "两列均固定一百万级输入、全局批量三十二、每个平台一千零二十四张卡，以及相同F配置、top32和输出两个词元。"
    "左列单位是秒，观察prefill的首词元时延；右列单位是毫秒，观察首个decode间隔。"
    "保留TP、减少EP的兼顾组合，计算主导时间基本不变，prefill收益主要来自通信传输缩短。"
    f"decode的主因则是通信启动：950DT和960DT分别减少{gain950}与{gain960}毫秒。"
    "下方表格保留四个主要分项与总计，条形长度采用原始精度，极窄分项也未人为放大。"
    "这里尤其要区分主导归因与计算工作量：计算和HBM服务有重叠，关键路径把这段服务归给主导资源，"
    "不能再把两个独立下界相加。decode计算主导为零，表示这些算子由HBM主导，并不表示没有计算或零FLOPs，"
    "原第5页的计算工作量仍然有效。调度等待、资源争用、kernel启动和其他服务已保留在底层，"
    "本次各阶段合计只有低于十亿分之一秒的浮点残差。分项分别保留三位小数，因此显示和可能与总计相差零点零零一。"
    "这些结果属于未经实机校准的固定批量模型预测，不代表稳态时延或生产环境P99。"
)

glossary = [
    ("prefill", "Prompt Prefill，提示预填充阶段", "处理完整输入上下文、生成首个输出词元之前的模型执行阶段。"),
    ("decode", "Autoregressive Decoding，自回归解码阶段", "在已有上下文与缓存基础上逐个生成后续词元；本页为首次解码间隔。"),
    ("TTFT", "Time To First Token，首词元时延", "本次模型中prefill阶段的完整关键路径时间，左列以秒展示。"),
    ("TPOT", "Time Per Output Token，每输出词元时延", "本页取生成第二个输出词元所需的一次decode间隔，右列以毫秒展示。"),
    ("TP", "Tensor Parallelism，张量并行", "将张量运算分摊到多张卡；本页布局的第一个数字同时表示TP和SP。"),
    ("SP", "Sequence Parallelism，序列并行", "本次六组配置中SP与TP相同；图中32/8/4等写法省略重复的SP值。"),
    ("EP", "Expert Parallelism，专家并行", "将专家分布到多张卡；布局的第二个数字表示EP。"),
    ("DP", "Data Parallelism，数据并行", "以多份模型处理不同数据批次；布局的第三个数字表示DP。"),
    ("HBM", "High Bandwidth Memory，高带宽存储器", "HBM主导服务表示计算与访存重叠后由存储带宽控制的服务时间。"),
    ("FLOPs", "Floating-Point Operations，浮点运算次数", "计算工作量指标；计算主导服务时间为零不等于浮点运算次数为零。"),
    ("1M", "1M Tokens，本次一百万级输入规模标签", "本仿真明确为1,048,576个输入词元，六组场景保持相同。"),
    ("bs32", "Batch Size 32，批量大小三十二", "页眉指全局批量32，各数据并行副本的本地批量随DP变化。"),
    ("F", "F，本次仿真配置标签", "沿用原仿真固定的F加速配置，本页未修改配置或另行推定其数值精度。"),
    ("top32", "Top-32 Expert Routing，前32个专家路由配置", "本次模型选择使用的专家路由数量配置，各比较方案保持一致。"),
    ("output2", "Output Length 2，输出长度两个词元", "生成首词元后再执行一次decode，因此本页TPOT是一个解码间隔。"),
    ("950DT", "950DT（平台型号）", "图表中的950对应原稿950DT硬件平台。"),
    ("960DT", "960DT（平台型号）", "图表中的960对应原稿960DT硬件平台。"),
    ("s", "Second，秒", "左侧prefill分项与总计的时间单位，也是其他归因残差的基准单位。"),
    ("ms", "Millisecond，毫秒", "右侧decode分项与总计的时间单位；1毫秒等于0.001秒。"),
    ("kernel", "Compute Kernel，计算内核", "底层算子执行单元；kernel启动与集合通信启动为不同归因项。"),
    ("P99", "99th Percentile，第99百分位", "生产请求时延分布的尾部指标，本次固定批量模型未模拟该指标。"),
    ("Huawei", "Huawei（公司名），华为", "沿用原稿页脚中的公司名称与商标。"),
    ("Huawei Proprietary - Restricted Distribution", "Huawei Proprietary - Restricted Distribution，华为专有信息—限制分发", "沿用原稿的固定页脚保密标识，不构成仿真数据证据。"),
]
notes = {
    "page": 6,
    "title": TITLE,
    "speech": speech,
    "transition": "作为新增补充页，回扣第4页的布局取舍与第5页的计算工作量；后续可按通信启动、带宽和算子主导资源逐项开展实机校准。",
    "glossary": [{"term": term, "full": full, "explain": explain} for term, full, explain in glossary],
    "red_notes": [
        {
            "quote": "保TP减EP",
            "explain": "来源为同第4页六组扫描结果；相对各平台原并行32/32/1，兼顾组合保持TP与SP为32，将950DT的EP降至8、960DT的EP降至4，并相应增加DP。",
        },
        {
            "quote": "传输缩短",
            "explain": "来源为latency-data.json的原生关键路径归因；兼顾组合相对各平台原并行，prefill通信传输由30.190降至27.597秒、由27.446降至21.945秒，而计算主导值不变。",
        },
        {
            "quote": "通信启动减少",
            "explain": "来源为collective_startup_s；兼顾组合相对同平台原并行减少集合通信启动耗时，是这两组decode总时延改善的主要贡献，通信传输与HBM服务仍分别保留。",
        },
        {
            "quote": "prefill / TTFT 时延拆解（s）",
            "explain": "左面板使用原始prefill关键路径归因，所有分项、总计和条形横轴均以秒为单位。",
        },
        {
            "quote": "decode / TPOT 时延拆解（ms）",
            "explain": "右面板使用一次decode关键路径归因，源秒值统一乘以1000后以毫秒展示，不能与左面板直接比较条长。",
        },
        {
            "quote": "通信传输",
            "explain": "浅红色条段来自communication_service_s，是关键路径上的通信数据传输服务；其原始值按各阶段共同线性刻度绘制，不人为放大。",
        },
        {
            "quote": "通信启动",
            "explain": "深红色条段来自collective_startup_s，表示集合通信启动开销；采用相同颜色突出decode受其主导，并与浅红色数据传输及其他归因分开。",
        },
        {
            "quote": "27.052",
            "explain": "950DT兼顾组合的decode通信启动为27.052毫秒；相对原并行36.326毫秒下降，表值由完整精度独立四舍五入。",
        },
        {
            "quote": "12.146",
            "explain": "960DT兼顾组合的decode通信启动为12.146毫秒；相对原并行17.298毫秒下降，表值由完整精度独立四舍五入。",
        },
        {
            "quote": gain_quote,
            "explain": f"源于latency-data.json的gains字段；950DT与960DT兼顾组合相对各自原并行的decode通信启动分别减少{gain950}与{gain960}毫秒，先以原始精度作差再舍入，非总时延减少量。",
        },
        {
            "quote": "并不代表没有计算",
            "explain": "零值表示模型把decode算子中重叠的计算与HBM服务归给主导的HBM资源，计算与FLOPs仍然存在；不得将独立计算和HBM下界相加，原第5页的计算工作量继续保留。",
        },
    ],
    "sources": [
        "input/latency-data.json：页面唯一数值来源；2026-10-20固定六组场景，12个阶段的原始完整精度总和已核对。",
        "content/content6.md：第6页论点、归因口径、零值与舍入说明。",
        "outputs/kimi-k3-10t-1m-acceleration-20261019/parallelism-20261020/summary.json",
        "outputs/kimi-k3-10t-1m-acceleration-20261019/parallelism-20261020/latency_breakdown.csv",
        "原稿第4页：六组平台/并行布局与TTFT/TPOT总值；原稿第5页：计算工作量。",
        "方法边界：原生关键路径与算子主导资源归因，未经实机校准的固定批量解析/离散事件模型；不含生产排队、稳态或P99。",
        "其他四项包括scheduler_wait_s、shared_resource_contention_s、kernel_startup_s、other_service_s；每阶段合计仅有低于1e-9秒的浮点残差。",
    ],
}
(WS / "notes/page6.json").write_text(json.dumps(notes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(WS / "qa/page6-red-scan.json").write_text(json.dumps({"pages": [notes]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

checks = []
for row in rows:
    for phase in ("prefill", "decode"):
        entry = row[phase]
        expected_display = [f"{x:.3f}" for x in entry["primary"]] + [f'{entry["total"]:.3f}']
        assert expected_display == entry["display"]
        factor = 1 if phase == "prefill" else 1000
        raw_total = sum(entry["raw_s"].values()) * factor
        assert math.isclose(raw_total, entry["total"], rel_tol=1e-10, abs_tol=1e-8)
        assert entry["other_total_s"] < 1e-9
        checks.append({
            "case": row["name"], "phase": phase, "display": entry["display"],
            "raw_sum_matches_total": True, "other_residual_below_1e_9_s": True,
        })
speech_cjk = len(re.findall(r"[\u4e00-\u9fff]", speech))
assert 350 <= speech_cjk <= 650, speech_cjk
qa = {
    "page": 6,
    "data_source_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
    "speech_cjk_character_count": speech_cjk,
    "table_checks": checks,
    "bars": bar_geometry,
    "decode_zero_width_compute_segments": 6,
    "other_display_semantics": "Numerical residual <1e-9 seconds per stage; explicitly explained in footnote and notes.",
    "mutation_scope": ["svg/page6.svg", "notes/page6.json", "qa/page6-red-scan.json", "qa/page6-data-check.json"],
}
(WS / "qa/page6-data-check.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Created page6.svg and notes/page6.json; {len(checks)} phase checks; speech CJK count {speech_cjk}.")
