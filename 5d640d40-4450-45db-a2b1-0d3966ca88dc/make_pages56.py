"""Pages 5-6: data-derived AR results, context/batch trends and physical work."""
from pathlib import Path
import json
import re

from deck_style import (
    WS, DATA, RED, BLACK, GRAY, LINE, PALE, BG,
    Slide, text, rect, line, panel, table, group,
)

AR = DATA["ar_cases"]
CONTEXTS = DATA["fixed_layout_contexts"]
BATCHES = DATA["fixed_layout_batches"]
CHIPS = ["950dt", "960dt"]
BASE = Path(DATA["source"])


def case(chip, objective):
    return next(r for r in AR if r["chip"] == chip and r["objective"] == objective)


def layout(r):
    return "/".join(str(r[k]) for k in ("tp", "ep", "dp", "sp"))


def label(r):
    return r["chip"].upper() + ("·参考" if r["objective"] == "reference" else "·优选")


def f2(v):
    return f"{v:.2f}"


def f3(v):
    return f"{v:.3f}"


def signed_pct(v):
    return f"+{100*v:.1f}%"


def source_list(names):
    return [
        {"path": str(BASE / name), "sha256": DATA["sources"].get(name, "")}
        for name in names
    ]


def notes_md(page):
    """Keep each assigned page's Markdown notes ready for the final builder."""
    data = json.loads((WS / f"notes/page{page}.json").read_text(encoding="utf8"))
    rows = [
        f"【第{page}页 · {data['title']}】",
        "",
        data["speech"],
        "",
        "过渡：" + data["transition"],
        "",
        "【本页名词表 · 英文简称：英文全称，中文全称——一句话解释】",
    ]
    rows += [f"- {x['term']}：{x['full']}——{x['explain']}" for x in data["glossary"]]
    rows += ["", "【标红内容详解 · 逐条一句话解释】"]
    rows += [f"- 「{x['quote']}」：{x['explain']}" for x in data["red_notes"]]
    rows += ["", "【数据来源】"]
    rows += [
        f"- {x['path']}" + (f"；SHA256={x['sha256']}" if x.get("sha256") else "")
        for x in data["sources"]
    ]
    out = WS / f"svg/notes/page{page}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rows) + "\n", encoding="utf8")


def validate_speech(speech, page):
    count = len(re.findall(r"[\u4e00-\u9fff]", speech))
    assert 500 <= count <= 1000, (page, count)
    return count


def page5():
    refs = {c: case(c, "reference") for c in CHIPS}
    best = {c: case(c, "fastest_decode") for c in CHIPS}
    gains = {c: best[c]["decode_tokens_s"] / refs[c]["decode_tokens_s"] - 1 for c in CHIPS}
    ttft_up = {c: best[c]["ttft_bs1_s"] / refs[c]["ttft_bs1_s"] - 1 for c in CHIPS}
    cuts = {c: 1 - best[c]["tpot_ms"] / refs[c]["tpot_ms"] for c in CHIPS}
    for c in CHIPS:
        constrained = next(
            r for r in DATA["recommendation_samples"]
            if r["chip"] == c and r["mode"] == "AR" and r["global_batch"] == 32
            and r["objective"] == "decode_without_TTFT_regression"
        )
        assert constrained["layout"] == refs[c]["layout"]
        assert constrained["tpot_ms"] == refs[c]["tpot_ms"]
    title = "调整并行组合可提升 1M 吞吐，代价是 TTFT 与驻留容量增加"
    core = [
        ("950DT / 960DT 吞吐分别提升 ", BLACK, True),
        (f"{signed_pct(gains['950dt'])} / {signed_pct(gains['960dt'])}", RED, True),
        ("，首 token 时间需独立权衡", BLACK, True),
    ]
    highlights = [
        [
            ("★ 吞吐收益 ", BLACK, False),
            (f"{signed_pct(gains['950dt'])} / {signed_pct(gains['960dt'])}", RED, True),
            ("；单请求 TTFT 代价 ", BLACK, False),
            (f"{signed_pct(ttft_up['950dt'])} / {signed_pct(ttft_up['960dt'])}", RED, True),
            ("（950DT / 960DT）", BLACK, False),
        ],
        "★ 约束 TTFT 不劣于参考布局时，本扫描未找到更低 TPOT 的替代布局",
    ]
    footnotes = [
        "来源：2026-10-20 本地解析仿真；summary.csv、prefill.csv、recommendations.csv。1M = 1,048,576 token；条件预测，非真机测量。",
        "口径：TTFT = prefill 全局 bs1；TPOT / 吞吐 = decode 全局 bs32。分阶段可行不等于 32 请求同时 prefill 的端到端容量保证。",
    ]
    s = Slide(
        5, title,
        "1024 卡、统一 FP8、AR 基线；在相同上下文和 decode batch 下比较参考布局与扫描内最小 TPOT 布局",
        core, highlights, footnotes,
    )

    # Left: four cases, independently labelled prefill and decode HBM peaks.
    out = panel(50, 172, 820, 366, "1M 四组代表结果：时延、吞吐与容量一起看", "table")
    widths = [112, 130, 102, 104, 110, 118, 118]
    x0 = 63
    out += rect(x0, 216, sum(widths), 43, BG, "none")
    headers = [
        ("平台 / 策略", ""),
        ("TP/EP/DP/SP", ""),
        ("TTFT", "bs1 · s"),
        ("TPOT", "bs32 · ms"),
        ("吞吐", "tokens/s"),
        ("HBM prefill", "bs1 · GB/卡"),
        ("HBM decode", "bs32 · GB/卡"),
    ]
    xx = x0
    for w, (first, second) in zip(widths, headers):
        out += text(xx + 8, 233, first, 12, BLACK, True)
        if second:
            out += text(xx + 8, 250, second, 12)
        xx += w
    red_notes = []
    for i, r in enumerate(AR):
        yy = 259 + i * 42
        isbest = r["objective"] == "fastest_decode"
        if isbest:
            out += rect(x0, yy, sum(widths), 42, PALE, "none")
        values = [
            label(r), layout(r), f2(r["ttft_bs1_s"]), f2(r["tpot_ms"]),
            f2(r["decode_tokens_s"]), f2(r["prefill_bs1_peak_GB"]), f2(r["hbm_peak_GB"]),
        ]
        xx = x0
        for j, (w, v) in enumerate(zip(widths, values)):
            color = RED if isbest and j in (3, 4) else BLACK
            out += text(xx + 8, yy + 26, v, 12, color, isbest)
            if color == RED:
                unit = "ms" if j == 3 else "tokens/s"
                red_notes.append((v, f"{r['chip'].upper()} 在 1M、AR、全局 decode bs32、布局 {layout(r)} 下的{'TPOT' if j == 3 else '全系统输出吞吐'}为 {v} {unit}，来自 summary.csv。"))
            xx += w
        out += line(x0, yy + 42, x0 + sum(widths), yy + 42)
    out += text(65, 457, "优选 = 扫描内最小 TPOT；不是同时最小 TTFT，也未覆盖全部潜在实现。", 12)
    out += text(65, 483, [
        ("950DT / 960DT 的 TPOT 降幅：", BLACK, False),
        (f"{100*cuts['950dt']:.1f}% / {100*cuts['960dt']:.1f}%", RED, True),
        ("；更多 DP 副本增加权重与 ngram 驻留。", BLACK, False),
    ], 12)
    out += text(65, 509, "预填充峰值单列；不能用较低的 decode HBM 峰值代替整个请求的容量检查。", 12)
    s.add("ar-result-table", out)

    # Right: one common horizontal scale for both hardware platforms.
    out = panel(890, 172, 340, 366, "相同负载下的系统吞吐", "chart-bar")
    out += rect(908, 214, 10, 10, "#BFBFBF", "none") + text(924, 223, "参考", 12)
    out += rect(992, 214, 10, 10, RED, "none") + text(1008, 223, "最小 TPOT", 12)
    chart_x, chart_w, chart_max = 948, 220, 7000
    for chip, top in zip(CHIPS, (249, 349)):
        out += text(907, top, [
            (chip.upper() + "   ", BLACK, True),
            (signed_pct(gains[chip]), RED, True),
        ], 15)
        for kind, r, dy, fill in (
            ("参考", refs[chip], 18, "#BFBFBF"),
            ("优选", best[chip], 50, RED),
        ):
            yy = top + dy
            out += text(907, yy + 12, kind, 12)
            out += rect(chart_x, yy, chart_w * r["decode_tokens_s"] / chart_max, 16, fill, "none")
            out += text(1216, yy + 12, f2(r["decode_tokens_s"]), 12, BLACK, kind == "优选", "end")
    out += line(chart_x, 430, chart_x + chart_w, 430)
    for tick in (0, 3500, 7000):
        px = chart_x + chart_w * tick / chart_max
        out += line(px, 430, px, 434)
        out += text(px, 449, str(tick), 10.5, GRAY, False, "middle")
    out += text(907, 477, "共同横轴：0–7000 tokens/s", 12)
    out += text(907, 501, "吞吐 = global bs / TPOT（秒）", 12)
    out += text(907, 524, "每轮输出 1 token / 序列", 12)
    s.add("same-load-throughput-chart", out)

    speech = (
        f"本页只讨论普通自回归基线。四组结果使用同一合成结构、统一八位浮点口径、相同一百万级上下文和全局三十二条生成序列，"
        f"变化的是并行布局。表中首词时间专门来自单请求预填充，生成时延和吞吐来自三十二条序列的生成阶段，二者没有混成整批预热指标。"
        f"两平台参考布局都是张量并行八、专家并行一百二十八、数据并行一。"
        f"950DT 把专家并行缩到 {best['950dt']['ep']}、数据副本增加到 {best['950dt']['dp']} 后，"
        f"每输出词时延由 {f2(refs['950dt']['tpot_ms'])} 降至 {f2(best['950dt']['tpot_ms'])} 毫秒，"
        f"吞吐由 {f2(refs['950dt']['decode_tokens_s'])} 升至 {f2(best['950dt']['decode_tokens_s'])} 词元每秒。"
        f"960DT 的优选布局为 {layout(best['960dt'])}，相应吞吐从 {f2(refs['960dt']['decode_tokens_s'])} "
        f"升至 {f2(best['960dt']['decode_tokens_s'])}，提升 {signed_pct(gains['960dt'])}。"
        "缩小专家组减少了分发与归并的串行通信轮数；已有请求又能填充多个互不争用的数据副本，因此这是真实固定负载下的布局收益。"
        "不过，优选目标只针对生成时延，并不代表请求全程更快。"
        f"950DT 与 960DT 的单请求首词时间分别增加 {signed_pct(ttft_up['950dt'])} 和 {signed_pct(ttft_up['960dt'])}。"
        "预填充只有一个活跃数据副本，增加副本不会自动加速该请求；缩小张量或专家组还会增大目标卡承担的工作与权重驻留。"
        f"容量也必须分阶段核对：950DT 优选的预填充与生成峰值分别为 {f2(best['950dt']['prefill_bs1_peak_GB'])} "
        f"和 {f2(best['950dt']['hbm_peak_GB'])} 吉字节每卡；960DT 对应为 {f2(best['960dt']['prefill_bs1_peak_GB'])} "
        f"和 {f2(best['960dt']['hbm_peak_GB'])}。这说明只看生成峰值会明显低估预填充压力。"
        "即使两阶段分别通过容量准入，也只证明单请求预填充以及指定生成批次可行，不能推出三十二请求同时预填充的端到端保证。"
        "右侧条形图共用同一横轴，避免不同尺度放大视觉差异。若增加首词时间不得退化的约束，推荐表返回的仍是参考布局，"
        "当前扫描没有找到同时改进生成时延的替代方案。所有结果都是锁定配置与调度假设后的解析预测，尚未用真机测量校准。"
    )
    validate_speech(speech, 5)
    for c in CHIPS:
        red_notes.append((signed_pct(gains[c]), f"{c.upper()} 同平台同负载的吞吐提升，按最小 TPOT 布局吞吐除以参考布局吞吐再减一计算；未使用跨平台或增加请求数来形成增益。"))
        red_notes.append((signed_pct(ttft_up[c]), f"{c.upper()} 单请求 prefill TTFT 的增加比例，按优选布局 TTFT 除以参考布局 TTFT 再减一计算。"))
    red_notes.append((f"{100*cuts['950dt']:.1f}% / {100*cuts['960dt']:.1f}%", "950DT / 960DT 的 AR TPOT 降幅，分别按一减去优选 TPOT 与参考 TPOT 的比值计算。"))
    red_notes.append(("红色吞吐条", "右图红条代表各平台扫描内最小 TPOT 布局，灰条代表 TP8/EP128/DP1/SP8 参考布局，两平台均使用同一 0–7000 tokens/s 横轴。"))
    content = (
        "<!-- 逻辑套路：问题分析；标题模式：总结型；三段式 -->\n"
        f"# {title}\n\n{speech}\n\n"
        "|平台/策略|TP/EP/DP/SP|TTFT prefill bs1 (s)|TPOT decode bs32 (ms)|tokens/s|prefill bs1 HBM GB/卡|decode bs32 HBM GB/卡|\n"
        "|---|---|---:|---:|---:|---:|---:|\n"
        + "\n".join("|" + "|".join([
            label(r), layout(r), f2(r["ttft_bs1_s"]), f2(r["tpot_ms"]),
            f2(r["decode_tokens_s"]), f2(r["prefill_bs1_peak_GB"]), f2(r["hbm_peak_GB"]),
        ]) + "|" for r in AR)
    )
    s.save(
        speech,
        "接下来保持这些生成优选布局不变，观察上下文与批大小的敏感性，再用物理工作账本解释吞吐和驻留的不同变化。",
        red_notes,
        source_list(["summary.csv", "prefill.csv", "recommendations.csv", "methodology.md", "experiment.json"]),
        content,
        glossary_extra=[
            {"term": "global", "full": "Global Batch，全局批次", "explain": "全系统已有的同时生成序列数，不是每个 DP 副本新增的负载。"},
        ],
    )
    notes_md(5)


def page6():
    best = {c: case(c, "fastest_decode") for c in CHIPS}
    context_by_chip = {c: [r for r in CONTEXTS if r["chip"] == c] for c in CHIPS}
    batch_by_chip = {c: [r for r in BATCHES if r["chip"] == c] for c in CHIPS}
    for c in CHIPS:
        assert len(context_by_chip[c]) == 4 and len(batch_by_chip[c]) == 6
        assert all(r["layout"] == best[c]["layout"] for r in context_by_chip[c] + batch_by_chip[c])
        assert all(r["status"] == "feasible" for r in context_by_chip[c] + batch_by_chip[c])
    length_ratio = CONTEXTS[3]["context_tokens"] / CONTEXTS[0]["context_tokens"]
    ttft_ratio = {c: context_by_chip[c][-1]["ttft_bs1_s"] / context_by_chip[c][0]["ttft_bs1_s"] for c in CHIPS}
    tpot_growth = {c: context_by_chip[c][-1]["tpot_ms"] / context_by_chip[c][0]["tpot_ms"] - 1 for c in CHIPS}
    idle = {c: batch_by_chip[c][0]["idle_replicas"] for c in CHIPS}
    title = "长上下文主要拉高 TTFT，小 batch 使 DP 副本闲置"
    core = [
        ("长度扩展主要推高 ", BLACK, True), ("TTFT", RED, True),
        ("；batch 扩展主要填充 ", BLACK, True), ("DP", RED, True),
        ("，物理工作量与驻留量必须分开", BLACK, True),
    ]
    highlights = [
        [
            (f"★ 上下文 {length_ratio:.0f} 倍：TTFT 为 ", BLACK, False),
            (f"{ttft_ratio['950dt']:.2f}× / {ttft_ratio['960dt']:.2f}×", RED, True),
            ("；TPOT 仅增 ", BLACK, False),
            (f"{100*tpot_growth['950dt']:.1f}% / {100*tpot_growth['960dt']:.1f}%", RED, True),
            ("（950DT / 960DT）", BLACK, False),
        ],
        [
            ("★ bs1 时闲置 DP 副本为 ", BLACK, False),
            (f"{idle['950dt']} / {idle['960dt']}", RED, True),
            ("；权重与 ngram 仍驻留，不能把闲置容量当成零成本", BLACK, False),
        ],
    ]
    footnotes = [
        "口径：上表 TTFT / 预填 HBM 为 bs1；TPOT / 吞吐 / 生成 HBM 为 bs32。固定布局，不对每个长度和 batch 重新选优；容量不代表 bs32 同时预填。",
        "来源：summary.csv、prefill.csv。下表为物理执行总量：生成 TFLOP = FP8 + FP32；生成流量按每轮计，ngram DRAM 为集群常驻总量。GB/TB 为十进制。",
    ]
    s = Slide(
        6, title,
        f"固定 1M、bs32 最小 TPOT 布局：950DT TP/EP/DP/SP = {layout(best['950dt'])}；960DT = {layout(best['960dt'])}",
        core, highlights, footnotes,
    )

    # Context sensitivity: eight rows; separate phase capacity rather than a misleading maximum.
    out = panel(50, 172, 600, 210, "上下文扫描：TTFT 对长度更敏感", "table")
    widths = [65, 57, 84, 85, 91, 88, 88]
    rows = []
    for r in CONTEXTS:
        rows.append([
            r["chip"].upper(), r["context"], f2(r["ttft_bs1_s"]), f2(r["tpot_ms"]),
            f2(r["decode_tokens_s"]), f2(r["prefill_bs1_peak_GB"]), f2(r["hbm_peak_GB"]),
        ])
    out += table(
        61, 203, widths,
        ["平台", "长度", "TTFT s", "TPOT ms", "吞吐 tok/s", "预填 GB", "生成 GB"],
        rows, row_h=18.75, head_h=24, red_rows=(3, 7), size=12,
    )
    s.add("fixed-layout-context-table", out)

    # Batch sensitivity: six powers-of-two on a categorical axis, all point labels shown.
    out = panel(670, 172, 560, 210, "1M batch 扫描：已有请求逐步填满副本", "chart-line")
    out += line(700, 213, 720, 213, "#8C8C8C") + text(728, 217, "950DT", 12)
    out += line(804, 213, 824, 213, RED) + text(832, 217, "960DT", 12)
    out += text(1215, 217, "单位：tokens/s", 10.5, GRAY, False, "end")
    plot_x, plot_right, plot_top, plot_bottom, ymax = 720, 1185, 234, 333, 7000
    xs = [plot_x + i * (plot_right - plot_x) / 5 for i in range(6)]
    for tick in (0, 3500, 7000):
        yy = plot_bottom - (plot_bottom - plot_top) * tick / ymax
        out += line(plot_x, yy, plot_right, yy, "#EBEBEB")
        out += text(708, yy + 4, str(tick), 10.5, GRAY, False, "end")
    out += line(plot_x, plot_top, plot_x, plot_bottom, LINE)
    red_notes = [
        ("TTFT", "核心思想强调首词时间对上下文长度的敏感性；此页所有 TTFT 均为 prefill 全局 bs1。"),
        ("DP", "核心思想强调固定副本数由已有请求填充：active DP=min(DP, global batch)，闲置副本仍保留权重和 ngram。"),
    ]
    for c, color in (("950dt", "#8C8C8C"), ("960dt", RED)):
        rows = batch_by_chip[c]
        points = [(x, plot_bottom - (plot_bottom - plot_top) * r["decode_tokens_s"] / ymax) for x, r in zip(xs, rows)]
        for p1, p2 in zip(points, points[1:]):
            out += line(*p1, *p2, color)
        for (xx, yy), r in zip(points, rows):
            out += f'<circle cx="{xx}" cy="{yy}" r="2.5" fill="{color}"/>'
            offset = 16 if c == "950dt" else -9
            out += text(xx, yy + offset, f2(r["decode_tokens_s"]), 12, color, c == "960dt", "middle")
            if c == "960dt":
                red_notes.append((f2(r["decode_tokens_s"]), f"红色 960DT 曲线在固定布局 {layout(r)}、1M、AR、全局 decode bs{r['global_batch']} 下的吞吐为 {f2(r['decode_tokens_s'])} tokens/s，来自 summary.csv；不是每个点重新选优。"))
    for xx, r in zip(xs, batch_by_chip["950dt"]):
        out += text(xx, 370, str(r["global_batch"]), 12, BLACK, False, "middle")
    out += text(686, 370, "bs", 12)
    s.add("fixed-layout-batch-chart", out)

    # The lower table reports physical executed work, not native envelope work.
    out = panel(50, 395, 1180, 143, "1M 物理账本：预填 bs1 与生成 bs32 分列，存储计入全部 DP 副本", "database")
    widths = [136, 170, 170, 160, 160, 180, 180]
    headers = ["平台 / 策略", "预填 FP8 EFLOP", "生成 TFLOP/轮", "预填 SU TB", "生成 SU GB/轮", "生成 HBM TB/轮", "ngram DRAM TB"]
    rows = [[
        label(r), f3(r["prefill"]["flops_fp8"] / 1e18), f3(r["decode_TFLOPs_round"]),
        f3(r["prefill"]["su_physical_bytes"] / 1e12), f3(r["SU_GB_round"]),
        f3(r["HBM_TB_round"]), f3(r["cluster_ngram_DRAM_TB"]),
    ] for r in AR]
    out += table(62, 425, widths, headers, rows, row_h=21, head_h=25, red_rows=(1, 3), size=12)
    s.add("physical-work-ledger", out)

    speech = (
        "本页固定前页选出的生成优选布局，只改变上下文长度或已有生成请求数，不在每个点重新搜索布局。"
        "左表覆盖两平台各四个上下文长度，首词时间和预填充显存来自单请求预填充，生成时延、吞吐和生成显存则对应全局三十二条序列。"
        f"从十三万级扩展到一百万级，上下文恰好增加 {length_ratio:.0f} 倍，950DT 与 960DT 的首词时间分别变为 "
        f"{ttft_ratio['950dt']:.2f} 倍与 {ttft_ratio['960dt']:.2f} 倍；生成时延只增加 "
        f"{100*tpot_growth['950dt']:.1f}% 与 {100*tpot_growth['960dt']:.1f}%。"
        "这与因果编码器处理全部提示词、稀疏生成只读取部分历史的结构差异一致，但首词时间并非按长度简单线性外推。"
        "预填充显存峰值随长度增长也更显著，因此表中继续分列两个阶段，避免把生成峰值当成请求容量上界。"
        "右图只扫描一百万级上下文，横轴六档批次采用等间距离散刻度，每个点标出全系统吞吐。"
        "数据并行不会创造新请求；活跃副本数等于副本总数与已有批次的较小者。"
        f"在单请求下，950DT 的 {best['950dt']['dp']} 个副本只有一个工作，闲置 {idle['950dt']} 个；"
        f"960DT 的 {best['960dt']['dp']} 个副本闲置 {idle['960dt']} 个。"
        "批次填充副本的区间内，每个活跃副本本地批次保持一，时延基本不变而全局吞吐随请求数上升；副本填满后，本地工作增加，增长不再严格线性。"
        "下表把时延解释进一步落到物理工作量。预填充浮点运算只列八位矩阵部分，生成浮点运算包含八位和三十二位操作；"
        "两者在表头与注释中明确区分，不能拿单位不同的数直接相除。互联字节是实际物理传输总量，生成显存字节是每轮访问量，均不是容量占用。"
        "最后一列才是全体数据副本的词组表常驻内存，闲置副本也计入，因此其值随副本数增长，不能从活跃副本数反推总驻留。"
        f"950DT 优选共驻留 {f3(best['950dt']['cluster_ngram_DRAM_TB'])} 太字节，960DT 优选为 "
        f"{f3(best['960dt']['cluster_ngram_DRAM_TB'])} 太字节。"
        "表内工作量来自执行阶段账本而非缩约资源包络，独立舍入仅服务展示。以上敏感性仍受固定路由、无计算通信重叠等假设约束，"
        "阶段可行与吞吐预测不构成请求到达、整批预热和在线排队的服务保证。"
    )
    validate_speech(speech, 6)
    red_notes += [
        (f"{ttft_ratio['950dt']:.2f}× / {ttft_ratio['960dt']:.2f}×", "分别为固定优选布局下，950DT / 960DT 的 1M 单请求 TTFT 除以 128k 单请求 TTFT；上下文长度之比为 8。"),
        (f"{100*tpot_growth['950dt']:.1f}% / {100*tpot_growth['960dt']:.1f}%", "分别为固定优选布局下，950DT / 960DT 的 1M 与 128k 的 AR bs32 TPOT 比值减一；批次与布局保持不变。"),
        (f"{idle['950dt']} / {idle['960dt']}", "950DT / 960DT 在固定优选布局、全局 bs1 时闲置的数据副本数；原始 idle_replicas 字段分别为 3 和 15，权重及 ngram 仍驻留。"),
        ("红色 960DT 曲线", "固定 TP4/EP16/DP16/SP4 的 960DT 结果系列；与灰色 950DT 系列共享吞吐纵轴，横轴是 1、2、4、8、16、32 六档离散批次。"),
    ]
    content = (
        "<!-- 逻辑套路：问题分析；标题模式：总结型；三段式 -->\n"
        f"# {title}\n\n{speech}\n\n"
        "## 固定布局的上下文扫描\n"
        "|平台|长度|TTFT bs1 s|TPOT bs32 ms|tokens/s|prefill bs1 HBM GB/卡|decode bs32 HBM GB/卡|\n"
        "|---|---|---:|---:|---:|---:|---:|\n"
        + "\n".join("|" + "|".join([
            r["chip"].upper(), r["context"], f2(r["ttft_bs1_s"]), f2(r["tpot_ms"]),
            f2(r["decode_tokens_s"]), f2(r["prefill_bs1_peak_GB"]), f2(r["hbm_peak_GB"]),
        ]) + "|" for r in CONTEXTS)
        + "\n\n## 1M 固定布局的批次扫描\n|平台|全局 bs|TPOT ms|tokens/s|闲置 DP|\n|---|---:|---:|---:|---:|\n"
        + "\n".join("|" + "|".join([
            r["chip"].upper(), str(r["global_batch"]), f2(r["tpot_ms"]),
            f2(r["decode_tokens_s"]), str(r["idle_replicas"]),
        ]) + "|" for r in BATCHES)
        + "\n\n## 1M 物理账本\n|平台/策略|预填 FP8 EFLOP|生成 TFLOP/轮|预填 SU TB|生成 SU GB/轮|生成 HBM TB/轮|ngram DRAM TB|\n"
        "|---|---:|---:|---:|---:|---:|---:|\n"
        + "\n".join("|" + "|".join(row) + "|" for row in rows)
    )
    s.save(
        speech,
        "在普通自回归的布局与负载边界明确后，再检查三层草稿机制的额外成本，以及每轮实际提交数量对收益门槛的影响。",
        red_notes,
        source_list(["summary.csv", "prefill.csv", "recommendations.csv", "methodology.md", "experiment.json"]),
        content,
    )
    notes_md(6)


if __name__ == "__main__":
    page5()
    page6()
    print("Generated pages 5-6: SVG, content Markdown, notes JSON and notes Markdown.")
