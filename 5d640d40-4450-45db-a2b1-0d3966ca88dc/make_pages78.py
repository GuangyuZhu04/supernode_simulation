"""Pages 7–8: DSpark conditions and additive latency attribution.

Reads the shared, frozen dataset.  No simulation is run by this script.
"""
from __future__ import annotations

import json
import math
import re

from deck_style import (
    WS, DATA, Slide, text, rect, line, arrow, icon, panel, table, group,
    RED, BLACK, GRAY, LINE, PALE, BG,
)

SOURCE = "outputs/deepseek-v41-pro-10t-20261020/"
AR = DATA["ar_cases"]
DS = [r for r in DATA["dspark_cases"] if r["objective"] == "fastest_decode"]
SENS = DATA["dspark_sensitivity"]
MODEL = DATA["model"]


def best_ar(chip):
    return next(r for r in AR if r["chip"] == chip and r["objective"] == "fastest_decode")


def ref_ar(chip):
    return next(r for r in AR if r["chip"] == chip and r["objective"] == "reference")


def short_chip(chip):
    return chip.replace("dt", "DT")


def layout_text(row):
    return "/".join(str(row[k]) for k in ("tp", "ep", "dp", "sp"))


def runs(*parts):
    return [(p, RED if emph else BLACK, emph) for p, emph in parts]


def redcell(value):
    return {"text": value, "color": RED, "bold": True}


def write_extra(n, **extra):
    path = WS / f"notes/page{n}.json"
    notes = json.loads(path.read_text(encoding="utf8"))
    notes.update(extra)
    path.write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf8")


def make_page7():
    c4 = {r["chip"]: r for r in DS}
    d95, d96 = c4["950dt"], c4["960dt"]
    s95, s96 = [d["speedup_vs_same_layout_AR"] for d in (d95, d96)]
    sens = {(r["chip"], r["committed"]): r for r in SENS}
    draft_positions = MODEL["dspark_block_size"]
    target_queries = draft_positions + 1
    layers = MODEL["n_mtp_layers"]
    c = d95["committed"]
    threshold = min(r["committed"] for r in SENS if r["speedup_vs_same_layout_AR"] > 1)
    assert threshold == d95["min_committed_to_beat_same_AR"] == d96["min_committed_to_beat_same_AR"]
    for row in DS:
        assert row["committed"] == c
        assert math.isclose(sum(row[k] for k in ("draft_ms", "verify_ms", "commit_ms")),
                            row["round_ms"], rel_tol=1e-12, abs_tol=1e-10)
        assert math.isclose(row["round_ms"] / c, row["tpot_ms"], rel_tol=1e-12)
        assert row["commit_ms"] > 0
        assert row["layout"] == best_ar(row["chip"])["layout"]
        assert math.isclose(best_ar(row["chip"])["tpot_ms"] / row["tpot_ms"],
                            row["speedup_vs_same_layout_AR"], rel_tol=1e-12)

    title = f"DSpark 在 C={c} 条件下，同布局折算提速 {s95:.2f}× / {s96:.2f}×"
    slide = Slide(
        7, title,
        "按草稿、目标验证、状态提交三阶段计费；固定 1M、decode bs32，对照同平台的 AR 最优布局",
        runs((f"{draft_positions} 个草稿位置、{target_queries} 个目标查询共同完成一轮；", False),
             (f"C≥{threshold}", True), (" 才在当前条件下超过同布局 AR", False)),
        [
            runs((f"★ C={c} 条件收益：950DT ", False), (f"{s95:.2f}×", True),
                 ("；960DT ", False), (f"{s96:.2f}×", True), ("，不混入布局优化收益", False)),
            runs(("★ C 是提交数；TPOT=round/C 是折算服务成本，", False),
                 ("不等于客户端 ITL", True)),
        ],
        [
            f"布局 TP/EP/DP/SP：950DT {layout_text(d95)}；960DT {layout_text(d96)}。C∈[1,{target_queries}] 为条件输入，接受率与文本质量未实测。",
            "来源：2026-10-20 仿真 summary.csv、dspark_round_breakdown.csv、methodology.md；循环路由条件，无真实 serving 校准。",
        ],
    )
    slide.add("dspark-process-panel", panel(50, 172, 570, 366, "一轮包含完整验证与非零状态提交", "arrows-repeat"))
    node_specs = [
        (66, "草稿 draft", "bolt", f"{layers} 层 / {draft_positions} 个草稿位置", f"{draft_positions} 次马尔可夫校正"),
        (252, "验证 verify", "layers", f"{target_queries} 个目标查询", f"完整 {MODEL['n_layers']} 层 + KV 写入"),
        (438, "提交 commit", "circle-checkmark", f"提交 C∈[1,{target_queries}]", "截断、恢复、追赶窗口"),
    ]
    for i, (x, heading, ic, l1, l2) in enumerate(node_specs):
        node = rect(x, 215, 164, 76, "white", LINE, 3)
        node += icon(ic, x + 10, 223, 18, GRAY)
        node += text(x + 34, 237, heading, 12, BLACK, True)
        node += text(x + 12, 259, l1, 12)
        node += text(x + 12, 280, l2, 12)
        slide.add(f"dspark-stage-{i + 1}", node)
    slide.add("dspark-flow-arrows", arrow(231, 254, 247, 254) + arrow(417, 254, 433, 254))
    content = text(66, 313, "验证后的隐藏态追赶草稿窗口；替换或奖励输出留给下一轮", 12)
    content += text(66, 337, f"C={c} 阶段成本（ms）", 12, BLACK, True)
    stage_rows = []
    for label, key in [
        ("草稿 draft", "draft_ms"),
        ("目标验证 verify", "verify_ms"),
        ("状态提交 commit", "commit_ms"),
        ("单轮 round", "round_ms"),
        ("折算 TPOT", "tpot_ms"),
    ]:
        vals = [f"{r[key]:.3f}" for r in (d95, d96)]
        stage_rows.append([label] + ([redcell(v) for v in vals] if key == "tpot_ms" else vals))
    content += table(
        66, 347, [212, 160, 166],
        ["阶段", "950DT", "960DT"], stage_rows,
        row_h=26, head_h=25, red_rows=(4,), aligns=["start", "end", "end"],
    )
    content += text(66, 525, "round 为成批输出间隔；TPOT=round/C 只折算服务成本", 12)
    slide.add("dspark-stage-cost", content)

    sensitivity = panel(640, 172, 590, 366, "C=1…6 的条件敏感性：AR 基线 = 1.00×", "chart-line")
    sensitivity += text(656, 219, "每平台固定上表布局；加速比 = 同布局 AR TPOT ÷ DSpark TPOT", 12)
    rows = []
    for committed in range(1, target_queries + 1):
        r95, r96 = sens[("950dt", committed)], sens[("960dt", committed)]
        values = [
            f"C={committed}",
            f"{r95['tpot_ms']:.3f}",
            f"{r95['speedup_vs_same_layout_AR']:.2f}×",
            f"{r96['tpot_ms']:.3f}",
            f"{r96['speedup_vs_same_layout_AR']:.2f}×",
        ]
        rows.append([redcell(v) for v in values] if committed == c else values)
    sensitivity += table(
        656, 231, [62, 145, 99, 145, 107],
        ["提交数", "950 TPOT(ms)", "加速比", "960 TPOT(ms)", "加速比"],
        rows, row_h=31, head_h=28, red_rows=(c - 1,),
        aligns=["start", "end", "end", "end", "end"],
    )
    sensitivity += rect(656, 460, 558, 63, BG, "none", 3)
    c1, c2 = [sens[("950dt", cc)] for cc in (1, 2)]
    d1, d2 = [sens[("960dt", cc)] for cc in (1, 2)]
    sensitivity += text(670, 483, runs(
        (f"收益门槛 C≥{threshold}", True),
        (f"：C=1 仅 {c1['speedup_vs_same_layout_AR']:.2f}× / {d1['speedup_vs_same_layout_AR']:.2f}×", False),
    ), 15)
    sensitivity += text(670, 508,
                        f"C=2 达 {c2['speedup_vs_same_layout_AR']:.2f}× / {d2['speedup_vs_same_layout_AR']:.2f}×；额外轮成本开始被提交数摊薄", 12)
    slide.add("dspark-sensitivity", sensitivity)

    speech = (
        f"这一页在固定的 1M 上下文和 decode 全局 batch32 下，单独看 DSpark 带来的条件收益。"
        f"950DT 固定 TP/EP/DP/SP={layout_text(d95)}，960DT 固定 {layout_text(d96)}，"
        "两者都使用前面已经选出的 AR 最小 TPOT 布局。因而这里的加速比只回答：在相同硬件、负载和布局上，"
        "加入草稿机制后，提交多个输出是否足以覆盖额外工作。布局优化的收益已经在前页展示，此处不再重复计入。\n\n"
        f"执行过程分成三个明确的阶段。第一阶段用 {layers} 层草稿网络同时产生 {draft_positions} 个位置，"
        f"并执行 {draft_positions} 次马尔可夫串行校正及置信度处理。第二阶段让完整目标网络验证 {target_queries} 个输入位置，"
        "其中也包含当前待处理输入；因此不能把验证成本按五个草稿位置计数。第三阶段进行接受判断和状态提交。"
        "全局 KV 可以追加后截断，滑窗和池化状态要保留候选前缀的版本并按提交结果恢复，已验证的目标隐藏态还要投影并追赶草稿窗口。"
        "替换或奖励输出尚无对应隐藏态，它作为下一轮输入继续处理，不能免费补出这部分状态。\n\n"
        f"C={c} 时，950DT 的草稿、目标验证、状态提交分别为 {d95['draft_ms']:.6f}、"
        f"{d95['verify_ms']:.6f}、{d95['commit_ms']:.6f} ms，合计 {d95['round_ms']:.6f} ms；"
        f"960DT 对应为 {d96['draft_ms']:.6f}、{d96['verify_ms']:.6f}、{d96['commit_ms']:.6f} ms，"
        f"合计 {d96['round_ms']:.6f} ms。状态提交成本虽小却非零，目标验证是主要时间开销。"
        f"每轮时间除以提交数后，TPOT 为 {d95['tpot_ms']:.3f} 和 {d96['tpot_ms']:.3f} ms，"
        f"同布局 AR 比值分别是 {s95:.2f}× 和 {s96:.2f}×。\n\n"
        f"右表继续把 C 从 1 扫到 {target_queries}。C=1 时，两平台比值都小于一，新增的草稿和验证工作无法被摊薄；"
        f"到 C={threshold} 才开始超过同布局 AR。C 表示每轮最终提交的输出数，包含替换或奖励输出，它不是接受率，也不能直接换算成接受概率。"
        "本次循环路由是解析工作负载条件，真实文本中的路由相关性、接受情况和模型质量尚未测量。"
        "最后区分两个时间概念：round 是一轮成批完成的间隔，TPOT 是将其除以 C 得到的输出服务成本。"
        "客户端会如何逐个收到 token，还取决于输出交付和调度；不能把该折算值当成逐 token ITL。"
    )
    reds = [
        (f"C≥{threshold}", "summary.csv 的同布局条件扫描中，两平台 C=1 的比值均小于一，C=2 才开始超过 AR。"),
        (f"收益门槛 C≥{threshold}", "该门槛仅对当前 1M、decode bs32、所选布局与循环路由仿真条件成立。"),
        (f"{s95:.2f}×", f"950DT：同布局 AR {best_ar('950dt')['tpot_ms']:.12f} ms / DSpark C=4 {d95['tpot_ms']:.12f} ms，未混入布局改善。"),
        (f"{s96:.2f}×", f"960DT：同布局 AR {best_ar('960dt')['tpot_ms']:.12f} ms / DSpark C=4 {d96['tpot_ms']:.12f} ms，未混入布局改善。"),
        ("不等于客户端 ITL", "methodology.md 将 TPOT 定义为 round/C 的折算服务成本，round 是成批输出完成间隔，未建模客户端逐 token 交付。"),
        (f"C={c}", "这里每轮提交数固定为四，是仿真条件输入，包含替换或奖励输出，不代表实测接受率。"),
        (f"{d95['tpot_ms']:.3f}", "950DT 的 C=4 轮时间除以四，单位为毫秒；其状态提交成本已经计入。"),
        (f"{d96['tpot_ms']:.3f}", "960DT 的 C=4 轮时间除以四，单位为毫秒；其状态提交成本已经计入。"),
        ("一轮包含完整验证与非零状态提交", "三阶段成本来自 dspark_round_breakdown.csv；该面板说明本地状态维护没有作为免费操作。"),
        ("C=1…6 的条件敏感性：AR 基线 = 1.00×", "比值以每个平台相同布局的 AR TPOT 为分子，各 C 值是独立的条件输入。"),
    ]
    sources = [
        SOURCE + "summary.csv",
        SOURCE + "dspark_round_breakdown.csv",
        SOURCE + "methodology.md",
        SOURCE + "model.json",
        "input/deck-data.json: dspark_cases / dspark_sensitivity / ar_cases",
    ]
    slide.save(
        speech,
        "下一页回到普通 AR，将 prefill 和 decode 的延迟拆成可加资源项，解释布局优化实际省在哪里、又付出了什么代价。",
        reds, sources,
        glossary_extra=[
            {"term": "round", "full": "Round，草稿验证提交的一轮",
             "explain": "包含 draft、target verify 与 state commit，完成后可成批提交 C 个输出。"},
            {"term": "serving", "full": "Inference Serving，推理在线服务",
             "explain": "接收请求、调度推理并向客户端交付结果的系统；本实验没有进行真实在线服务校准。"},
        ],
    )
    write_extra(7, dspark_stage_costs=DS, sensitivity_rows=SENS)


CATEGORIES = [
    ("compute_bound", "计算", "#8C8C8C"),
    ("hbm_bound", "HBM", "#D9D9D9"),
    ("su_transfer", "SU传输", "#F1B4B7"),
    ("communication_startup", "启动", RED),
    ("host", "主机*", "#7FAAD4"),
]
RAW_CATEGORIES = [
    "compute_bound", "hbm_bound", "su_transfer",
    "communication_startup", "host_lookup_hash", "host_transfer",
]


def raw_latency(row, phase):
    if phase == "prefill":
        return {k: row["prefill"][f"latency_{k}_s"] for k in RAW_CATEGORIES}
    return {k: row["decode_latency"][f"{k}_ms"] for k in RAW_CATEGORIES}


def grouped_latency(row, phase):
    raw = raw_latency(row, phase)
    return {k: (raw["host_lookup_hash"] + raw["host_transfer"] if k == "host" else raw[k])
            for k, _, _ in CATEGORIES}


def latency_panel(x, phase):
    prefill = phase == "prefill"
    unit = "s" if prefill else "ms"
    heading = f"prefill bs1｜单请求首 token（{unit}）" if prefill else f"decode bs32｜AR 单轮（{unit}）"
    out = panel(x, 172, 578, 366, heading, "chart-bar")
    plot_x, plot_w = x + 92, 420
    for tick in range(0, 51, 10):
        tx = plot_x + plot_w * tick / 50
        out += line(tx, 229, tx, 342, "#EBEBEB")
        out += text(tx, 220, str(tick), 12, BLACK, False, "middle")
    out += line(plot_x, 342, plot_x + plot_w, 342)
    table_rows = []
    raw_rows = []
    for idx, row in enumerate(AR):
        label = row["chip"].replace("dt", "") + ("参考" if row["objective"] == "reference" else "最优")
        total = row["ttft_bs1_s"] if prefill else row["tpot_ms"]
        raw = raw_latency(row, phase)
        grouped = grouped_latency(row, phase)
        assert math.isclose(sum(raw.values()), total, rel_tol=1e-11, abs_tol=1e-10), (phase, row["layout"])
        assert math.isclose(sum(grouped.values()), total, rel_tol=1e-11, abs_tol=1e-10)
        yy = 238 + idx * 28
        out += text(x + 16, yy + 12, label, 12)
        bx = plot_x
        for key, _, color in CATEGORIES:
            width = grouped[key] / 50 * plot_w
            # Do not inflate the very small nonzero components for visibility.
            if width > 0:
                out += rect(bx, yy, width, 15, color, "none")
            bx += width
        out += text(x + 560, yy + 12, f"{total:.3f}", 12,
                    RED if row["objective"] == "fastest_decode" else BLACK,
                    row["objective"] == "fastest_decode", "end")
        values = []
        for key, _, _ in CATEGORIES:
            precision = 6 if (not prefill and key in ("compute_bound", "host")) else 3
            values.append(f"{grouped[key]:.{precision}f}")
        table_rows.append([label] + values + [
            redcell(f"{total:.3f}") if row["objective"] == "fastest_decode" else f"{total:.3f}"
        ])
        raw_rows.append({
            "phase": phase, "unit": unit,
            "chip": row["chip"], "objective": row["objective"], "layout": row["layout"],
            "context_tokens": row["context_tokens"],
            "global_batch": 1 if prefill else row["global_batch"],
            "components": raw, "total": total,
            "check_sum": sum(raw.values()),
        })
    legend_x = x + 18
    for key, name, color in CATEGORIES:
        out += rect(legend_x, 356, 10, 10, color, "none")
        out += text(legend_x + 15, 365, name, 12)
        legend_x += 108 if key in ("compute_bound", "hbm_bound", "su_transfer") else 100
    out += table(
        x + 12, 380, [70, 80, 67, 76, 75, 91, 95],
        ["布局", "计算", "HBM", "SU传输", "启动", "主机*", "合计"],
        table_rows, row_h=24, head_h=26, red_rows=(1, 3),
        aligns=["start", "end", "end", "end", "end", "end", "end"],
    )
    if prefill:
        inc95 = raw_latency(best_ar("950dt"), phase)["compute_bound"] - raw_latency(ref_ar("950dt"), phase)["compute_bound"]
        inc96 = raw_latency(best_ar("960dt"), phase)["compute_bound"] - raw_latency(ref_ar("960dt"), phase)["compute_bound"]
        out += text(x + 16, 525, runs(
            ("最优布局的计算归因增加：950 ", False), (f"+{inc95:.3f} s", True),
            ("；960 ", False), (f"+{inc96:.3f} s", True)), 12)
    else:
        dec95 = raw_latency(ref_ar("950dt"), phase)["communication_startup"] - raw_latency(best_ar("950dt"), phase)["communication_startup"]
        dec96 = raw_latency(ref_ar("960dt"), phase)["communication_startup"] - raw_latency(best_ar("960dt"), phase)["communication_startup"]
        out += text(x + 16, 525, runs(
            ("最优布局的通信启动减少：950 ", False), (f"{dec95:.3f} ms", True),
            ("；960 ", False), (f"{dec96:.3f} ms", True)), 12)
    return out, raw_rows


def make_page8():
    a95, b95, a96, b96 = AR
    dec95 = a95["decode_latency"]["communication_startup_ms"] - b95["decode_latency"]["communication_startup_ms"]
    dec96 = a96["decode_latency"]["communication_startup_ms"] - b96["decode_latency"]["communication_startup_ms"]
    inc95 = b95["prefill"]["latency_compute_bound_s"] - a95["prefill"]["latency_compute_bound_s"]
    inc96 = b96["prefill"]["latency_compute_bound_s"] - a96["prefill"]["latency_compute_bound_s"]
    pre95 = b95["ttft_bs1_s"] / a95["ttft_bs1_s"] - 1
    pre96 = b96["ttft_bs1_s"] / a96["ttft_bs1_s"] - 1
    suf95 = a95["prefill"]["latency_su_transfer_s"] / a95["ttft_bs1_s"] * 100
    suf96 = a96["prefill"]["latency_su_transfer_s"] / a96["ttft_bs1_s"] * 100
    slide = Slide(
        8,
        "decode 收益主要来自通信启动减少，prefill 需承担计算代价",
        "四组 AR 布局按资源归因拆解；prefill bs1 用秒，decode bs32 用毫秒，分别保留阶段负载与单位",
        runs(("减少 EP 可压低 decode 启动成本；", False),
             ("prefill 计算归因增加", True), ("，使最小 TPOT 布局的 TTFT 上升", False)),
        [
            runs(("★ decode 启动节省：950DT ", False), (f"{dec95:.4f} ms", True),
                 ("；960DT ", False), (f"{dec96:.3f} ms", True), ("，驱动单轮延迟下降", False)),
            runs(("★ prefill TTFT 仍上升 ", False), (f"{pre95:.1%} / {pre96:.1%}", True),
                 ("；计算与 HBM 每个 kernel 只归因较慢者", False)),
        ],
        [
            f"布局顺序 TP/EP/DP/SP：参考均为 {layout_text(a95)}；最优为 950DT {layout_text(b95)}、960DT {layout_text(b96)}。1M=1,048,576 token。",
            "来源：prefill.csv / latency_breakdown.csv。主机*=查表/hash+传输；六项原值在备注，独立取整有尾差；微小项未放大绘图。",
        ],
    )
    left, raw_pre = latency_panel(50, "prefill")
    right, raw_dec = latency_panel(652, "decode")
    slide.add("prefill-latency", left)
    slide.add("decode-latency", right)

    speech = (
        "这一页用相同的四组 AR 结果回答布局优化实际改变了哪类延迟。左边是 1M 单请求 prefill，"
        "全局 batch 为一，单位是秒；右边是已经完成预填充后的全局 batch32 自回归 decode，单位是毫秒。"
        "上下文相同并不意味着两列代表同一批请求的端到端时间，也不能把单请求首 token 时间视为三十二条请求同时预填充的耗时。"
        "每个平台的两行分别是统一参考布局和该平台最小 TPOT 布局。\n\n"
        "原始账本有六项可加归因：计算受限、HBM 受限、SU 传输、通信启动、主机查表与哈希、主机传输。"
        "图中为节约列宽，把最后两项合并为主机，原始数值完整保留在备注附表。每个矩阵 kernel 同时比较计算和 HBM 的 roofline 时间，"
        "只将较慢的一项归入该阶段，二者不会再次相加。不同算子阶段和通信阶段按仿真调度串行累加，因此六项总和等于阶段延迟。"
        "组件维度拆解是另一个可加视角，不能再与此处资源归因相加。\n\n"
        f"先看 decode，950DT 的通信启动由 {a95['decode_latency']['communication_startup_ms']:.3f} ms"
        f"降到 {b95['decode_latency']['communication_startup_ms']:.3f} ms，净减 {dec95:.4f} ms；"
        f"960DT 由 {a96['decode_latency']['communication_startup_ms']:.3f} 降到"
        f" {b96['decode_latency']['communication_startup_ms']:.3f} ms，净减 {dec96:.3f} ms。"
        "减少 EP 压低了 dispatch/combine 串行轮数，解释了本轮时延的大部分改善。"
        "这不是把全网带宽利用率直接换算成性能；每个端点的注入和每轮启动仍受显式资源约束。\n\n"
        f"再看 prefill，参考布局的 SU 服务分别占总时间 {suf95:.2f}% 和 {suf96:.2f}%，"
        "长上下文索引与通信仍是重要成本。切到 decode 最优布局后，950DT 计算归因"
        f"从 {a95['prefill']['latency_compute_bound_s']:.3f} 增到 {b95['prefill']['latency_compute_bound_s']:.3f} s；"
        f"960DT 从 {a96['prefill']['latency_compute_bound_s']:.3f} 增到 {b96['prefill']['latency_compute_bound_s']:.3f} s。"
        f"这使总 TTFT 分别上升 {pre95:.1%} 和 {pre96:.1%}。单请求只激活一个 DP 副本，"
        "更多 DP 不会自动缩短这一个请求；较少并行资源与每目标 rank 的工作变化，会抵消传输和启动的节省。"
        "因此应同时约束首 token 时间、decode 服务成本和容量，不能用一个最小 TPOT 结果概括所有目标。"
        "本页是条件仿真的资源归因，未进行真机 kernel 校准；表格独立取整会出现尾差，原值的加和检查已经通过。"
    )
    raw_all = raw_pre + raw_dec
    raw_lines = []
    for r in raw_all:
        fields = "；".join(f"{k}={v:.12g}" for k, v in r["components"].items())
        raw_lines.append(f"原始六项附表｜{r['phase']} bs{r['global_batch']} {r['chip']} {r['objective']} "
                         f"{r['layout']}（{r['unit']}）：{fields}；total={r['total']:.12g}")
    reds = [
        ("prefill 计算归因增加", "prefill.csv 中最小 TPOT 布局的 compute-bound 归因相对参考布局上升，其他资源项不能完全抵消这部分增长。"),
        (f"{dec95:.4f} ms", "950DT decode bs32：参考通信启动减去最优通信启动，等于 43.2132−16.602=26.6112 ms。"),
        (f"{dec95:.3f} ms", "950DT decode bs32 通信启动净减少值，图内保留三位小数，原值为 26.6112 ms。"),
        (f"{dec96:.3f} ms", "960DT decode bs32：通信启动由 20.584 ms 降至 3.594 ms，净减少 16.990 ms。"),
        (f"+{inc95:.3f} s", f"950DT prefill bs1：计算受限归因差为 {inc95:.12g} s，带来 TTFT 取舍。"),
        (f"+{inc96:.3f} s", f"960DT prefill bs1：计算受限归因差为 {inc96:.12g} s，带来 TTFT 取舍。"),
        (f"{pre95:.1%} / {pre96:.1%}", "950DT/960DT 的最小 TPOT 布局 TTFT，相对各自参考布局分别按最优/参考−1 计算。"),
        ("prefill bs1｜单请求首 token（s）", "左图使用独立 prefill 全局 batch1 账本，单位为秒；其六项之和等于 TTFT。"),
        ("decode bs32｜AR 单轮（ms）", "右图使用 1M、AR、decode 全局 batch32 账本，单位为毫秒；AR 每轮提交一个输出。"),
    ]
    for r in (b95, b96):
        reds += [
            (f"{r['ttft_bs1_s']:.3f}", f"{short_chip(r['chip'])} 最小 TPOT 布局的 prefill bs1 总时间，单位为秒，六项原始归因加和验证通过。"),
            (f"{r['tpot_ms']:.3f}", f"{short_chip(r['chip'])} 最小 TPOT 布局的 AR decode bs32 单轮时间，单位为毫秒，六项原始归因加和验证通过。"),
        ]
    sources = [
        SOURCE + "prefill.csv",
        SOURCE + "latency_breakdown.csv",
        SOURCE + "summary.csv",
        SOURCE + "methodology.md",
        "input/deck-data.json: ar_cases[*].prefill / decode_latency",
    ] + raw_lines
    slide.save(
        speech,
        "材料至此完成从架构、资源约束、布局扫描到阶段归因的闭环；下一步以真实 kernel、路由和接受情况校准这些条件预测。",
        reds, sources,
        glossary_extra=[
            {"term": "hash", "full": "Hash，哈希寻址", "explain": "主机按 ngram 索引计算查表地址，算入主机查表与哈希延迟。"},
            {"term": "kernel", "full": "Compute Kernel，计算内核", "explain": "每个内核按计算与 HBM roofline 较慢者计费，二者不重复归因。"},
        ],
    )
    write_extra(8, raw_latency_components=raw_all, raw_latency_components_order=RAW_CATEGORIES)


if __name__ == "__main__":
    make_page7()
    make_page8()
    for page in (7, 8):
        notes = json.loads((WS / f"notes/page{page}.json").read_text(encoding="utf8"))
        cn_len = len(re.findall(r"[\u4e00-\u9fff]", notes["speech"]))
        print(f"page{page}: speech Chinese characters={cn_len}; SVG/notes/content generated")
