"""Scope-only updates to the original three introduction/method slides."""
from pathlib import Path
import json

WS = Path(__file__).resolve().parent
changes = {
    1: {
        "当前 Kimi K3": "公开 Kimi K3 快照",
        "最新实验采用 EP32，每个专家并行分片对应 104 个专家": "并行扫描以 EP32 为起点，并向 EP16 / 8 / 4 / 2 扩展",
        "来源：官方配置快照及 2026-10-19 仿真清单。": "来源：官方配置快照与2026-10-20并行扫描。",
    },
    2: {
        "02 集群要求 · 已仿真的部署条件；以下资源指标采用 1M 输入、batch=1、top-32、output=2":
        "02 基线集群要求 · 原并行配置的资源锚点；1M、batch=1、top-32、output=2",
        "，SP32 共用 TP ranks；PP=DP=1（未搜索最小卡数）":
        "，SP32共用TP；PP=DP=1为基线，最新并行扫描见后两页",
        "来源：2026-10-19 最新 report.md、summary.csv、phase_metrics.csv、latency_breakdown.csv 与配置。硬件、拓扑及效率均为未实测校准的仿真条件。":
        "来源：2026-10-19基线报告及配置；2026-10-20并行扫描见第4–5页。硬件、拓扑与效率未实测校准。",
    },
    3: {
        "本次：结构化流式推理": "本次：副本级流式 + DP聚合",
        "边界：固定 batch，无线上排队/P99；未做实机 kernel 校准；解析下界不能证明可行。来源：项目仿真技能说明与系统架构文档。":
        "边界：固定batch、无排队/P99；DP按单逻辑非阻塞SU独立副本聚合，并经原生整图核对；未做真机校准。",
    },
}
audit = []
for n, mapping in changes.items():
    p = WS / f"svg/page{n}.svg"
    text = p.read_text(encoding="utf-8")
    for before, after in mapping.items():
        assert text.count(before) == 1, (n, before)
        text = text.replace(before, after)
        audit.append({"page": n, "before": before, "after": after})
    p.write_text(text, encoding="utf-8")
    note_path = WS / f"notes/page{n}.json"
    note = json.loads(note_path.read_text(encoding="utf-8"))
    if isinstance(note["speech"], list):
        note["speech"] = "\n\n".join(note["speech"])
    if n == 1:
        note["speech"] += (
            "本次新增并行度扫描仍使用同一个10.171T合成模型与top-32，"
            "以EP32为起点，向EP16、EP8、EP4、EP2等候选扩展，并同步调整TP和DP；"
            "专家规模及质量假设没有随布局改变。结果在第4至5页比较。"
        )
    if n == 2:
        note["speech"] = note["speech"].replace("已仿真的部署是", "本页沿用的原并行基线是")
        note["speech"] += (
            "这些数字专门作为batch=1、TP32/EP32/DP1的资源锚点；"
            "并不表示所有最新候选仍使用DP1。后两页在固定1024卡和全局请求数下比较新布局，"
            "峰值和最终驻留随副本复制及上下文变化，不能套用本页61.44GB到全部布局。"
        )
        for g in note["glossary"]:
            if g["term"] == "DP":
                g["explain"] = "不同副本处理不同输入；本页基线DP1，后两页扫描更大的DP，SP不额外相乘。"
            if g["term"] == "PP":
                g["explain"] = "沿模型层划分流水段；本次基线及全部并行扫描均为1。"
    if n == 3:
        old = "本次使用结构化流式推理，保留性能、流量、阶段资源和并发峰值，而不是据导出明细多少判断仿真精度。"
        new = (
            "本次每个DP副本使用原生DP1结构化流式推理。"
            "在单逻辑非阻塞SU、无跨副本共享瓶颈的明确条件下，各副本资源独立："
            "时延取副本最大值，流量与计算量累加，空闲副本仍计静态驻留。"
            "副本拆分已用1024卡原生整图和完整93层样例核对；"
            "没有把流式引擎改为通用DP执行器。"
        )
        assert old in note["speech"]
        note["speech"] = note["speech"].replace(old, new)
        note["transition"] = "接下来两页先比较不同并行组合，再展示上下文、batch和资源账本的变化。"
        for r in note["red_notes"]:
            if r["quote"] == "本次：结构化流式推理":
                r["quote"] = "本次：副本级流式 + DP聚合"
                r["explain"] = "当前单逻辑非阻塞SU下，独立DP副本按最大时延和累计工作量聚合，已与原生整图核对；只适用于该资源独立条件。"
        note["glossary"].append({"term": "DP", "full": "Data Parallelism，数据并行",
            "explain": "不同副本接收不同请求；本次以经验证的独立副本方式聚合，而非放开流式引擎的DP1限制。"})
    note["sources"].append("2026-10-20补充：outputs/kimi-k3-10t-1m-acceleration-20261019/parallelism-20261020/verification.json及audit/verification.json")
    note_path.write_text(json.dumps(note, ensure_ascii=False, indent=2), encoding="utf-8")
(WS / "qa/scope-updates.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
print({"scoped_text_updates": len(audit)})
