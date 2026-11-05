"""Build an offline walkthrough from the current POD/DFlash result and ImageGen assets.

Run the native case first, then this script. No model parameters or engine code change.
The HTML and SVG embed the original generated PNGs without editing their pixels.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "outputs/deepseek-v41-process-20261103"


def data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def snapshot(result: dict) -> dict:
    hw = result["hardware"]
    npu, cpu = hw["npu"], hw["cpu"]
    rates = {f"NPU.{d}": npu["flops"][d] * npu["efficiency"] for d in ("fp8", "fp32")}
    rates.update({
        "NPU.hbm": npu["hbm_Bps"],
        "SU.tx": min(hw["scale_up"]["bandwidth_Bps"], hw["scale_up"]["injection_Bps"]) * hw["scale_up"]["efficiency"],
        "HOST": hw["host_link"]["injection_Bps"] * hw["host_link"]["efficiency"],
        "CPU.dram": cpu["dram_Bps"], "CPU.fp32": cpu["flops"] * cpu["efficiency"],
        "CPU.iops": cpu["iops"],
    })
    rates["SU.rx"] = rates["SU.tx"]
    phases = {}
    for name, phase in result["phases"].items():
        clock = 0.0
        records = []
        for op in phase["stages"]:
            row = {k: op[k] for k in ("name", "component", "layer", "work", "startup_s", "dtype", "duration_s", "bottleneck")}
            row["start_s"] = clock
            clock += op["duration_s"]
            row["end_s"] = clock
            predicted = max((v / rates[k] for k, v in op["work"].items()), default=0) + op["startup_s"]
            assert abs(predicted - op["duration_s"]) < 1e-7
            records.append(row)
        assert abs(clock - phase["duration_s"]) < 1e-7
        assert len(records) == phase["native_completed_events"]
        assert abs(sum(phase["memory"]["peak_components_bytes"].values()) - phase["memory"]["hbm_peak_bytes"]) < 0.1
        phases[name] = {k: phase[k] for k in ("duration_s", "breakdown_s", "component_latency_s", "memory", "physical_work", "native_completed_events")}
        phases[name]["stages"] = records
    assert abs(sum(phases[p]["duration_s"] for p in ("draft", "verify", "commit")) - result["metrics"]["round_s"]) < 1e-10
    return {
        "generated": "2026-11-03", "config": result["config"], "layout": result["layout"],
        "metrics": result["metrics"], "phases": phases, "rates": rates,
        "provenance": result["provenance"], "hardware": hw, "draft": result["draft"],
        "model": {k: result["model"][k] for k in ("name", "n_layers", "dim", "n_routed_experts", "n_activated_experts", "engram_layer_ids", "kv_source_layers", "index_source_layers", "ngram_params")},
    }


def overview(data: dict, images: dict[str, str]) -> str:
    """Readable vector text and paths; the three component renders remain PNGs."""
    s = ['<svg xmlns="http://www.w3.org/2000/svg" width="1800" height="1630" viewBox="0 0 1800 1630" role="img" aria-labelledby="title desc">',
         '<title id="title">DeepSeek V4.1 Pro 10T 当前 FP4 / DFlash 仿真过程</title>',
         '<desc id="desc">512 卡 TP4 EP128 的合成外推模型，从配置、四阶段构图、原生资源调度到真实仿真结果。部件效果图为概念图。</desc>',
         '<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 10 5 0 10z" fill="#16877c"/></marker></defs>',
         '<style>text{font-family:Segoe UI,Microsoft YaHei,Arial,sans-serif;fill:#17323a}.small{font-size:17px;fill:#5a7075}.label{font-size:20px}.title{font-size:27px;font-weight:650}.mono{font-family:Consolas,monospace;font-size:17px;fill:#476368}</style>',
         '<rect width="1800" height="1630" fill="#f6f8f6"/>']

    def text(x, y, value, cls="label", fill=None):
        extra = f' style="fill:{fill}"' if fill else ""
        s.append(f'<text x="{x}" y="{y}" class="{cls}"{extra}>{html.escape(str(value))}</text>')

    def box(x, y, w, h, fill="#ffffff", stroke="#dae5e1", r=14):
        s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}"/>')

    def line(x1, y1, x2, y2, arrow=True):
        s.append(f'<path d="M{x1} {y1} L{x2} {y2}" fill="none" stroke="#16877c" stroke-width="2"'+(' marker-end="url(#arr)"' if arrow else '')+'/>')

    text(60, 52, "SUPERNODE SIMULATION  /  CURRENT CASE  /  2026.11.03", "mono")
    text(60, 112, "DeepSeek V4.1 Pro 10T · 仿真过程", "title")
    text(60, 152, "Synthetic 64 层 / hidden 7168 / 1792 experts / top-24 · 路由专家 FP4 · 其余精度沿用案例", "label")
    box(60, 179, 1680, 45, "#e5f1ec", "#e5f1ec", 8)
    text(79, 209, "512 卡 = TP4 × EP128 × DP1 × PP1  |  SP1  |  上下文 131072  |  Prefill B1 / Decode B32  |  DFlash 净接受 6，块长 8", "label")
    hardware = [
        (60, "pod", "逻辑 Full-mesh SU POD", ["64 hosts × 8 NPUs · 单个 SU 域", "SU 有效 800 GB/s / 方向 · 无 SO 流量"]),
        (635, "npu", "NPU + HBM", ["950DT 案例：96 GB / 卡，4 TB/s", "FP8 有效算力 500 TFLOP/s / 卡"]),
        (1210, "host", "Host DRAM / Engram", ["3.4398T 查表参数，含 scale 共 3.547 TB", "64 主机分片；HBM 不常驻整张查询表"]),
    ]
    for x, name, title, lines in hardware:
        box(x, 246, 530, 245)
        s.append(f'<image x="{x+12}" y="252" width="205" height="170" href="{images[name]}"/>')
        text(x+226, 292, title, "label")
        # Long hardware lines use the full card width beneath the render.
        text(x+20, 444, lines[0], "small")
        text(x+20, 475, lines[1], "small")
    text(60, 521, "ImageGen 部件为概念效果图，机柜/封装结构不表示实际产品规格；硬件容量、效率与拓扑按本案例输入。", "small")

    text(60, 569, "01  配置与结构 → 四阶段分别构图并执行", "title")
    text(60, 600, "CaseConfig → architecture.inventory() / model.hardware() → simulate_case() → run_phase()", "mono")
    names = [("Prefill", "prefill"), ("Draft", "draft"), ("Verify", "verify"), ("Commit", "commit")]
    details = [
        ["B1 · 131072 输入 token", "Embedding → Encoder 32 层", "Decoder KV 投影 → 128-token 重放", "首 token → DFlash KV 初始化"],
        ["B32 · 块长 8", "已知前缀 + 7 个候选位置", "5 层 Dense / 全上下文 attention", "共享目标 head → 草稿候选"],
        ["B32 · 每序列 8 个位置", "目标模型完整 64 层验证", "CSA2 + MoE + Engram + mHC", "输出预测；保留前缀状态版本"],
        ["B32 · 净接受 6（外部给定）", "恢复 / 截断前缀状态", "TP 接受数同步与 token 比较", "草稿 KV 追赶 → 下一轮"],
    ]
    for i, ((name, key), lines) in enumerate(zip(names, details)):
        x = 60 + i*425
        box(x, 624, 405, 246, "#ffffff")
        text(x+20, 664, f"{i+1:02}  {name}", "title")
        for j, value in enumerate(lines):
            text(x+20, 703+j*29, value, "small")
        duration = data["phases"][key]["duration_s"]
        text(x+20, 851, (f"{duration:.6f} s" if key == "prefill" else f"{duration*1000:.6f} ms") + f" · {data['phases'][key]['native_completed_events']} 事件", "mono")
        if i < 3:
            line(x+406, 655, x+422, 655)
    text(60, 901, "计时分开：TTFT = Prefill（含草稿 KV seed）；Decode 一轮 = Draft + Verify + Commit。B1 与 B32 不拼成同一请求时间线。", "small")

    text(60, 952, "02  每个目标层的资源路径", "title")
    chain = [("CSA2 / mHC", "投影、索引、稀疏注意力"), ("MoE router", "1792 experts → top-24"), ("EP dispatch", "SU pairwise 发送"), ("FP4 → FP8", "读 packed/scale，写缓冲"), ("Expert GEMM", "FP8 计算 + TP reduce"), ("EP combine", "返回 origin + shared expert")]
    for i, (a, b) in enumerate(chain):
        x = 60 + i*285
        box(x, 976, 260, 90, "#eaf3ef")
        text(x+15, 1010, a, "label")
        text(x+15, 1043, b, "small")
        if i < 5:
            line(x+262, 1021, x+280, 1021)
    text(60, 1098, "Engram 在 L2 / L22：CPU hash → H2D 地址 → SU 分发 → DRAM 行查询 → H2D 返回 → SU / TP 分发 → 投影。层号从 0 开始。", "small")

    text(60, 1151, "03  Phase.run() → 原生 engine.simulate()", "title")
    engine = [("容量准入", "阶段完整资源包络预留"), ("依赖就绪 / 启动", "串行 Event 链"), ("资源速率 / 推进时钟", "max(work ÷ rate) + startup"), ("记录与聚合", "trace / 物理流量 / 峰值")]
    for i, (a,b) in enumerate(engine):
        x=60+i*425
        box(x,1175,405,88)
        text(x+18,1209,a,"label")
        text(x+18,1242,b,"small")
        if i<3: line(x+407,1218,x+421,1218)
    text(60,1295,"各算子使用最忙 lane / host 的资源需求包络；全机物理工作量单独累计。当前案例为串行阶段，不模拟生产在线排队。","small")

    text(60,1347,"04  当前源码复跑结果", "title")
    m=data["metrics"]
    vals=[("TTFT · Prefill B1",f"{m['ttft_s']:.3f} s"),("固定上下文 TPOT · B32",f"{m['tpot_s']*1000:.3f} ms"),("Decode 一轮 / 净接受数",f"{m['round_s']*1000:.3f} ms / 6"),("每卡 HBM 峰值 · P / D",f"{m['prefill_hbm_GB']:.2f} / {m['decode_hbm_GB']:.2f} GB")]
    for i,(label,value) in enumerate(vals):
        x=60+i*425
        box(x,1372,405,104,"#173b3b","#173b3b")
        text(x+20,1404,label,"small","#b8d1ca")
        text(x+20,1451,value,"title","#ffffff")
    text(60,1515,"依据：当前仓库架构 / 函数流程图、v41_fp4 案例源码、2026-11-03 原生复跑 result.json。", "small")
    text(60,1545,"边界：10T 为合成外推，非官方 10T checkpoint；接受数、硬件效率及容量是输入。结果不等同实测、服务 P99 或长输出平均 ITL。", "small")
    text(60,1580,"源码指纹  " + data["provenance"]["numeric_fingerprint"][:24] + "…  |  打开配套 index.html 可逐阶段、逐算子查看。", "mono")
    s.append("</svg>")
    return "\n".join(s)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT)
    args = parser.parse_args()
    folder = args.directory.resolve()
    raw = (folder / "simulation/result.json").read_bytes()
    result = json.loads(raw)
    if result["status"] != "feasible":
        raise ValueError("This walkthrough requires a complete feasible case result.")
    expected = {"cards": 512, "tp": 4, "ep": 128, "sp": 1, "context_tokens": 131072, "prefill_batch": 1, "decode_batch": 32, "accepted": 6, "block": 8, "chip": "950dt"}
    if any(result["config"][k] != v for k,v in expected.items()):
        raise ValueError("The annotated walkthrough is specific to the current 950DT TP4/EP128 baseline.")
    data = snapshot(result)
    images = {k: data_uri(folder / "assets" / (k + ".png")) for k in ("pod", "npu", "host")}
    template = Path(__file__).with_name("deepseek_v41_process.html").read_text(encoding="utf-8")
    for key, value in images.items():
        template = template.replace("__" + key.upper() + "__", value)
    template = template.replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"))
    (folder / "index.html").write_text(template, encoding="utf-8")
    (folder / "overview.svg").write_text(overview(data, images), encoding="utf-8")
    (folder / "walkthrough-data.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "created_date": "2026-11-03", "result_sha256": hashlib.sha256(raw).hexdigest(),
        "numeric_fingerprint": data["provenance"]["numeric_fingerprint"],
        "reference_documents": ["docs/simulation-architecture.md", "docs/simulation-function-flows.md", "docs/diagrams/08-event-engine.mmd", "docs/fp4-512-case.md"],
        "source_files": ["supernode_sim/case_studies/v41_fp4/__init__.py", "supernode_sim/case_studies/v41_fp4/architecture.py", "supernode_sim/case_studies/v41_fp4/model.py", "supernode_sim/case_studies/v41_fp4/target.py", "supernode_sim/engine.py"],
        "assets": {k: {"file": f"assets/{k}.png", "sha256": hashlib.sha256((folder/"assets"/(k+".png")).read_bytes()).hexdigest(), "kind": "imagegen conceptual illustration; original pixels and alpha preserved"} for k in images},
        "checks": {"all_event_roofline_durations_match": True, "all_phase_duration_sums_match": True, "all_phase_memory_sums_match": True, "event_count": sum(p["native_completed_events"] for p in data["phases"].values())},
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (folder / "README.txt").write_text(
        "DeepSeek V4.1 Pro 10T 当前案例仿真过程\n\n"
        "index.html：自包含离线交互演示，图片、数据、样式与脚本均已内嵌。\n"
        "overview.svg：可缩放静态总览，文字与连线为矢量，部件为内嵌 ImageGen PNG。\n"
        "assets/：三张透明背景部件效果图原件；imagegen-prompts.json：完整提示词。\n"
        "simulation/：当前源码的原生仿真结果；manifest.json：来源与校验。\n\n"
        "复跑（仓库根目录）：\n"
        ".\\.venv\\Scripts\\python.exe -m supernode_sim.case_studies.v41_fp4 examples/deepseek_v41_fp4_512_ep128_950dt.json -o outputs/deepseek-v41-process-20261103/simulation\n"
        ".\\.venv\\Scripts\\python.exe scripts/render_deepseek_v41_process.py\n\n"
        "模型为合成外推，硬件与接受数为条件输入；不代表官方 checkpoint 或实机测量。\n", encoding="utf-8")
    print(json.dumps({"html": str(folder/"index.html"), "svg": str(folder/"overview.svg"), "events": manifest["checks"]["event_count"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
