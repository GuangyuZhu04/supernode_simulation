from pathlib import Path
import copy, hashlib, json, math, shutil

WS = Path(__file__).resolve().parent
ROOT = WS.parent
OLD = ROOT / "3c05eebe-30bc-4f4a-99d9-d62a72c8f29a"
SIM = ROOT / "outputs/kimi-k3-10t-1m-acceleration-20261019/parallelism-20261020"
SOURCE = ROOT / "outputs/presentations/kimi-k3-10t-brief-20261020.pptx"
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text(encoding="utf-8"))
assert sha(SOURCE) == read(OLD / "qa/final-delivery.json")["sha256"]
shutil.copy2(SOURCE, WS / "input/source.pptx")
shutil.copy2(OLD / "svg/logo.png", WS / "svg/logo.png")
summary = {r["name"]: r for r in read(SIM / "summary.json")}
prior = read(OLD / "input/selected-data.json")
primary = ["compute_bound_service_s", "hbm_bound_service_s",
           "communication_service_s", "collective_startup_s"]
other = ["scheduler_wait_s", "shared_resource_contention_s",
         "kernel_startup_s", "other_service_s"]
result = {"date": "2026-10-20", "units": {"prefill": "s", "decode": "ms"},
          "primary_keys": primary, "other_keys": other, "rows": [], "checks": []}
for selection in prior["selected"]:
    r = summary[selection["name"]]
    assert r["status"] == "feasible" and r["global_batch"] == 32 and r["context"] == "1M"
    entry = {k: r[k] for k in ("name", "chip", "tp", "sp", "ep", "dp", "local_batch",
                               "global_batch", "ttft_s", "tpot_ms")}
    entry.update(label=selection["label"], layout=selection["layout_display"])
    entry["short_label"] = r["chip"][:3] + {"原并行": "原", "兼顾时延": "兼", "解码优先": "解"}[selection["label"]]
    entry["chart_label"] = entry["short_label"] + " " + entry["layout"]
    for phase, factor, total in (("prefill", 1, r["ttft_s"]), ("decode", 1000, r["tpot_ms"])):
        raw = r["latency_" + phase]
        assert set(raw) <= set(primary + other)
        assert raw == selection["latency_" + phase]
        summed = sum(raw.values()) * factor
        assert math.isclose(summed, total, rel_tol=1e-10, abs_tol=1e-8)
        values = [raw.get(k, 0) * factor for k in primary]
        other_total_s = sum(raw.get(k, 0) for k in other)
        assert other_total_s < 1e-9
        entry[phase] = {"raw_s": raw, "primary": values, "other_total_s": other_total_s,
                        "total": total, "unit": result["units"][phase],
                        "display": [f"{v:.3f}" for v in values] + [f"{total:.3f}"],
                        "display_rounding_difference": sum(round(v, 3) for v in values) - round(total, 3)}
        result["checks"].append({"case": r["name"], "phase": phase,
            "sum_components": summed, "reported_total": total,
            "absolute_difference": abs(summed-total), "unit": result["units"][phase]})
    result["rows"].append(entry)
result["gains"] = {}
for chip in ("950dt", "960dt"):
    baseline = next(r for r in result["rows"] if r["chip"] == chip and r["label"] == "原并行")
    balanced = next(r for r in result["rows"] if r["chip"] == chip and r["label"] == "兼顾时延")
    aggressive = next(r for r in result["rows"] if r["chip"] == chip and r["label"] == "解码优先")
    result["gains"][chip] = {
        "balanced_prefill_s_saved": baseline["ttft_s"] - balanced["ttft_s"],
        "balanced_decode_ms_saved": baseline["tpot_ms"] - balanced["tpot_ms"],
        "balanced_decode_startup_ms_saved": baseline["decode"]["primary"][3] - balanced["decode"]["primary"][3],
        "aggressive_prefill_compute_s_added": aggressive["prefill"]["primary"][0] - baseline["prefill"]["primary"][0],
    }
result["scope"] = {
    "workload": "F,1M=1048576,global_bs32,top32,output2,1024 cards/platform,PP1; exact six combinations on original slide4",
    "attribution": "Native critical-path attribution: a roofline event's overlapping compute/HBM service is assigned to its controlling resource, not summed twice.",
    "decode_zero_compute": "Compute-bound service=0 means decode kernels are HBM-bound in this model; computation and FLOPs still exist and are included in original slide5.",
    "other": "Scheduler wait + shared-resource contention + kernel startup + other service are zero up to floating-point residuals below1e-9 s per stage in these six cases.",
    "rounding": "Totals use full-precision simulation; independently rounded components can differ from displayed total by0.001 s or ms.",
    "limit": "Uncalibrated model prediction, fixed batch, first decode interval; no queueing or production P99.",
}
(WS / "input/latency-data.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
source_proof = {"source_pptx": str(SOURCE), "source_sha256": sha(SOURCE),
    "simulation_files": {n: sha(SIM/n) for n in ("summary.json", "latency_breakdown.csv", "verification.json")},
    "original_slides": 5, "new_slide": 6, "preservation": "Append only; keep every original slide/notes/media/master part byte-for-byte.",
    "component_sum_checks": result["checks"]}
(WS / "qa/source-proof.json").write_text(json.dumps(source_proof, ensure_ascii=False, indent=2), encoding="utf-8")
(WS / "input/request.md").write_text(
    "# 用户请求\n更新刚刚的PPT，在保留原有全部数据不删除的情况下，加入仿真时延拆解。\n"
    "方案：原有5页及其备注/图形不修改，新增第6页，同6组组合的prefill和decode关键路径分解；沿用华为风格。\n",
    encoding="utf-8")
content = """<!-- 逻辑套路：问题分析；沿用原稿V4三段式 -->
# 时延拆解显示：prefill重计算与传输，decode主要受通信启动约束
06 时延拆解 · 与第4页同六组组合：1M、全局bs32、每平台1024卡、F加速、top32、output2
> 保TP减EP：prefill收益来自传输缩短，decode收益主要来自通信启动减少。
## 左：prefill / TTFT 关键路径（单位s）
|平台/策略|TP/SP,EP,DP|计算主导|HBM主导|通信传输|通信启动|TTFT合计|
|---|---|---:|---:|---:|---:|---:|
"""
for r in result["rows"]:
    content += f"|{r['short_label']}|{r['layout']}|" + "|".join(r["prefill"]["display"]) + "|\n"
content += """## 右：decode / TPOT 关键路径（单位ms）
|平台/策略|TP/SP,EP,DP|计算主导|HBM主导|通信传输|通信启动|TPOT合计|
|---|---|---:|---:|---:|---:|---:|
"""
for r in result["rows"]:
    content += f"|{r['short_label']}|{r['layout']}|" + "|".join(r["decode"]["display"]) + "|\n"
content += """## 方法与归因
- **时间分项**：计算主导服务、HBM主导服务、通信传输、通信启动为四个主要分项；其余四项也被保留于底层数据。
- **其余归因**：调度等待、共享资源争用、kernel启动、其他服务，在这六组配置内合计仅有低于10^-9秒的浮点残差，显示约0。
- **不要双计重叠**：按原生关键路径与算子主导资源归因；不能把计算和HBM两个独立下界再相加。
- **零值含义**：decode“计算主导=0”表示本模型中算子服务由HBM主导，不代表零计算或零FLOPs；原第5页仍保留计算工作量。
- **可比性**：固定1M、全局bs32、F精度、相同6组平台/布局，全部整程可行；TPOT为一次decode间隔，非稳态/P99。
- **精度**：分项独立四舍五入，显示总和可能差0.001；用原始完整精度已核对12个阶段总和。
## 本页亮点
兼顾组合：950DT/960DT的decode通信启动分别减少9.274 / 5.152 ms。
计算与HBM按主导项归因；decode计算主导为0并不代表没有计算。
来源：2026-10-20并行度扫描summary.json、latency_breakdown.csv；未实机校准。
"""
(WS / "content/content6.md").write_text(content, encoding="utf-8")
print(json.dumps({"workspace": str(WS), "rows": len(result["rows"]),
                  "stage_sums_verified": len(result["checks"]), "gains": result["gains"]}, ensure_ascii=False, indent=2))
