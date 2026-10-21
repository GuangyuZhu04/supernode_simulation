from pathlib import Path
import csv, json, hashlib
from collections import Counter
from zipfile import ZipFile
from lxml import etree as E

WS=Path(__file__).resolve().parent
SRC=WS.parent/"outputs/deepseek-v41-pro-10t-20261020"
def read_csv(name):
    out=[]
    with (SRC/name).open(encoding="utf-8-sig",newline="") as f:
        for row in csv.DictReader(f):
            for k,v in row.items():
                if v=="": row[k]=None
                else:
                    try: row[k]=float(v) if any(c in v for c in ".eE") else int(v)
                    except (ValueError,TypeError): pass
            out.append(row)
    return out
summary=read_csv("summary.csv")
recs=read_csv("recommendations.csv")
prefill=read_csv("prefill.csv")
latency=read_csv("latency_breakdown.csv")
def selected(chip, objective, mode="AR", c=1, context="1M", batch=32):
    rr=[r.copy() for r in recs if r["chip"]==chip and r["context"]==context and r["global_batch"]==batch and r["mode"]==mode and r["committed"]==c and r["objective"]==objective]
    if len(rr)!=1: raise ValueError((chip,objective,mode,c,len(rr),sorted({r["objective"] for r in recs})))
    r=rr[0]
    pp=[p for p in prefill if p["chip"]==chip and p["context"]==context and p["global_prefill_batch"]==1 and p["layout"]==r["layout"] and p["dspark"]==str(mode!="AR")]
    assert len(pp)==1
    r["prefill"]=pp[0]
    r["decode_latency"]={v["item"]:v["amortized_ms"] for v in latency if v["chip"]==chip and v["context"]==context and v["global_batch"]==batch and v["layout"]==r["layout"] and v["mode"]==mode and v["committed"]==c}
    assert abs(sum(r["decode_latency"].values())-r["tpot_ms"])<1e-8
    r["request_peak_bs1_GB"]=max(r["hbm_peak_GB"],r["prefill_bs1_peak_GB"])
    return r

print("OBJECTIVES",sorted({r["objective"] for r in recs}))
data={"source":str(SRC),"status_counts":dict(Counter(r["status"] for r in summary))}
data["architecture"]=json.loads((SRC/"architecture-audit.json").read_text(encoding="utf8"))
data["model"]=json.loads((SRC/"model.json").read_text(encoding="utf8"))
data["experiment"]=json.loads((SRC/"experiment.json").read_text(encoding="utf8"))
data["recommendation_samples"]=[r for r in recs if r["context"]=="1M" and r["global_batch"]==32 and (r["mode"]=="AR" or r["committed"]==4)]
data["summary_samples"]=[r for r in summary if r["context"]=="1M" and r["global_batch"]==32 and r["layout"] in ("tp8-ep128-dp1-sp8","tp8-ep32-dp4-sp8","tp4-ep16-dp16-sp4") and r["status"]=="feasible"]
data["sources"]={n:hashlib.sha256((SRC/n).read_bytes()).hexdigest() for n in ["summary.csv","prefill.csv","recommendations.csv","latency_breakdown.csv","methodology.md","architecture-audit.json","experiment.json","verification.json","result-verification.json"]}
data["ar_cases"]=[selected(chip,obj) for chip in ("950dt","960dt") for obj in ("reference","fastest_decode")]
data["dspark_cases"]=[selected(chip,obj,"DSpark3",4) for chip in ("950dt","960dt") for obj in ("reference","fastest_decode")]
data["fixed_layout_contexts"]=[]
data["fixed_layout_batches"]=[]
data["dspark_sensitivity"]=[]
for chip in ("950dt","960dt"):
    base=selected(chip,"reference")
    best=selected(chip,"fastest_decode")
    layout=best["layout"]
    metrics={
        "chip":chip,
        "throughput_gain":best["decode_tokens_s"]/base["decode_tokens_s"]-1,
        "ttft_increase":best["ttft_bs1_s"]/base["ttft_bs1_s"]-1,
        "tpot_reduction":1-best["tpot_ms"]/base["tpot_ms"],
        "peak_percent":best["request_peak_bs1_GB"]/(144 if chip=="950dt" else 288)*100,
    }
    data.setdefault("gains",[]).append(metrics)
    for context in ("128k","256k","512k","1M"):
        row=next(r.copy() for r in summary if r["chip"]==chip and r["layout"]==layout and r["context"]==context and r["global_batch"]==32 and r["mode"]=="AR")
        row["request_peak_bs1_GB"]=max(row["hbm_peak_GB"],row["prefill_bs1_peak_GB"])
        data["fixed_layout_contexts"].append(row)
    for batch in (1,2,4,8,16,32):
        row=next(r.copy() for r in summary if r["chip"]==chip and r["layout"]==layout and r["context"]=="1M" and r["global_batch"]==batch and r["mode"]=="AR")
        data["fixed_layout_batches"].append(row)
    for c in (1,2,3,4,5,6):
        row=next(r.copy() for r in summary if r["chip"]==chip and r["layout"]==layout and r["context"]=="1M" and r["global_batch"]==32 and r["mode"]=="DSpark3" and r["committed"]==c)
        data["dspark_sensitivity"].append(row)
data["route_reuse"]=read_csv("route-reuse-sensitivity.csv")
(WS/"input/deck-data.json").write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf8")
with ZipFile(WS/"input/source.pptx") as z:
    texts=[]
    for i in range(1,7):
        t=E.fromstring(z.read(f"ppt/slides/slide{i}.xml")).xpath("//a:t/text()",namespaces={"a":"http://schemas.openxmlformats.org/drawingml/2006/main"})
        texts.append(f"# 第{i}页\n\n"+"\n\n".join(t))
    (WS/"input/reference-content.md").write_text("\n\n".join(texts),encoding="utf8")
(WS/"input/data-initial.json").write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf8")
for r in data["recommendation_samples"]:
    print(r["chip"],r["mode"],r["objective"],r["layout"],*[f"{r[k]:.4f}" for k in ["ttft_bs1_s","tpot_ms","decode_tokens_s","hbm_peak_GB","prefill_bs1_peak_GB"]])
for r in data["ar_cases"]:
    print("\nAR CASE",r["chip"],r["objective"])
    print("Decode",json.dumps(r["decode_latency"]))
    print("Prefill",json.dumps({k:v for k,v in r["prefill"].items() if k.startswith("latency")}))
print("\nGAINS",json.dumps(data["gains"]))
for r in data["fixed_layout_contexts"]:
    print("CONTEXT",r["chip"],r["context"],*[round(r[k],4) for k in ["ttft_bs1_s","tpot_ms","decode_tokens_s","request_peak_bs1_GB"]])
for r in data["fixed_layout_batches"]:
    print("BATCH",r["chip"],r["global_batch"],*[round(r[k],4) for k in ["tpot_ms","decode_tokens_s"]])
for r in data["dspark_sensitivity"]:
    print("DSPARK",r["chip"],r["committed"],*[round(r[k],4) for k in ["round_ms","tpot_ms","decode_tokens_s","draft_ms","verify_ms","commit_ms","speedup_vs_same_layout_AR"]])
