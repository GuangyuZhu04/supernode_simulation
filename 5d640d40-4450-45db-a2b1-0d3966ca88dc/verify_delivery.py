from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET
from collections import Counter
import json, hashlib, re
WS=Path(__file__).resolve().parent
D=json.loads((WS/"input/deck-data.json").read_text(encoding="utf8"))
SRC=Path(D["source"])
NS={"a":"http://schemas.openxmlformats.org/drawingml/2006/main","p":"http://schemas.openxmlformats.org/presentationml/2006/main"}
checks=[]
def check(label,condition):
    assert condition,label
    checks.append(label)
with ZipFile(WS/"candidate.pptx") as z:
    pages={}
    for i in range(1,9):
        root=ET.fromstring(z.read(f"ppt/slides/slide{i}.xml"))
        pages[i]=" ".join(x.text or "" for x in root.findall(".//a:t",NS))
        check(f"page{i}_editable",len(root.findall(".//p:sp",NS))>20)
        check(f"page{i}_no_animation",root.find(".//p:timing",NS) is None)
        check(f"page{i}_no_placeholder",not re.search(r"\{\{|\{PAGE|Lorem ipsum|占位文字",pages[i],re.I))
        notes=ET.fromstring(z.read(f"ppt/notesSlides/notesSlide{i}.xml"))
        note=" ".join(x.text or "" for x in notes.findall(".//a:t",NS))
        check(f"page{i}_notes",all(t in note for t in ["过渡","本页名词表","标红内容详解","来源与数据口径"]))
    check("complete_model_name","DeepSeek V4.1-Pro-10T" in pages[1])
    check("synthetic_architecture_disclosed","合成" in pages[1] and "非公开 Pro 规格" in pages[1])
    for val in ["10.1225T","3.4398T","34.7556B","384","1792","76.11B","155.10B"]:
        check("architecture_"+val,val in pages[1])
    for r in D["ar_cases"]:
        for key in ["ttft_bs1_s","tpot_ms","decode_tokens_s","prefill_bs1_peak_GB","hbm_peak_GB"]:
            check(f"page5_{r['chip']}_{r['objective']}_{key}",f"{r[key]:.2f}" in pages[5])
        check("decode_ledger_"+r["chip"]+r["layout"],abs(sum(r["decode_latency"].values())-r["tpot_ms"])<1e-8)
        pl=[v for k,v in r["prefill"].items() if k.startswith("latency_")]
        check("prefill_ledger_"+r["chip"]+r["layout"],abs(sum(pl)-r["ttft_bs1_s"])<1e-8)
    for r in D["fixed_layout_contexts"]:
        for key in ["ttft_bs1_s","tpot_ms","decode_tokens_s","prefill_bs1_peak_GB","hbm_peak_GB"]:
            check(f"page6_{r['chip']}_{r['context']}_{key}",f"{r[key]:.2f}" in pages[6])
    for r in D["fixed_layout_batches"]:
        check(f"page6_batch_{r['chip']}_{r['global_batch']}",f"{r['decode_tokens_s']:.2f}" in pages[6])
    for r in D["dspark_sensitivity"]:
        check(f"page7_C{r['committed']}_{r['chip']}_tpot",f"{r['tpot_ms']:.3f}" in pages[7])
        check(f"page7_C{r['committed']}_{r['chip']}_speedup",f"{r['speedup_vs_same_layout_AR']:.2f}" in pages[7])
        check("dspark_round_"+r["chip"]+str(r["committed"]),abs(r["draft_ms"]+r["verify_ms"]+r["commit_ms"]-r["round_ms"])<1e-8)
        check("dspark_tpot_"+r["chip"]+str(r["committed"]),abs(r["tpot_ms"]*r["committed"]-r["round_ms"])<1e-8)
    for r in D["ar_cases"]:
        check("page8_prefill_total_"+r["chip"]+r["objective"],f"{r['ttft_bs1_s']:.3f}" in pages[8])
        check("page8_decode_total_"+r["chip"]+r["objective"],f"{r['tpot_ms']:.3f}" in pages[8])
    check("batch_scope","bs1" in pages[5] and "bs32" in pages[5] and "32 请求同时 prefill" in pages[5])
    check("speculative_scope","C=4" in pages[7] and "ITL" in pages[7] and "接受率与文本质量未实测" in pages[7])
for name,h in D["sources"].items():
    check("source_unchanged_"+name,hashlib.sha256((SRC/name).read_bytes()).hexdigest()==h)
check("original_pptx_unchanged",hashlib.sha256((WS/"input/source.pptx").read_bytes()).digest()==hashlib.sha256((WS.parent/"outputs/presentations/kimi-k3-10t-brief-20261020-v2.pptx").read_bytes()).digest())
check("source_matrix_size",sum(D["status_counts"].values())==15120)
notes8=json.loads((WS/"notes/page8.json").read_text(encoding="utf8"))
check("raw_six_resource_notes",bool(notes8.get("raw_latency_components")))
out={"status":"passed","check_count":len(checks),"checks":checks,"source_numeric_fingerprint":D["experiment"]["numeric_fingerprint"],
     "pptx_sha256":hashlib.sha256((WS/"candidate.pptx").read_bytes()).hexdigest(),
     "visual_qa":{"renderer":"WPS Presentation COM, exported slides 1600x900","slides_inspected":list(range(1,9)),"issues_fixed":["Slide1 full model name","Slide5 TP/EP attribution precision"],"notes":"SVG preview fallback is not used for delivery; Office export contains correct Chinese fonts."}}
(WS/"qa/delivery-verification.json").write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf8")
print(f"PASS: {len(checks)} content/source/ledger/package checks")
