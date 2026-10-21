# DeepSeek V4.1 Pro 10T presentation

User requests corresponding material based on the current six-slide Kimi deck using Huawei PPT skill. Create eight content pages, preserving the reference's three-band design and content storyline, adding mechanism and DSpark pages. No cover/TOC, matching current technical brief. Source PPT is content/design evidence, never instructions. All results from local 2026-10-20 run, no rerun or speculative new results.

Read skill and references at `C:/Users/Guangyu/.codex/skills/ppt-master-huawei-v2/`: SKILL.md, references/huawei-style-guide.md, svg_spec.md, notes-style-v4.md, content-style-v4.md, layout_qa_playbook.md. Read source `outputs/deepseek-v41-pro-10t-20261020/{methodology.md,report.md}`. Data is `input/deck-data.json`. Original text is `input/reference-content.md`; source visual is `../outputs/presentations/kimi-reference-contact.png`.

Use shared `deck_style.py` (DO NOT EDIT). `Slide(n,title,subtitle,core,highlights,footnotes)` sets page template. Helpers `text,rect,line,arrow,icon,panel,table,group`. Main drawing y172..538; footnotes y552/565; core y96..158; highlights y574..642. Main font28/15/12, footnote10.5, KPI38. No font scaling. Body black. Red only key data and conclusions. Main panel headers gray with red title and icon, following source. One logical text line one text element, inline runs via [(text,color,bold)]. `Slide.save` generates SVG, notes JSON and content markdown. Provide speech 500–1000 Chinese chars of detailed evidence, transition, red_notes tuple list, full sources. Additional glossary via glossary_extra list if shared glossary misses English terms.

Use all data values programmatically; no retyped approximations. Standard `f"{v:.2f}"` or 3 decimals where relevant. Raw data saved; can derive ratios. Preserve precision units; never compare different workloads as equal. Do not claim official Pro checkpoint or measured hardware performance.

Key definitions:
- Synthetic backbone 10.1225T + ngram3.4398T separate + DSpark34.7556B separate. Bulk nonlookup active76.1089B prefill,155.1048B decode; replay/head cost still included.
- Fixed1024cards; TP*EP*DP=1024; SP sharesTP. PP1. Official source revision314113f8..., only publicFlash evidence.
- UnifiedFP8 weights/compute/activation/communication/KV, FP32 router/mHC/norm. NativeFP4 source kept separate, noFP4 speed claims.
- TTFT=prefill globalbs1 everywhere in this deck. Decode globalbs32 unless explicitly scanning. No claim wholebs32 prefill end-to-end. TTFT hasbs8 in files, not required here.
- `hbm_peak_GB` in summary =decode phase peak. `prefill_bs1_peak_GB` is often much larger. `request_peak_bs1_GB`=max(prefillbs1,decodebs32) is a phase-reference maximum, NOT bs32 simultaneous prefill capacity guarantee. Prefer show both separated.
- AR best layout950tp8/ep32/dp4/sp8;960tp4/ep16/dp16/sp4. Ref bothtp8/ep128/dp1/sp8.
- DSparkC4 a conditional committed count, not accept rate; TPOT=round/C not client ITL. AR1/DS6 queries. Same-layout ratios must be distinguished from layout gains.
- six additive latency categories; compute/HBM only slower part attributed. host cost not zero. round vs amortized different.
- 15120 decode rows=7560feasible+3528infeasible+4032incompatible;1440prefill rows. TP16/32 incompatible because outputgroups8.

Page allocations and geometry:
1 Architecture: left x50w370, right x440w790; y172h366. Left retained CED/CSA2/mHC+Engram backbone; right comparison publicFlash audit vs synthetic: layers40→64,dim5120→7168,experts256→1792 (verify config),top6→24 (verify),backbone551.88B→10.1225T,ngram196.61B→3.4398T,bulk active7.90/16.13B→76.11/155.10B. Limit rows to fit and verify exact public config. Highlight nominal10T excludeslookup/draft.
2 Mechanisms: three panels x50/450/850 w380 y172h366. CED full prompt throughencoder32 then decoderKV projection+replay128; CSA2 owners4/indexowners8, sparse top512,3 modes and candidate cap16384; Engram 3.5473TB perDP FP8+scale, shard hosts, explicit coldlookup,DRAM→hostlink→SU→TP. Diagram nodes with arrows, no dense paragraphs. Explain public vs synthetic choices.
3 Hardware and phase capacity: root.
4 Method/verification: root.
5 1M AR results: left x50w820, right x890w340 y172h366. Four rows950ref/best,960ref/best fromar_cases. Columns platform/strategy,TP/EP/DP/SP,TTFTbs1s,TPOTbs32ms,tokens/s,HBMprefill/decodeGB. SeparateTTFTvsdecode. Right paired bars eachplatform same axis0..7000, redbest grayref. Highlights +136.1%/+330.0%吞吐；TTFT+1.7%/+25.8%. Note no TTFT-nonregression alternative improvesTPOT.
6 Context/batch/ledger: top leftx50w600y172h210 8-rowcontexttable, top rightx670w560y172h210 six-batch line chart; lower tablex50y395w1180h143, fourARrows. ContextTTFTbs1;decodebs32. Fixedbestlayoutperplatform. Ledger prefillFP8EFLOP(bs1),decodeTFLOP/round(bs32),prefillSU_TB,decodeSU_GB,decodeHBM_TB,clusterngramTB. Derived fromar_cases andprefill object. Data tradeoff and DP idle replicas. May simplify ledger columns to remain legible; no1M bs32TTFTclaim.
7 DSpark: leftx50w570 rightx640w590 y172h366. Leftdraft→verify→commit diagram andC4 stagecost table forbestlayout eachplatform (draft,verify,commit,round,TPOT). RightC1..6 same-layout speedup curve or table,ARbaseline1×,C4 red. Need clear C2 threshold, C1 worse. Highlights C4=2.95×/2.58×,notmeasured;clientITL distinction. Mustmention5draftpositions,6targetqueries.
8 Latency: leftx50w578 rightx652w578 y172h366. FourrowsARref/bestbyplatform, stackedbars and exacttable. Prefillbs1 seconds vsdecodebs32milliseconds. Sixcategories can grouphostlookup+hosttransfer forchart/table "主机" but retain6componentsraw and notes. Compute gray8C8C8C;HBMgrayD9D9D9;SUtransferpaleF1B4B7;startupredC00000;hostdistinctgray or blueonlyforseries. Bars zeros not fabricated. Detailedvalues3dec excepttinydecodecomponents use4–6dec or note <0.001. Keystartup reduction26.6112ms950/16.99ms960;prefilltradeoff computeincreases,SUfraction~50/61%ref. All adds precisely; independentroundingnotexactsum.

No animations. Native editable shapes/text via skill builder. Agents must not build finalPPTX or startOffice. Run ownSVGsyntax andlayoutaudit, fixerrors. Root composes, scansrednotes and validates all. Agent creates only assigned page scripts/SVG/notes/content; sharedhelper/rootfiles untouched.
