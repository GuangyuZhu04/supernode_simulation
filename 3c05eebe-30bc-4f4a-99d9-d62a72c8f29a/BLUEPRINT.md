# Existing-deck update: exactly two new results pages (page4, page5)

Read the Huawei skill references before writing. Skill root:
C:/Users/Guangyu/.codex/skills/ppt-master-huawei-v2

Authoritative content: content/content4.md, content/content5.md.
Authoritative numbers: input/selected-data.json. Never invent values or round one metric into another.
The first three original slides retain their established V4 three-band design. Follow that design for both new slides.
User requests comparisons of TTFT, TPOT, throughput and resource occupancy; these are simulation predictions, not measured hardware benchmarks.

## Shared exact skeleton

Canvas1280×720, white. Fonts 微软雅黑, Arial, sans-serif.
Colors: #C00000 accent; #1D1D1A main text; #666666 notes; #D9D9D9 line; #F2F2F2 gray fill; #FFF1EF pale red.
Three primary type levels: title28, panel/core15, body12. Auxiliary10.5 and KPI exceptions only. Numeric cells can be15 if room.
Stroke1, accent1.5 maximum. No animations. 3–8 core semantic groups with IDs; header/footer groups excluded.
Footer preserved exactly:
<g id="footer-group" font-family="Arial, sans-serif" font-size="10.5" fill="#1D1D1A">
<text x="77" y="683">PAGE</text>
<text x="125" y="683">Huawei Proprietary - Restricted Distribution</text>
<image href="logo.png" x="1070" y="663" width="133" height="28" preserveAspectRatio="xMidYMid meet"/>
</g>
Header group id=page-header:
title x50 y50 size28 bold red; subtitle x50 y77 size12 gray.
Core group:
rect(50,96,1180,62), pale red, stroke red1.5, rx4;
target-arrow icon x70y115 w22h22 red;
核心思想 x104y132 font15 red bold; divider x180y110..146 gray;
core text x200y132 font15 black bold, use inline red tspan for meaningful key claims.
Highlight:
rect(50,574,1180,68) gray fill, border gray1;
本页亮点 x72y601 font15 redbold;
two concise lines atx180 y601/626 font15 black, red tspan keywords.
Sources/definitions can use10.5 at y551/565, never overlap core table/chart.

Safe icons confirmed:
chunk-filled/target-arrow, chart-bar, chart-line, database, server, arrows-repeat, circle-checkmark.
Use <use data-icon="chunk-filled/chart-bar" .../>. Root will expand natively.

Inline rules: escape &, <, >. One logical line = one text with tspan runs; inline tspan never x/y/dy.
Text box budgets: CJK1em; uppercase1.06em conservative; digits.66em bold; otherASCII.62em.
Keep text ≥12–16px from card edges. No text stretching, no shrinking below readable12px body.
Top-level groups must be meaningful, not a group per cell.

## Page4 geometry and content

Title: 保留TP32、降低EP，可同时降低时延并提升吞吐
Subtitle includes2026-10-20, F,1024卡/平台,1M,全局bs32,top32,output2.
Core: 兼顾布局decode吞吐+27.1% / +34.0%，TTFT同步下降；解码优先有首词时延代价。
Main left panel x50 y172 w820 h356; right x885 y172 w345 h356.
Left table: 6 rows = 950DT original/balanced/decode-priority, then960DT same.
Use source labels 原并行 / 兼顾时延 / 解码优先. All6 full-request feasible.
Columns: platform/strategy; TP/SP,EP,DP (second line local bs); TTFT/s; TPOT/ms; decode tokens/s; peak HBM GB / occupancy%.
Suggested widths140/120/100/100/130/200 =790 inside820. Header begins y219; rows spaced42px, row baselines255,297,339,381,423,465. Adjust to avoid row labels colliding.
Use red for balanced-row key result metrics, black for other numbers, pale red row fill optional. Do not make a giant red wall.
Right panel: two platform-specific horizontal throughput charts, each3 bars; gray original, deep red balanced, pale red decode-priority.
Show numeric labels (742.9/944.5/1292.3 and1586.2/2125.3/4407.6).
Can use independent per-platform axes only if clearly labeled; a shared0–4500 axis also fine.
Chart labels in12px, data labels12 or15. Include icons20px and red header15.
Sources/definitions y551/565: "HBM占比=峰值/144或288GB；数据为F配置、同全局负载；TPOT为一次decode间隔，未校准预测。"
Highlight emphasizes balanced gains and TTFT/memory cost of lowest TPOT.

## Page5 geometry and content

Title: 增大batch可近线性提升吞吐，扩容DP仍需核对资源代价
Main top-left x50y172w600h220: 8-row context table fixed layouts from page4, globalbs32.
Header15 with icon; columns 平台/上下文/TTFT(s)/TPOT(ms)/吞吐(tok/s)/峰值HBM(GB).
Rows use12px,22px spacing. Display8 exact data rows. Fixed layouts noted in header/subtitle or notes.
Main top-right x665y172w565h220: batch throughput chart, 2 lines (950DT gray,960DT red),6 batch levels1/2/4/8/16/32.
X uses log2 batch slots, explicitly marked; Y tokens/s linear. Show all12 numeric data labels, offset950below and960above; edge labels start/end anchored so no clipping.
At bottom of chart show TPOT endpoints for both platforms (source batch_growth), clearly forbs1→32.
Resource table full-width x50y406w1180h146:
5 rows = shared original baseline;950balanced;950decode-priority;960balanced;960decode-priority.
Columns (suggestion): 方案180 / prefill计算EFLOP150 / decode计算TFLOP/步150 / prefill SU PB150 / decode SU GB/步150 / decode HBM TB/步170 / 全系统驻留TB180. Total1130 inside1180.
Header two short lines, rows baselines471/489/507/525/543. Data body12.
All values in selected-data.resources; baseline row legitimately common to both platforms, asserted equal by root.
Footer definition/source line y565 font10.5. "工作量≠利用率；驻留=decode完成态，非峰值之和；GB/TB/PB十进制；模型预测。"
Highlight: bs1→32吞吐31.4–31.5×，TPOT增幅1.6%–2.0%；1008组合的534/66/408 feasibility evidence. Exact950:31.3766×,1.9869%;960:31.4893×,1.6218%.

## Notes/output contract

Each agent writes ONLY its svg/pageN.svg and notes/pageN.json.
Notes JSON: {page:N,title,speech,transition,glossary:[{term,full,explain}],red_notes:[{quote,explain}],sources:[...]}.
Speech roughly350–650Chinese chars, include interpretation and fair global-batch boundary.
Glossary covers all visible English including units/boilerplate. Every meaningful red number/phrase/series gets a red_notes explanation with source/baseline/meaning.
Cite simulation summary.json and exact selected case names, date2026-10-20. Preserve first-decode interval, uncalibrated, noP99.
Run skill syntax/layout audits on your slide, repair errors (up to3 rounds); root will run whole-deck audits and red-note scan.
Do not build PPTX, start browsers/Office, edit other slide, or edit shared data/blueprint.
Respond DONE pageN; send blocking findings separately if any.
