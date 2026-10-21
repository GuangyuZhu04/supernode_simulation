# Add one latency breakdown slide; preserve original five slides unchanged

Read C:/Users/Guangyu/.codex/skills/ppt-master-huawei-v2/SKILL.md and its svg_spec, huawei-style-guide, notes-style-v4, layout_qa_playbook.
Authoritative content: content/content6.md. Data: input/latency-data.json.
Same six cases as original slide4. Do not re-simulate or change numbers. Raw stage sums checked exactly.
Only create svg/page6.svg and notes/page6.json. Root will append natively to existing PPTX with original parts unchanged.

## Exact shared style

1280×720 white; font-family="微软雅黑, Arial, sans-serif".
Title28 red #C00000 bold; core/panel15; body12; auxiliary10.5.
Main black#1D1D1A; auxiliary#666666; line#D9D9D9; gray background#F2F2F2; core pale#FFF1EF.
Stroke1 maximum1.5. No animations. 3–8 meaningful core groups; header/footer excluded.
Header id=page-header titlex50y50; subtitlex50y77 12gray.
Core group rect(50,96,1180,62) rx4 pale red stroke1.5red; target-arrow icon(70,115,22,22);
核心思想 x104y132 15boldred; dividerx180y110..146; one textx200y132 15boldblack with red tspan.
Core: 保TP减EP：prefill收益来自传输缩短，decode收益主要来自通信启动减少。
Highlight rect(50,574,1180,68) grayfill graystroke; 本页亮点 x72y601 15redbold;
two lines atx180y601/626 15black, highlight meaningful phrases/data red.
Footer EXACT:
<g id="footer-group" font-family="Arial, sans-serif" font-size="10.5" fill="#1D1D1A">
<text x="77" y="683">6</text>
<text x="125" y="683">Huawei Proprietary - Restricted Distribution</text>
<image href="logo.png" x="1070" y="663" width="133" height="28" preserveAspectRatio="xMidYMid meet"/>
</g>
Icon whitelist confirmed: chunk-filled/target-arrow, chart-bar, chart-line, database, server.

## Content geometry

Two parallel phase panels: LEFT x50y172w578h380, RIGHT x652y172w578h380.
Panel titles atx90/692y191 15boldred, icon20px at65/667y176.
Left title "prefill / TTFT 时延拆解（s）"; right "decode / TPOT 时延拆解（ms）".
Same category colors in both:
计算主导 #8C8C8C; HBM主导 #D9D9D9; 通信传输 #F1B4B7; 通信启动 #C00000.
Legend row y215, 10.5 or12; four items fit inside578px.
Horizontal stacked bars (6 rows, native vector geometry):
row center/baselines roughly y237,258,279,311,332,353.
Labels source chart_label e.g"950原 32/32/1", width130 at x66(left) /668(right), 10.5–12.
Plot starts x180(left)/782(right), width385, shared scales within each panel.
Left x-axis0..180s; right0..45ms. Axis ticks y377.
Bars12px high; totals labeled3dec outside end; reserve enough right margin (panel ends628/1230). No fabricated segment sizes.
For decode compute-bound=0 so its segment is zero width, NOT an indication of zero FLOPs.
Small segments may be too small for internal labels; exact values must appear in the table immediately below.

Numeric table within each panel: x66/668, width546, y389..544.
Columns: 平台/策略80; 计算主导93; HBM主导83; 通信传输93; 通信启动93; 合计104.
One header row baseline402; six data rows baselines424/446/468/490/512/534.
Values from phase.display (4 components + full-precision total, rounded independently3dec).
Rows short_label "950原/950兼/950解/960原/960兼/960解". Labels original/balanced/decode-priority explained in notes/subtitle.
All values visible, editable text; 12px body. Keep main cells black, balanced row or communication-startup key cells red sparingly.
Panel box ends552; table last descender must leave≥6px padding.
Footnote y565,10.5, one line: "其他=等待/争用/kernel启动/其他服务，合计<10^-9s；分项独立舍入。主导服务归因不双计计算/HBM重叠；模型预测。"
If too long, compress without losing zero-compute caveat (which belongs in highlight/notes).
Highlight line1: "兼顾组合：950DT/960DT的decode通信启动分别减少9.274 / 5.152 ms。"
Highlight line2: "计算与HBM按主导项归因；decode计算主导为0，并不代表没有计算。"

Title proposed: 时延拆解显示：prefill重计算与传输，decode主要受通信启动约束
Subtitle must state same6 cases fromslide4, 1M,globalbs32,1024cards/platform,F,top32,output2,date2026-10-20.

## Safety/QA/content discipline

One logical line = one text; inline tspan has no x/y/dy. Escape &,<,>.
No stretch tricks or smaller-than12px body. Numeric table units are LEFT seconds and RIGHT milliseconds, never mix.
Do not add independent FLOP/HBM lower bounds; data is native critical-path attribution.
Other categories retained in raw JSON, effectively0 with numerical residual<1e-9seconds; explain them rather than inventing values.
Native stage totals (prefill TTFT, decode first TPOT) must match original slide4 numbers within rounding.
Source scope uncalibrated fixed-batch analytical/discrete-event prediction, not measured or steady/P99.
Notes JSON {page:6,title,speech,transition,glossary:[{term,full,explain}],red_notes:[{quote,explain}],sources:[...]}.
Speech350–650Chinese chars, whole glossary (includingunits/boilerplate), all meaningful red phrases/data/series explained with source/baseline.
Run SVG syntax/layout audits; fix errors. Root will run notes red scan, native conversion, preservation and data checks.
Use .venv/Scripts/python.exe -X utf8, exec_command sandbox_permissions=require_escalated if needed.
Never edit original slides or shared files. Do not build PPTX/startOffice. Reply DONE page6.
