# 四页共同蓝图
用户明确要求 ppt-master-huawei-v2，简短4页，无封面目录。采用V4三问锚点。阅读技能references/svg_spec.md、huawei-style-guide.md、notes-style-v4.md。技能目录 C:/Users/Guangyu/.codex/skills/ppt-master-huawei-v2。
Canvas1280×720 white; font family 微软雅黑, Arial, sans-serif; title28px red, panel/core15px, body15px or12px (main hierarchy28/15/12), note10.5px, KPIs38-42px. Stroke1px max1.5. Body#1D1D1A, gray#666666, red#C00000, pale#FFF1EF, line#D9D9D9.
Fixed header group id=page-header: title x50 y50, subtitle x50 y77 size12 gray.
Core group: rect x50 y96 w1180 h62 fill#FFF1EF stroke#C00000 1.5; icon target-arrow x70y11522; label 核心思想 x104 y13215 red; divider x180y110..146; core text x200y13215bold with red tspan key fact. One text/tspan per line, no x/y in inline tspan.
Content y172..560 x50..1230. Key cards icons20px. 3–8 semantic groups. No corner decoration. All text has padding15px. Core/highlight box rx4; main panels rect not rounded.
Highlight group rect(50,574,1180,68) fill#F2F2F2 stroke#D9D9D9; label 本页亮点 red15bold x72y601; max2 lines text x180y601/626 size15; separate source/boundary line y55110.5.
Footer group id=footer-group; page numeral at77,683 10.5 black; text Huawei Proprietary - Restricted Distribution x125y68310.5; image href=logo.png x1070y663w133h28 preserveAspectRatio=xMidYMid meet.
Icons confirmed: chunk-filled/target-arrow, layers, cube, server, database, chart-bar, chart-line, code-block, file, circle-checkmark, arrows-repeat. Use <use data-icon=...> format. Parent expands/renders.
Write svg/pageN.svg and notes/pageN.json ONLY. JSON shape {page:N,title,speech,transition,glossary:[{term,full,explain}],red_notes:[{quote,explain}],sources:[...]}. Speech350-650Chinese chars with caveats; glossary covers all English on page, units and boilerplate. red_notes explains all meaningful red data/phrases and highlighted chart series. Root merges and generates four-section notes Markdown.
No animations. No data invented. Files unique per page. Do not edit shared sources or blueprint.
Filesystem via node_repl read/write works; exec_command default cannot start but require_escalated works. No browser/PowerPoint from agents. Parent QA.
