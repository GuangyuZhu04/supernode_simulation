"""Draw an exact-value batch comparison from the audited complete summary."""
from pathlib import Path
import json
import math
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/kimi-k3-10t-950dt-dspark'
data = json.loads((OUT / 'summary.json').read_text(encoding='utf-8'))
assert data['aggregate']['complete'], 'Do not present an incomplete matrix as final.'
rows = {r['case']: r for r in data['rows']}
font_file = 'C:/Windows/Fonts/msyh.ttc'
font = lambda size: ImageFont.truetype(font_file, size=size)
image = Image.new('RGB', (1600, 970), '#f3f6fa')
draw = ImageDraw.Draw(image)
ink, muted, blue, teal, red = '#132235', '#65768a', '#7d93ae', '#098e85', '#bf4b57'
draw.rectangle((0, 0, 1600, 170), fill='#102b49')
draw.rectangle((0, 0, 1600, 6), fill='#19b3a7')
draw.text((55, 30), 'Kimi K3 10T · batch 16 / 32 / 64', font=font(38), fill='white')
draw.text((55, 94), '128K 上下文 · TP8 · 输出 8 token · 完整请求的平均 TPOT / ms', font=font(24), fill='#d1dfeb')
draw.text((55, 187), '全局 Batch；权重 / 计算 / 激活 / 通信 / KV 为 FP8，KDA 循环状态为 FP32。', font=font(22), fill=muted)
draw.rectangle((55, 238, 81, 264), fill=blue)
draw.text((94, 234), '关闭 DSpark', font=font(22), fill=ink)
draw.rectangle((300, 238, 326, 264), fill=teal)
draw.text((339, 234), '启用 DSpark（draft 7 / 每轮提交目标 4）', font=font(22), fill=ink)

for cards, ep, left in [(128, 16, 55), (256, 32, 825)]:
    right, top, bottom = left + 715, 365, 755
    draw.rounded_rectangle((left, 292, right, 835), radius=12, fill='white', outline='#dce4ec', width=2)
    draw.text((left + 24, 308), f'{cards} 卡 · TP8 × EP{ep}', font=font(27), fill=ink)
    x0, x1 = left + 70, right - 30
    selected = [rows[f'n{cards}-tp8-ep{ep}-b{b}-128K'] for b in (16, 32, 64)]
    values = [v for r in selected for v in (r['tpot_ms'], r['baseline_tpot_ms']) if v is not None]
    ymax = max(35, math.ceil(max(values, default=30) / 5) * 5 + 5)
    y = lambda value: bottom - value / ymax * (bottom - top)
    for tick in range(0, ymax + 1, 5):
        yy = y(tick)
        draw.line((x0, yy, x1, yy), fill='#e4ebf1', width=1)
        draw.text((x0 - 47, yy - 12), str(tick), font=font(19), fill=muted)
    yy = y(30)
    for x in range(x0, x1, 16):
        draw.line((x, yy, min(x + 8, x1), yy), fill=red, width=2)
    draw.text((x1 - 122, yy - 29), '30 ms 目标', font=font(18), fill=red)
    for i, (batch, row) in enumerate(zip((16, 32, 64), selected)):
        center = x0 + (i + .5) * (x1 - x0) / 3
        pair = [(row['baseline_tpot_ms'], blue, center - 48), (row['tpot_ms'], teal, center + 6)]
        if all(value is None for value, _, _ in pair):
            draw.text((center - 55, bottom - 125), '完整请求', font=font(23), fill=muted)
            draw.text((center - 55, bottom - 88), '容量不足', font=font(23), fill=red)
        else:
            for value, color, x in pair:
                if value is None:
                    draw.text((x - 2, bottom - 45), '不可用', font=font(17), fill=red)
                    continue
                draw.rounded_rectangle((x, y(value), x + 42, bottom), radius=3, fill=color)
                label = f'{value:.2f}'
                box = draw.textbbox((0, 0), label, font=font(20))
                draw.text((x + 21 - (box[2] - box[0]) / 2, y(value) - 31), label, font=font(20), fill=ink)
        draw.text((center - 31, bottom + 19), f'B={batch}', font=font(22), fill=ink)
        if row['dspark_speedup']:
            draw.text((center - 43, bottom + 51), f"{row['dspark_speedup']:.2f}×", font=font(20), fill=teal)
    draw.line((x0, bottom, x1, bottom), fill='#a3b4c4', width=2)

draw.text((55, 863), '容量失败不使用独立 warm Decode 补值。128K = 131,072 token；加速比为同配置关闭 / 开启 DSpark。', font=font(20), fill=muted)
draw.text((55, 907), '数据来自已审计仿真；草稿资源和接受长度为显式假设，非实机测量或模型质量验证。', font=font(20), fill=muted)
image.save(OUT / 'batch_16_32_64_comparison.png')
print(OUT / 'batch_16_32_64_comparison.png')
