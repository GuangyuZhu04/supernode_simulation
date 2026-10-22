"""Draw SP8/TP8 versus SP16/TP16 from complete, audited simulation results."""
from pathlib import Path
import json
import math
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/kimi-k3-10t-950dt-dspark'
data = json.loads((OUT / 'summary.json').read_text(encoding='utf-8'))
assert data['aggregate']['complete']
pairs = [r for r in data['sp_comparisons'] if r['group'] == 'main' and r['variant'] == 'dspark'
         and r['context'] == '128K' and r['batch'] in (16, 32, 64) and r['cards'] in (128, 256)]
assert len(pairs) == 6
font = lambda size: ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', size=size)
im = Image.new('RGB', (1600, 980), '#f3f6fa')
d = ImageDraw.Draw(im)
ink, muted, teal, orange = '#132235', '#65768a', '#098e85', '#d58528'
d.rectangle((0, 0, 1600, 172), fill='#102b49')
d.rectangle((0, 0, 1600, 6), fill=orange)
d.text((55, 30), 'Kimi K3 10T · SP8 / SP16 并行策略对照', font=font(36), fill='white')
d.text((55, 94), '相同硬件与负载 · 128K · DSpark 开启 · 完整请求 TPOT / ms', font=font(24), fill='#d1dfeb')
d.text((55, 188), 'SP与TP共用rank；EP = 总卡数 / TP。FP8组合、物理拓扑、输出8 token及草稿假设沿用。', font=font(21), fill=muted)
for x, color, label in [(55, teal, 'SP8 / TP8'), (360, orange, 'SP16 / TP16')]:
    d.rectangle((x, 240, x+26, 266), fill=color)
    d.text((x+40, 236), label, font=font(22), fill=ink)
values = [v for r in pairs for v in (r['sp8_tpot_ms'], r['sp16_tpot_ms']) if v is not None]
ymax = max(12, math.ceil(max(values) / 2) * 2 + 2)
for cards, left in [(128, 55), (256, 825)]:
    right, top, bottom = left+715, 370, 755
    x0, x1 = left+65, right-30
    d.rounded_rectangle((left, 293, right, 838), radius=12, fill='white', outline='#dce4ec', width=2)
    d.text((left+24, 309), f'{cards}卡 · EP{cards//8} → EP{cards//16}', font=font(27), fill=ink)
    y = lambda v: bottom - v/ymax*(bottom-top)
    for tick in range(0, ymax+1, 2):
        d.line((x0, y(tick), x1, y(tick)), fill='#e4ebf1', width=1)
        d.text((x0-42, y(tick)-13), str(tick), font=font(19), fill=muted)
    for i, batch in enumerate((16, 32, 64)):
        r = next(r for r in pairs if r['cards'] == cards and r['batch'] == batch)
        center = x0+(i+.5)*(x1-x0)/3
        bars = [(r['sp8_tpot_ms'], teal, center-49), (r['sp16_tpot_ms'], orange, center+7)]
        if all(v is None for v, _, _ in bars):
            d.text((center-56, bottom-123), '完整请求', font=font(23), fill=muted)
            d.text((center-56, bottom-87), '容量不足', font=font(23), fill='#bf4b57')
        else:
            for v, color, x in bars:
                if v is None:
                    d.text((x-4, bottom-42), '不可用', font=font(18), fill='#bf4b57')
                    continue
                d.rounded_rectangle((x, y(v), x+42, bottom), radius=3, fill=color)
                label = f'{v:.2f}'
                bounds = d.textbbox((0, 0), label, font=font(20))
                d.text((x+21-(bounds[2]-bounds[0])/2, y(v)-31), label, font=font(20), fill=ink)
        d.text((center-32, bottom+19), f'B={batch}', font=font(22), fill=ink)
        speedup = r['sp16_decode_speedup']
        if speedup is not None:
            d.text((center-47, bottom+51), f'{speedup:.2f}×', font=font(20), fill=orange)
    d.line((x0, bottom, x1, bottom), fill='#a3b4c4', width=2)
d.text((55, 864), '倍数 = SP8 TPOT / SP16 TPOT；>1 表示SP16更快。容量失败不以独立warm Decode替代。', font=font(21), fill=muted)
d.text((55, 909), '草稿7、每轮提交目标4为显式假设。更换EP后按相同公式重算局部草稿成本；结果未实机校准。', font=font(21), fill=muted)
im.save(OUT / 'sp8_sp16_comparison.png')
print(OUT / 'sp8_sp16_comparison.png')
