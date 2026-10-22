"""Render standalone precision and SP figures directly from audited results."""
from pathlib import Path
import argparse
import json
import math
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'outputs/kimi-k3-10t-950dt-dspark-fp4'
INK, MUTED, FP8, FP4 = '#132235', '#65768a', '#899db1', '#098e85'
BG, GRID, ORANGE = '#f3f6fa', '#e4ebf1', '#d58528'


def font(size):
    return ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', size=size)


def draw_panel(draw, box, title, rows, keys, colors, ymax):
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=12, fill='white', outline='#dce4ec', width=2)
    draw.text((left + 23, top + 17), title, font=font(25), fill=INK)
    x0, x1, y0, y1 = left + 56, right - 24, top + 87, bottom - 81
    y = lambda value: y1 - value / ymax * (y1 - y0)
    tick_step = max(2, math.ceil(ymax / 6 / 2) * 2)
    for tick in range(0, ymax + 1, tick_step):
        draw.line((x0, y(tick), x1, y(tick)), fill=GRID)
        draw.text((left + 13, y(tick) - 12), str(tick), font=font(18), fill=MUTED)
    for i, row in enumerate(rows):
        center = x0 + (i + 0.5) * (x1 - x0) / len(rows)
        for j, key in enumerate(keys):
            value = row[key]
            x = center - 50 + 64 * j
            if value is None:
                draw.text((x - 2, y1 - 29), 'OOM', font=font(17), fill='#b94b59')
                continue
            draw.rounded_rectangle((x, y(value), x + 37, y1), radius=3, fill=colors[j])
            text = f'{value:.2f}'
            width = draw.textbbox((0, 0), text, font=font(18))[2]
            draw.text((x + 18 - width / 2, y(value) - 27), text, font=font(18), fill=INK)
        draw.text((center - 32, y1 + 12), f"B={row['batch']}", font=font(21), fill=INK)
        a, b = row[keys[0]], row[keys[1]]
        if a is not None and b is not None:
            text = f'{a/b:.2f}×'
            width = draw.textbbox((0, 0), text, font=font(18))[2]
            draw.text((center - width / 2, y1 + 43), text, font=font(18), fill=colors[1])
    draw.line((x0, y1, x1, y1), fill='#a3b4c4', width=2)


def render(directory):
    payload = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
    assert payload['aggregate']['complete']
    comparison = payload['precision_comparison']
    rows = [r for r in comparison['rows'] if r['group'] == 'main' and r['variant'] == 'dspark'
        and r['context'] == '128K' and r['tp'] in (8, 16) and r['global_batch'] in (16, 32, 64)]
    assert len(rows) == 18
    im = Image.new('RGB', (1800, 1370), BG)
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, 1800, 170), fill='#102b49')
    d.rectangle((0, 0, 1800, 6), fill=FP4)
    d.text((45, 28), 'Kimi K3 10T · 路由专家 FP4 / 其余权重 FP8', font=font(36), fill='white')
    d.text((45, 94), '相同硬件、拓扑与 DSpark 假设 · 上下文128K · 输出8 · 完整请求 TPOT / ms', font=font(24), fill='#d1dfeb')
    for x, color, label in ((45, FP8, '原全FP8'), (300, FP4, '路由FP4 / 其他FP8')):
        d.rectangle((x, 198, x + 24, 222), fill=color)
        d.text((x + 37, 192), label, font=font(23), fill=INK)
    d.text((775, 195), '倍率 = FP8 / FP4；>1 表示 FP4 更快', font=font(22), fill=MUTED)
    ymax = math.ceil(max(r[key] for r in rows for key in ('fp8_tpot_ms', 'mixed_tpot_ms')
        if r[key] is not None) / 5) * 5 + 5
    for row_index, tp in enumerate((8, 16)):
        for col, cards in enumerate((64, 128, 256)):
            values = [{**r, 'batch': r['global_batch']} for r in rows if r['tp'] == tp and r['cards'] == cards]
            values.sort(key=lambda r: r['batch'])
            left, top = 45 + col * 580, 249 + row_index * 454
            draw_panel(d, (left, top, left + 550, top + 428),
                f'{cards}卡 · SP{tp}/TP{tp} × EP{cards//tp}', values,
                ('fp8_tpot_ms', 'mixed_tpot_ms'), (FP8, FP4), ymax)
    q, stats = comparison['precision_assumptions'], comparison['aggregate']
    d.text((45, 1185), f"逻辑权重本体：{q['fp8_logical_weight_bytes_excluding_scales']/1e12:.3f} → {q['mixed_logical_weight_bytes_excluding_scales']/1e12:.3f} TB；量化元数据保持不变。", font=font(24), fill=INK)
    d.text((45, 1233), '显式计入 FP4→FP8 转换、临时缓冲和访存；没有融合解量化或 FP4 算力翻倍假设。', font=font(22), fill=MUTED)
    d.text((45, 1277), 'OOM 表示完整请求容量失败，不以 warm decode 替代。DSpark草稿7/每轮提交目标4；结果未实机校准。', font=font(22), fill=MUTED)
    im.save(directory / 'precision_comparison.png')

    pairs = [r for r in payload['sp_comparisons'] if r['group'] == 'main' and r['variant'] == 'dspark'
        and r['context'] == '128K' and r['batch'] in (16, 32, 64)]
    assert len(pairs) == 9
    im = Image.new('RGB', (1800, 860), BG)
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, 1800, 169), fill='#102b49')
    d.rectangle((0, 0, 1800, 6), fill=ORANGE)
    d.text((45, 28), 'Kimi K3 10T · 混合 FP4 的 SP8 / SP16 对照', font=font(36), fill='white')
    d.text((45, 94), '路由专家FP4，其余权重FP8 · 128K · DSpark开启 · 完整请求 TPOT / ms', font=font(24), fill='#d1dfeb')
    for x, color, label in ((45, FP4, 'SP8 / TP8'), (360, ORANGE, 'SP16 / TP16')):
        d.rectangle((x, 198, x + 24, 222), fill=color)
        d.text((x + 37, 192), label, font=font(23), fill=INK)
    ymax = math.ceil(max(r[key] for r in pairs for key in ('sp8_tpot_ms', 'sp16_tpot_ms')
        if r[key] is not None) / 5) * 5 + 5
    for col, cards in enumerate((64, 128, 256)):
        values = sorted((r for r in pairs if r['cards'] == cards), key=lambda r: r['batch'])
        left = 45 + col * 580
        draw_panel(d, (left, 249, left + 550, 683), f'{cards}卡 · EP{cards//8} → EP{cards//16}',
                   values, ('sp8_tpot_ms', 'sp16_tpot_ms'), (FP4, ORANGE), ymax)
    d.text((45, 722), '倍率 = SP8 / SP16；>1 表示SP16更快。SP与TP共用rank，卡数=TP×EP；物理拓扑不变。', font=font(22), fill=MUTED)
    d.text((45, 767), '草稿预算按相同公式随EP重算；OOM不以warm decode替代。未实机校准，未验证量化质量。', font=font(22), fill=MUTED)
    im.save(directory / 'sp8_sp16_comparison.png')
    print({'precision_figure': str(directory / 'precision_comparison.png'), 'representative_pairs': len(rows)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    render(parser.parse_args().output)
