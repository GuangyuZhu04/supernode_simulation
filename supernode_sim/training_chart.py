"""Standalone, dependency-free 100% stacked bars for training reports."""
from math import fsum
from xml.etree import ElementTree as ET


LABELS = {
    'compute_only': '仅计算', 'compute_and_transfer': '计算与通信重叠', 'exposed_transfer': '暴露通信',
    'other_local': '其他本地活动', 'idle': '空泡', 'compute_s': '计算 / 本地访存',
    'communication_node_s': '节点内通信', 'communication_scaleup_s': 'Scale-up 通信',
    'communication_scaleout_s': 'Scale-out 通信', 'communication_other_s': '其他通信',
    'synchronization_wait_s': '同步等待', 'scheduling_wait_s': '调度等待',
    'memory_wait_s': '内存准入等待', 'resource_contention_s': '资源争用', 'other_s': '其他时延',
    'weights': '权重', 'gradients': '梯度', 'activations': '激活', 'buffers': '缓冲区',
    'optimizer_momentum': '优化器动量', 'optimizer_variance': '优化器方差',
    'gathered_weight': '聚合后权重', 'master_weights': '主权重', 'momentum': '动量',
    'variance': '二阶矩', 'unsharded_gradient_buffer': '未分片梯度缓冲', 'activation': '激活',
    'before_first_activity': '首次活动前', 'after_last_activity': '末次活动后',
    'between_activities': '活动间隙', 'inactive_rank': '全程空闲卡',
}
STATE_COLORS = {'compute_only': '#176b81', 'compute_and_transfer': '#42a99a',
                'exposed_transfer': '#bd5b29', 'other_local': '#a98ac2', 'idle': '#94a3b8'}
PALETTE = ('#176b81', '#c07732', '#678a51', '#77649d', '#418dab', '#ab556a', '#96a5b2')


def training_chart(summary):
    root = ET.Element('svg', {'xmlns': 'http://www.w3.org/2000/svg', 'width': '1000',
                              'font-family': 'Segoe UI, Microsoft YaHei, sans-serif', 'role': 'img',
                              'aria-labelledby': 'chart-title'})
    ET.SubElement(root, 'title', id='chart-title').text = '训练仿真 · 分段百分比'
    ET.SubElement(root, 'rect', width='100%', height='100%', fill='#ffffff')

    def text(x, y, value, size=15, **attrs):
        ET.SubElement(root, 'text', {'x': str(x), 'y': str(y), 'font-size': str(size),
                                     'fill': '#263c49', **attrs}).text = value

    def rect(x, y, width, height, color):
        return ET.SubElement(root, 'rect', x=str(x), y=str(y), width=str(width), height=str(height), fill=color)

    def amount(value, unit):
        if unit == 'byte':
            return f'{value / 2**20:,.2f} MiB'
        return f'{value * 1000:,.3f} ' + ('ms' if unit == 's' else 'NPU·ms')

    text(36, 45, '训练仿真 · 占比总览', 28, **{'font-weight': 'bold'})
    peak = summary['peak_hbm']
    text(36, 76, f"{summary['model']}   ·   {summary['npu_count']} NPU   ·   PP {summary['pp']}")
    text(36, 102, f"峰值卡 {peak['pool']}   ·   物理容量 {amount(peak['physical_bytes'], 'byte')}   ·   "
         f"峰值预留 {amount(peak['reserved_bytes'], 'byte')}")
    y = 146
    for group in summary['groups'].values():
        text(36, y, group['title'], 19, **{'font-weight': 'bold'})
        text(964, y, amount(group['total'], group['unit']), **{'text-anchor': 'end'})
        rows = [s for s in group['segments'] if s['value'] > 0]
        if len(rows) > 7:
            rows = [*rows[:6], {'name': f'其余 {len(rows)-6} 项',
                                'percent': fsum(r['percent'] for r in rows[6:])}]
        x = 36
        rect(x, y+13, 928, 22, '#edf1f4')
        for i, row in enumerate(rows):
            label = LABELS.get(row['name'], row['name'])
            pct = row['percent']
            caption = f'{label} · {pct:.2f}%' if pct >= .01 else f'{label} · <0.01%'
            color = STATE_COLORS.get(row['name'], PALETTE[i])
            width = 9.28 * pct
            ET.SubElement(rect(x, y+13, width, 22, color), 'title').text = caption
            x += width
            lx, ly = 36 + i % 2 * 466, y + 61 + i // 2 * 24
            rect(lx, ly-11, 12, 12, color)
            if len(label) > 43:
                caption = caption.replace(label, label[:40] + '…')
            text(lx+20, ly, caption)
        if not rows:
            text(36, y+61, '无该类活动，占比不适用')
        y += 87 + max(1, (len(rows)+1)//2) * 24
    root.set('height', str(y))
    root.set('viewBox', f'0 0 1000 {y}')
    return ET.tostring(root, encoding='unicode')
