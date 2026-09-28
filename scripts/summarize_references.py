"""Build a small index from completed reference reports, without rerunning models."""
from __future__ import annotations

import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    cases = [
        ('FP8 推理', 'kimi-k3-ascend950-fp8'),
        ('FP16 推理', 'kimi-k3-ascend950-fp16'),
        ('MXFP4 routed / BF16 其余权重推理', 'kimi-k3-ascend950-mxfp4'),
        ('64 NPU · Muon optimizer offload', 'kimi-k3-ascend950-training-muon'),
        ('128 NPU · FSDP2 + Muon offload', 'kimi-k3-ascend950-training-muon-128'),
    ]
    lines = ['# Kimi K3 × Ascend 950DT：参考场景对照', '',
        '模型采用官方公开文本骨干结构，硬件采用 950DT 公开名义规格。CPU/DDR/H2D、互连方向拆分、利用率及工作负载为显式假设，结果未经整机实测校准。', '',
        '每张 NPU 配置 144 GB HBM、4 TB/s 名义内存带宽；64 NPU 是参考柜切片，不是 Atlas 950 最大域规模。', '',
        '| 场景 | NPU | 最大 HBM 峰值或初始需求 / 卡 GB | 最大 DDR 峰值或初始需求 / 主机 GB | 结果 |',
        '|---|---:|---:|---:|---|']
    details = []
    for label, case in cases:
        directory = root / 'outputs' / case
        required = ('report.md', 'result.json', 'component_memory.csv', 'state_memory.csv', 'model_parameters.csv', 'tensors.csv')
        if not all((directory / name).exists() for name in required):
            lines.append(f'| {label} | — | — | — | 尚未生成完整报告 |')
            continue
        result = json.loads((directory / 'result.json').read_text(encoding='utf-8'))
        pools = result['memory_breakdown']['pools']
        hbm = max((r['peak_bytes'] for r in pools if r['pool_kind'] == 'device_hbm'), default=0) / 1e9
        dram = max((r['peak_bytes'] for r in pools if r['pool_kind'] == 'host_dram'), default=0) / 1e9
        count = result['config']['hardware']['npu_count']
        status = {'feasible': '在当前模型下可行', 'infeasible': '容量/调度不可行', 'slo_failed': 'SLO 不满足'}.get(result['status'], result['status'])
        lines.append(f'| [{label}]({case}/report.md) | {count} | {hbm:,.3f} | {dram:,.3f} | {status} |')
        details.append(f'- **{label}**：[{case}/component_memory.csv]({case}/component_memory.csv)；[参数形状]({case}/model_parameters.csv)；[JSON]({case}/result.json)。')
    lines += ['', '初始状态已超过容量的案例不能执行，其表中数值是必需驻留量，不是成功运行的实测峰值。不同设备/内存类型的容量不能互相替代。', '',
        '## 明细内容', '',
        '每个报告按 WordEmbedding、PreNorm、PostNorm、MLA、KDA、Dense、MoE、Output 和 AttnRes 分解：权重、激活、梯度、主权重、Adam 动量/方差、Muon 动量、KV、循环状态、量化元数据及缓冲。', '',
        '同一时刻的组成可以相加；各组件独立峰值不能相加。推理没有训练梯度和优化器状态，对应明细在训练报告中。', '', *details, '',
        '## 适用范围', '',
        '- 文本骨干：2,779,484,476,000 参数，每 token 活跃 104,189,619,808 参数；视觉编码器/projector 与训练 MTP 不在本场景内。',
        '- MLA 采用压缩 latent 缓存假设；`architecture.mla.cache_layout: expanded` 可切换官方 HF 参考 K/V 布局。KDA 循环矩阵与卷积历史单独计算。',
        '- FP8/FP16 是重新设定精度的设计对照；官方混合权重案例单列 routed expert MXFP4 与其余 BF16。',
        '- 官方 K3 的 MoonEP、Pipeline ZeRO-2、远程 activation offload、分布式 Muon 正交化等未完整复现。Muon 状态容量可核算，训练耗时是代理模型结果。',
        '- [公开资料和本地快照](../docs/sources/README.md)；[使用说明](../README.md)；[Skill](../skills/supernode-modeling/SKILL.md)。', '']
    output = root / 'outputs/kimi-k3-ascend950-comparison.md'
    output.parent.mkdir(exist_ok=True)
    output.write_text('\n'.join(lines), encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
