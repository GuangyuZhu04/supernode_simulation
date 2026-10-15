# 控制仿真程序的内存与磁盘占用

本页控制运行 Python 仿真器的主机资源；不改变所模拟设备的 HBM/DDR、精度、并行布局、负载或调度策略。不能为了降低主机开销而缩小用户场景、跳过失败案例或把独立 decode 当作端到端成功。

## 执行方式

- 结构化推理（`model.architecture`，DP=PP=1）只需要性能、流量和设备容量峰值时，使用 `supernode_sim.streaming.run_streaming`。它逐组件执行并释放事件图；支持 SP。返回数值 JSON，不提供普通报告的完整组件/张量生命周期。阶段解释见 [inference.md](inference.md#大规模结构化推理)。
- 训练、其他并行布局、完整组件/张量审计使用普通 `estimate` / `simulate`。默认大型导出精简事件与临时缓冲记录，完整汇总仍保留。
- 需要中断后恢复时使用 `scripts/run_reference_experiment.py`。检查点需要额外磁盘；普通 CLI 不额外保存检查点。先检查已有结果和指纹，避免重复运行同一场景。

流式 API 示例（从项目根目录、项目解释器运行）：

```python
import json
from pathlib import Path
from supernode_sim.cache import BoundedCache
from supernode_sim.config import load_config
from supernode_sim.streaming import run_streaming

config = load_config('path/to/scenario.yaml')
cache = BoundedCache(max_bytes=64 * 1024 * 1024, max_entries=256)
result = run_streaming(config, cache=cache)
output = Path('outputs/my-streaming-case')
output.mkdir(parents=True, exist_ok=True)
snapshot = config.model_dump()
snapshot['calibration'] = None  # profile 已解析进快照
for name, value in [('config.resolved.json', snapshot), ('result.json', result)]:
    with (output / name).open('w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
```

## 缓存和批量实验

流式时序缓存默认最多 64 MiB 的 Python 对象计量、256 项；淘汰只触发重新计算，不放宽显存准入。跨场景共享时使用 `BoundedCache`；显式传入普通 `dict` 仍沿用调用者管理的无界行为。预算为 0 时关闭时序缓存。

两个 Kimi 批处理脚本支持 `--cache-mib`，例如：

```text
python scripts/run_kimi_10t_inference.py --run --cache-mib 32
python scripts/run_kimi_sp_inference.py --run --cache-mib 32
```

这些命令要求对应目录已有 manifest、配置和硬件 profile。缓存预算按每个进程计，不是进程 RSS 上限；事件、拓扑、结果和 Python 运行时另占内存。默认保持串行，增加 shard 进程数前按各进程峰值评估总内存。

静态拓扑的 pair 路由缓存最多 16 MiB / 65,536 项，各路由模式的源树缓存最多 16 MiB / 128 项。内置 `full_mesh` 会惰性增加 SU 边，其 pair 缓存保留历史语义，暂不限制；源树仍有界。拓扑本身的边和资源不受这些缓存预算约束。

## 文件大小与恢复

大型默认导出中，超过 50,000 行的保留张量表写入 `tensors.csv.gz`，保留 CSV 的列与全部导出行；实际文件名、导出和省略行数见 `result.json` 的 `artifact_export` 和报告开头。小表仍是 `tensors.csv`。按行读取压缩表：

```python
import csv, gzip
with gzip.open('outputs/my-case/tensors.csv.gz', 'rt', encoding='utf-8-sig', newline='') as handle:
    for row in csv.DictReader(handle):
        # 在此筛选或累计，不把整个文件转成 list。
        pass
```

`--full-detail` 明确请求完整逐事件和张量记录，仍输出普通 `tensors.csv`，会明显增加内存和磁盘需求。大型 JSON 使用紧凑编码，批处理 JSON 逐块写入，不先创建整份字符串。

参考实验检查点默认 gzip level 6，可用 `--checkpoint-compression 1` 换取更快写入；其 `--full-detail` 使用独立的 `-full` 检查点。恢复支持流式读取压缩日志，代码/配置指纹不匹配时重新建检查点，不能强行复用。修改引擎后旧实验结果仍可阅读，但不属于当前代码的已验证缓存。

本次优化只影响新运行和导出，不自动清理历史输出。重新导出优先使用新目录，避免把旧 `tensors.csv` 与新 `tensors.csv.gz` 混为同一次结果。需要清理时先列出大小、用途和可重建性，保留配置、来源、摘要及仍需恢复的检查点，按用户指定范围处理。
