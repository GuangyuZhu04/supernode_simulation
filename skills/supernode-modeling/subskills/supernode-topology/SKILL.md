---
name: supernode-topology
description: Generate supernode scale-up and scale-out topologies from natural-language requirements, including SO domains and directed domain links. Produce a validated topology specification, native supernode_sim hardware configuration, and Mermaid schematic; use for topology design or topology input preparation for resource and traffic simulation.
---

# 自然语言生成超节点拓扑

把用户描述转为结构化 SU/SO 拓扑，调用本 skill 的确定性编译器生成引擎原生硬件配置，并交付可检查的 Mermaid 图。自然语言理解由调用本 skill 的 Codex 完成；编译器仅处理结构化 YAML/JSON。

需要 `supernode_simulation` 仓库及其 Python 环境，或已安装的 `supernode_sim` 包。项目位置优先使用用户给定路径，其次检查 `SUPERNODE_SIM_ROOT`、当前目录及父目录中的 `supernode_sim/` 和 `pyproject.toml`。本 skill 嵌套于仓库时，其四级父目录是项目根目录；被复制到全局 skills 后不能据此定位引擎。命令中的脚本路径应替换为当前 skill 的实际位置。

## 生成与交付

1. 读 [specification.md](references/specification.md)，整理卡数、SU 域大小、主机映射、SU 拓扑、SO 域成员、SO 域内拓扑、域间连接及链路参数。把原话中的“域”落实为 SU 域或 SO 域，保留 `sources` 与 `assumptions`。仅追问影响连接关系或参数含义且无法从已有资料确定的缺项；用户接受的演示参数应标为合成假设。
2. 写出 `topology-spec.yaml`。保存用户已知参数；缺少速率/延迟/效率时明确列出缺项，或使用用户已接受的假设，不把引擎默认值当作真实硬件规格。检查 bps/Bps、单向/双向、单链路/端口聚合口径。
3. 从项目根目录运行编译器。只有拓扑需求时输出硬件片段；用户提供完整场景时用 `--base` 保留模型、负载、设备 profile 与执行策略，随后用引擎 `validate` 检查完整配置。不要为了画拓扑补造模型或负载。
4. 从成功编译的 `hardware.explicit_links` 生成 `topology.mmd`：按 SU/SO 域分组，显示主机映射、网关、交换节点、链路方向、带宽和域间 link ID。图使用原生 Mermaid 文本，可直接内嵌展示；大规模拓扑可折叠重复节点，但注明范围、数量及折叠规则。图是结构示意，连通性以编译校验和实际路径为准。
5. 交付结构化输入、原生输出、图及可重跑命令，说明采用的假设、缺参和模型边界。若继续仿真，把原生输出交回父 skill；没有负载时到拓扑交付为止。

从仓库根目录运行的示例：

```text
python skills/supernode-modeling/subskills/supernode-topology/scripts/build_topology.py --schema
python skills/supernode-modeling/subskills/supernode-topology/scripts/build_topology.py topology-spec.yaml --output outputs/topology.yaml
python skills/supernode-modeling/subskills/supernode-topology/scripts/build_topology.py topology-spec.yaml --base scenario.yaml --output outputs/scenario-topology.yaml
python -m supernode_sim validate outputs/scenario-topology.yaml
```

可直接编译的 8 卡示例见 [two_domains.yaml](examples/two_domains.yaml)，其自然语言描述与适配完整场景的限制见 schema 参考。

## 保持模型语义

- SU 域由连续 NPU 索引和统一 `su_size` 决定；SO 域是这些 SU 域的分区，两者不等同。v1 不表达不等大小 SU 域或任意主机映射；遇到这类需求明确报告限制，不静默改写。
- 编译结果使用 `topology: explicit` 与 `routing_policy: su_so`，同 SU 通信限定在 SU 链路，跨 SU 通信限定在 SO 链路，防止最短跳经过 host/SO 捷径改变语义。
- Clos/Fat-tree 是带共享容量的两层流体抽象，多个 spine 采用确定性最短跳，不能宣称支持生产级交换芯片、ECMP、拥塞控制、报文 buffer 或 P99 建模。
- 生成的显式图依赖卡数、域大小与成员集合；更改卡数或进行 `size-su` 搜索时逐候选重新编译，不能仅修改 `npu_count/su_size` 后沿用旧图。
