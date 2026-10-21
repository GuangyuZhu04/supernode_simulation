"""Author pages 1–2 only; locked inputs, native SVG elements and explicit notes."""
from pathlib import Path
import json
import re

from deck_style import (
    WS, DATA, RED, BLACK, GRAY, LINE, PALE, BG,
    Slide, text, rect, line, arrow, icon, panel, table, group,
)

SRC = Path(DATA["source"])
PUBLIC_CONFIG = SRC / "sources/config.json"
PUBLIC = json.loads(PUBLIC_CONFIG.read_text(encoding="utf8"))["text_config"]
MODEL = DATA["model"]
AUDIT = DATA["architecture"]
FLASH = AUDIT["flash_reference"]
PRO = AUDIT["synthetic_pro"]
REV = AUDIT["source_revision"]
PUBLIC_URL = f"https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/{REV}/config.json"
MODEL_URL = f"https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/{REV}/inference/model.py"
assert PUBLIC["n_routed_experts"] == 384
assert PUBLIC["num_experts_per_tok"] == 6
assert MODEL["n_routed_experts"] == 1792
assert MODEL["n_activated_experts"] == 24


def red(value):
    return {"text": value, "color": RED, "bold": True}


def run(value):
    return (value, RED, True)


def normal(value):
    return (value, BLACK, False)


def sources(*names):
    return [str((SRC / name).resolve()) for name in names] + [PUBLIC_URL, MODEL_URL]


def mechanism_card(x, y, title, detail, ic):
    return (
        rect(x, y, 338, 68, "white", LINE, 3)
        + icon(ic, x + 12, y + 10, 20)
        + text(x + 42, y + 27, title, 15, BLACK, True)
        + text(x + 16, y + 51, detail, 12)
    )


def build_page1():
    backbone = f"{PRO['backbone_params'] / 1e12:.4f}T"
    ngram = f"{PRO['ngram_params'] / 1e12:.4f}T"
    dspark = f"{PRO['dspark_additional_params'] / 1e9:.4f}B"
    prefill = f"{PRO['prefill_bulk_active_params'] / 1e9:.2f}B"
    decode = f"{PRO['decode_active_params'] / 1e9:.2f}B"
    ratio = PRO["backbone_params"] / FLASH["backbone_params"]
    title = f"DeepSeek V4.1-Pro-10T 合成骨干{PRO['backbone_params'] / 1e12:.2f}T，查表与草稿另计"
    slide = Slide(
        1, title,
        "01 模型演进 · 基于 DeepSeek V4.1 Flash 公开机制与本地合成结构，独立审计文本骨干",
        [
            normal("名义10T仅指骨干；"), run(backbone),
            normal("、ngram 与 DSpark 分列，分别约束容量与活跃计算。"),
        ],
        [
            [
                normal("★ 骨干 "), run(backbone), normal("；ngram "),
                run(ngram), normal(" 与 DSpark "), run(dspark), normal(" 另计。"),
            ],
            [
                normal("★ bulk 活跃 "), run(f"{prefill} / {decode}"),
                normal("；有界重放与输出头成本仍全部保留。"),
            ],
        ],
        [
            "边界：Pro-10T 为合成研究结构，非公开 Pro 规格；仅文本骨干，不含视觉及投影器，未验证模型质量。",
            "来源：2026-10-20 架构审计、model.json；公开 Flash 配置锁定 revision 314113f8…，参数为结构计数。",
        ],
    )
    left = panel(50, 172, 370, 366, "保留计算骨架，分开核算规模", "layers")
    left += mechanism_card(
        66, 216, "CED：编码—解码分工",
        f"{MODEL['encoder_layers']}层编码器 → {MODEL['n_layers'] - MODEL['encoder_layers']}层解码器",
        "layers",
    )
    left += arrow(235, 284, 235, 302)
    left += mechanism_card(
        66, 303, "CSA2：跨层共享与稀疏注意力",
        f"{len(MODEL['kv_source_layers'])}个 KV owner · {len(MODEL['index_source_layers'])}个索引 owner",
        "card-stack",
    )
    left += arrow(235, 371, 235, 389)
    left += mechanism_card(
        66, 390, "mHC + Engram：连接与记忆",
        f"{MODEL['hc_mult']}路残差流 · 主机 ngram 表",
        "database",
    )
    left += text(
        66, 487,
        f"保留共享专家{MODEL['n_shared_experts']}、SWA={MODEL['window_size']}、输出groups={MODEL['o_groups']}",
        12,
    )
    left += text(66, 511, "合成层位映射与模型质量尚未验证", 12)
    slide.add("preserved-backbone", left)

    rows = [
        [
            "层数 / 隐藏维",
            f"{PUBLIC['num_hidden_layers']} / {PUBLIC['hidden_size']}",
            f"{MODEL['n_layers']} / {MODEL['dim']}",
        ],
        [
            "路由专家 / 每token选择",
            f"{PUBLIC['n_routed_experts']} / top-{PUBLIC['num_experts_per_tok']}",
            f"{MODEL['n_routed_experts']} / top-{MODEL['n_activated_experts']}",
        ],
        [
            "专家中间维",
            str(PUBLIC["moe_intermediate_size"]),
            str(MODEL["moe_inter_dim"]),
        ],
        [
            "骨干参数",
            f"{FLASH['backbone_params'] / 1e9:.2f}B",
            red(backbone),
        ],
        [
            "ngram 参数（另计）",
            f"{FLASH['ngram_params'] / 1e9:.2f}B",
            red(ngram),
        ],
        [
            "bulk 活跃：prefill / decode",
            f"{FLASH['prefill_bulk_active_params'] / 1e9:.2f}B / {FLASH['decode_active_params'] / 1e9:.2f}B",
            red(f"{prefill} / {decode}"),
        ],
        [
            "DSpark 额外参数（另计）",
            f"{FLASH['dspark_additional_params'] / 1e9:.4f}B",
            red(dspark),
        ],
    ]
    right = panel(440, 172, 790, 366, "公开 Flash 审计 → 本地合成 Pro-10T", "chart-bar")
    right += table(
        456, 216, [252, 240, 266],
        ["比较维度", "公开 Flash 配置 / 审计", "合成 Pro-10T"],
        rows, row_h=34, head_h=28,
    )
    right += text(
        456, 506,
        [
            normal("骨干规模约为公开 Flash 的 "), run(f"{ratio:.2f}倍"),
            normal("；扩展深度、隐藏维与专家池。"),
        ],
        12,
    )
    right += text(456, 526, "bulk 活跃计数不等于激活字节，也不含巨型 lookup 表。", 12)
    slide.add("architecture-parameter-audit", right)

    speech = (
        "这一页先明确后续仿真的模型对象。材料中的DeepSeek V4.1 Pro-10T是本地研究用合成结构，"
        "公开证据来自锁定版本的Flash配置和参考实现，不能把它称为已发布的Pro模型，也没有训练后质量的证明。"
        "结构保留了因果编码器—解码器、第二代压缩稀疏注意力、流形约束超连接和条件记忆机制，"
        f"再按研究约束扩展深度、隐藏维和专家池。公开Flash配置实际是{PUBLIC['num_hidden_layers']}层、隐藏维{PUBLIC['hidden_size']}、"
        f"{PUBLIC['n_routed_experts']}个路由专家、每token选{PUBLIC['num_experts_per_tok']}个专家；"
        f"本地结构为{MODEL['n_layers']}层、隐藏维{MODEL['dim']}、{MODEL['n_routed_experts']}个路由专家、"
        f"每token选{MODEL['n_activated_experts']}个专家，专家中间维由{PUBLIC['moe_intermediate_size']}扩至{MODEL['moe_inter_dim']}。"
        f"编码器和解码器各{MODEL['encoder_layers']}层，跨层共享状态的持有层由公开结构按相对深度映射，这种映射没有做质量验证。\n\n"
        f"参数审计采用逐矩阵结构计数。合成骨干为{backbone}，是公开Flash骨干的{ratio:.2f}倍；"
        f"名义10T只表示骨干规模。两张ngram查表总量{ngram}和三层DSpark草稿额外参数{dspark}均单独列账，"
        "从而避免把主机查表容量或草稿驻留误计入骨干。公开Flash与合成结构均按同一审计定义对照，"
        "表中的公开参数也是配置推导结果，而不是重新下载或测量权重文件。\n\n"
        f"计算侧则用非lookup活跃参数描述主体工作：prefill为{prefill}，decode为{decode}。"
        "这两个数字不是activation内存大小，也不是每阶段全部执行成本；prefill的有限解码器重放与末位置输出头"
        "没有进入bulk活跃锚点，但仍在仿真中真实执行。共享专家、窗口长度、输出分组和四路残差等机制保留，"
        f"其中输出分组为{MODEL['o_groups']}，后续也会约束可兼容的张量并行候选。后续所有时延和容量结论只对应这组明确的合成结构与硬件假设。"
        "本页把公开来源与合成推演并列展示，是为了让评审能够沿参数定义复核容量与工作量，"
        "并为后续比较布局、上下文和批量提供稳定基准，不能据此推断相同任务质量或业务收益。"
    )
    red_notes = [
        ("保留计算骨架，分开核算规模", "计算骨架取自锁定的Flash参考，深度、维度和专家数由本地model.json推演，参数口径以architecture-audit.json审计为准。"),
        ("公开 Flash 审计 → 本地合成 Pro-10T", "左列依据Flash官方配置及本地结构审计，右列是合成研究配置，两者不是官方产品版本等价对照。"),
        (backbone, f"architecture-audit.json逐矩阵计数得到骨干{PRO['backbone_params']:,}个参数，名义10T与实际计数的差异为{(PRO['backbone_params'] / 1e13 - 1) * 100:.3f}%。"),
        (ngram, f"model.json中两张ngram表合计{PRO['ngram_params']:,}个参数，作为主机查表容量单独核算，不计入名义10T。"),
        (dspark, f"architecture-audit.json给出DSpark额外{PRO['dspark_additional_params']:,}个参数，已排除共享主模型embedding与输出头的重复计数。"),
        (f"{prefill} / {decode}", "architecture-audit.json定义的非lookup主体活跃参数分别对应prefill与decode；有限重放和输出头虽然不进入bulk锚点，仍执行并计费。"),
        (f"{ratio:.2f}倍", "由合成骨干参数除以同一审计定义下的Flash骨干参数计算，仅表达参数容量倍率，不表示模型质量或实机速度倍率。"),
    ]
    slide.save(
        speech,
        "接下来把这些保留机制展开成可计算的阶段与资源路径，说明为什么不能只凭活跃参数估算时延。",
        red_notes,
        sources("architecture-audit.json", "model.json", "methodology.md", "sources/config.json", "sources/manifest.json"),
        glossary_extra=[
            {"term": "groups", "full": "Output Projection Groups，输出投影分组", "explain": "保留的输出投影分组数为8，限制本实验兼容的TP配置。"},
            {"term": "bulk", "full": "Bulk Phase，主体批量阶段", "explain": "活跃参数锚点只统计主体路径中的非lookup参数；额外重放和输出头仍计执行成本。"},
            {"term": "lookup", "full": "Table Lookup，查表", "explain": "从ngram表按地址读取行，属于条件记忆容量与访存成本。"},
            {"term": "top-6", "full": "Top-6 Expert Routing，六选专家路由", "explain": "锁定的公开Flash配置每token选择6个路由专家。"},
            {"term": "Pro", "full": "（研究称谓）Pro结构", "explain": "本材料仅指基于Flash机制的合成研究结构，没有公开Pro规格或权重的证据。"},
            {"term": "model.json", "full": "Model Configuration JSON，模型配置文件", "explain": "保存本地合成结构、参数与假设的机器可读文件。"},
            {"term": "revision", "full": "Repository Revision，仓库固定版本", "explain": "公开Flash来源锁定到完整提交标识314113f8a7dadb388215396ba5fbee2c610a04d6。"},
        ],
    )
    return len(re.findall(r"[\u4e00-\u9fff]", speech))


def build_page2():
    enc = MODEL["encoder_layers"]
    dec = MODEL["n_layers"] - enc
    window = MODEL["window_size"]
    kv_owners = len(MODEL["kv_source_layers"])
    index_owners = len(MODEL["index_source_layers"])
    topk = MODEL["index_topk"]
    candidates = MODEL["candidate_topk_blocks"] * MODEL["candidate_block_size"]
    # FP8 lookup has one byte per value plus one scale byte per 32 values.
    host_tb = PRO["ngram_params"] * (1 + 1 / 32) / 1e12
    host_value = f"{host_tb:.4f}"
    title = "CED、CSA2 与 Engram 改写资源路径，阶段成本逐项计入"
    slide = Slide(
        2, title,
        "02 机制落地 · 合成 Pro-10T 的长提示、共享缓存与主机冷查表均进入资源账本",
        [
            normal("CED缩短全量路径，CSA2复用状态，Engram下沉主机；"),
            run("阶段成本逐项计费"), normal("。"),
        ],
        [
            [
                normal("★ 长提示只经过编码器；有界重放、索引与输出头的 "),
                run("成本全部保留"), normal("。"),
            ],
            [
                normal("★ 状态共享与冷查表同时入账；候选提前聚集属于 "),
                run("仿真假设"), normal("。"),
            ],
        ],
        [
            f"边界：{enc}+{dec}层与共享层位属于合成推演；统一FP8，公开原生FP4存储另记；未证明等质量或真机性能。",
            "来源：methodology.md、model.json、Flash inference/model.py；Engram容量含scale，TB为十进制。",
        ],
    )

    ced = panel(50, 172, 380, 366, "CED：长提示与有界重放", "layers")
    ced += text(66, 222, "全量提示词（长度 L）", 12, BLACK, True)
    ced += rect(80, 235, 320, 44, BG, LINE, 3)
    ced += text(240, 263, f"{enc}层编码器处理全量提示词", 15, BLACK, True, "middle")
    ced += arrow(240, 279, 240, 297)
    ced += rect(80, 297, 320, 44, "white", LINE, 3)
    ced += text(240, 325, "编码器终态 → 解码器全局 KV", 15, BLACK, True, "middle")
    ced += arrow(240, 341, 240, 359)
    ced += rect(80, 359, 320, 56, PALE, "#F1B4B7", 3)
    ced += text(
        240, 381, [
            normal("末尾 "), run(f"{window} token"),
            normal(f" × {dec}层解码器重放"),
        ], 15, BLACK, True, "middle",
    )
    ced += text(240, 402, "建立各层 SWA 与首token状态", 12, BLACK, False, "middle")
    ced += arrow(240, 415, 240, 432)
    ced += rect(80, 432, 320, 38, BG, LINE, 3)
    ced += text(240, 456, "输出头 → 首 token", 15, BLACK, True, "middle")
    ced += text(66, 493, "TTFT包含重放、索引与输出头", 12)
    ced += text(66, 515, "不能将旧模型时延简单除二", 12)
    slide.add("ced-prefill-route", ced)

    csa = panel(450, 172, 380, 366, "CSA2：共享状态，逐级稀疏", "card-stack")
    csa += text(466, 222, f"{kv_owners}个 KV owner · {index_owners}个索引 owner", 12, BLACK, True)
    modes = [
        (466, "Full", "生成共享KV", "生成索引"),
        (582, "Reindex", "复用共享KV", "重算索引"),
        (698, "Reuse", "复用共享KV", "复用稀疏位"),
    ]
    for x, label, first, second in modes:
        csa += rect(x, 237, 108, 76, BG, LINE, 3)
        csa += text(x + 54, 259, label, 15, BLACK, True, "middle")
        csa += text(x + 54, 282, first, 12, BLACK, False, "middle")
        csa += text(x + 54, 302, second, 12, BLACK, False, "middle")
    csa += text(640, 337, "各打分tile先跨TP做FP32全归约", 12, BLACK, True, "middle")
    csa += arrow(640, 343, 640, 355)
    csa += rect(466, 355, 348, 50, "white", LINE, 3)
    csa += text(640, 385, f"全历史初级索引 → top-{topk}", 15, BLACK, True, "middle")
    csa += arrow(640, 405, 640, 425)
    csa += rect(466, 426, 348, 47, PALE, "#F1B4B7", 3)
    csa += text(
        640, 446,
        [normal("后续解码器候选上限 "), run(f"{candidates:,}")],
        15, BLACK, True, "middle",
    )
    csa += text(
        640, 465,
        f"{MODEL['candidate_topk_blocks']}块 × {MODEL['candidate_block_size']}位置",
        12, BLACK, False, "middle",
    )
    csa += text(466, 494, "候选提前聚集：本仿真优化假设", 12)
    csa += text(466, 516, "公开参考：先全量打分，再屏蔽", 12)
    slide.add("csa2-sharing-indexing", csa)

    eng = panel(850, 172, 380, 366, "Engram：主机分片，冷查表计费", "database")
    eng += text(1040, 250, host_value, 38, RED, True, "middle")
    eng += text(1040, 275, "TB / DP 副本（FP8 + scale）", 12, BLACK, True, "middle")
    eng += text(866, 302, "两张ngram表，按行分片至所属主机", 12)
    eng += text(866, 324, "选中行返回路径", 12, BLACK, True)
    nodes = [
        (866, "DRAM", "随机查表"),
        (954, "host link", "行传输"),
        (1042, "SU", "跨主机返回"),
        (1130, "TP", "张量分发"),
    ]
    for i, (x, label, detail) in enumerate(nodes):
        eng += rect(x, 338, 76, 59, BG, LINE, 3)
        eng += text(x + 38, 360, label, 12, BLACK, True, "middle")
        eng += text(x + 38, 384, detail, 12, BLACK, False, "middle")
        if i < len(nodes) - 1:
            eng += arrow(x + 76, 368, x + 88, 368)
    eng += text(866, 424, "请求：主机哈希 → H2D → SU → D2H", 12)
    eng += text(866, 447, "返回：行 H2D → SU → 本机 → TP", 12)
    eng += line(866, 465, 1214, 465)
    eng += text(866, 490, "主场景：无缓存命中、无免费预取重叠", 12)
    eng += text(866, 514, "CPU / DRAM / IOPS / host / SU 均计费", 12)
    slide.add("engram-host-route", eng)

    speech = (
        "这一页解释三个机制如何进入仿真，而不是用总参数或活跃参数乘上一个经验系数。"
        f"左侧CED首先让全量提示词只经过{enc}层编码器，然后将编码器最终状态投影成解码器的全局键值缓存。"
        f"它仍要让末尾{window}个token完整重放{dec}层解码器，以建立各层滑动窗口状态并产生首token。"
        "因此首token时间包含这段有界重放、索引和输出头，不能把旧模型的首token时延直接除以二。"
        "启用DSpark时，初始预填充还包含草稿网络末窗口的种子状态，具体成本在后页分开解释。\n\n"
        f"中间CSA2只有{kv_owners}个全局KV持有层和{index_owners}个索引持有层，按Full、Reindex、Reuse三种模式生成或复用状态。"
        "这些位置由公开Flash层位映射到合成结构，缓存共享质量没有实测。初级索引面对全历史打分，"
        f"每个分块都先跨张量并行分片进行FP32全归约，再执行top-{topk}选择，不能让各分片各自选top-k后假装等价。"
        f"后续解码器索引限定在{candidates:,}个候选位置。这里提前聚集候选后再打分是本地仿真的优化执行假设；"
        "锁定的公开参考实现采用全量打分后屏蔽。因此PPT把这种策略直接标为假设，不将其包装成已验证实现性能。\n\n"
        f"右侧Engram的ngram表采用FP8值和每32元素一字节scale，每个数据并行副本共{host_value}TB。"
        "表按行分片放到该副本所属主机，而不是每主机重复一整份。主场景是冷查表，没有缓存命中或免费预取重叠。"
        "请求从原主机哈希出地址，经过主机到设备、域内互联与设备到主机到达持有行的主机，"
        "查到的行再通过主机链路返回并在张量并行组内分发。本地直达不会重复计域内传输。"
        "CPU、主机DRAM带宽与随机访问、主机链路、互联及设备字节均单独入账，图中压缩了路由细节但没有省略成本。"
        "这些机制共同说明，参数规模决定一部分容量，最终时延还取决于阶段顺序、状态驻留和每条数据路径。"
    )
    red_notes = [
        ("CED：长提示与有界重放", "依据methodology.md的CED阶段定义，编码器主体、解码器KV投影、有界重放及输出头依次执行。"),
        ("CSA2：共享状态，逐级稀疏", "公开参考保留跨层共享和三种模式，合成结构映射共享层位；候选提前聚集单列为本地执行假设。"),
        ("Engram：主机分片，冷查表计费", "本实验每DP副本持有一份ngram表并按行分片到所属主机，冷查表路径包含CPU、DRAM、host与SU实际成本。"),
        ("阶段成本逐项计费", "methodology.md按计算、缓存、索引、通信、主机查表和传输显式建模，没有用参数倍率代替新增机制成本。"),
        (f"{window} token", "model.json的window_size为128，CED用该末窗口完整重放32层解码器以建立SWA和首token状态。"),
        (f"{candidates:,}", "model.json设置candidate_topk_blocks=2048、candidate_block_size=8，乘积为16384；提前聚集候选是仿真优化假设。"),
        (host_value, f"由ngram参数{PRO['ngram_params']:,}乘以FP8每元素1字节及1/32 scale开销得到每DP副本{host_tb:.8f}TB，不是每主机一份。"),
        ("成本全部保留", "bulk活跃参数口径没有包括有限重放与末位置输出头，但执行图包含这些阶段，TTFT并非仅编码器主体耗时。"),
        ("仿真假设", "公开PyTorch参考先全量打分再屏蔽，本地采用候选提前聚集；此假设与层位映射、FP8统一精度一起限定结果适用范围。"),
    ]
    slide.save(
        speech,
        "明确资源路径后，再看同一1024卡硬件输入下的容量准入，区分预填充峰值与生成峰值。",
        red_notes,
        sources("methodology.md", "model.json", "sources/inference/model.py", "sources/inference/engram.py", "sources/config.json"),
        glossary_extra=[
            {"term": "L", "full": "Sequence Length，序列长度", "explain": "图中全量输入提示词的token数量。"},
            {"term": "tile", "full": "Tiled Computation Block，分块计算单元", "explain": "索引打分以有限工作区内的分块执行，每块归约及启动均计费。"},
            {"term": "scale", "full": "Quantization Scale，量化缩放因子", "explain": "lookup表按每32个FP8元素保存一个scale字节，计入主机容量。"},
            {"term": "host link", "full": "Host–Device Link，主机设备链路", "explain": "模型中主机与设备之间的双向传输资源，具有独立带宽和启动开销。"},
            {"term": "host", "full": "Host，主机", "explain": "持有部分ngram表并处理哈希、查表与设备传输的CPU主机。"},
            {"term": "methodology.md", "full": "Methodology Document，仿真方法文档", "explain": "给出架构来源、执行阶段、精度、资源路径和不确定性边界。"},
            {"term": "model.json", "full": "Model Configuration JSON，模型配置文件", "explain": "保存本地合成模型的结构、共享层位与候选索引参数。"},
            {"term": "inference/model.py", "full": "Reference Inference Model，参考推理模型实现", "explain": "来自锁定Flash仓库，仅作为公开机制证据，未执行下载的代码。"},
        ],
    )
    return len(re.findall(r"[\u4e00-\u9fff]", speech))


if __name__ == "__main__":
    counts = {"page1_chinese_speech_chars": build_page1(),
              "page2_chinese_speech_chars": build_page2()}
    print(json.dumps(counts, ensure_ascii=False))
