# -*- coding: utf-8 -*-
"""Single source of truth for the paper (paper.pdf + paper.tex)."""

TITLE = "MOSS-Image 后训练研究：奖励驱动的在线微调与少步蒸馏"
TITLE_EN = "Post-Training MOSS-Image: Reward-Driven Online Fine-Tuning and Few-Step Distillation"

AUTHORS = "soyamilk, Cloveriris"
AUTHORS_NOTE = "两位作者对本文工作贡献相同（equal contribution）"
AFFIL = "独立研究（Independent Research）　·　实验日期：2026 年 10–11 月　·　报告版本：2026-11-22"

ABS_CN = (
    "本文报告在开源文生图模型 MOSS-Image-Base 上进行后训练的系统性实验。我们比较了两条路线："
    "奖励驱动的在线微调（DiffusionART）与基于 BA 采样器的少步蒸馏。前者在 4 张 16 GB 显卡上"
    "完成 26 个训练 run，后者完成 2 个。实验揭示了三个可迁移的结论。第一，rollout 的数值精度"
    "是决定性变量：在其余配置基本一致时，采用校准 MXFP8 的 1024 分辨率 run 在固定评估集上取得"
    "融合奖励 +0.266 的提升，而未校准 MXFP8 的同系列 run 下滑 −0.128。第二，奖励融合的方差"
    "主要由单一分量主导：权重仅 0.2 的 HPS 项贡献了约 70% 的融合方差，将其移出融合后，仅 20 次"
    "优化器更新即可获得 +0.203 的稳定提升。第三，少步蒸馏的可行性取决于学生容量：15 M 参数的"
    "轻量学生相对零修正基线的改善仅 0.2%，而 101.8 M 的 MOSS 形状学生在 64% 预算处已达 17.2%，"
    "但单位算力收益远低于奖励后训练。受多项目共享算力所限，全部 200 更新预算的长跑均在 16–46 步"
    "被中断，因此本文给出的是方向性证据而非最终结论。"
)

ABS_EN = (
    "We report a systematic post-training study on the open-source text-to-image model "
    "MOSS-Image-Base. Two routes are compared: reward-driven online fine-tuning (DiffusionART) "
    "and few-step distillation built on the BA sampler. Three transferable findings emerge. "
    "First, rollout numerical precision dominates: with otherwise comparable settings, a 1024px run "
    "using calibrated MXFP8 improves the fused reward by +0.266 on a fixed evaluation set, while an "
    "uncalibrated MXFP8 counterpart degrades by -0.128. Second, fusion variance is dominated by a "
    "single component: an HPS term carrying only 0.2 weight accounts for roughly 70% of the variance; "
    "removing it yields a clean +0.203 gain within 20 optimizer updates. Third, distillation "
    "feasibility is capacity-bound: a 15M lightweight student improves over the zero-correction "
    "baseline by only 0.2%, whereas a 101.8M MOSS-shaped student reaches 17.2% at 64% of budget, "
    "though at a far worse return per GPU-hour than reward fine-tuning. Shared cluster contention "
    "interrupted every 200-update run between step 16 and 46, so we present directional evidence "
    "rather than a settled conclusion. Code, configuration and analysis are released."
)

KEYWORDS = "后训练；扩散模型；审美奖励；少步蒸馏；可复现性"

SECTIONS = [
    ("1 引言", [
        ("p", "对齐文生图模型的审美偏好，通常有两条现实路径：用奖励信号做在线强化式微调，"
              "或把多步采样器蒸馏成少步。前者直接优化人类偏好，代价是每次更新都要在线采样并打分；"
              "后者压缩推理成本，但需要额外训练一个学生网络。二者都昂贵，而在消费级显卡上，"
              "真正决定「能不能做」的往往不是算法，而是显存与卡时。"),
        ("p", "本文的目标不是提出新算法，而是回答一个工程问题：在 4 张 16 GB 显卡上，"
              "把 MOSS-Image-Base 往「更好看」的方向推，哪些设计选择真正重要？我们复现并改造了 "
              "DiffusionART 风格的在线闭环，配合三个公开审美模型构成奖励，并实现了两个规模的 "
              "BA 少步蒸馏学生。所有 run 的配置、日志与指标均已归档，可逐项复核。"),
    ]),
    ("2 方法", [
        ("p", "在线闭环。每个 collection 取 P 个 prompt，每个采样 K 张图（48 步、CFG 3.0、"
              "shift 5.0），由外部奖励服务打分，在组内做 rank-sqrt 归一化得到优势，再在 T 个"
              "时间点上对 LoRA 做梯度更新。标准配置 P24×K8×T4 对应每次 collection 192 条轨迹、"
              "768 个训练点，A/B 各半（FullDiT 与谱采样 1:1）。LoRA 秩 48、alpha 48，仅注入注意力；"
              "AdamW 学习率 1e-4、预热 4 步。"),
        ("p", "奖励融合。三个分量分别归一化后加权：Kawai 分档分类器 0.4、DeepGHS 审美百分位 0.4、"
              "HPSv3++ 0.2，末端以 0.5 为基准做仿射缩放。归一化的 μ/σ 来自固定的 128 张分层校准集，"
              "随 run 冻结。优势估计有三个版本：直接使用融合分、rank-sqrt 组内排序、以及在其上再做"
              "分量重排（rerank）。"),
        ("p", "少步蒸馏。教师为冻结的 BF16 全量 DiT。学生不替代教师，而是作为 BA 采样器每个区间的"
              "速度残差修正器：输入为 latent 与教师速度，输出随带符号步长缩放的速度残差，末层零初始化"
              "以保证 step-0 修正恒为零。我们训练了两个规模：15 M 的轻量 SideNet，与 101.8 M 的 "
              "MOSS 形状学生（11 个 joint block、宽度 768）。DP4 下 4 卡各算一条轨迹，"
              "经主机内存平均梯度后单次更新。"),
    ]),
    ("3 实验设置", [
        ("p", "硬件为 4 × GB203（SM120，16 GB GDDR7）加 1 × GB206（SM120，16 GB），20 核 CPU、"
              "46 GiB 内存，Ubuntu 24.04。训练栈为 JAX 0.10.2 / flax 0.12.7 / optax 0.2.8，"
              "奖励侧为 PyTorch 2.11.0+cu130。数据为自建审美集：2 000 条源记录经长度过滤后，"
              "1 991 条进入训练，另固定 4 条 prompt × 4 个 seed 作为跨 run 可比的评估集，"
              "两侧无 ID 与文本重叠。"),
        ("p", "评估协议。每个检查点（每 2 次更新）用同一套奖励管线对 16 张图打分，报告融合值以及"
              "kawai、deepGHS、HPS 三个分量的均值。文中所有 Δ 均为「末次 − 首次」，只做同 run 内部比较。"),
    ]),
    ("4 结果", [
        ("p", "校准精度。两个 1024 分辨率、同奖励管线、同 LoRA 与优化器配置的 run，唯一显著差异是 "
              "rollout 精度。校准 MXFP8 的 run 融合奖励从 0.334 升至 0.600（+0.266），deepGHS 分量"
              "+0.380；未校准 MXFP8 的 run 从 0.334 跌至 0.205（−0.128），HPS 分量 −5.49。"
              "这一分化与工程侧「训练禁止使用未校准 FP8 rollout」的约定完全一致，可作为该约定的直接证据。"),
        ("fig", "figures/fig1-fused-reward-curves.png",
         "图 1　固定 4 prompt × 4 seed 评估集上融合奖励随优化器更新的变化。蓝线为校准 FP8，"
         "红线为同配置未校准 FP8，二者在第 12 步附近分道扬镳。"),
        ("p", "奖励分量的噪声。完整跑满 20 次更新的 768 分辨率 pilot 中，kawai +0.328、deepGHS +0.036 "
              "均为正，但 HPS 为 −6.82，使融合值反成 −0.038。原因是 HPSv3++ 输出未归一化的原始 μ，"
              "单图动态范围可达 −0.09 至 +1.2；归一化后单点贡献 −4.31，尽管权重只有 0.2，"
              "却贡献了融合值约 0.1 的变化量，约占总方差的七成。改用只含 kawai 与 deepGHS（1:1）的 "
              "rank-sqrt 配置后，20 次更新即得到 +0.203 的融合提升与 +0.666 的 kawai 提升，"
              "信号明显更干净。"),
        ("table", {
            "caption": "表 1　固定评估集上的分量变化（Δ = 末次 − 首次）",
            "header": ["run", "分辨率", "精度", "Δ融合", "Δkawai", "ΔdeepGHS", "ΔHPS"],
            "rows": [
                ["rank-sqrt（kawai+deepGHS）", "1024", "校准", "+0.203", "+0.666", "+0.236", "—"],
                ["rank-sqrt rerank", "1024", "校准", "+0.266", "—", "+0.380", "+3.35"],
                ["rank-sqrt rerank", "1024", "未校准", "-0.128", "—", "-0.005", "-5.49"],
                ["reward3 融合（完整 20 步）", "768", "未校准", "-0.038", "+0.328", "+0.036", "-6.82"],
            ],
        }),
        ("p", "预算与产出的现实。26 个 run 目录中 16 个是正式后训练 run，但只有 3 个达成自身配置的"
              "更新预算；8 个 200 更新预算的长跑最远到 46 步，多数停在 16–22 步。直接原因是计算卡被"
              "另一条预训练线长期占满。按 10–21 min/更新的实测速度，200 次更新需 33–70 小时独占 4 卡，"
              "这构成了本研究的硬约束。"),
        ("fig", "figures/fig3-budget-vs-actual.png",
         "图 2　各正式 run 的更新预算（灰）与实跑步数（蓝）。200 预算一栏的缺口即是本研究的现实边界。"),
        ("p", "少步蒸馏。以 latent MSE 相对零修正基线的改善衡量：15 M 的 SideNet 末段 0.0884 对 "
              "0.0886，改善仅 0.21%，曲线几乎与基线重合，说明 128 维文本压缩加轻量卷积不足以拟合"
              "固定 CFG 后的速度场残差；101.8 M 的 MOSS 形状学生末段 0.0682 对 0.0828，改善 17.2%，"
              "且在停止前仍有下降趋势。该 run 在 1264/1972 处被主动停止。"),
        ("fig", "figures/fig4-distill-mse.png",
         "图 3　少步蒸馏的 latent MSE 滑动平均（窗口 25）。实线为学生修正后，浅色虚线为零修正基线；"
         "两者间距即学生的真实贡献。"),
        ("table", {
            "caption": "表 2　少步蒸馏结果",
            "header": ["学生", "参数量", "更新/计划", "末段 mse / 基线", "改善"],
            "rows": [
                ["SideNet v2", "15 M", "1016 / 1972", "0.0884 / 0.0886", "+0.21%"],
                ["MOSS 形状", "101.8 M", "1264 / 1972", "0.0682 / 0.0828", "+17.22%"],
            ],
        }),
    ]),
    ("5 讨论与局限", [
        ("p", "本研究最重要的局限是预算。所有结论都建立在 20 次更新的 pilot 与不足四分之一预算的长跑上，"
              "16 张图的固定评估在如此小的样本下置信区间很宽，±0.03 量级的差异不宜过度解读。"
              "其次是消融不严格：校准与未校准两组之间，奖励管线版本与参考项系数仍有差异，"
              "因此我们只能断言「高度相关」，而非严格因果。第三，蒸馏缺少终点评估，17.2% 是训练期"
              "MSE 指标，尚未转化为 FID 或人类偏好分数。"),
        ("p", "另一个值得记录的负面结果：早期版本存在奖励 wrapper 动态加载导致的延迟导入缺陷，"
              "以及训练模板会把 rollout 精度退回未校准模式的问题。前者造成过若干次无效重启，"
              "后者直接导致了一次评估倒退的 run。这类「工程性噪声」在文献中很少被报告，"
              "但对小团队的复现成本影响巨大，因此我们在发布包中保留了完整的失败记录。"),
    ]),
    ("6 结论", [
        ("p", "在 4 张 16 GB 显卡上，把 MOSS-Image-Base 的审美表现推高是可行的，但真正的瓶颈依次是："
              "数值精度、奖励设计的信噪比、以及可持续的卡时预算。校准 FP8 与去掉高方差奖励分量，"
              "能让 20 次更新就产生可测量的提升；而要把这种提升确认下来，需要一次约 50 小时的独占训练。"
              "少步蒸馏在 100 M 量级学生上初见成效，但单位算力收益远低于奖励后训练。"
              "我们希望这份包含失败记录的完整数据，能让后来者少走几段弯路。"),
    ]),
]

REFS = [
    "OpenMOSS Team. MOSS-Image Technical Report. arXiv:2610.23374, 2026.",
    "OpenMOSS. MOSS-Image-Base / MOSS-Image-SFT. Hugging Face, Apache-2.0, 2026.",
    "HKUST-LongGroup. BA-solver: reference implementation of the BA sampler. GitHub, 2026.",
    "R. Black et al. DiffusionART: online reward fine-tuning for diffusion models, 2024.",
    "X. Wu et al. HPSv3: towards wide-spectrum human preference scoring. arXiv, 2025.",
    "S. Kirstain et al. Pick-a-Pic: an open dataset of user preferences for text-to-image generation. NeurIPS, 2023.",
    "T. Karras et al. Elucidating the design space of diffusion-based generative models. NeurIPS, 2022.",
    "C. Song et al. Consistency models. ICML, 2023.",
]
