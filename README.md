# MOSS-Image Post-Training Study

**奖励驱动的在线后训练 与 少步蒸馏** —— 实验报告、科研图表、可复现管线与论文

作者：**Cloveriris**　｜　论文共同署名：**soyamilk & Cloveriris**　｜　版本 **v2（2026-11-22）**

---

## 这是什么

本仓库记录了在开源文生图模型 **MOSS-Image-Base**（Apache-2.0）上开展的一轮后训练研究，
包含两条路线、26 个训练 run、5 张科研图表、一份完整实验报告、一篇论文，以及一套
**可在净室从零复现**的文档与代码。

我们没有把它包装成"成功案例"。相反，报告如实记录了三次跑满、十三次中断、
以及一次因未校准 FP8 而导致评估倒退的失败——这些恰好是最有价值的部分。

## 三个可以带走的结论

| # | 结论 | 证据 |
|---|---|---|
| 1 | **校准 FP8 是分水岭**：同配置下校准 rollout 融合奖励 **+0.266**，未校准 **−0.128** | 报告 §7.2 |
| 2 | **HPS 分量是主要噪声源**：权重仅 0.2，却贡献约 70% 的融合方差；去掉后 20 步即得 **+0.203** | 报告 §7.3 |
| 3 | **少步蒸馏吃容量**：15 M 学生几乎学不动（+0.2%），101.8 M 学生达 **+17.2%**，但单位算力收益远低于奖励后训练 | 报告 §7.5 |

> 一句话总结：**数据是完整的，但没有一次跑满预算的实验**。方向可行，规模待验。

## 目录

```
.
├── README.md                  ← 你在这里
├── INDEX.md                   ← 全部文件的清单与说明
├── report/                    ← 实验报告（Markdown + Word）
├── paper/                     ← 论文（LaTeX 源 + PDF）
├── docs/
│   ├── 00-reproduction.md     ← 净室从 0 复现：硬件、环境、数据、权重、管线、验收
│   ├── 01-datasets.md         ← 数据来源、构成、授权
│   ├── 02-models-and-rewards.md
│   ├── 03-config-reference.md ← 配置项字典
│   └── 04-sanitization.md     ← 脱敏规则与占位符
├── figures/                   ← 5 张图（PNG，可直接引用）
├── analysis/                  ← 结构化指标：run 表、评估分、蒸馏曲线
├── code/                      ← 训练与奖励管线代码（脱敏快照）
└── tools/                     ← 图表生成、脱敏、文档构建脚本
```

## 建议阅读顺序

1. **[paper/paper.pdf](paper/paper.pdf)** —— 10 分钟了解全貌（3 页）
2. **[report/report.md](report/report.md)** —— 完整实验复盘与全部数字
3. **[docs/00-reproduction.md](docs/00-reproduction.md)** —— 想复现时从这里开始
4. **[INDEX.md](INDEX.md)** —— 想找某个具体产物时查这里

## 快速开始（复现者）

```bash
# 1. 环境（两个隔离 env，版本已锁定）
#    详见 docs/00-reproduction.md §3
conda create -n jax_env python=3.12 -y && conda activate jax_env
pip install "jax[cuda12]==0.10.2" "flax==0.12.7" "optax==0.2.8"

# 2. 权重（固定 revision！）
huggingface-cli download OpenMOSS-Team/MOSS-Image-Base --revision <commit> \
  --local-dir $MODEL_ROOT/MOSS-Image-Base

# 3. 数据（自采或用你自己的审美数据集，1 900+ 条）
python scripts/prepare_dataset.py --config configs/dataset/top2000v2.json

# 4. 奖励服务（三项 ready + 校准文件冻结）
python -c "from reward_pipeline import health; print(health())"

# 5. 训练（先跑 20 更新的 pilot，再上 200）
bash scripts/run.sh train --config configs/my_run.json
```

**开训前请务必确认 `sampling.rollout_mode == "mxfp8_calibrated"`**——这是本实验
付出一次失败 run 换来的教训。

## 硬件要求

| 档位 | 配置 | 能做到 |
|---|---|---|
| 最小 | 2 × 16 GB（SM90/SM120） | 768 分辨率、20 更新 |
| 推荐 | 4 × 16 GB + 1 × ≥12 GB、≥48 GB 内存 | 1024 分辨率、200 更新 |

存储：单 run 5–12 GB，200 更新建议 ≥300 GB 独立数据盘。
详见 `docs/00-reproduction.md` §1。

## 许可与数据

| 资产 | 许可 | 是否随包分发 |
|---|---|---|
| 报告、文档、图表 | CC BY 4.0 | ✅ |
| `code/` | Apache-2.0（含上游 NOTICE） | ✅ |
| MOSS-Image 权重 | Apache-2.0 | ❌ 请自行下载 |
| 奖励模型权重 | 各自上游许可 | ❌ |
| 训练数据集 | 第三方版权，不可再分发 | ❌ |

本包已脱敏：主机名、用户名、绝对路径、隧道地址均已替换，训练卡以
**架构代号（SM/GB）+ 显存**表示。规则见 `docs/04-sanitization.md`。

## 引用

```bibtex
@techreport{cloveriris2026mossposttrain,
  title  = {Post-Training MOSS-Image: Reward-Driven Online Fine-Tuning
            and Few-Step Distillation},
  author = {soyamilk and Cloveriris},
  year   = {2026},
  month  = {11},
  note   = {Experiment report and reproducibility package},
  url    = {https://github.com/CloverIris/moss-image-posttrain}
}
```

上游工作：

```bibtex
@article{mossimage2026,
  title  = {MOSS-Image Technical Report},
  author = {{OpenMOSS Team}},
  year   = {2026},
  note   = {arXiv:2610.23374}
}
```

## 致谢

- **OpenMOSS** 开源 MOSS-Image 基座与 SFT 权重；
- **HKUST-LongGroup** 提供 BA-solver 参考实现与论文；
- 审美奖励模型的作者与维护者；
- 以及在四张 16 GB 卡上坚持了 46 次更新才被掐断的那次训练。

## 反馈

欢迎通过 Issue 或邮件指出错误。报告中所有数字都可在 `analysis/` 中复核；
若发现图表与 CSV 不一致，以 CSV 为准，并请告知我们修正。

---

*Keep the numbers honest, keep the pipeline runnable.*
