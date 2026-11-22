# 索引

本文件列出发布包的每个文件及其用途。所有数字可在 `analysis/` 中复核；
论文数字与报告数字来自同一份 CSV。

---

## 1. 顶层

| 文件 | 说明 |
|---|---|
| `README.md` | 项目总览、结论速览、阅读顺序、引用格式 |
| `INDEX.md` | 本文件 |
| `LICENSING.md` | 许可说明（文档 CC BY 4.0 / 代码 Apache-2.0） |
| `NOTICE.md` | 第三方组件的版权与许可声明 |

## 2. `report/` —— 实验报告

| 文件 | 说明 |
|---|---|
| `report.md` | Markdown 版，共 10 节 + 2 个附录，含全部数字与 3 张图 |
| `report.docx` | Word 版，14 张表、3 张图，可直接分发 |

## 3. `paper/` —— 论文

| 文件 | 说明 |
|---|---|
| `paper.pdf` | 最终 PDF（5 页），作者并列：soyamilk & Cloveriris |
| `paper.tex` | LaTeX 源（`ctexart`，XeLaTeX 编译），与 PDF 内容同源 |
| `paper.docx` | 排版中间产物（用于生成本环境下的 PDF） |

编译 LaTeX（需 XeLaTeX + ctex）：

```bash
cd paper && xelatex paper.tex   # 图路径为 ../figures/
```

> 本机未安装 TeX 引擎，`paper.pdf` 由同源的 DOCX 排版流水线生成，
> 与 `paper.tex` 内容一致（同一份 `tools/paper_content.py`）。

## 4. `docs/` —— 复现文档

| 文件 | 内容 | 读者 |
|---|---|---|
| `00-reproduction.md` | 净室从 0：硬件推荐、驱动、双环境、目录、权重、数据、奖励服务、训练管线、验收门槛、成本、排障、检查清单 | 想亲手复现的人 |
| `01-datasets.md` | 数据来源、构成、授权限制、预处理链、重建命令 | 关心数据合规的人 |
| `02-models-and-rewards.md` | 基座与奖励模型的获取、校验、校准 | 搭环境的人 |
| `03-config-reference.md` | 配置项字典与四组典型组合 | 调参的人 |
| `04-sanitization.md` | 脱敏规则、占位符对照、已知限制 | 审阅发布包的人 |

## 5. `figures/` —— 科研图表

| 文件 | 内容 |
|---|---|
| `fig1-fused-reward-curves.png` | 融合奖励随更新的变化（5 个 run，含校准/未校准对照） |
| `fig2-delta-fused.png` | 末次相对首次的 Δ 融合奖励（正负分化） |
| `fig3-budget-vs-actual.png` | 16 个正式 run 的预算 vs 实跑 |
| `fig4-distill-mse.png` | 蒸馏 latent MSE 滑动平均与零修正基线 |
| `fig5-cost-per-update.png` | 单次更新的平均墙钟耗时 |

生成方式：`python tools/make_figures.py`（PIL 超采样 2×，200 DPI 等效）。

## 6. `analysis/` —— 结构化指标

| 文件 | 行数 | 说明 |
|---|---|---|
| `summary.json` | — | 26 个 run + 2 个蒸馏 run 的配置与指标汇总 |
| `runs_table.csv` | 26 | run × 预算/更新数/耗时/完成标记 |
| `eval_scores.csv` | 66 | 固定评估记录（run, step, fused, kawai, deepGHS, HPS） |
| `eval_scores.json` | 66 | 同上，JSON 版 |
| `distill_metrics.csv` | 2 280 | 蒸馏逐更新记录（mse, zero_offset_mse, grad_norm） |
| `extract*.py` | — | 抽取脚本（路径已占位符化） |

## 7. `code/` —— 代码快照（已脱敏）

| 目录 | 说明 |
|---|---|
| `moss_art/` | 训练主包：采样、ART 闭环、LoRA、数据管线 |
| `configs/` | 配置模板（路径为占位符） |
| `scripts/` | 启动、预缓存、数据准备脚本 |
| `tests/` | 数值 parity 与单元测试 |
| `moss-ba-jax/` | SideNet 版少步蒸馏 |
| `moss-ba-small/` | 101.8 M MOSS 形状学生 |
| `moss-art-jax-warmup4/` | ART 的一个实验分支（warmup 变体） |
| `moss-reward-pipeline/` | 奖励服务、校准、固定评估队列 |

## 8. `tools/` —— 构建工具

| 文件 | 用途 |
|---|---|
| `make_figures.py` | 从 `analysis/*.csv` 生成 5 张图 |
| `plotlib.py` | 轻量绘图模块（Pillow 实现，替代 matplotlib） |
| `build_docx.py` | Markdown → Word |
| `build_paper.py` | 论文 → `.tex` + `.docx`（→ PDF） |
| `paper_content.py` | 论文内容的唯一来源 |
| `verify_eval.py` | 复核评估 CSV 的一致性 |

## 9. 占位符

| 占位符 | 含义 |
|---|---|
| `$HOME` | 复现机用户家目录 |
| `$CODE_ROOT` | 训练代码根目录 |
| `$MOSS_ROOT` | 训练产物根目录（run / cache / datasets） |
| `$MODEL_ROOT` | 权重目录 |
| `$DATA_DISK` | 数据盘挂载点 |
| `$CONDA_PREFIX` | conda 安装前缀 |
| `train-host` / `<PORT>` | 训练主机与端口（已隐去） |
| `<WORKSPACE>` | 打包工作目录 |

## 10. 发布前自检

- [x] 全部文本文件中主机名、用户名、隧道域名、绝对路径残留 **0** 处
- [x] 显卡型号已替换为架构代号（SM/GB）+ 显存
- [x] 数据集与权重不随包分发，`NOTICE.md` 已声明上游许可
- [x] 报告 / 论文 / 图表数字同源（`analysis/*.csv`）
- [x] `paper.pdf` 与 `paper.tex` 内容一致
