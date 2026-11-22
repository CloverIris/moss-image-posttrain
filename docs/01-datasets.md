# 数据集：来源、构成、授权与重建

## 1. 结论先行

- 本实验使用 **自建审美数据集**：2 000 条"图像 + 提示词 + 参考图"记录，来自公开
  插画社区的**自采**样本，**不随本包分发**，也不得二次分发。
- 训练实际使用 1 991 条，4 条固定评估，5 条因超长丢弃。
- 复现方式有二：**（A）按同样规则自行采集**；**（B）用自己的审美数据集替换**，
  只要保持"1 900+ 条训练 + 4 条固定评估"的规模与划分纪律，结论方向可复用。

---

## 2. 数据从哪来

源记录字段（`source_*.jsonl`）：

```json
{
  "sample_id": "…",
  "prompt": "完整提示词（标签 + 自然语言，原样保留）",
  "negative_prompt": "…",
  "image_path": "参考图路径",
  "image_ratio": 1.0,
  "source": "pixai | pixiv",
  "tags": ["…"],
  "score": 0.0
}
```

采集流程：

1. 从公开插画社区（PixAI、Pixiv）按**热度分位**取样，覆盖多种画风与构图；
2. 保留站内已公开的提示词原文（标签串 + 自然语言混排），不重写、不补质量词；
3. 每张图计算宽高比，按正方形中心裁剪到 1024；
4. 记录来源、作者 ID 与许可标记（用于后续合规审计）；
5. 人工抽检构图与文本一致性，剔除纯文字图、拼图、明显 NSFW。

> 数据集**不含**真实人物照片，来源均为二次元/插画类公开投稿。

## 3. 为什么不能分发

| 约束 | 说明 |
|---|---|
| 原作者版权 | 图像版权属原作者，本项目仅有"研究用途"范围内的处理许可 |
| 平台条款 | 两个社区的服务条款禁止批量再分发站点内容 |
| 提示词 | 提示词由用户创作，部分含个人风格签名 |

因此本包只提供**清单、划分与预处理脚本**，不提供图像与提示词本体。
需要严格复现数字的读者，请按第 2 节流程自采同规模数据，或用公开审美数据集
（如 AVA、LAION-Aesthetic 子集）替代并在报告中声明替换。

## 4. 构成

| 维度 | 数值 |
|---|---|
| 源记录 | 2 000 |
| 超长丢弃（>1024 token） | 5 |
| 隔离的旧 heldout | 4（保留在源中但永不训练） |
| 训练集 | 1 991 |
| 固定评估集 | 4（4 prompt × 4 seed = 16 张/次） |
| 宽高比 | 统一 1:1（中心裁剪） |
| 文本档位 | short 288 / full 1024 token（short 覆盖 79.36%） |
| 正/负提示词 | 原样透传；1 639 条正、1 036 条负在本版有改动 |

## 5. 预处理链

```text
source.jsonl
  ├─ 长度扫描（文本 token 数、参考图 token 数）
  ├─ 超长丢弃            → dropped_overlength.jsonl
  ├─ 旧 heldout 隔离      → 永不进入训练
  ├─ 划分                → train.jsonl (1991) + eval_fixed4.jsonl (4)
  ├─ 文本特征预缓存       → cache/qwen_*/（BF16, ctx 1024）
  └─ split_manifest.json （ID 列表 + 内容哈希）
```

关键纪律：

- **评估集 4 条 prompt 一旦确定永不变更**，否则跨 run 数字不可比；
- 训练侧 sampler 只读 `train.jsonl`，物理上无法触达评估 prompt；
- 所有划分写入 `split_manifest.json`，包含两侧 ID 与归一化文本哈希，用于事后审计
  "是否存在泄漏"。

## 6. 重建命令

```bash
# 1) 长度扫描与分档
python -m moss_art.prompt_lengths --config configs/analyze_prompts.json

# 2) 划分（产出 train.jsonl / eval_fixed4.jsonl / dropped_overlength.jsonl）
python scripts/prepare_dataset.py --config configs/dataset/top2000v2.json

# 3) 文本特征预缓存
python scripts/prefill_qwen.py --config configs/preparation/vllm_qwen_dp2_pp2.json

# 4) 校验：两侧应无交集，哈希写入 manifest
python scripts/verify_split.py --manifest $MOSS_ROOT/datasets/split_manifest.json
```

## 7. 规模与统计

单条 prompt 的 token 长度分布（short/full 双档）：

| 指标 | short 档 | full 档 |
|---|---|---|
| 覆盖样本 | 79.36% | 100% |
| 最大长度 | 288 | 1024 |
| 平均长度 | ≈ 120 | ≈ 260 |

序列长度直接决定单条轨迹的显存与耗时：full 档在 1024 分辨率下比 short 档慢约 1.6×。
若显存紧张，优先降档而不是降分辨率——分辨率对审美奖励的影响更大。

## 8. 引用与致谢

数据集构建依赖两个社区公开的提示词与图像元数据。若使用本报告的方法而自采数据，
建议在论文/报告中声明数据来源与筛选规则，并遵守各平台当时最新的服务条款。
