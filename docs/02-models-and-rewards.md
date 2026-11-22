# 权重与奖励模型：来源与获取

## 1. 基座权重

| 模型 | 来源 | 许可 | 用途 |
|---|---|---|---|
| MOSS-Image-Base | Hugging Face `OpenMOSS-Team/MOSS-Image-Base` | Apache-2.0 | 后训练起点（ART 与蒸馏教师） |
| MOSS-Image-SFT | Hugging Face `OpenMOSS-Team/MOSS-Image-SFT` | Apache-2.0 | 对照基线，含预量化 MXFP8 主干 |

镜像：ModelScope `openmoss/MOSS-Image-Base`、`openmoss/MOSS-Image-SFT`。
技术报告：arXiv 2610.23374；代码：`github.com/OpenMOSS/MOSS-Image`。

```bash
# 固定 revision，避免上游更新导致数值漂移
REV=<commit-sha>
huggingface-cli download OpenMOSS-Team/MOSS-Image-Base --revision $REV \
  --local-dir $MODEL_ROOT/MOSS-Image-Base

# 国内镜像
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download \
  OpenMOSS-Team/MOSS-Image-Base --revision $REV --local-dir $MODEL_ROOT/MOSS-Image-Base
```

### 1.1 组成（由 `model_index.json` 声明）

| 组件 | 类 | 说明 |
|---|---|---|
| text_encoder | `Qwen3_5Model` | 文本编码器，输出 4096 维序列特征 |
| tokenizer | `Qwen2Tokenizer` | 官方 chat template |
| transformer | `MOSSImageTransformer` | 扩散主干，BF16 约 5.08 B 参数，3 个 shard |
| vae | `AutoencoderKLFlux2` | 1024 → latent 8 通道 |
| scheduler | `MOSSImageScheduler` | 非均匀 shift=5.0 时间表 |

### 1.2 完整性校验

下载完成后按 shard 计算 SHA256，与训练日志中的 `input_audit.json` 比对：

```bash
sha256sum $MODEL_ROOT/MOSS-Image-Base/transformer/*.safetensors
# 三个 shard 的哈希必须与 run 内 input_audit.json 的 transformer_weights 完全一致
```

哈希不一致意味着上游 revision 变更或下载损坏，**不要开始训练**——数值会整体漂移。

---

## 2. 奖励模型

奖励侧采用三个公开审美模型 + 一个瑕疵检测（本批关闭），由 `moss-reward-pipeline`
统一调度。**具体权重文件与版本号记录在每个 run 的 `reward_pipeline.config.json` 中**，
复现时应以该文件为准，而不是以本文档的家族名为准。

| 分量 | 家族 | 输出 | 备注 |
|---|---|---|---|
| Kawai | 动漫插画审美分类器（5 档 SS/S/A/B/C） | 类别概率 → 加权等级分 | 对"完成度"敏感 |
| DeepGHS | 动漫审美百分位回归 | 百分位 [0,1] | 对"构图/色彩"敏感 |
| HPSv3++ | 人类偏好打分（Qwen3-VL-8B 主干，film_hybrid 结构） | μ / logσ | 原始 μ 动态范围极大，必须截断 |
| SDG artifact | 瑕疵/伪影检测 | 瑕疵概率 | 本批权重置 0 |

### 2.1 加载与健康检查

```bash
# 三项都应为 ready，并打印执行设备与分片情况
python - <<'PY'
from reward_pipeline import health
import json; print(json.dumps(health(), indent=1, ensure_ascii=False))
PY
```

典型输出要点：`hpsv3.status == "ready"`、`torch_cuda` 与驱动匹配、`device_map`
把 45 个模块分到两张卡。

### 2.2 校准（不可跳过）

三个分量的原始尺度差异极大，必须先归一化再融合：

```text
z(x) = clip((x - μ) / σ, -3, 3)
fused = 0.5 + gain * (0.4*z_kawai + 0.4*z_deepghs + 0.2*z_hps)
```

μ/σ 来自**固定校准集**（本实验 128 张人工分层样本，覆盖 S/A/B 三档），
校准结果随 run 冻结写入 `calibration.json`，训练期间不得更新。

> 教训：HPS 不做截断时，单图 μ 可从 −0.09 跨到 +1.2，归一化后单点贡献 −4.3，
> 0.2 的权重会主导整个融合值，使训练退化为噪声优化（报告 7.3）。

### 2.3 版本管理

```
moss-reward-pipeline/
├── versions/
│   ├── kawai_deepghs_rank_sqrt_lazy_import_fix_<date>/   ← 修复延迟导入后的稳定版
│   └── rank_sqrt_rerank_<date>/                          ← rerank 变体
└── checkpoint_eval/
    └── run_reward_with_validation.py                     ← 训练侧调用的入口
```

每个 run 的 `config.json` 里 `reward.command` 记录的就是**绝对版本路径**。
复现时二选一：锁定同一版本目录，或用 `versions/` 下的同名代码 + 同一权重。

---

## 3. 许可与合规

| 资产 | 许可 | 再分发 |
|---|---|---|
| MOSS-Image-Base / SFT | Apache-2.0 | 允许，需保留 NOTICE |
| 奖励模型权重 | 各自上游许可（研究用途为主） | **不建议**随包分发，请自行下载 |
| 自建数据集 | 不适用（第三方版权） | **禁止**，见 `01-datasets.md` |

本包 `code/` 与 `NOTICE` 保留 Apache-2.0 要求的声明；论文引用请同时引用
MOSS-Image 技术报告与 BA-solver 原文。
