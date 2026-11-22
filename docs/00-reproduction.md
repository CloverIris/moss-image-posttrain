# 净室复现指南：从 0 到一次完整的 MOSS-Image 后训练

本文件面向**没有本实验任何残留环境**的复现者。按顺序执行即可从一台空机器走到
"固定评估集出分"。所有路径用占位符表示，含义见 `04-sanitization.md`。

预计总时长：环境 1–2 h，数据 1–3 h（取决于是否自行采集），一次 20 更新 pilot 约 4 h，
一次 200 更新正式实验约 50 h（4 卡独占）。

---

## 1. 硬件推荐

本实验的瓶颈是**单卡显存**与**卡数**，不是算力峰值。1024×1024、48 步 rollout 在
BF16 全量 DiT 下约占 9–10 GB，再加上 LoRA 反传与 VAE 解码，16 GB 是硬门槛。

| 档位 | 配置 | 能跑到哪一步 |
|---|---|---|
| **最小可跑** | 2 × 16 GB（SM90 或 SM120），64 GB 系统内存 | 768 分辨率、P8/K8/T4，20 更新（约 6 h） |
| **推荐** | 4 × 16 GB 计算卡 + 1 × ≥12 GB 卡（专职学习器/评估），≥48 GB 内存 | 1024 分辨率、P24/K8/T4，200 更新（约 50 h） |
| **等价复现** | 4 × 24 GB（SM90，如 GB100/A100 级） | 同上，可放宽 microbatch 与 resolution mixing |

存储：单 run 20 更新产生 5–12 GB（rollout 图 + checkpoint）；200 更新建议预留
**≥300 GB NVMe**。系统盘与数据盘分离，避免系统盘写满导致训练中断（本实验踩过）。

内存：DP4 采用**主机内存梯度汇聚**，不使用 NCCL。每条轨迹的 LoRA 梯度约数百 MB，
4 卡并发建议 ≥48 GB，否则交换会显著拖慢更新。

网络：单机多卡即可，无跨机需求。若用多机，需自行替换梯度汇聚实现。

---

## 2. 系统与驱动

```bash
# 验证：应看到 SM120 或 SM90 的设备
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
```

| 组件 | 版本要求 | 说明 |
|---|---|---|
| OS | Ubuntu 22.04 / 24.04，内核 ≥ 6.5 | 需要 io_uring 与 cgroup v2 |
| NVIDIA 驱动 | ≥ 550（SM120 需 6xx 分支） | 本实验实测 620.70.06 |
| CUDA Runtime | 13.0（torch 2.11+cu130 自带） | 驱动侧只要求向下兼容 |
| cuDNN | 9.x | VAE 卷积后端使用 `cudnn_bounded` |
| 磁盘 | ≥ 2 TB NVMe | 数据 + 产物分离挂载 |

---

## 3. Python 环境（两个隔离 env）

### 3.1 训练环境 `jax_env`

```bash
conda create -n jax_env python=3.12 -y
conda activate jax_env
pip install --upgrade pip

# JAX 与训练栈（版本必须锁定，数值 parity 依赖它们）
pip install "jax[cuda12]==0.10.2" "jaxlib==0.10.2" \
            "flax==0.12.7" "optax==0.2.8" \
            "numpy==2.3.5" "safetensors==0.5.0" "msgpack==1.1.0" \
            "pillow==11.0.0" "msgpack-numpy==0.4.8"
pip install "transformers==4.57.0" "tokenizers==0.21.0"
```

> JAX 的 CUDA 轮子按 `cuda12`/`cuda13` 分发，与本机驱动分支不一致时以
> **能跑通 `jax.devices()` 且 dtype 为 bfloat16** 为准；本实验的 parity 基线
> 建立在 0.10.2 上，换版本需重跑 9.1 的数值验收。

### 3.2 奖励环境 `torch_env`

```bash
conda create -n torch_env python=3.12 -y
conda activate torch_env
pip install "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu130
pip install "transformers==4.57.0" "accelerate==1.2.0" \
            "diffusers==0.38.0" "safetensors==0.5.0" \
            "pillow==11.0.0" "numpy==2.3.5"
```

### 3.3 特征预缓存环境（可选）`vllm_env`

文本编码器特征可离线预缓存，避免训练时持有编码器。若显存充裕，也可让编码器常驻
第 5 张卡（本实验的 `qwen_resident` 模式）。

```bash
conda create -n vllm_env python=3.12 -y
conda activate vllm_env
pip install "vllm==0.11.0" "transformers==4.57.0"
```

---

## 4. 目录布局

```bash
export CODE_ROOT=$HOME/moss          # 代码
export MOSS_ROOT=/data/moss-art      # 训练产物（大容量盘）
export MODEL_ROOT=/data/models       # 权重
export DATA_DISK=/data               # 数据集与缓存

mkdir -p $CODE_ROOT $MOSS_ROOT/{runs,cache,datasets,evaluations,distillation} $MODEL_ROOT
```

本包 `code/` 目录即为 `$CODE_ROOT/moss-art-jax` 的脱敏快照；其余三个代码目录
（`moss-ba-jax`、`moss-ba-small`、`moss-reward-pipeline`）同样在 `code/` 下。

---

## 5. 权重获取

MOSS-Image 由 OpenMOSS 发布（Apache-2.0）：

```bash
huggingface-cli download OpenMOSS-Team/MOSS-Image-Base \
  --local-dir $MODEL_ROOT/MOSS-Image-Base
huggingface-cli download OpenMOSS-Team/MOSS-Image-SFT \
  --local-dir $MODEL_ROOT/MOSS-Image-SFT

# 国内网络可用镜像
export HF_ENDPOINT=https://hf-mirror.com
```

ModelScope 镜像：`openmoss/MOSS-Image-Base`、`openmoss/MOSS-Image-SFT`。

**务必固定 revision**（`--revision <commit>`）并记录到实验日志。管线组件由
`model_index.json` 声明：Qwen3.5 文本编码器、MOSSImageTransformer、FLUX.2 系
`AutoencoderKLFlux2` VAE，diffusers 版本 0.38.0。

校验：权重下载后计算主干 shard 的 SHA256，与首次训练时 `input_audit.json` 中
记录的三个 transformer shard 哈希比对，一致才继续。

---

## 6. 数据准备

本实验使用**自建审美数据集**（2 000 条 prompt + 一条参考图），不随本包分发。
完整来源与授权说明见 `01-datasets.md`。渲染自己的数据时，格式为 JSONL：

```json
{"sample_id":"cat-001","prompt":"A cat wearing a red hat.","negative_prompt":""}
```

### 6.1 长度扫描与分档

```bash
python -m moss_art.prompt_lengths --config configs/analyze_prompts.json
```

产出 short/full 双档（本实验 288 / 1024 token，short 覆盖 79.36%），超长样本写入
`dropped_overlength.jsonl`（本实验丢弃 5 条）。

### 6.2 划分

- 训练集：1 991 条；
- 固定评估集：4 条，**跨轮次永不改动**，且与训练集无 ID 重叠、无归一化文本重叠；
- 写出 `split_manifest.json` 记录两侧 ID 与哈希，训练侧只读训练 JSONL。

> 评估集固定是全部结论可比的前提。任何"换一批评估 prompt"的改动都会让跨 run
> 的数字失效——本实验早期就因为换过验证集，导致旧 run 与新 run 不可直接比较。

### 6.3 文本特征缓存

```bash
python scripts/prefill_qwen.py --config configs/preparation/vllm_qwen_dp2_pp2.json
```

缓存 key 为 `(prompt_text, negative_text, bucket)`，命中即复用；训练阶段不再持有编码器。

---

## 7. 奖励服务搭建

奖励是一个独立进程，通过命令行协议被训练进程调用（`reward.kind = "command"`）。
它需要加载三个审美模型并输出融合分：

| 分量 | 输出 | 归一化 | 权重 |
|---|---|---|---|
| Kawai | 5 类分档概率 → 加权等级分 | 线性映射到 [0,1] | 0.4 |
| DeepGHS | 审美百分位 | 已是 [0,1] | 0.4 |
| HPSv3++ | μ（原始分） | 以校准集 μ/σ 做 z-score 后截断 | 0.2 |
| SDG artifact | 瑕疵概率 | 反向 | 0（关闭） |

融合公式：

```text
fused = 0.5 + gain * (0.4*z(kawai) + 0.4*z(deepghs) + 0.2*z(hps))
```

**校准**是这里最容易忽略的一步：`z()` 的 μ/σ 必须来自固定校准集（本实验用 128 张
人工分层样本），校准文件随 run 冻结，不能在线更新，否则奖励尺度会漂移。

> 经验：HPS 的原始 μ 动态范围极大（本实验中单图 −0.09 ~ +1.2），若不做截断，
> 0.2 的权重会贡献 ~70% 的融合方差，把训练变成噪声优化。详见报告 7.3。

启动后先做健康检查：

```bash
python -c "from reward_pipeline import health; print(health())"   # 三个模型都应为 ready
python scripts/reward_smoke_test.py --config configs/reward/smoke.json
```

---

## 8. 训练管线

### 8.1 一次更新做什么

```text
采样 P×K 张图（48 步、CFG 3.0、MXFP8 校准 rollout）
  → 奖励服务打分（并行 4 卡）
  → 组内 rank-sqrt 归一化 → 优势
  → 在每个 prompt 的 T 个时间点构造训练点
  → LoRA 前反传（FullDiT / Spectrum 1:1）
  → AdamW 更新 2 次（每 K 张图）
  → 写 metrics.jsonl / checkpoint / rollout 图
```

### 8.2 最小配置

```json
{
  "model":   { "path": "$MODEL_ROOT/MOSS-Image-Base", "dtype": "bfloat16",
               "vae_precision": "float16" },
  "lora":    { "rank": 48, "alpha": 48.0, "preset": "attn", "seed": 42 },
  "sampling":{ "rollout_mode": "mxfp8_calibrated", "height": 1024, "width": 1024,
               "steps": 48, "group_size": 8, "guidance_scale": 3.0,
               "skip_layers": [22], "train_timesteps": [0, 15, 31, 47] },
  "art":     { "noise_level": 0.8, "reference_mse_beta": 0.0, "feedback": "scalar" },
  "optimizer":{ "name": "adamw", "learning_rate": 1e-4, "warmup_steps": 4 },
  "training":{ "max_updates": 20, "prompts_per_update": 24,
               "rollouts_per_prompt": 8, "training_points_per_trajectory": 4,
               "checkpoint_every": 2, "keep_checkpoints": 2 }
}
```

字段全集见 `03-config-reference.md`。

### 8.3 启动与监控

```bash
cd $CODE_ROOT/moss-art-jax
bash scripts/run.sh train --config configs/my_run.json

# 进度（读取 run 目录下的 metrics.jsonl 与 checkpoints）
python report.py
```

`$MOSS_ROOT/runs/<run_name>/` 下应有：

| 路径 | 内容 |
|---|---|
| `config.json` | 冻结后的**生效**配置（含未填项的默认值） |
| `training/metrics.jsonl` | 每次更新一行 |
| `training/rewards/rollout_*.json` | 每批的逐图奖励明细 |
| `training/checkpoints/step_*/` | LoRA + 优化器状态（`arrays.npz` + `manifest.json`） |
| `logs/controller.log` | 事件流，含每步的分量奖励与梯度范数 |
| `fixed_eval_5060/` | 固定评估的 outbox/ack 队列与结果 |

### 8.4 断点与恢复

检查点是原子的：先写临时目录，再 rename 成 `step_XXXXXXX`，最后写 `COMMITTED`。
恢复时只认最新一个**带 COMMITTED 标记**的目录，避免读到半截权重。

```bash
ls $MOSS_ROOT/runs/<run>/training/checkpoints/ | tail -3
# 从最新完整检查点续跑
bash scripts/run.sh train --config configs/my_run.json --resume
```

---

## 9. 验收门槛

不要跳过这三关，否则后面所有数字都不可信。

1. **数值 parity**：用官方 PyTorch 参考实现与小张量 oracle 对拍采样器（1/5/8/12 NFE）。
   本实验记录在 `code/` 的 `tests/` 下，应全部通过。
2. **精度一致性**：`rollout_mode` 必须是 `mxfp8_calibrated`。检查 `config.json` 里
   实际生效值，而不是模板值——历史上出现过模板退回未校准模式、导致评估反向下滑的事故（报告 7.2）。
3. **固定评估基线**：训练前先对 step 0（纯 Base）跑一次固定评估，确认分数落在历史
   区间内（本实验 768 分辨率 fused ≈ 0.36，1024 分辨率 ≈ 0.33）。若偏离 > 0.05，
   先查权重哈希与奖励校准，不要开训。

---

## 10. 成本估算

| 项目 | 20 更新 pilot | 200 更新正式实验 |
|---|---|---|
| 训练墙钟 | 3.5–6 h | 33–70 h |
| 卡时（4 卡） | 14–24 卡时 | 132–280 卡时 |
| 产物体积 | 5–12 GB | 60–120 GB |
| 奖励服务 | 与训练同机，占用 1 卡 | 同左 |

单次更新 10.1–20.8 min 的差异主要来自分辨率（768 vs 1024）、K（8 vs 16）与
是否开启谱采样。排期时按 **15 min/更新 × 预算** 估算最稳。

---

## 11. 故障排查

| 现象 | 原因 | 处理 |
|---|---|---|
| `rollout_mode` 被重置为 `mxfp8` | 模板默认值覆盖 | 显式写 `mxfp8_calibrated`，启动后读生效配置复核 |
| 奖励分数方差极大 | HPS 未截断 / 校准文件不匹配 | 检查校准 SHA；对 HPS 做 ±3σ 截断 |
| 更新耗时突然翻倍 | 主机内存不足触发交换 | 降低 `rollouts_per_update` 或加内存 |
| `checkpoint` 读不出来 | 上次写盘中断 | 只加载带 `COMMITTED` 的目录，回退一格 |
| 系统盘写满 | 产物写入默认路径 | 把 `$MOSS_ROOT` 指到数据盘 |
| 评估服务卡住 | outbox 有未消费任务 | 清 `.claim.lock`，重投递 ack |

---

## 12. 复现检查清单

- [ ] `nvidia-smi` 显示 ≥4 张 16 GB 卡，驱动 ≥550
- [ ] `jax.devices()` 返回 5 张设备，bfloat16 可用
- [ ] 权重 revision 已固定并记录 SHA256
- [ ] 训练/评估集划分 manifest 已生成，两侧无重叠
- [ ] 文本特征缓存命中率 ≥95%
- [ ] 奖励服务三项 `ready`，校准文件已冻结
- [ ] 数值 parity 测试全绿
- [ ] step 0 固定评估分数在历史区间内
- [ ] `max_updates`、`checkpoint_every`、`keep_checkpoints` 已写入运行日志
- [ ] 产物目录在数据盘上，系统盘剩余 ≥20%
