# Rollout only：提示词文件批量生图

只执行文本编码、DiT 采样和可选 VAE 解码。不会计算 reward、反向传播、优化器更新或保存训练轨迹。
底模默认 BF16，可设置 `model.base_precision: "mxfp8"`；attention 默认 cuDNN，VAE 默认 FP32 + tiled_gemm 卷积。
全部参数来自 JSON：

```bash
cd $CODE_ROOT/moss-art-jax
$CONDA_PREFIX/envs/jax_env/bin/python run.py rollout --config configs/rollout/default.json
# rollout-only 是同一入口的别名
```

## 配置

```json
{
  "data": {
    "prompts": "$HOME/my_prompts.txt",
    "cache": "$CODE_ROOT/moss-art-jax/cache/my_rollout",
    "negative_prompt": "",
    "short_context_length": 512,
    "context_length": 4096
  },
  "rollout": {
    "count": 8,
    "seed": 42,
    "prompts_per_batch": 4,
    "decode": true,
    "save_latents": false
  },
  "sampling": {
    "height": 1024,
    "width": 1024,
    "steps": 50,
    "sampler": "res_multistep",
    "scheduler": "beta",
    "guidance_scale": 3.0,
    "skip_layers": [22],
    "beta_alpha": 0.6,
    "beta_beta": 0.6
  },
  "preparation": {
    "device": 4,
    "async_prefetch": true
  },
  "runtime": {
    "devices": [0, 1, 2, 3],
    "output_dir": "$CODE_ROOT/moss-art-jax/outputs/my_rollout"
  }
}
```

- `count` 是**每条保留的提示词生成几张图**。例如 100 条 × count=8，共 800 张；支持 1 和奇数。
- `seed` 是起始随机种子。同一文件顺序、count 和 seed 下，种子不随卡数、提示词批大小改变。
  第 i 条源提示词、第 j 张图的 seed 为 `(seed + i*count + j) % 2^32`，i、j 从零开始。
- `prompts_per_batch` 是每个 Qwen 缓冲批包含的提示词数，不是 DiT 的张量 batch size。
  DiT 每卡每次生成一张，多张独立图片并行分配；同一个提示词也可分到不同 GPU。
- Qwen 使用物理 GPU4（第 5 张卡），DiT 使用 GPU0～3。Qwen 编码结果缓存到磁盘，
  每条提示词的多次采样复用结果；当前批生图时，另一个缓冲预计算下一批提示词。
  `async_prefetch=false` 则先完成整份文件的缓存预处理，再开始生图。
- 超过 `context_length` 的条件会输出丢弃日志，并记录在 `run_info.json`。
  更换 short/full 配置时使用新的 cache 目录，避免复用不兼容的 Qwen 缓存。
- `decode=true` 保存 PNG；`save_latents=true` 另存 packed latent 的 NPY。至少开启一项。
- 分辨率必须是 16 的倍数。采样器支持 `euler`、`res_multistep`；调度器支持 `moss`、`beta`。
  `moss` 可设 `shift` / `training_shift`；`beta` 使用 beta_alpha/beta_beta，training_shift 保持 1。
- 省略 `run.adapter` 时使用底模。可加载导出的 LoRA：

```json
{
  "run": {"adapter": "/path/to/adapter.safetensors"},
  "lora": {"rank": 4, "alpha": 4.0, "preset": "attn-mlp"}
}
```

LoRA 配置需匹配适配器。这个入口不接收优化器、训练 checkpoint 或 reward 配置。
模型路径、VAE 精度与卷积选项可在 `model` 中配置，沿用主项目的字段和校验。

## 提示词文件

**MOSS 层提示词完整透传**：不解释、增删或整理正文，不根据 Prompt/Negative Prompt 标签拆分条件。
仅做一次约定的转义处理，再由编码器添加官方 Qwen 模板。用户原文件不自动改写。

TXT 每个非空行是一条提示词，支持 UTF-8 BOM：

```text
A watercolor painting of a white cat sitting by a window.
A small wooden cabin beside a lake at sunset.
```

JSONL / NDJSON 支持独立 ID 和负面提示词，未填写 negative_prompt 时使用 data.negative_prompt：

```jsonl
{"id":"cat","prompt":"A white cat sitting by a window.","negative_prompt":"blurry"}
{"id":"cabin","prompt":"A wooden cabin beside a lake."}
```

ID 必须唯一。此入口的输入是文本条件；带参考图的缓存采样沿用 SFT sample 入口。

## 输出

```text
output_dir/
  config.json               # 原始输入配置合并默认值
  prompts.jsonl             # 标准化的提示词快照
  run_info.json             # 输入/保留数量、丢弃 ID、LoRA 身份
  samples.jsonl             # 每张图的 prompt、seed、参数及相对文件路径
  p000000-r000000-s42.png
  p000000-r000000-s42.json   # 单图元数据
  p000000-r000000-s42.npy    # 开启 save_latents 时生成，[C,H/16,W/16]
  index.html                # 开启 decode 时生成的本地浏览页
  complete.json             # 全部成功后才写入
```

输出目录必须为空（允许遗留的锁文件），已有结果不会覆盖。失败时保留已完成文件，
当前没有自动续跑功能；重新启动请选择新目录。相对路径以命令运行目录为基准。
GPU 映射使用 nvidia-smi 的物理索引，沿用已有 UUID 映射，不修改 XLA 全局优化配置。


## 验证记录

`configs/rollout/smoke_1024.json` 已实际生成两条提示词各 3 张，共 6 张 1024×1024 PNG，
使用 BF16 DiT + cuDNN attention、50-step res_multistep + beta、FP32 tiled_gemm VAE。
同时验证了 seed/元数据、latent 有限性和 HTML 浏览页。
Qwen 总共编码 3 份条件（两条正面提示词和共用的空负面提示词），第二批等待仅约 0.03 ms。
36 项相关测试通过，包括 CPU 单卡/多卡、同步/双缓冲结果一致、奇数尾批以及输出防覆盖。
详细结果在 `reports/rollout_only_validation.json`。


### 手写换行与标签

读取每条记录后，会把 prompt 和 negative_prompt 字符串中的字面量 `\n` 转换成真实换行。
TXT 仍按文件的实际行分隔记录，所以一行内写 `\n` 可以构造一条多行提示词：

```text
Prompt: 一只白猫\nNegative Prompt: blurry, low quality
```

送入 Qwen 的正文为：

```text
Prompt: 一只白猫
Negative Prompt: blurry, low quality
```

`Prompt:` / `Negative Prompt:` 等标签原样保留，不解析、不拆分，不会自动设置 CFG 的负面条件。
需要独立的 CFG 负面条件时，仍使用 JSONL 的 negative_prompt 字段或 data.negative_prompt 配置。
JSONL 中标准 JSON 转义 `\n` 本身就会解码成换行；双重转义 `\\n` 解码后也会被本次转换处理。
转义按从左到右处理一次，支持：

| 解析前的字符 | 结果 |
| --- | --- |
| `\n`（一个反斜杠加 n） | 真实换行 |
| `\\`（两个反斜杠） | 一个字面反斜杠 |
| `\\n`（两个反斜杠加 n） | 字面量 `\n`，不换行 |

以上规则作用于 TXT 正文或 JSON 解码后的字符串。JSONL 文件若要最终保留字面量 `\n`，
prompt 值中需写 `\\\\n`（四个反斜杠加 n）：JSON 解码后剩两个，本解析器再转成一个。
未知转义（例如反斜杠加 t）和末尾单独的反斜杠原样保留；中文、ID 和 metadata 不变。


### MXFP8 DiT 推理

```json
{"model": {"base_precision": "mxfp8"}}
```

复用已验证的 JAX MXFP8 线性层：42 个 DiT block 共 252 个投影使用 E4M3 + E8M0 分块缩放，
权重加载时量化，输入激活动态量化，CUDA 上执行 cuDNN blockScaledDot。
attention、归一化等仍使用 BF16/原有精度；不是把所有算子都强制改为 FP8。
rollout-only 没有反向传播，不保留 MXFP8 转置权重副本，也不更新参数。
`worker_info.json` 和启动日志记录各卡实际 base_precision、量化层数和底座存储字节数；
逐图元数据也记录精度。生产路径要求 Blackwell CUDA 卡，CPU 仅作 tiny 数值参考。
