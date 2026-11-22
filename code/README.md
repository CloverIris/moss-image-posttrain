# MOSS Image：JAX DiffusionART 与 SFT

服务器工程：`$CODE_ROOT/moss-art-jax`，使用 `5070Ti_REMOTE`。
DiffusionART 是在线生成、奖励评分、两次 LoRA 更新的闭环；SFT 是有目标图片的
flow matching，支持文生图以及参考图＋文本生图。两条入口独立，TP2、大 microbatch
目前接在 SFT；short/full 条件 padding 已接入 ART 和 SFT。

## JSON 配置入口

所有训练、采样、评估、预处理参数都从 `--config XXX.json` 读取。
命令行只保留操作名和配置文件；不再接受参数覆盖。
旧 YAML 文件留作历史输入兼容，维护中的启动示例和配置使用 JSON。
相对路径相对于进程工作目录；推荐从工程根目录运行 `scripts/run.sh`。
JSON 不允许注释、重复字段、NaN/Infinity、未知字段或错误类型。
未填写字段使用默认值；实际生效配置写入运行目录的 `config.json`。

```bash
cd $CODE_ROOT/moss-art-jax
bash scripts/run.sh train --config configs/moss_lora_4gpu.json
bash scripts/run.sh train --config configs/moss_http_4gpu.json
bash scripts/run.sh sft train --config configs/sft/train_1024.json
bash scripts/run.sh sft sample --config configs/sft/sample_1024.json
```

先编辑配置中的数据路径和输出目录。HTTP 的 `reward.url` 必须填写真实评分服务；
示例 URL 并不代表服务器正在运行。完整字段与运行状态配置见 [JSON 配置说明](docs/CONFIGURATION.md)。

| 用途 | JSON 文件 |
| --- | --- |
| 四卡 ART / SigLIP | `configs/moss_lora_4gpu.json` |
| 四卡 ART / HTTP POST 奖励 | `configs/moss_http_4gpu.json` |
| 单卡 1024 ART 验证 | `configs/moss_train_1024.json` |
| 四卡 DP SFT | `configs/sft/train_1024.json` |
| MXFP8 LoRA 全链路验证 | `configs/sft/mxfp8_e2e_train.json` |
| 两组 TP2 SFT | `configs/sft/train_1024_tp2.json` |
| TP2 每组 microbatch 3 | `configs/sft/train_1024_tp2_mb3.json` |
| SFT 图文缓存预处理 | `configs/sft/prepare.json` |
| SFT 条件采样 | `configs/sft/sample_1024.json` |
| CPU ART 流程验证 | `configs/smoke_cpu.json` |

## 数据预处理

先用纯 CPU 扫描脚本统计长度、计算参考图 token，选择 short/full：

```bash
$CONDA_PREFIX/envs/jax_env/bin/python -m moss_art.prompt_lengths --config configs/analyze_prompts.json
```

扫描配置、超长丢弃和 GPU4 Qwen 缓存复用见 [双档长度说明](docs/PROMPT_BUCKETS.md)。
ART 默认配置启用 GPU4 常驻 Qwen 与计算卡双缓冲预取，见 [异步预取说明](docs/ASYNC_PREFETCH.md)。

### vLLM 离线 Qwen prefill

可设置 `preparation.backend: vllm`，通过独立脚本提前批量缓存 Qwen 的完整最终
归一化特征，训练阶段只读已完成缓存。四卡配置使用 DP2 × PP2 × TP1，按 PCIe
拓扑分为 `[1,2]` 与 `[0,3]` 两组，BF16 计算与存储：

```bash
$CONDA_PREFIX/envs/vllm_env/bin/python scripts/prefill_qwen.py --config configs/preparation/vllm_qwen_dp2_pp2.json
```

增加 `--plan` 可只检查设备顺序。使用现有 `vllm_env`，训练不持有 Qwen；配置、
断点复用及验证记录见 [vLLM 预处理说明](docs/vllm_prefill.md)。


ART 提示词使用 JSONL，每行例如：

```json
{"sample_id":"cat-001","prompt":"A cat wearing a red hat.","negative_prompt":""}
```

在 ART 配置中设置 `data.prompts`、`data.cache`、`preparation.device`，然后：

```bash
bash scripts/run.sh prepare-text --config configs/moss_lora_4gpu.json
bash scripts/run.sh doctor --config configs/moss_lora_4gpu.json
```

SFT 数据包含目标图和可选参考图，格式和缓存机制见 [SFT 文档](docs/SFT.md)。
预处理入口使用现有 Torch 环境处理 CPU processor/VAE；Qwen 在 GPU4 的独立 JAX 子进程做一次 prefill，训练本身使用 JAX：

```bash
$CONDA_PREFIX/envs/torch_env/bin/python -m moss_art.sft_prepare --config configs/sft/prepare.json
bash scripts/run.sh sft doctor --config configs/sft/train_1024.json
```

## 恢复、初始 LoRA 与评估

编辑同一训练 JSON 的 `run` 字段，而不是追加命令行参数。

```json
{"run":{"resume":"outputs/my_run"},"training":{"max_updates":100}}
```

上面是需要合并到训练配置中的字段。精确恢复仍检查模型、数据、算法配置和源码；
只切换恢复路径不会改变训练身份。切换算法/源码时，使用 `run.init_adapter`
从独立 LoRA 开始新优化器状态，并设置新的 `runtime.output_dir`。
`run.resume`、`run.init_adapter`、`run.checkpoint`、`run.adapter` 互斥；未用的路径设为 `""`。

ART 评估配置使用 `run.checkpoint` 或 `run.adapter`，再执行：

```bash
bash scripts/run.sh evaluate --config configs/my_evaluation.json
```

SFT 采样使用 `run.sample_index`、`run.seed`，以及可选的 `run.adapter` 或 `run.checkpoint`。
不填二者时使用底模。训练随机种子在 `training.seed`。

## 环境与运行结果

工程 Python 是 `$CONDA_PREFIX/envs/jax_env/bin/python`，底模在 `$MODEL_ROOT/MOSS-Image-SFT`。
项目直接使用已有 Conda `jax_env`，不再创建 `.venv`。
`bash scripts/setup.sh --check-only` 检查依赖；`bash scripts/setup.sh` 仅以 `--no-deps` 安装项目入口。
启动脚本默认选择 `jax_env`，也可通过 `MOSS_ART_PYTHON` 指定解释器。预处理与奖励模型复用已有 Torch 环境。
不自动下载权重，不终止其他 GPU 任务。`runtime.devices` 使用 nvidia-smi 物理编号，
运行时转换成 UUID。当前 TP2 配对为 `[0,3]`、`[1,2]`。

ART 输出包含 `config.json`、`run_info.json`、`metrics.jsonl`、检查点、独立
`adapter.safetensors` 及完成标记。一次有效 rollout 做两次更新，因此 ART 的
`training.max_updates` 必须为正偶数。SFT 没有这个限制。
历史测试报告只说明执行正确性、显存与短程速度，不证明奖励训练后的图像质量。

## 调整参数

- `model.attention_backend: cudnn` 显式启用 DiT cuDNN FlashAttention；`native` 保留原 XLA attention。没有静默 fallback。
- `model.dit_autotune_level: 4` 只作用于 DiT 顶层 forward/VJP。值 0 表示不传局部 compiler_options、继承 XLA 默认或调用者环境；项目不设置全局 autotune。切换后端或编译策略应从适配器开启新实验。

- `model.vae_conv_backend: cudnn_bounded` 使用精度受控的 cuDNN 卷积；`vae_conv_workspace_mib: 256` 限制单个计划 workspace。需要显式构建 native 库；默认值仍是 `xla`，新 fast 配置启用它。见 [卷积优化](docs/vae_convolution_optimization.md)。

- VAE 默认 FP32。可选 `model.vae_precision: float16` 和 `model.vae_low_precision_stages: [up1, up2, up3]`，仅后三级上采样卷积用 FP16，敏感运算保持 FP32。见 [分段精度实测](docs/vae_precision.md)、[全 FP16 与 attention 后端核查](docs/vae_all_fp16.md)、[VAE 逐层与层内耗时分析](docs/vae_layer_profile.md) 与 `configs/moss_train_1024_vae_fp16.json`；HTTP/SigLIP 奖励和独立评估使用相同配置。



- `lora.preset: attn` 训练每个块的 q/k/v/o；`attn-mlp` 再加入 fused FFN up/gate 和 FFN down。默认 rank=4、alpha=4。原始 BF16 主干冻结；LoRA 参数、梯度、优化器主副本为 FP32。
- 先从 256×256、K=2、短提示词、rank=2 或 4 开始。单卡完整保存约 9.5GiB BF16 主干，四卡不会把这一份主干分摊为四份。激活、VJP 重算、LoRA 与 VAE 还需要额外空间。
- `sampling.train_timesteps` 指实际 MOSS 调度器网格中的采样步索引。配置中的步数经网格吸附与去重后，不能保证仍有同样数量的有效步；越界会报错。
- CFG 默认 3.0；`skip_layers: [22]` 是零起算的第 23 层，仅跳过负面分支。不要把它当作正面分支也跳层。
- 默认 SigLIP 是图文语义匹配奖励，不是二次元审美、角色身份或文字准确率的专门评分器。奖励上升不等价于全部画质指标提升。
- `toy` 是图片平均亮度；`toy_latent` 是接近常数 latent 的合成分数。它们只供测试。持续优化它们会学到无意义的奖励偏好。

更多实现细节见 [架构与 ART 语义](docs/architecture.md)、[自定义奖励协议](docs/rewards.md)、[运行与排错](docs/operations.md)。上游来源与许可证见 [NOTICE](NOTICE)；最终验证结果查看 [`reports/`](reports/)。


## Supervised LoRA training

Text-to-image and full image-plus-text conditioning SFT are available through
python run.py sft train. See [SFT setup, data and inference](docs/SFT.md).
Training uses native JAX; frozen offline vision/text/VAE encoding uses Torch.

Muon / AdamW 可通过训练 JSON 的 `optimizer.name` 选择，SFT 和 DiffusionART 共用。
Muon 的分组、参数与四卡 MXFP8 验证配置见 [SFT 文档](docs/SFT.md#muon-优化器sft-与-diffusionart-共用)。


### Rollout only

```bash
$CONDA_PREFIX/envs/jax_env/bin/python run.py rollout --config configs/rollout/default.json
```

从 TXT/JSONL 提示词文件批量生图，`rollout.count` 设置每条提示词的生成张数，
`sampling` 设置采样参数，`runtime.output_dir` 设置输出位置。
支持 GPU4 Qwen 双缓冲、四卡并行、可选 LoRA 与 latent 保存；不计算 reward 或执行训练。
配置和文件格式见 [Rollout-only 文档](docs/ROLLOUT_ONLY.md)。


## Optional Spectrum / Chebyshev rollout

Set sampling.spectrum.enabled=true in JSON to enable head-feature forecasting.
The default is false. BF16 and MXFP8 precision modes are independent of this
option. ART replay points always use complete DiT forwards.
See [configuration, implementation and measured results](docs/spectrum_chebyshev.md).
Runnable Base/MXFP8 preset: configs/rollout/spectrum_chebyshev.json.


## Per-prompt full-DiT / Spectrum training mixture

Set sampling.rollout_mixing.enabled=true and configure full_dit_weight and
spectrum_weight. Logical batches enforce prompt quotas (P=8, 1:1 gives 4+4);
each prompt keeps its sampler across all trajectories, devices and resumes.
See [configuration and per-method TensorBoard logging](docs/rollout_mixing.md).
Training preset: configs/art/silva_anchor_mxfp8_calibrated_sampler_mix.json.
