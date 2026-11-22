# 1024 VAE 卷积优化

> 更新：本文测量属于此前设置全局 autotune=0 的历史运行。项目现已移除全局调优和精度覆盖；原生 XLA 默认调优本身也能达到亚秒解码。历史约 65× 只相对受限基线，不能代表相对正常 XLA 的收益。当前策略与复测见 [恢复 XLA 默认优化](xla_defaults.md)。

2026-10-20，5070Ti_REMOTE / GB203(SM120) 16GB，JAX/jaxlib 0.10.2，cuDNN 9.23.2。

**推荐全 FP32 的 `cudnn_bounded`：1024 完整热解码约 0.491 秒/张；同卡、同 latent 的原实现约 31.757 秒，约 64.85× 加速。** 这次主要收益来自卷积执行计划选择。已经接入训练、HTTP/SigLIP 解码与独立评估，并完成真实 MOSS 两次 ART 更新。

## 实测结果

独立 VAE 测试保持 9.6GiB 活跃容量占位数组，模拟主干常驻的显存压力；占位数组不是实际 DiT。解码逐张执行，四张固定 latent（cat seed 42/43、girl seed 1051/1052），两轮。热耗时使用第二轮，包含 latent H2D、同步解码和 raw D2H，不含首次编译、权重加载、PNG 保存、指标计算或奖励。

| 后端 / 精度 | 热解码秒/张 | 编译临时显存 | 累积 JAX 活跃峰值（含 9.6GiB 占位） |
| --- | ---: | ---: | ---: |
| 原 XLA 路径 / FP32 | 31.757 | 1.875GiB | 11.593GiB |
| **有 workspace 上限的 cuDNN / FP32** | **0.491** | **1.875GiB** | **11.593GiB** |
| cuDNN / 仅 up1、up2、up3 卷积 FP16 | 0.145 | 1.125GiB | 10.843GiB |
| 分块 GEMM / FP32（实验性） | 0.774 | 4.125GiB | 13.924GiB，达到分配器上限 |

原实现本次复测的是 sample 3，两轮；其结果与此前 FP32 参考逐位一致。其余三行是四图第二轮平均。严格同 GPU3、同 sample 3 的对比是 **31.757452 → 0.489675 秒（64.85×）**。FP16 与分块 GEMM 使用其他同型号卡，较小差异不宜解释为严格受控比较。compiled temp 是整图临时分配，不等于 cuDNN workspace；峰值是该进程的累计记录，不能只相加某个子算子的 workspace 来估算容量。

![性能和实际流程](../reports/vae_conv_optimization.png)

四图对原 FP32 输出的最差值：

| 后端 / 精度 | raw 最大绝对误差 | 最低 RGB SSIM | 8-bit 最大像素差 |
| --- | ---: | ---: | ---: |
| cuDNN FP32 | 1.523e-5 | 0.999999325 | 1 |
| cuDNN 后三级卷积 FP16 | 0.002294 | 0.999904144 | 1 |
| 分块 GEMM FP32 | 8.226e-6 | 0.999999413 | 1 |

全部输出有限；同一后端两轮 raw 输出逐位一致。推荐配置保持全 FP32。145ms 方案仅后三级卷积操作数改为 FP16，归一化、attention、残差等敏感计算仍为 FP32，**不是全 FP16 VAE**。四张图的误差检查不代表任意图像或长期训练收敛保证。

## 改了什么

[先前逐层分析](vae_layer_profile.md)发现原实现 36 个卷积占整图 GPU kernel 时间的 99.667%，高分辨率卷积使用非常慢的 direct kernel。全局 XLA autotune=0 当时用于避开超大 workspace 的候选算法；只改精度没有解决计划选择。

新增 [cuDNN Python 接口](../moss_art/cudnn_conv.py)和 [typed JAX FFI 实现](../moss_art/native/cudnn_conv_ffi.cc)：

- 在主机准备阶段查询 cuDNN heuristic，选第一个满足精度及 workspace 条件的计划；不是运行时穷举测速。执行阶段只查询已建计划并调用 cuDNN，不构建 plan/handle，也不自行 cudaMalloc。
- 外部 NCHW/OIHW，FFI 内采用 NHWC 布局。支持 batch=1、1×1/3×3、SAME、stride/dilation=1；生产解码逐张 slicing。FP32/FP16 操作数使用 FP32 compute，bias 在原有 FP32 路径加回。
- 拒绝 `DOWN_CONVERT_INPUTS` 和 `REDUCED_PRECISION_REDUCTION`。FP32 还拒绝 `TENSOR_CORE` 计划，防止此版本中未标记 DOWN_CONVERT 的 TF32 路径。FP16 允许 Tensor Core 计划与 FP32 累加。
- workspace 作为显式 JAX FFI U8 输出由 XLA 管理，按计划实际所需分配；配置上限 0–512MiB，推荐 256MiB。本次全部计划最大仅 **26.42MiB**。零 workspace 计划使用 1 byte 未使用的 scratch 占位；上限 0 不表示允许任意内存。
- 每进程只能看到一张 GPU，生产每卡独立 worker 符合要求。库不存在、源码/二进制指纹不匹配或无合格计划时明确报错，不自动构建或静默回退。FFI 未声明 command-buffer-compatible，也不支持对 VAE 反向传播；在线 ART 只训练 DiT LoRA，不对图片奖励/VAE 求梯度。
- 全局 autotune 仍为 0，DiT 的 cuDNN FlashAttention 和局部 autotune 4 保持原配置。此次没有改 VAE attention 实现。

完整预热 JIT 图 CUPTI 核验：**36/36 个卷积走新 cuDNN FFI**，实际主要 kernel 为 `convolve_common_engine_float_NHWC`、`implicit_convolve_sgemm`，35 次 engine 28、1 次 engine 1。卷积 GPU 时间从原约 31.623s 变为 **373.886ms，占优化后 432.082ms kernel 总时长的 86.53%**。设备 kernel 总时长与上表完整解码墙钟 491ms 的口径不同，不用它计算 64.85×。精度结论依据计划 metadata 与数值实测，不把 kernel 名称当作 SASS 指令审计。

保留了 [纯 JAX 分块 GEMM](../moss_art/tiled_conv.py) 作为实验性替代。它的整图编译临时内存明显大于卷积微测量，并在 9.6GiB 占位下触及分配器上限，因此不作为 16GB 生产推荐。`vae_conv_workspace_mib` 只约束 cuDNN 计划，不约束 GEMM 整图内存。该路径独立将无卷积的 VAE 图局部 autotune 设为 4。

## 实际 rollout 与训练

严格匹配历史对照：同 rank=4 LoRA（SHA256 `13dc08e4eb1cf07b3e1f3fc377d12b0932f2113915d3f9c09a1d55fc555c4acb`）、cat/girl、相同四个 seed、10 步与 scheduler。本次 GPU2，历史对照 GPU1，均为 GB203(SM120)。两轮四图均通过，热第二轮每张：

| 阶段 | 原 FP32 VAE | 新 FP32 cuDNN VAE |
| --- | ---: | ---: |
| DiT rollout | 13.399s | 13.335s |
| VAE 解码 | 32.168s | 0.501s |
| CPU SigLIP reward | 4.582s | 4.235s |
| 三阶段之和 | **50.149s** | **18.071s** |

这是测得三个阶段之和，不包含模型/进程初始化、全部文件 I/O 或训练反向；总流程约 **2.78×**，主要减少 VAE 时间。对照来自历史运行，小幅 DiT/reward 变化不归因于卷积优化。实际新 rollout JAX 活跃峰值 **11.659GiB**，nvidia-smi 观察到最大 **14739MiB**；后者含分配器保留等开销。

相同四图的 SigLIP reward 最大绝对变化 **7.957e-6**，组内排序、ART mask 不变，两组 ART 方向余弦均大于 0.999999999999。完整证据见 [奖励与 ART](../reports/vae_conv_rewards.json) 和 [严格匹配 rollout 核验](../reports/vae_conv_matched_rollout_validation.json)。另有 `outputs/rollout_1024_fast_vae` 是零初始化 LoRA、cat/city 容量验证，不能混为同提示词/适配器的画质对照。

真实训练使用 [FP32 快速配置](../configs/moss_train_1024_fast_vae.json)，单卡 GPU3、1024、rank4、girl/K2、两个 ART 窗口各 5 个训练时间步：
**121.73s 完成两次优化器更新**，两次梯度范数 0.029621 / 0.027618，336 个 LoRA 张量均有变化；JAX 活跃峰值 **11.658GiB**。完整 checkpoint 重载、导出逐位一致、参数和优化器全部有限，Adam count=2。核验包含不兼容 backend/workspace/source/binary 拒绝续训；本次未再跑额外恢复后的 GPU 更新，也不声称两步训练证明长期画质提升。[训练验证](../reports/train_1024_fast_vae_validation.json)。

## 使用

服务器本项目已经构建好 native 库。新 checkout/新源码首次使用时执行：

```bash
cd $CODE_ROOT/moss-art-jax
$CONDA_PREFIX/envs/jax_env/bin/python scripts/build_cudnn_conv.py
```

构建只使用现有 g++、JAX FFI headers、cuDNN/CUDA13 库，不安装或升级环境。默认产物位于 `outputs/native`；自定义目录使用构建 `--output-dir` 并设置 `MOSS_CUDNN_CONV_BUILD=/absolute/path/build.json`。此构建脚本针对已验证的 Linux CUDA13/cuDNN9 环境。

推荐从新实验目录运行：

```bash
# 参数已写入下方 JSON；运行前检查数据、输出目录和可选 reward URL。
bash scripts/run.sh train --config configs/examples/vae_convolution_optimization-1.json

# 使用已有 LoRA 开始新的 ART 实验，可选 HTTP URL
# 参数已写入下方 JSON；运行前检查数据、输出目录和可选 reward URL。
bash scripts/run.sh train --config configs/examples/vae_convolution_optimization-2.json
```

将 GPU 与 URL 换成实际空闲设备和服务地址。原有默认配置保持 `vae_conv_backend: xla`；推荐新配置显式启用：

```yaml
model:
  vae_precision: fp32
  vae_conv_backend: cudnn_bounded
  vae_conv_workspace_mib: 256
```

可选 [后三级 FP16 配置](../configs/moss_train_1024_fast_vae_fp16.json) 在此基础上设置 `vae_precision: float16`、`vae_low_precision_stages: [up1, up2, up3]`。该精度/后端组合已做四图完整解码核验；本次完整训练验证使用 FP32 配置。

现有多卡配置也可添加以上 model 字段，各 worker 独立持有模型、VAE 和计划；本次新增后端的真实训练实测为单卡，不将原版本四卡结果算作新 FFI 四卡实测。reward 的 GPU/CPU 配置不改变解码后端；`reward.cpu_decode: true` 与 cuDNN backend 不兼容。

续训绑定 backend、workspace、核心源码、native build manifest 和二进制 SHA256。**已有库无需每次启动重建；重建会改变 manifest 身份。** 更换配置/实现或重建后用独立 LoRA 开启新实验。同一构建和配置可用 JSON 字段 `run.resume="outputs/my_fast_vae_run"`、`training.max_updates=20`，不要同时给 `run.init_adapter`。已验证检查点为 `outputs/train_1024_fast_vae/checkpoints/step_00000002`。

## 复现与验证证据

```bash
# 同原测试的四张 latent，完整解码；选择空闲卡和新的输出目录
$CONDA_PREFIX/envs/jax_env/bin/python scripts/benchmark_vae_precision.py \
  --device 3 --precision fp32 --conv-backend cudnn_bounded \
  --conv-workspace-mib 256 --samples all --repeats 2 --reserve-dit-gib 9.6 \
  --latents outputs/rollout_1024_cudnn_tuned/repeat_00/endpoints.npy \
  --reference outputs/vae_precision/fp32_reference \
  --output outputs/vae_conv_optimization/new_fp32

# 同适配器/提示词的真实 rollout
$CONDA_PREFIX/envs/jax_env/bin/python scripts/benchmark_rollout.py \
  --config configs/moss_rollout_1024_fast_vae.json \
  --adapter outputs/moss_siglip_4gpu/adapter.safetensors --prompt-ids cat,girl \
  --output outputs/new_fast_vae_matched --repeats 2

# 已保存数据汇总，不使用 GPU
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  $CONDA_PREFIX/envs/jax_env/bin/python scripts/diagnostics/summarize_vae_conv_optimization.py --overwrite
```

- CPU 完整回归 **218 tests + 74 subtests 通过**；随后新增的 FFI API 与资产身份定向回归 **24 tests 通过**，其中资产测试与全量有重叠。独立集成/边界回归 82 tests 通过。日志在 `reports/vae_conv_regression.log`、`reports/vae_conv_identity_api_tests.log`。
- 生产 FFI 真实 GPU 8 个小形状边界用例通过，包含 FP32/FP16、奇数空间与通道、1×1/3×3、1 像素空间、0/1MiB cap。[边界报告](../reports/cudnn_conv_independent_edges.json)。
- [主汇总 JSON](../reports/vae_conv_optimization.json)、[优化后整图 trace](../reports/vae_conv_optimized_trace.json)、[全部卷积 CSV](../reports/vae_conv_optimized_trace.csv)、[独立审查](../reports/vae_conv_backend_independent_review.json)、[trace 独立复算](../reports/vae_conv_trace_independent_review.json)。
- 原始 HLO/trace/npy/PNG 在 `outputs/vae_conv_optimization`、`outputs/vae_conv_optimized_trace`；真实训练/rollout 在 `outputs/train_1024_fast_vae`、`outputs/rollout_1024_fast_vae_matched`，均保留服务器。native 二进制和大型产物不进入 Git。
- [benchmark 源码元数据变更记录](../reports/vae_conv_benchmark_source_transition.json)保留早期 baseline/tiled 测试使用的脚本快照；后续仅增加 source/plan 元数据，计时逻辑未改。原始报告保留各自来源哈希。
