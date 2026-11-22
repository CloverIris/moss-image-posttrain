# 恢复 XLA 默认优化

2026-10-20，5070Ti_REMOTE / GB203(SM120) 16GB，JAX/jaxlib 0.10.2。

**已移除项目注入的全部全局编译和精度覆盖。当前推荐原生 XLA 默认优化，1024 全 FP32 VAE 四图热平均 0.3855 秒/张。** 此前 31 秒基线使用了全局 autotune=0；相对该基线的约 65× 不能解释为自定义 FFI 优于正常工作的 XLA。

## 原因与修正

此前为避开 VAE 调优/编译期间超大临时分配日志，在进程级注入了 `--xla_gpu_autotune_level=0`。作用范围过大，限制了整条 XLA 图的优化。本次原生默认调优测试出现大分配被拒绝后自行恢复并完成，候选拒绝日志不作为运行失败；以最终编译、同步执行和输出验证为准。

生产 runtime、独立 evaluate、VAE benchmark 与诊断脚本均不再写入或覆写：

- `XLA_FLAGS`，包括全局 autotune=0。
- `TF_FP16_CONV_USE_FP32_COMPUTE` 等全局卷积精度覆盖。
- 项目没有注入 JIT 禁用或优化 pass 禁用设置。

调用者已有环境原样保留，未设置的变量保持未设置；实际新 benchmark 的 `XLA_FLAGS` 与 TF 精度变量均为 null。SSH profile、Conda activate/state、systemd 用户环境亦未发现这些注入。[环境与源码独立审计](../reports/xla_defaults_independent_review.json)。

设备可见性/platform、分配器预分配/显存预算、CPU 线程/亲和性用于资源调度；编译缓存和 `JAX_ENABLE_X64=true` 保留。它们不关闭 XLA 优化 pass，显式 BF16/FP32 参数不因开启 x64 自动变为 FP64。

`dit_autotune_level: 4` 仍是局部 DiT JIT 选项。配置值 0 表示不传 compiler_options、继承默认或调用者环境；不等于全局禁用调优。诊断中的显式局部 level0/4 对照仍可用，且 metadata 准确记录实际环境；历史 reports 与 measured_sources 未改写。

## VAE 复测

使用相同四张已保存 latent：cat seed42/43、girl seed1051/1052，batch1、1024×1024、全 FP32，9.6GiB 活跃占位缓冲模拟主干显存。占位不是实际 DiT。两轮，第二轮为热运行；计时包含 latent H2D、同步 decode、raw D2H，排除编译、模型加载、图像保存、指标和奖励。

| 当前后端 | 设备 | 四图热平均 | 对历史 FP32 参考最差 raw 误差 | 最低 SSIM |
| --- | --- | ---: | ---: | ---: |
| **原生 XLA 默认优化** | GPU2 | **0.385545s** | 1.030e-5 | 0.999999486 |
| cuDNN bounded，XLA 默认优化 | GPU1 | 0.433747s | 1.276e-5 | 0.999999292 |

全部输出有限，两轮 raw 逐位一致，8-bit 最大像素差均为 1。独立同 GPU1、同 sample3 的结果：原生 0.382101s、bounded 0.433914s；不同运行时段，未给统计置信区间。

原生最终 HLO 有 36 个 cuDNN 卷积，并有 14 个融合 residual side input。FP32 output、HIGHEST operands；固定版本源码核验表明 FP32 accumulation，并在该路径过滤 TF32/Tensor Core 和输入降精度计划，未声称审计 SASS。最大选中单卷积 workspace 约 396MiB，最终 executable 临时内存约 1.875GiB。初次独立运行的累积活跃峰值 13.506GiB 包含调优及 9.6GiB 占位；它不是热解码独立峰值。调优期间的约 73GiB 分配日志不是最终显存需求。

bounded 后端保留为显式 workspace 控制选项，其 FP32 compute 与 workspace 上限只作用于该 FFI 卷积。现在没有证据把它推荐为比原生默认 XLA 更快的路径。

## 真实训练和 rollout

两种后端均完成 1024、rank4、girl/K2、两个 ART 窗口各5个时间步、两次优化器更新：

| 路径 | 总运行时间（含加载/编译等） | JAX 活跃峰值 | 验证 |
| --- | ---: | ---: | --- |
| 原生 XLA 默认 | 127.62s | 11.658GiB | Adam=2，336个LoRA张量变化，梯度/优化器有限 |
| bounded + XLA 默认 | 201.95s | 11.658GiB | 同上 |

进程首次编译、缓存、并发和CPU评分负载不同，**这两个总运行时间不作为后端纯性能比**。完整 checkpoint 重读及导出逐位一致，current/run/checkpoint 身份匹配；未运行额外续训 GPU 更新。

- [原生训练验证](../reports/train_1024_native_xla_defaults_validation.json)
- [bounded 训练验证](../reports/train_1024_xla_defaults_validation.json)

另完成同适配器、cat/girl、相同四个seed的 bounded 实际 rollout。新运行两轮 latent/PNG 逐位一致，热阶段每张为 DiT 13.296s、decode 0.751s、CPU reward 5.009s，三阶段和 19.055s；该进程同时存在其他卡验证和 CPU 回归负载，不能将与历史微小时间差解释为纯后端变化。

**全局编译策略变化也影响了 DiT 的数值执行路径**，新 rollout 与旧全局 level0 的 latent 不再逐位相同（相对 L2 约0.1167）。因此不能沿用前一报告“reward 仅变化8e-6”的结论；该旧结论只用于同一份固定 latent 的卷积方案比较。新运行自身重复稳定，固定 latent 的 VAE 数值误差独立列在上表。测试范围是四张固定图和两次更新，不证明长期收敛或画质改善。

## 当前使用方式

原生默认路径无需构建 FFI：

```bash
cd $CODE_ROOT/moss-art-jax
# 参数已写入下方 JSON；运行前检查数据、输出目录和可选 reward URL。
bash scripts/run.sh train --config configs/examples/xla_defaults-1.json
```

`model.vae_conv_backend` 默认即 `xla`，VAE 默认 FP32；不要再添加关闭调优的全局变量。需要 HTTP 评分，在 JSON 中设置 `reward.kind="http"`、`reward.decode=true`、`reward.url="http://REWARD_HOST:8765/reward"`。显式需要 bounded workspace 时使用 `configs/moss_train_1024_fast_vae.json`；该配置的名称沿用之前版本，并不表示当前比原生更快。

源码已变更，旧完整 checkpoint 的严格身份会拒绝跨版本恢复。使用已有独立 LoRA 的 `run.init_adapter` 开启新实验；新验证检查点分别在 `outputs/train_1024_native_xla_defaults` 和 `outputs/train_1024_xla_defaults`。

## 验证与复现

完整 CPU 回归：**255 tests、74 subtests 通过**。新增环境策略测试覆盖 runtime unset/empty/用户值逐字保留，以及所有生产/诊断入口的全局覆盖回归。最终定向环境/编译测试另行通过。大输出保留服务器，历史报告保留真实测量条件。

```bash
# 四图原生默认 VAE；选择空闲GPU和新输出目录
$CONDA_PREFIX/envs/jax_env/bin/python scripts/benchmark_vae_precision.py \
  --device 2 --precision fp32 --conv-backend xla \
  --latents outputs/rollout_1024_cudnn_tuned/repeat_00/endpoints.npy \
  --samples all --repeats 2 --reserve-dit-gib 9.6 \
  --reference outputs/vae_precision/fp32_reference \
  --output outputs/vae_xla_defaults/new_native_four

# 已保存数据汇总，CPU-only
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=2 \
  $CONDA_PREFIX/envs/jax_env/bin/python scripts/diagnostics/summarize_xla_defaults.py
```

[机器可读汇总](../reports/xla_defaults_validation.json)包含源报告 SHA256、实际环境、精度、计时口径、训练和新旧 rollout 差异。历史全局 level0 测量来源为提交 `a0dde46` 及更早版本；当前入口不会重新注入那个限制。
