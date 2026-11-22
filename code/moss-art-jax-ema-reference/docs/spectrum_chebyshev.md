# 可选的 Spectrum / Chebyshev rollout 加速

在服务器 `$CODE_ROOT/moss-art-jax` 使用 `jax_env`。BF16、MXFP8、
MXFP8 calibrated 三种 rollout 精度均可叠加此选项；默认关闭。
纯 rollout、RL 的 rollout、单卡 worker 和单进程 JAX Mesh DP 共用实现。

## 开启

在现有 JSON 配置中加入：

```json
{
  "sampling": {
    "spectrum": {"enabled": true}
  }
}
```

关闭只需 `enabled: false`。关闭时调用原来的 `model.predict`，不分配特征历史；
旧 checkpoint 的配置指纹在禁用时兼容。

可运行的 Base / MXFP8 / 1024 / 48 步 / 每条 8 张参考配置：
`configs/rollout/spectrum_chebyshev.json`。先设置其中的 `data.prompts`、
`data.cache` 和 `runtime.output_dir`，然后：

```bash
cd $CODE_ROOT/moss-art-jax
$CONDA_PREFIX/envs/jax_env/bin/python run.py rollout --config configs/rollout/spectrum_chebyshev.json
```

## 参数与上游的对应

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| enabled | false | 是否启用预测快路径 |
| degree | 3 | Chebyshev 多项式次数，对应上游 cheby_degree，共 degree+1 个基函数 |
| ridge_lambda | 0.1 | 岭回归正则项，必须大于 0 |
| blend_weight | 0.3 | Chebyshev 权重；剩余权重用于一阶 Taylor 外推 |
| window_size | 2.0 | 初始刷新间隔 |
| flex_window | 0.5 | warmup 后每次真实前向增加的刷新间隔 |
| warmup_steps | 6 | 开头强制完整 DiT 的步数 |
| tail_actual_steps | 3 | 末尾强制完整 DiT 的步数 |
| history_size | 100 | 最大观测历史，至少 degree+2 |

默认参数与上游 MOSS 实现一致。**纯 Chebyshev** 设置 `blend_weight: 1.0`；
`0.0` 是纯一阶 Taylor。`blend_weight: 0.3` 是上游默认的混合方式。

调小 flex_window、增加 warmup/tail 会增加真实计算。关闭预测不会改变
sampler、scheduler 或调度网格；开启后仍执行全部调度步，只替换部分网络计算。

## 实现与训练点

真实步捕获 `final_norm_video` 之前的图像特征，正负 CFG 分支分别保存。
预测步用 Chebyshev 岭回归与 Taylor 混合预测这些特征，然后调用原 MOSS
时间条件与输出头。跳过 text refiner 及全部 42 层 DiT。

时间坐标和窗口刷新规则按上游使用**采样步索引**，并在 scheduler 去重之后
根据实际步数创建计划。模型输入的时间条件仍使用当前 `sigma * 1000`。
每个 prompt/seed batch、分辨率或参数版本都创建独立历史，不跨批次复用。

当 `keep_cache=true`，所有 `sampling.train_timesteps` 都强制完整前向，
因此 ART 的缓存 old velocity 来自实际 DiT。训练、梯度、BF16 anchor 的
前向接口保持完整计算。rollout 的中间轨迹仍会因预测而改变，效果需比较。

小规模系数求解在 CPU 上用 NumPy FP64 完成；大特征历史、FP32 加权累积和
输出头都在 JAX 设备上计算，生产路径不导入 PyTorch 或 ComfyUI。
不增加 XLA 全局限制。固定形状的 donated 环形缓冲保留 DP/TP 分片，
不复制整段 FP32 历史到内存，也不随观测数变化重新编译预测核。

历史容量取 `min(history_size, 计划真实步数)`。默认 1024²、CFG、单卡 BZ=1、
48 步需要约 **864 MiB/卡** 的 BF16 特征历史。显存不足时可减少 history_size
（最低 5），但拟合窗口变化也可能影响画质。

## 观测与验证

每轮日志输出 `spectrum_rollout`，包括 actual_steps、forecast_steps、
actual_step_indices、forced_training_steps 和 feature_history_bytes。
rollout-only 的单图 JSON 及 samples.jsonl 也保留 spectrum 字段。
原有 actual_steps 仍指调度器实际步数，不改作完整 DiT 次数。

上游固定版本：
`1baeab6f9ee94a75011353c6ef41963888cc8979`
（https://github.com/sorryhyun/ComfyUI-Spectrum-KSampler）。
MIT 许可保存在 `docs/licenses/Spectrum-MIT.txt`。
参考论文：https://arxiv.org/abs/2603.01623。

验证包括：27 组 Torch 上游系数对齐、上游窗口计划、环形历史、
BF16/FP32 + CFG + padding + LoRA 的逐位输出头重入、训练点真实 velocity、
跨 seed 历史隔离、单进程四卡 Mesh 结果与串行对齐、现有 ART/精度/驻留回归。

真实权重基准可复现：

```bash
$CONDA_PREFIX/envs/jax_env/bin/python scripts/diagnostics/benchmark_spectrum.py --config configs/spectrum_smoke.json
```

2026-10-25：MOSS Base、MXFP8、cuDNN attention、1024²、48 RES multistep / beta、
四张 GB203(SM120)、每卡 BZ=1，预热后同样输入比较：

| 路径 | 四张图一批的 DiT 采样时间 | 完整 DiT 步数 |
| --- | --- | --- |
| 完整计算 | 39.918 秒 | 48 |
| Spectrum 默认混合 | 15.277 秒 | 18 |

加速 **2.613×**，其余 30 步使用预测。重用真实特征时输出头逐位一致；
两路 endpoint 均有限。单卡观测峰值约 6.9 GiB，包含模型与预测历史。
该基准使用合成 128-token BF16 文本嵌入，不含 Qwen/VAE、奖励或训练，
不能作为画质/reward 对比。报告位于
`$MOSS_ROOT/reports/spectrum_chebyshev_20261025/report.json`。
