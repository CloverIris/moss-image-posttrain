# 按 prompt 混合完整 DiT 与默认 Spectrum 采样

在服务器 `$CODE_ROOT/moss-art-jax` 的训练 JSON 中设置：

```json
{
  "sampling": {
    "rollout_mixing": {
      "enabled": true,
      "full_dit_weight": 1.0,
      "spectrum_weight": 1.0
    }
  }
}
```

配比按 **prompt 数量** 计算。每条 prompt 的全部 K 张轨迹、所有 GPU 分片、
梯度累积和 ART 两个更新窗口都保持同一种采样方式。

完整 DiT 指每个采样步计算完整网络；权重精度仍由 `sampling.rollout_mode`
决定，所以两组可以同时使用 BF16、MXFP8 或 MXFP8 calibrated。
采样步数、sampler、scheduler、CFG、分辨率等沿用同一配置。

## 配比与分组

| full_dit_weight | spectrum_weight | 8 条 prompt 的组成 |
| --- | --- | --- |
| 1 | 1 | 4 条完整 DiT + 4 条默认预测 |
| 3 | 1 | 6 条完整 DiT + 2 条默认预测 |
| 1 | 3 | 2 条完整 DiT + 6 条默认预测 |
| 1 | 0 | 全部完整 DiT |
| 0 | 1 | 全部默认预测 |

默认关闭此混合功能。开启后，训练种子与原始正负条件文本的哈希将每个
prompt 固定分入两类。相同文本即使 ID 不同，也保持相同方式；改变数据行顺序、
分辨率、轨迹 seed、卡号或重启进程都不会重新分配。

使用 `training.prompts_per_update > 0` 的逻辑批次时，从两类 prompt 中按配比
选取本轮的不同 prompt。整数比例严格满足，例如 P=8、1:1 一定是 4+4。
不能整除时使用可复现的系统舍入，例如 P=8、2:1 每轮为 5+3 或 6+2，
长期按 2:1 分布。分辨率混合先确定当前分辨率桶，再在该桶内选两种方式。

这需要每个分辨率桶内两类 prompt 都足够；启动时检查配额所需的最少数量。
只有一条 prompt 的过拟合数据只能属于一种方式；空的方式桶会被排除。
旧的非逻辑批次配置仍支持固定哈希分组，但每轮不保证整数配额。

断点恢复不需要额外 RNG 状态，使用原有 prompt cursor / resolution cursors。
修改混合权重、训练种子或启用状态会改变实验指纹，应作为新实验运行。
混合关闭时仍兼容此前 checkpoint 的配置指纹。

## 默认预测分支

该分支固定为 **30% Chebyshev + 70% 一阶 Taylor**；配置要求
`sampling.spectrum.blend_weight = 0.3`（默认值）。多项式次数、历史大小、
warmup、tail 和窗口参数仍使用 `sampling.spectrum`，详见
[Spectrum 说明](spectrum_chebyshev.md)。混合功能开启时，按 prompt 的分组
控制是否预测，优先于全局 `sampling.spectrum.enabled`。

两类均在 ART 的 `sampling.train_timesteps` 上计算真实 DiT velocity。
预测分支的其他步骤可以走快路径，因而训练阶段的实际完整 DiT 步数可能
比纯 rollout 基准的 18/48 更多；逐轨迹的实际计数均有记录。

## 记录与 TensorBoard

- `sampling_assignments.json`：每条记录的 ID、文本哈希、固定采样方式与桶计数；不复制提示词正文。
- `sampling/rollout_XXXXXX.json`：每轮各组 reward、prompt/图片数，以及每张图的 seed、方式、完整/预测步数。
- `rollouts.jsonl`：full_dit_prompts、spectrum_default_prompts、图片数和实际 prompt 比例。
- 导出图片目录的 `samples.json`：每张图增加 sampling_method、dit_full_steps、dit_forecast_steps。
- TensorBoard：`rollout/full_dit/reward_mean`、`rollout/spectrum_default/reward_mean`，以及各组标准差与数量。

原来的总 reward 曲线和每条 prompt 内的 reward 归一化仍然保留。
两组对应不同 prompt，分组曲线用于监控各自趋势；比较采样器画质时应另做同 prompt、同 seed 对照。

## 示例训练配置

`configs/art/silva_anchor_mxfp8_calibrated_sampler_mix.json` 在原 Silva 配置上
启用 1:1：P=8、K=16、T=4，两个 ART 更新窗口，768/1024 分辨率按原来的
2:1 配置。每轮 4 条 prompt 使用完整 DiT，另 4 条使用默认预测。
数据、优化器、LoRA、模型精度等沿用原配置，输出和 TB 指向新的 nvme2T 目录。

```bash
cd $CODE_ROOT/moss-art-jax
$CONDA_PREFIX/envs/jax_env/bin/python run.py train --config configs/art/silva_anchor_mxfp8_calibrated_sampler_mix.json
```

这份配置用于新实验；本次实现只执行合成数据的小模型训练/恢复验证，没有启动生产训练。
