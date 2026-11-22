# VAE 分段混合精度（1024×1024）

> 更新：本文测量属于此前设置全局 autotune=0 的历史运行。项目现已移除全局调优和精度覆盖；原生 XLA 默认调优本身也能达到亚秒解码。历史约 65× 只相对受限基线，不能代表相对正常 XLA 的收益。当前策略与复测见 [恢复 XLA 默认优化](xla_defaults.md)。

测试日期：2026-10-20；服务器：5070Ti_REMOTE；工作目录：$CODE_ROOT/moss-art-jax。
结论：选用后三级上采样卷积 FP16，保留入口、中间块、up0 和输出层 FP32。
这是可选策略，原有配置与默认值继续使用 FP32。

## 切换范围

`model.vae_precision` 支持 `fp32 / bfloat16 / float16`。
`model.vae_low_precision_stages` 从下列阶段显式选择；半精度不能为空，FP32 必须为空。

| stage | 范围 |
| --- | --- |
| input | post_quant_conv 与 decoder.conv_in |
| mid | 两个中间 ResNet 的卷积及 attention 的 q/k/v/out 投影 |
| up0…up3 | 对应上采样块的 ResNet、shortcut 和 upsampler 卷积 |
| output | decoder.conv_out |

推荐配置：

```yaml
model:
  attention_backend: cudnn
  dit_autotune_level: 4
  vae_precision: float16
  vae_low_precision_stages: [up1, up2, up3]
```

所有阶段的 BN 解包、GroupNorm、SiLU、残差相加、attention 的 QK/softmax/AV 和 tile blending 保持 FP32。
选中的卷积使用半精度输入、权重和输出，随后立即转回 FP32 再加 bias。
这不是将整个 VAE 或其敏感计算整体降为 FP16。
权重仍以 FP32 加载；本项主要减少卷积临时空间，并没有把全部模型权重减半。

训练、独立评估、GPU reward decode 和 CPU reward decode 共用该策略，HTTP 和 SigLIP 都接收解码后的 uint8 RGB。
策略进入配置指纹，helper 进入源码身份。修改策略或升级源码后，应以独立 adapter 开启新实验，不能冒充旧实验精确续训。

## 实验方法

使用同一份已生成的四个 endpoint：
`outputs/rollout_1024_cudnn_tuned/repeat_00/endpoints.npy`，形状 [4,128,64,64]。
cat 的种子为 42/43，girl 为 1051/1052。所有 VAE 对比都解码完全相同的 latent，不重新采样 DiT。
FP32 参考是原 vendor 解码路径；vendor 文件未修改。

先在 sample 3 上扫描 BF16/FP16 的 7 个单阶段、early(input+mid+up0)、late(up1+up2+up3)、ups(up0…up3) 及 all，共 22 个策略。
组合/单阶段扫描各一次，首轮包含编译，因此只用于筛选；all 另有一次热轮。
late 对两精度都比 ups 更准确，二者首轮平均差仅 0.102 秒，因此选四图复验的统一策略 late。

正式参考与两候选各解码 4 图×2 次，第二轮四图为热时间样本；重复不是新增图像样本。
FP32/BF16/FP16 分别在物理 GPU1/2/3 的同型号 RTX5070Ti 上，存在并行背景负载，因此小幅速度差应视为本机观测。
独立 VAE 测试各持有 9.6GiB GPU 缓冲区，模拟 DiT 的容量压力；这不等同于真实模型计算。
另以真实 DiT、LoRA 与 ART rollout cache 驻留进行了 GPU1 完整 rollout/decode/SigLIP 测试。

每次保存 clipping 前的 FP32 raw NCHW、最终 PNG、显存采样、软件版本、文件 SHA256 和 optimized HLO。
先检查 raw finite，再 clipping；±Inf 不能再被裁剪成合法 RGB 而漏检。
指标在相同 sample index 上计算，RGB MAE 使用 0–255 单位，PSNR/SSIM 基于最终 uint8 RGB。

## 数值与速度结果

正式四图结果见 `reports/vae_precision_four_samples.json`。
奖励与 ART 比较见 `reports/vae_precision_rewards.json`。
真实模型共存结果见 `reports/vae_precision_rollout.json`。

| 配置 | 热解码均值/图 | RGB MAE均值 | 最差PSNR | 最差SSIM | 最大像素差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP32参考 | 31.90s | 0.000000 | ∞ | 1.000000 | 0 |
| BF16 late | 29.01s | 0.060554 | 59.78dB | 0.999294 | 3 |
| FP16 late | 29.26s | 0.007687 | 68.56dB | 0.999904 | 1 |

FP16 late 的热解码耗时降低约 8.28%（速度为参考的 1.090×）。四图无 NaN/Inf，最大RGB差为1；每个策略的重复raw输出逐位一致。
JAX活跃显存峰值从12,448,124,928降至11,642,820,352 bytes，减少805,304,576 bytes，约0.750GiB；这组数字包含9.6GiB占位。nvidia-smi含分配池，不能当作活跃内存。

真实DiT驻留测试：cat42/43、10steps、CFG3、rank4，两轮均passed。warm每图rollout13.154s、VAE29.264s、CPU SigLIP4.105s，三阶段合计46.523s；实际JAX活跃峰值10.909GiB。
它与历史FP32完整循环的提示词数不同，因此不将两者整体耗时相除宣传加速。两cat endpoint与四图参考的前两项逐位相等。

本地SigLIP参考分数重算与原benchmark逐位一致。两候选的两组内排序、ART group/row active masks 均与FP32相同，所有组仍有效。
FP16最大绝对分数变化为0.00030854，BF16为0.00045064：

| 策略 | cat42 | cat43 | girl1051 | girl1052 |
| --- | ---: | ---: | ---: | ---: |
| FP32 | 0.26165378 | 0.26485318 | 0.24326664 | 0.22739808 |
| BF16 late | 0.26176286 | 0.26440254 | 0.24335441 | 0.22708175 |
| FP16 late | 0.26196232 | 0.26494306 | 0.24332325 | 0.22716886 |

FP16的cat组reward gap缩小6.834%、girl组gap绝对值增大1.801%；BF16分别为-17.494%/+2.547%。
真实ART covariance函数算出的direction cosine均接近1（FP16最小0.999999999864），但本例K=2，这不能证明K>2或临界小差值样本同样稳定。
真实rollout输出的两张cat PNG与独立FP16 late解码逐像素、文件SHA256完全一致，见 `reports/vae_precision_rollout_standalone_pixels.json`。

单图筛选（sample 3；以下数值不是四图平均）：

| 策略 | BF16 像素 MAE | FP16 像素 MAE |
| --- | ---: | ---: |
| 仅 input | 0.064237 | 0.011756 |
| 仅 mid | 0.004983 | 0.000645 |
| 仅 up0 | 0.019179 | 0.002515 |
| 仅 up1 | 0.018525 | 0.002432 |
| 仅 up2 | 0.027696 | 0.003684 |
| 仅 up3 | 0.024966 | 0.003134 |
| 仅 output | 0.086303 | 0.010282 |
| late：up1+up2+up3 | 0.051795 | 0.006173 |
| ups：up0…up3 | 0.056028 | 0.006972 |
| all | 0.168996 | 0.015965 |

在这些样本上 FP16 比 BF16 更准确；它不保证任意 prompt/latent 都安全。FP16 输出仍可能溢出，FP32 内部累加不能消除半精度输出的范围限制。
当前检查会对非有限 raw 输出明确报错，不会静默回退或把它交给 reward。
四张图及本地 SigLIP 的结论不能代表任意远端 reward 的稳定性。

## 为什么收益有限

独立分段 FP32 profile 的热中位数：up1 7.354s、up2 12.340s、up3 9.885s，合计占独立 stage 时间 94.04%。
这支持优先优化后三级，而非入口层。
这些 stage 分开编译，合计 31.451s 不是完整 decoder 的端到端计时。

VAE attention 的 FP32 全局 level0 仅 74.48ms；将其单独 level4 调优降至 18.06ms，输出 bitwise 相等，但只省 56ms，因此没有为此拆分生产解码器。
DiT 的 cuDNN FlashAttention 与此处 VAE attention 是不同算子；VAE 的单头384维 attention 仍走原 FP32 路径。

HLO 显示 late 策略有 22 个半精度 cuDNN conv、14 个 FP32 conv。
FP32 路径可融合 bias；为了在半精度卷积后保留 FP32 bias，半精度路径使用 convForward 后再转换/加 bias。
因此半精度计算并不意味着完整解码速度翻倍。VAE 仍沿用全局 autotune level0；本轮没有开启曾产生巨大 workspace 申请的全局卷积调优。

## 编译与累加精度证据

直接设置 BF16/FP16 卷积输入并请求 FP32 preferred output，会被本机 XLA 提升为整个 FP32 cuDNN 卷积。
小型实测 FP32 为 8.54ms，BF16/FP16 输入+FP32输出仍约 8.5ms；
半精度 conv output 后转回 FP32 才成为真实半精度 cuDNN conv，约 5.4ms。
具体 HLO 和时间见 `reports/vae_conv_dtype_lowering.json`；不能只凭 Python cast 宣称成功使用半精度卷积。

本报告历史测量时，训练入口、benchmark 与独立 evaluate CLI 在 JAX/CUDA 初始化前设置
`TF_FP16_CONV_USE_FP32_COMPUTE=1`。本机对应 XLA 源码由该开关选择 FP16/BF16 卷积的 FP32 累加，
见 [固定版本 GetConvAccumulatorType](https://github.com/openxla/xla/blob/5a9e73cbd92530cac2ac36f4736a774b2412afe2/xla/stream_executor/cuda/cuda_dnn.cc#L2458-L2562)。
HLO 证明输入/输出 dtype，不单独证明库内部 accumulator；累加结论来自运行时开关与对应源码。
当前入口已移除上述全局覆盖；原生 XLA 路径使用其默认或调用者设置，bounded FFI 在每个卷积计划中显式声明 FP32 compute。历史精度结论只适用于记录的测量条件。

## 复现与使用

```bash
cd $CODE_ROOT/moss-art-jax

# 独立四图精度/时间复验；选择空闲的物理GPU和全新输出目录。
$CONDA_PREFIX/envs/jax_env/bin/python scripts/benchmark_vae_precision.py \
  --device 1 --precision float16 --stages up1,up2,up3 \
  --latents outputs/rollout_1024_cudnn_tuned/repeat_00/endpoints.npy \
  --samples 0,1,2,3 --repeats 2 --reserve-dit-gib 9.6 \
  --reference outputs/vae_precision/fp32_reference \
  --output outputs/my_vae_fp16_test

# 真实DiT驻留的两图rollout、解码和本地SigLIP；不更新优化器。
$CONDA_PREFIX/envs/jax_env/bin/python scripts/benchmark_rollout.py \
  --config configs/moss_rollout_1024_vae_fp16.json \
  --adapter outputs/moss_siglip_4gpu/adapter.safetensors \
  --repeats 2 --output outputs/my_fp16_rollout

# 可选FP16策略的1024训练配置，完整两个ART窗口。
# 参数已写入下方 JSON；运行前检查数据、输出目录和可选 reward URL。
bash scripts/run.sh train --config configs/examples/vae_precision-1.json
```

本轮执行了真实 rollout/decode/reward，没有用新混合精度重新进行优化器更新实验。
此前完整 FP32 VAE 的两窗口训练见 `docs/cudnn_attention.md`；此次 VAE 仍是 detached reward 路径。
CPU 回归：160 tests + 74 subtests，通过。辅助脚本与完整证据在 `scripts/diagnostics/`、`reports/vae_*`；
较大的 raw、HLO 和 PNG 保留在服务器 `outputs/vae_precision/`，筛选对照图在忽略追踪的 `reports/vae_precision_panels/`。
