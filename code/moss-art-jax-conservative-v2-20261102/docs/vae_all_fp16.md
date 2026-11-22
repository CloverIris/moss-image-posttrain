# VAE 全 FP16 与 attention 后端核查

> 更新：本文测量属于此前设置全局 autotune=0 的历史运行。项目现已移除全局调优和精度覆盖；原生 XLA 默认调优本身也能达到亚秒解码。历史约 65× 只相对受限基线，不能代表相对正常 XLA 的收益。当前策略与复测见 [恢复 XLA 默认优化](xla_defaults.md)。

2026-10-20，服务器 `5070Ti_REMOTE`。已测完全部卷积/投影半精度与权重、层间激活连续FP16的完整decoder。后者四图热均值28.920秒/图，最大像素差1，没有NaN/Inf；归一化统计、softmax等仍使用FP32。默认训练策略未更改。

现有 `all` 策略四图、两轮全部通过：热解码平均 **29.020 秒/图**，FP32 为 **31.901 秒/图**。原始输出全部有限，同一输入两轮逐位一致。与之前的 `late` FP16 相比，本次只少 0.239 秒/图，而且跨物理卡测量；这个差值不足以证明稳定的额外收益。全阶段策略的平均像素误差高于 `late`。

## “all” 的精确范围

本次调用 `moss_art.vae_precision.decode_latents`，选择 `input,mid,up0,up1,up2,up3,output` 全部阶段：

- 选中卷积：输入与权重转 FP16，卷积输出 FP16，随后转回 FP32 加 bias。
- 选中 attention 线性投影：输入与权重转 FP16，要求 FP32 输出。
- BN latent 反归一化、GroupNorm、SiLU、残差相加、attention 的 QK/softmax/AV、tile 拼接与混合保留 FP32。
- 参数的常驻副本仍为 FP32。

因此，`all` 表示所有可选卷积/线性层采用低精度策略，**不表示整个 VAE 的中间激活一直保持 FP16**。优化 HLO 中 36 个卷积调用的输出均为 `f16`；不能仅靠配置名称判断实际内核精度。

本次在加载 JAX 前显式设置 `TF_FP16_CONV_USE_FP32_COMPUTE=1`。该运行时的 XLA/cuDNN 卷积实现据此请求 FP32 内部计算；卷积最终输出仍舍入为 FP16。HLO 的 `f16` 输出本身不证明累加精度，也不排除其他输入出现半精度溢出。实现和源码依据见 [VAE 混合精度说明](vae_precision.md)。

## 测量方法

使用原 1024 rollout 保存的同一份四个 endpoints，不重新运行 DiT：cat 两图 seed 42/43；girl 两图 seed 1051/1052。输入 SHA256 为 `c8f303ccb1f3dff5b289b25294c7eba7953aaa9d0fa6879c1bdaf357225d26f3`。

每个策略逐图运行两轮，下面时间全部取第二轮。计时包含 latent 设备传输、同步解码和原始结果回传；不含模型加载、编译、PNG 编码、对照指标及 reward。第一轮以及独立编译耗时保留在原始 JSON，不作为热性能结论；共享编译缓存意味着编译时间也不代表完全冷启动。

三次测试均设置全局 XLA autotune level 0、禁止预分配、allocator fraction 0.9，并保留 **9.6 GiB = 10,307,921,510 bytes** 的设备占位缓冲区。占位用于接近 DiT 常驻的内存压力，但没有加载真正的 DiT 权重。FP32 用 GPU1，`late` 用 GPU3，`all` 用 GPU2；其他任务可能同时在别的卡运行。JAX/jaxlib 均为 0.10.2。

## 四图热解码性能与显存

| 策略 | 第二轮四图时间（秒） | 平均 / 中位数（秒） | 相对 FP32 | JAX 活跃分配峰值（bytes） | HLO 卷积输出 |
| --- | --- | --- | --- | --- | --- |
| FP32 | 32.422, 31.982, 31.595, 31.607 | 31.901 / 31.795 | 1.000× | 12,448,124,928 | 36 f32 |
| late FP16：up1/up2/up3 | 29.244, 29.259, 29.283, 29.248 | 29.259 / 29.254 | 1.090× | 11,642,820,352 | 22 f16 + 14 f32 |
| all FP16：全部阶段 | 29.033, 29.010, 29.030, 29.006 | 29.020 / 29.020 | 1.099× | 11,642,820,352 | 36 f16 |

峰值是进程生命周期内 JAX 活跃分配的累计峰值，包含 9.6 GiB 占位。不能把峰值直接叫作 VAE 独占内存，也不能把扣除占位后的差值当成真实训练峰值。NVIDIA 采样峰值分别为 14,607 / 14,543 / 14,607 MiB，包含 allocator 内存池，和 JAX 活跃分配不是同一口径。

## 与 FP32 的输出误差

PNG 指标在相同的 clamp、线性转换到 0–255、round 后计算。SSIM 使用 11×11 Gaussian、sigma 1.5、RGB 三通道平均、population covariance、有效区域裁去 5 像素边缘、K1=0.01/K2=0.03。原始相对 L2 使用未裁剪 FP32 NCHW 输出；global 值把四图平方误差合并再开根号。

| 指标 | late FP16 | all FP16 |
| --- | --- | --- |
| 四图平均 MAE（0–255） | 0.007687 | 0.019042 |
| 最差单图 MAE | 0.009062 | 0.021794 |
| 所有像素最大绝对差 | 1 | 1 |
| 平均 / 最差 PSNR（dB） | 69.328 / 68.559 | 65.372 / 64.747 |
| 平均 / 最差 RGB SSIM | 0.99991694 / 0.99990352 | 0.99979294 / 0.99975153 |
| raw global relative L2 | 0.000114866 | 0.000270745 |
| raw 最大绝对差 | 0.00175822 | 0.00361010 |
| 8 次未裁剪输出全部有限 | 是 | 是 |
| 每图两轮逐位相同 | 是 | 是 |

所有 `all` 的 PNG 最大差为 1，但更大范围的数据、其他模型参数或更极端激活未测，不能据此保证任意输入的稳定性。完整逐图数值和 SHA 在 [四图汇总](../reports/vae_all_fp16_four_samples.json)。

## Reward 与真实 ART 函数

使用原有 CPU SigLIP2 模型、cosine metric、batch size 1 重新评分，FP32 的四个分数与原始 rollout 完全一致。这次只新增 `all` 候选与 FP32 的对照；`late` 结果沿用 [先前独立报告](../reports/vae_precision_rewards.json)。

| 图 | FP32 reward | all FP16 reward | 差值 |
| --- | --- | --- | --- |
| cat 42 | 0.261653781 | 0.261737168 | +0.000083387 |
| cat 43 | 0.264853179 | 0.264767379 | -0.000085801 |
| girl 1051 | 0.243266642 | 0.243219405 | -0.000047237 |
| girl 1052 | 0.227398083 | 0.227101952 | -0.000296131 |

最大绝对 reward 差为 **0.000296131**。cat 的两样本差距从 0.003199399 降为 0.003030211，缩小 5.288%；girl 的绝对差距从 0.015868559 增为 0.016117454，扩大 1.568%。两组排序保持，未出现 tie。

另外在 CPU 上直接调用项目的 `moss_art.art.covariance`（scalar，group_size=2，seed=42，eps=1e-5），使用同一份 endpoints，而非重新实现公式。两组 direction 对 FP32 的余弦相似度为 0.999999999869 / 0.999999999999，相对 L2 差为 1.62e-5 / 1.37e-6；group_active 与 row_active 完全相同，方向全部有限。cat covariance RMS 从 0.001274550 变为 0.001207150；girl 从 0.007465933 变为 0.007583034。

这说明本次四图的 reward 排序与 ART 归一化方向保持接近。K=2 时方向很大程度取决于排序，不等于证明更大 K、更多 prompt 或长期训练没有变化；本对照未执行模型反传或 optimizer update。详细证据见 [全阶段 reward/ART 报告](../reports/vae_all_fp16_rewards.json)。

## 复现与证据

在仓库根目录使用新输出目录；脚本会拒绝覆盖已有测量和占用繁忙 GPU。

```bash
$CONDA_PREFIX/envs/jax_env/bin/python scripts/benchmark_vae_precision.py \
  --device 2 --precision float16 \
  --stages input,mid,up0,up1,up2,up3,output \
  --latents outputs/rollout_1024_cudnn_tuned/repeat_00/endpoints.npy \
  --samples 0,1,2,3 --repeats 2 --reserve-dit-gib 9.6 \
  --reference outputs/vae_precision/fp32_reference \
  --output outputs/vae_precision/fp16_all_four_NEW

$CONDA_PREFIX/envs/jax_env/bin/python scripts/diagnostics/compare_vae_rewards.py \
  --reference outputs/vae_precision/fp32_reference \
  --candidate outputs/vae_precision/fp16_all_four_NEW \
  --output reports/vae_all_fp16_rewards_NEW.json
```

- [原始测量 summary](../outputs/vae_precision/fp16_all_four/summary.json)、[优化 HLO](../outputs/vae_precision/fp16_all_four/optimized.hlo.txt)、[HLO dtype 证据](../outputs/vae_precision/fp16_all_four/hlo_evidence.json)。
- [性能、质量、显存汇总](../reports/vae_all_fp16_four_samples.json) 与 [汇总脚本](../scripts/diagnostics/summarize_vae_all_fp16.py)。
- [CPU reward 与 ART 报告](../reports/vae_all_fp16_rewards.json)。
- 原始 NPY、PNG 及逐像素差异面板保留在 `outputs/vae_precision/fp16_all_four/repeat_00/`，不纳入 Git。
- JSON 中保留输入、源码、报告与 HLO SHA、软件版本和环境策略。权重身份采用路径、文件大小、mtime、safetensors header SHA；不是全权重 payload SHA。

## 连续 FP16 decoder：独立四图实测

实验代码：`scripts/diagnostics/vae_full_fp16.py` 与 `benchmark_vae_full_fp16.py`。
全部decoder常驻权重和跨层激活为FP16，包括卷积/线性输出、bias、SiLU、残差与attention输入输出。
BN反归一化/解包在decoder边界之前保持FP32；GroupNorm统计和affine为FP32，存储的半精度scale/bias提升后计算、结果转回FP16。
QK用半精度输入与FP32 logits，softmax在FP32计算，概率转FP16后做PV，FP32累加再转FP16。
没有宣称每条中间指令都是FP16，也没有改变原单头attention结构。

同一份4张图×2轮、9.6GiB占位，本策略在GPU1测试。
所有8次raw输出有限，每图热重复逐位相同。汇总见 [四种策略对比](../reports/vae_full_fp16_comparison.json)。

| 方案 | 热均值/图 | RGB MAE均值（0–255） | 最差SSIM | JAX活跃峰值（含占位） |
| --- | ---: | ---: | ---: | ---: |
| FP32 | 31.901s | 0 | 1 | 12,448,124,928B |
| late FP16 | 29.259s | 0.007687 | 0.99990352 | 11,642,820,352B |
| all卷积/投影FP16 | 29.020s | 0.019042 | 0.99975153 | 11,642,820,352B |
| 连续FP16 decoder | 28.920s | 0.020026 | 0.99974012 | 11,385,299,456B |

连续FP16比FP32少约9.35%解码时间，活跃峰值少约0.990GiB；
与late相比只少0.339s/图、再少约0.240GiB。背景负载和卡不同，很小的额外速度差不宜视为保证。
最差单图MAE为0.023356，平均/最差PSNR为65.175/64.447dB，最大RGB差仍为1。
raw全局相对L2为0.000302122，最大绝对raw误差为0.00492784。

GPU HLO确认36个F16 `__cudnn$convBiasActivationForward`，恢复半精度bias融合。
140个decoder参数张量为FP16、3个BN buffer为FP32，总55,895,498B；
编译临时空间1,006,642,472B。这些指标与总活跃峰值、显卡内存池不同。
本实验仍为单独VAE加容量占位，未执行新的真实DiT共存训练或优化器更新。

本地CPU SigLIP四图分数为
`[0.2613703609, 0.2649127543, 0.2431753576, 0.2270236313]`，
相对FP32最大绝对变化0.0003744513。
两组排序、所有group/row active masks保持，但cat组reward gap增加10.7206%，girl绝对gap增加1.78445%。
真实ART covariance方向cosine分别为0.999999999904 / 0.999999999999。
K=2时方向接近很大程度反映排序保持，不能证明更大K或长期训练等价。
见 [连续FP16 reward/ART报告](../reports/vae_full_fp16_rewards.json)。

复现连续FP16实验：

```bash
$CONDA_PREFIX/envs/jax_env/bin/python scripts/diagnostics/benchmark_vae_full_fp16.py \
  --device 1 --precision float16 \
  --latents outputs/rollout_1024_cudnn_tuned/repeat_00/endpoints.npy \
  --samples 0,1,2,3 --repeats 2 --reserve-dit-gib 9.6 \
  --reference outputs/vae_precision/fp32_reference \
  --output outputs/vae_precision/full_fp16_NEW
```

这是独立实验路径。生产配置的 `vae_precision: float16` + 全部stages仍代表“all卷积/投影”，不会自动选择连续FP16 decoder。
本轮新增6项CPU回归通过，检查FP32对照、半精度跨层dtype、GN安全统计、未裁剪输出与tile转换。
测量时helper/runner源码快照和SHA在 `reports/measured_sources/` 及 `reports/vae_full_fp16_measured_sources.json`。
测量后只修复未触发的tile显式转换及FP32控制分支的描述，1024 direct数值路径未改。

## 当前实际使用的 attention 后端

**这套流程的 DiT 使用 cuDNN FlashAttention；VAE 没有使用 cuDNN FlashAttention。**

生产/近期实测配置如 `moss_lora_4gpu.yaml`、`moss_http_4gpu.yaml`、`moss_train_1024.yaml`、
`moss_rollout_1024_vae_fp16.yaml` 显式配置 `attention_backend: cudnn`。
实际DiT forward HLO出现 `__cudnn$fmhaSoftmax`；
VJP还出现 `__cudnn$fmhaSoftmaxBackward`，两者 `is_flash_attention=true`。
代码仍保留native对照配置和CPU路径，不能把所有配置都称作cuDNN。
这次精度实验只改VAE；DiT仍为BF16，LoRA/优化器主副本仍为FP32。

VAE在1024图像下为16384 tokens、1个head、head_dim=384。
原FP32、late/all半精度和连续FP16的实际HLO均无FMHA调用。
卷积确实使用cuDNN，但attention仍通过query分块限制内存，softmax覆盖全部keys。

GPU3用head_dim=384、tokens16做显式 `implementation="cudnn"` 测试：
FP16/BF16前向和VJP全部被当前JAX拒绝：

```text
NotImplementedError: The head dim must be <= 256 and a multiple of 8, but got 384.
```

安装源码 `jax/_src/cudnn/fused_attention_stablehlo.py` 的约束、版本、SHA、错误和实际HLO证据见
[后端审计](../reports/vae_attention_backend_audit.json)；复现脚本 `scripts/diagnostics/probe_vae_cudnn.py`。
该结论仅针对本机JAX显式cuDNN接口；没有把384维拆成多个head绕过限制，因为那会改变模型语义。
此前实测VAE FP32 attention约74.48ms，占独立阶段耗时约0.24%；主要瓶颈在上采样卷积。
