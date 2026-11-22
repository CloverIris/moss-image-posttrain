# 1024×1024 原生 attention 基线评估

本报告记录切换 cuDNN 之前的 XLA attention 基线。对应实现提交 `796a6a3`；之后的 cuDNN 优化与实际后端证据另见 [cuDNN 评估](cudnn_attention.md)。

在 `5070Ti_REMOTE` 的单张 GB203(SM120) 16GB 上，真实 5B MOSS 的 1024 rollout、JAX VAE 和单项 ART LoRA 反传均通过，无 OOM。原始单项反传的活跃显存峰值几乎到达分配上限；加入 query-chunk 重计算后，整流程活跃峰值从 **13.92GiB 降到 11.66GiB**，减少 **2.26GiB（16.27%）**。

## 实测条件

- BF16 冻结主干；FP32 LoRA，rank=4、alpha=4、q/k/v/o。使用此前在 256 分辨率仅更新 4 次的 adapter。
- 1024×1024、10 个实际 Euler 步、CFG=3、负面 skip layer 22；每个 prompt K=2，保留 `[0,2,4,6,8]` 的回放缓存。
- 主测试 cat/girl 共 4 张唯一图，seed 为 42、43、1051、1052；相同 seed 重复只验证性能与确定性。
- 梯度探针用 girl 的 seed 42、43，先取完整 K=2 组的真实 CPU SigLIP2 奖励，再对 row 0、step 4（sigma=0.863）执行一次 ART VJP。
- 主测试 GPU 1，梯度探针对照先后使用 GPU 2；每项只用一张卡，不是 FSDP。另一任务使用 GPU 0，没有操作该任务。
- 持久编译缓存开启，首次运行不等于清空编译缓存的冷启动。并行任务同机运行，以下是实际服务器观测，并非隔离环境的硬件极限吞吐。

## 性能与容量

| 测试 | 图数 | rollout 总秒数 | rollout 秒/张 | VAE 秒数 | CPU reward 秒数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 主测试首轮 | 4 | 472.38 | 118.09 | 130.86 | 23.97 |
| 同进程重复相同 seed | 同4张 | 467.99 | 117.00 | 131.75 | 14.48 |
| 原实现梯度探针的采样 | 2 | 237.67 | 118.84 | 65.96 | 13.42 |
| 重计算梯度探针的采样 | 同2张 | 237.25 | 118.62 | 66.43 | 15.13 |

主测试热轮合计 rollout、VAE 和 reward 约 **153.6 秒/张**，不含加载与 PNG 写盘。整个两轮测试耗时 1246.32 秒。

| JAX 活跃分配口径 | 峰值 |
| --- | ---: |
| 首次 rollout，尚未进行 VAE | 10.12GiB |
| VAE 后累计峰值 | 11.66GiB |
| 原实现单项反传后累计峰值 | 13.92GiB |
| 重计算单项反传后累计峰值 | 11.66GiB |
| 重计算后距 allocator limit 的余量 | 2.26GiB |

JAX peak 是进程累计高水位。重计算后的峰值已在 VAE 达到，只能说单项反传峰值不超过 11.66GiB，不能据此给出独立梯度阶段峰值。第二轮 rollout 的 peak 也包含首轮 VAE。原始峰值距 allocator limit 仅 24,320 bytes，因此虽然通过，仍补充了重计算。

`nvidia-smi` 含分配器保留与 CUDA 开销：主测试最多 14617MiB，原梯度测试最多 14842MiB，重计算测试最多 14627MiB。活跃峰值降低后保留显存仍约 14.3GiB，不能据 11.66GiB 安排同卡其他模型。

ART NumPy 回放缓存：4 张图约 88MiB，2 张图约 44MiB，均在 CPU。反传块边界激活同样保存到 CPU。原始报告保留 host RSS，但加载时 mmap/page cache 与其他任务会影响它，未将 RSS 差值归因为重计算成本。

## 反传及数值验证

| 单项 ART 回放 | 原实现 | query-chunk 重计算 |
| --- | ---: | ---: |
| 回放与 VJP 秒数 | 63.41 | 64.47 |
| active / masked / denominator | 1 / 0 / 1 | 1 / 0 / 1 |
| 梯度 L2 范数 | 0.0582328102 | 0.0584797330 |
| 有限且非零梯度元素 | 4128768 | 4128768 |
| optimizer 更新次数 | 0 | 0 |

主测试两次 latent 与 PNG 完全一致。重计算前后的两张完整模型输出也是 latent 逐位一致、PNG 逐像素一致、奖励一致。

全模型 GPU 梯度范数变化约 0.424%；没有保存基线完整梯度数组，不能据范数声称全部梯度向量一致。CPU 对照覆盖 FP32/BF16 × 17/128/129/257 tokens，forward 和 Q/K/V VJP 的最大绝对误差全部为 0。额外真实形状 `[1,4123,3072]` GPU attention 对照中，FP32 VJP 相对 L2 小于 1.60e-7；BF16 Q/K/V VJP 最大相对 L2 为 **0.002775%**，并非逐位一致。局部误差不能替代完整模型梯度向量比较。

重计算只改变中间值保存策略，保留全局 attention、FP32 scores/softmax 与原生舍入。vendor 快照没有修改，新 helper 纳入源码身份。此版本完整 CPU 回归 **96 项测试及 74 项子测试通过，65.17 秒**。

## 图像评估

| 提示词 | seed | SigLIP2 |
| --- | ---: | ---: |
| 白猫、水彩、蓝茶杯 | 42 | 0.261533 |
| 白猫、水彩、蓝茶杯 | 43 | 0.265577 |
| 银发人物、书本、自然光 | 1051 | 0.235701 |
| 银发人物、书本、自然光 | 1052 | 0.225931 |

均值 **0.247186**，标准差 **0.016791**。额外 girl 两图为 0.232765、0.247590。

人工查看：主体、配色及风格符合提示词，没有黑图或明显残留噪声。人物倾向看镜头，“正在读书”的动作表达较弱。不同 seed 有构图和细节变化。这些示例也曾用于短训练，不是独立验证集；不能据少量图、10 步和四次 LoRA 更新推断一般画质提升。SigLIP2 是图文匹配分数，不是准确率。

[原始四图与分数](../outputs/rollout_1024/repeat_00/index.html) · [重计算探针两图](../outputs/gradient_1024_remat/repeat_00/index.html)

![四张原图的缩略图](../outputs/rollout_1024/repeat_00/contact_sheet.jpg)

## 重跑原生后端

```bash
cd $CODE_ROOT/moss-art-jax
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 $CONDA_PREFIX/envs/jax_env/bin/python scripts/benchmark_rollout.py \
  --config configs/moss_rollout_1024.json --prompt-ids cat,girl --repeats 2 \
  --adapter outputs/moss_siglip_4gpu/adapter.safetensors --output outputs/my_native_1024

OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 $CONDA_PREFIX/envs/jax_env/bin/python scripts/benchmark_rollout.py \
  --config configs/moss_gradient_1024.json --prompt-ids girl --repeats 1 \
  --gradient-probe --adapter outputs/moss_siglip_4gpu/adapter.safetensors \
  --output outputs/my_native_gradient_1024
```

当前两个配置显式选择 native。输出目录存在报告时拒绝覆盖，GPU 被占用时预检拒绝运行。更改源码或 attention backend 后，使用旧 adapter 开始新实验，不能绕过严格 checkpoint 身份校验。

本报告记录的 native 阶段没有执行 1024 完整双窗口 optimizer 更新、续训或长时间训练。后续 cuDNN + DiT 调优版本已完成两次 1024 完整窗口更新，结果见 [加速后端报告](cudnn_attention.md)。原始 JSON、按阶段 GPU CSV、CPU/GPU parity 与日志都保存在 `reports/`；[机器可读汇总](../reports/rollout_1024_summary.json) 区分了单项探针与训练。
