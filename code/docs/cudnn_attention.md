# cuDNN FlashAttention 与 DiT 调优

> 更新：本文测量属于此前设置全局 autotune=0 的历史运行。项目现已移除全局调优和精度覆盖；原生 XLA 默认调优本身也能达到亚秒解码。历史约 65× 只相对受限基线，不能代表相对正常 XLA 的收益。当前策略与复测见 [恢复 XLA 默认优化](xla_defaults.md)。

## 确认结果

原先本工程使用显式 FP32 einsum/softmax 的 XLA attention。现在生产配置在 42 个 DiT block 的前向和反向中显式调用 `jax.nn.dot_product_attention(..., implementation="cudnn")`。没有自动 fallback；不满足 CUDA/BF16 或受支持形状时直接报错。

这仍是 JAX/XLA 编译的模型，但 attention 内核交给 cuDNN FlashAttention，不再执行原来的 XLA einsum/softmax attention。两个短文本 embedder block 和 VAE 保留 vendor 实现，不能将整条模型中的所有 attention 都称为已替换。

真实 5B 模型 1024 运行的 optimized HLO 包含：

- 前向：`__cudnn$fmhaSoftmax`。
- 反向：`__cudnn$fmhaSoftmaxBackward`。
- tuned DiT：同时包含 `__cublas$lt$matmul`，证明矩阵乘法的内核选择也已变化。

证据来自实际 `jit__dit_block` / `jit__block_vjp` 编译产物，不是配置值推断。见 [原始完整模型 HLO 摘要](../reports/cudnn_full_model_hlo_evidence.json)，tuned 编译原文保存在 `outputs/gradient_1024_cudnn_tuned/hlo/`。

## 两项独立优化

```yaml
model:
  attention_backend: cudnn
  dit_autotune_level: 4
```

1. FlashAttention：复用原 RoPE，不再重复应用 native split scaling，只传 `scale=head_dim**-0.5`。当前 JAX 0.10.2 的无 bias 奇数序列 VJP 存在长度检查问题，因此在 RoPE 后补一个 dummy token，通过原始 query/key sequence lengths 屏蔽，输出裁回原长。真实 4123 tokens 补为 4124；没有把有效序列改成局部 attention，也没有构造全量 dense mask。
2. DiT 局部调优：`dit_autotune_level: 4` 通过顶层 block JIT 的 `compiler_options` 开启 GEMM 调优。进程全局仍为 `--xla_gpu_autotune_level=0`，VAE 卷积保持原策略，避开曾经尝试约 73GiB workspace 的候选。level=0 继承进程设置，用于 native/Flash 基线。

调优会增加首次遇到新文本长度的编译时间，之后复用编译缓存。不能将 per-JIT 选项放到内部 attention helper，也不要在 tuned 模型外再包 `jax.jit` 或 `jax.grad`；JAX 不允许带 compiler options 的 JIT 嵌套。训练使用现有手动块级 VJP / `predict_with_pullback`，每块激活卸载到 host 后重算。

## 实测范围与口径

物理 GB203(SM120) 16GB，JAX/jaxlib 0.10.2、CUDA runtime 13.3、cuDNN 9.23.2；模型为真实 MOSS BF16 约 5B，LoRA rank=4/alpha=4，1024×1024、10 个实际去噪步、CFG=3、负面分支跳 block 22。每个提示词 K=2；所有对照使用同一个已完成四次 256 更新的 LoRA，基模/提示词/seed 一致。

Rollout 时间只含生成和保留 ART cache，不含 VAE 解码、CPU SigLIP、加载。first-in-process 可能命中持久缓存，不能称为保证 cold compilation；warm 指同进程第二次使用相同 seed。重复图不是新的独立质量样本。GPU0 有其他任务，未触碰；全机背景负载使结果不应解释为严格隔离硬件跑分。

### Attention 微基准

真实形状 `[1,4123,24,128]`，BF16，20 次热运行中位数：

| 操作 | native chunk remat | cuDNN | 加速 |
| --- | ---: | ---: | ---: |
| attention 前向 | 63.899ms | 3.148ms | 20.30× |
| attention 前向与 Q/K/V VJP | 199.56ms | 11.695ms | 17.06× |

这只是 attention 算子，不能代表整轮生成。完整输出向量相对 L2 差异约 0.244%，Q/K/V 梯度约 0.282%/0.279%/0.255%，全部有限。融合 BF16 运算与原 FP32 概率中间值存在舍入差异，不能称为逐位等价。原始未补齐的奇数序列反向失败记录保留在 [原始试验](../reports/cudnn_attention_standalone.json)；[补齐后的完整实测](../reports/cudnn_attention_padded_standalone.json) 与 [不同长度验证](../reports/cudnn_attention_length_smokes.json) 均通过。

### GEMM 隔离微基准

每个 case 使用独立进程，关闭 persistent cache，避免 autotune 结果在 case 间传播：

| 实际 MOSS linear | level 0 | level 4 | 加速 |
| --- | ---: | ---: | ---: |
| 3072→3072，BF16 base + FP32 rank4 LoRA | 10.323ms | 1.178ms | 8.77× |
| 3072→16384，FFN up/gate | 12.453ms | 4.970ms | 2.51× |

两项完整输出 SHA256 相同；这是这些固定输入的结果，不保证整模型所有调优内核逐位一致。首轮共享 cache 的探索不用于性能结论；最终采用 [隔离测试报告](../reports/gemm_per_jit_autotune_isolated.json)。局部 compile options 不修改全局的依据见 [作用域验证](../reports/per_jit_compiler_options_scope.json)。

## 完整 1024 性能与图像结果

| 后端/DiT autotune | 首轮 rollout 秒/张 | 热轮 rollout 秒/张 | 热轮生成+VAE+reward 秒/张 |
| --- | ---: | ---: | ---: |
| native / 0 | 118.09 | 117.00 | 153.55 |
| cuDNN / 0 | 68.33 | 67.07 | 103.69 |
| cuDNN / 4（生产配置） | 16.94 | 13.40 | 50.15 |

以上三个主测试均在物理 GPU1，cat/girl 四张固定 seed，重复两轮。FlashAttention 单独使热 rollout 提升 **1.74×**；再加 DiT 局部调优，相对 native 提升 **8.73×**。含 VAE 和 CPU reward 的实际三阶段合计从 **153.55 秒/张降到 50.15 秒/张（3.06×）**，不含模型加载与 PNG 写盘。VAE 约 32 秒/张，现为主要耗时。原始分阶段 JSON、显存 CSV 及完整数字见 [机器可读汇总](../reports/cudnn_1024_summary.json)。

cuDNN + autotune 4 的四张图 SigLIP 分数为 **0.261654、0.264853、0.243267、0.227398**，均值 **0.249293**、标准差 **0.015088**。图像主体、风格、配色符合提示词，无黑图或明显噪声；人物仍倾向看镜头，读书动作表达有限。这里的评分不是准确率，也不能从这些与历史训练重叠的少量提示词断言画质改善。

- 同一后端/配置内第二轮重复：三个主测试全部 latent 逐位相同、PNG 逐像素相同。
- native→cuDNN：四张图都发生变化，PNG 像素 MAE（0–255）为 0.752、0.765、3.176、2.543，完整 latent 相对 L2 为 12.92%。这反映整个 10 步轨迹的差异，不是单个 attention 的误差。
- cuDNN level0→level4：前三张完全一致，girl seed1052 的像素 MAE=1.645；四图 latent 相对 L2=4.96%。另两张 girl seed42/43 完全一致。不可把固定微基准的逐位一致性推广到所有图或完整梯度。

[最终四张 1024 PNG 与评分](../outputs/rollout_1024_cudnn_tuned/repeat_00/index.html)

![最终四图缩略图](../outputs/rollout_1024_cudnn_tuned/repeat_00/contact_sheet.jpg)

## 真实训练与显存

所有完整模型基准均未 OOM。JAX 累计活跃分配峰值 **11.6589GiB** 已在 VAE 阶段达到；探针与完整训练未超过它。`nvidia-smi` 最高约 **14627MiB**，包含分配器保留与 CUDA 开销。初次 tuned rollout 额外的 GEMM 调优峰值约 10.40GiB，仍低于 VAE 峰值。本次不需要 FSDP；高 rank、长提示词和更大分辨率仍须单独测容量。

单项 girl/step4 真实 ART probe：native remat 64.47 秒、cuDNN level0 43.42 秒、cuDNN level4 首次 100.95 秒。最后一个包含首次反向编译/调优，不能把它当作热运行速度；三者都产生 4,128,768 个有限非零 LoRA 梯度。level4 的 norm 为 0.0519213，level0 为 0.0521100。这里只验证有限性与可训练性；没有保存完整梯度向量做严格等价证明，不同 attention 后端的轨迹也改变了 ART 目标。

另一次 GPU3 新进程探针为 104.37 秒，梯度 finite、L2=0.0519618，4,128,767/4,128,768 个元素非零。虽然持久缓存已启用，但未取得命中日志，不能将其标成热 VJP；JAX 0.10.2 从缓存键排除上述 XLA dump flags，因此也不能将 dump 路径差异当作原因。见 [独立进程记录](../reports/gradient_1024_cudnn_tuned_cached.json)。稳定训练性能以下面的后续完整窗口为证据。

额外执行了 **完整 1024 两窗口训练**，使用 girl/K=2、全部五个选中时间步、真实 VAE+SigLIP：

| 指标 | 第一个窗口 | 第二个窗口 |
| --- | ---: | ---: |
| optimizer update | 1 | 2 |
| 回放项 / active / masked | 5 / 5 / 0 | 5 / 5 / 0 |
| 梯度 L2 | 0.0274832 | 0.0280880 |
| adapter delta L2 | 0.0182004 | 0.0131479 |

整个短训练约 **209.94 秒**，包括加载、rollout、解码、reward、两次更新与 checkpoint 保存。第二窗口从前一更新日志到本次更新日志为 **42.27 秒**，包括 5 个回放项及 CPU 规约、Adam 和参数广播；它不是一个单项 VJP 的计时。Adam 计数确认为 2，导出 adapter 与 checkpoint 参数逐位一致，相对初始 adapter 的 L2 变化为 0.0279526，参数及全部优化器叶子有限，checkpoint payload 完整性和配置指纹校验通过。

[训练验证报告](../reports/train_1024_cudnn_tuned_validation.json) · [新适配器](../outputs/train_1024_cudnn_tuned/adapter.safetensors) · [完整 checkpoint](../outputs/train_1024_cudnn_tuned/checkpoints/step_00000002/)

这验证当前加速后端的训练闭环；本次未运行 1024 精确续训、tuned 多卡训练、长时间收敛或更新后的图像质量评估。此前 native 四卡训练与续训记录仍作为历史独立证据。完整 CPU 回归 **123 passed，74 subtests passed，70.32 秒**，见 [测试日志](../reports/cudnn_tuned_full_tests.log)。

完整训练复现（当前配置使用物理 GPU2；选择新的输出目录）：

```bash
# 参数已写入下方 JSON；运行前检查数据、输出目录和可选 reward URL。
bash scripts/run.sh train --config configs/examples/cudnn_attention-1.json
```

## 复现

在服务器工程根目录，确认目标卡空闲后执行；每次指定新的输出路径：

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 $CONDA_PREFIX/envs/jax_env/bin/python scripts/benchmark_rollout.py \
  --config configs/moss_rollout_1024_cudnn_tuned.json \
  --adapter outputs/moss_siglip_4gpu/adapter.safetensors \
  --prompt-ids cat,girl --repeats 2 --output outputs/my_flash_benchmark

OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 $CONDA_PREFIX/envs/jax_env/bin/python scripts/benchmark_rollout.py \
  --config configs/moss_gradient_1024_cudnn_tuned.json \
  --adapter outputs/moss_siglip_4gpu/adapter.safetensors \
  --prompt-ids girl --repeats 2 --gradient-probe --output outputs/my_flash_gradient

bash scripts/test_cpu.sh
```

该 benchmark 的 `--gradient-probe` 仅运行一个有效 ART row 的一个选中时间步，没有 optimizer update。完整训练入口仍为 `bash scripts/run.sh train`。诊断脚本位于 `scripts/diagnostics/`，支持显式指定设备。

配置/源码变更均纳入严格身份检查。旧 LoRA 可通过 `run.init_adapter` 用于新实验；不可将旧 native 检查点伪装成同一实验继续精确 resume。
