# 架构与 ART 实现语义

## 数据与执行边界

```text
JSONL 提示词
    ↓ 原生 tokenizers + JAX Qwen3.5，独立预处理进程
按原长度保存的文本特征缓存
    ↓
CPU coordinator ── LoRA 参数 ──→ N 个独立 GPU worker
    ↑                              每个 worker：完整 BF16 MOSS + FP32 LoRA
    │                              同一 prompt 的 K 个采样，CFG + res_multistep / beta
    │                              冻结 endpoint、选中状态、旧速度
    ├── host endpoint ←───────────┘
    │
    ├── 原生 JAX VAE → PNG → 外部 CPU SigLIP2（或 HTTP POST 远端评分服务）→ 标量奖励
    ├── 全局奖励/latent 协方差、两个平衡窗口
    ├── 每个 worker 对 LoRA 执行原生 JAX VJP
    ├── CPU 求和未归一化梯度，再除原始全局窗口项数
    └── CPU 上 JAX/Optax AdamW → 广播新 LoRA → 第二窗口
```

这是一机多进程的数据并行实现。它不使用 Torch DDP，不使用 NCCL all-reduce，也不是 FSDP、ZeRO 或模型分片。每张训练卡都保存完整冻结主干；只在 CPU 与各 worker 之间传输 LoRA、LoRA 梯度及 rollout 数据。CPU 规约牺牲一部分通信效率，便于隔离显存与在消费级显卡上验证正确的分母和更新次序。

模型的 block forward/VJP 分别 JIT；前向在块边界把激活保存到 host，反向按块重算。不会构建整条去噪链的反向图，也不会请求冻结 5B 基模的参数梯度。`MossModel` 的 host 控制循环不能整体再套一层 `jax.jit`。

生产 CUDA 配置使用 `model.attention_backend: cudnn`：42 个 DiT block 的前向和手动 VJP 显式调用 JAX 的 cuDNN FlashAttention，失败时报错，没有自动回退。原 RoPE 后将奇数序列补齐到偶数，并通过原始 query/key length 屏蔽 dummy token，最后裁回原长度；这是当前 JAX/cuDNN 反向接口的长度兼容处理。两个短序列 text embedder block 与 VAE 保持 vendor 实现。融合 attention 与原 FP32 scores/probabilities 实现存在舍入差异，详见 [内核证据与速度](cudnn_attention.md)。

`model.dit_autotune_level: 4` 只为顶层 DiT block forward/VJP 设置 JIT compiler options；不注入进程级 XLA 调优限制，VAE 使用默认调优。配置 level=0 表示继承默认编译选项，CPU 必须用 0。tuned 模型的训练使用 `predict_with_pullback` / `loss_and_grad` 手动块级 VJP；不要在它外面再包 `jax.jit` 或 `jax.grad`，JAX 不允许带 compiler options 的 JIT 嵌在其他变换内。

作为数值对照的 `model.attention_backend: native` 对超过 128 个 query 的序列再做一层重计算：每个 query chunk 在 `lax.map` 内使用 `jax.checkpoint`，反传时重算该块的 scores/softmax，避免同时保存全部 query 块的 softmax 中间值。所有 query 仍看到全部 key，FP32 分数、概率与 BF16 舍入规则保持原样；这不是局部 attention，也没有改用低精度概率。实现位于 `moss_art/attention.py`，原 vendor 快照保留作为数值对照；新增实现纳入严格源码身份。

## 与 MOSS 推理保持一致的部分

1. 直接 mmap 读取原始 safetensors，保持主干 BF16、PyTorch 权重布局；数值运算使用 JAX。
2. 输入为 `[B,128,H/16,W/16]` 的 packed latent，时间参数为 `1000 * sigma`。
3. 默认50步 beta 从 shift5 的1000点模型网格取样，FP32 res_multistep 状态更新；显式 `euler + moss` 可恢复旧调度与更新算法。见 [采样器与训练回放](sampling.md)。
4. CFG 为 `uncond + scale * (cond - uncond)`；默认仅负面分支跳过零起算块 22。
5. 文本逐条原长度编码，使用 MOSS/Qwen3.5 的固定 chat 模板。训练期间只读取缓存，不训练文本编码器。
6. VAE 解码先做 BN 反归一化与 2×2 空间解包，再执行原生 JAX FP32 decoder；最后将 [-1,1] 映射至 RGB PNG。

跨框架的相同随机 seed 不保证相同噪声；BF16 也有跨后端舍入差异。tiny 梯度测试、组件对照、完整真实训练各自证明不同层面，不互相替代。

## 一轮 ART 的目标

对同一个提示词的 K 个样本，最终 latent 为 `z_i`、外部奖励为 `r_i`。默认 scalar 模式先计算奖励中心化与样本协方差：

```text
r_mean = mean_i(r_i)
c = sum_i((r_i - r_mean) * z_i) / (K - 1)
mad = mean_i(abs(r_i - r_mean))
signal = c / (mad if mad > 0 else 1)
G = signal / max(RMS(signal), eps)
```

RMS 在单个 latent 的全部特征上求均值；不以 L2 范数替代。默认 guidance magnitude 为 1，常数奖励或零协方差组明确设为无信号。后续损失入口仍按 RMS 对方向归一化，保持上游的数值约定。

回放保存的状态，对当前模型速度 `v_theta` 使用冻结目标：

```text
target = stop_gradient(v_old - G)
loss_item = sigma * mean_features((v_theta - target)^2)
```

这里的负号来自本工程的 MOSS 速度与下降 sigma 定义。终点、奖励、旧速度、改进方向和所有 mask 都不求梯度；梯度只经过当前速度预测进入 LoRA。

每组 K 条轨迹在两个窗口中各占 K/2 条，每条轨迹携带所选的全部 timestep。第一窗口更新后广播新 LoRA，第二窗口用更新后的模型重算当前速度，但使用**同一轮保存的旧速度与目标**。这保留了 ART 的冻结目标与漂移修正效果。

梯度以每个窗口原始的全局 item 数作为分母，包括无信号或被 mask 的 item。不能只除以 active/mask 保留的数量，也不能让每卡先各自平均后再等权平均不同大小的批量。

## 漂移 mask、pairwise 与无信号

默认使用上游 scheduler-reference 速度漂移度量。它利用真实求解器的当前 velocity 系数 `C`（Euler 时等于 `delta_sigma`）、sigma、`art.noise_level` 和 `art.kl_threshold` 计算 KL 风格的代理量，再结合向外方向判断决定屏蔽。它不是两个完整生成分布之间精确计算的 KL。

当前默认 rollout 是确定性 ODE res_multistep；`art.noise_level=0.8` 用于漂移代理量，不表示每个采样步额外加入 0.8 随机噪声。

`feedback: pairwise` 支持 K=2/4/8，按确定性随机配对比较标量分数，平局标签为 0.5。在某处存在有效偏好时，平局行可以参与 restore anchor。全局 coordinator 统一确定偏好、窗口与 restore 条件，再下发各 worker，避免局部全平局 worker 丢失全局修正项。

所有组都无信号时跳过该轮更新并继续采样；达到 `max_no_signal_rollouts` 后报错停止，避免静默空转。一个更新窗口全部被 mask 时，梯度可能为零；已有 Adam 动量仍可能使参数变化，因此 `grad_norm=0` 与非零 `adapter_delta_norm` 不必然矛盾。

## LoRA 兼容范围

- 普通 additive LoRA：`delta_W = (alpha / rank) * up @ down`。
- down 使用 Kaiming uniform 初始化，up 从零开始，初始预测等同于基模。
- `attn`：q/k/v/o；`attn-mlp`：再包含 fused FFN up/gate 与 FFN down。
- 参数和主优化器状态为 FP32；残差矩阵乘法使用 FP32，在加到基模激活处转换回激活 dtype。
- 导出接受 Kohaku 生态中的 `diffusion_model.<module>.lora_down.weight` / `.lora_up.weight` 命名与 alpha 信息。
- 导入仅覆盖代码声明的普通 LoRA 格式，支持归一化其 per-module alpha；不是通用 LyCORIS 全格式加载器。DoRA、LoHa、LoKr 与 fused-qkv 等不支持的文件会被拒绝。

JSON 的 `run.init_adapter` 新建优化器并重置步数；`run.resume` 则恢复完整 Optax 状态与数据游标。二者用途不同。

## 证据边界

本工程把论文/上游 ART 核心和本机已有 JAX MOSS 推理组合为可运行的训练流程。没有声称复现论文中其他模型的数值结果，也没有证明 SigLIP2 是 MOSS 二次元目标的最佳奖励。正式训练应保留固定验证提示词和独立人工检查，比较基模及多个检查点的语义、角色一致性、审美和失败样例。
