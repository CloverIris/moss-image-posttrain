# 当前实际验证记录

本页依据已完成产物记录，不能以配置存在或代码支持代替执行结果。最终补充报告可放在 `reports/`。

## 原生三卡真实奖励训练：已完成

配置：`configs/moss_siglip_3gpu.json`。输出：`outputs/moss_siglip_3gpu`。日志：`reports/moss_5b_siglip_3gpu.log`。

| 项目 | 实际值 |
| --- | --- |
| 物理 GPU | 1、2、3 |
| 模型 | 真实 MOSS 约 5B 主干，BF16 冻结权重 |
| LoRA | attention q/k/v/o，rank=2，alpha=2，FP32 |
| 采样 | 256×256、3 步、每提示词 K=2、每轮 3 个提示词 |
| 回放时间步 | `[0]` |
| CFG / 负面跳层 | 3.0 / `[22]`，零起算 |
| 图像解码 | 原生 JAX VAE，各 worker GPU 解码 |
| 奖励 | 本地 SigLIP2、独立 PyTorch CPU 进程、cosine |
| 优化器更新 | 2 次，1 个 rollout |
| 总完成时间 | 147.117 秒，包含加载及前后固定评估 |
| rollout 和两次更新 | 45.057 秒 |
| JAX allocator 最大峰值 | 10,688,730,112 字节，约 10.69GB / 9.95GiB |
| 完成产物 | `complete.json`、step 2 checkpoint、`adapter.safetensors`、前后 PNG/HTML |

两次更新的全局 LoRA 梯度范数分别为 0.0272180 和 0.0284224，适配器变化范数分别为 0.0100986 和 0.0101067；两个窗口的 active fraction 都为 1.0。说明真正执行了有信号的模型反向与参数更新。

六张固定评估图片的平均 SigLIP cosine：

- 更新前：0.1462827。
- 更新后：0.1489434。

这只是该次短验证的固定样本结果，不是完整评测集，也不是画质优越性的证据。其目的主要是确认“真实文本→MOSS 采样→真实 VAE→真实奖励→ART→LoRA 梯度→多 worker 规约→checkpoint→固定评估”的完整链路。

可查看 [训练前画廊](../outputs/moss_siglip_3gpu/evaluation/step_00000000/index.html) 和 [训练后画廊](../outputs/moss_siglip_3gpu/evaluation/step_00000002/index.html)。

## 四卡默认配置与严格续训：已完成

依据 [机器可读最终报告](../reports/validation.json)，`configs/moss_lora_4gpu.json` 在物理 GPU 0/1/2/3 上完成默认路径，并成功严格恢复。

| 项目 | 已验证结果 |
| --- | --- |
| 生成与回放 | 256×256，10 个采样步，回放 `[0,2,4,6,8]`，每轮 4 个提示词，K=2 |
| LoRA | rank=4，alpha=4，attention q/k/v/o，4,128,768 个参数 |
| 完整图像奖励链 | 原生 JAX 文本/MOSS/VAE，独立 PyTorch CPU SigLIP2 cosine |
| 首次训练 | 2 次优化器更新，188.2119 秒，含加载与评估 |
| 严格恢复 | 第 2 步恢复到第 4 步，145.0531 秒，含加载与评估 |
| 恢复后状态 | Adam step=4，prompt cursor=8，rollout=2 |
| JAX 最大活跃分配 | 10,714,465,024 字节，9.9786GiB |
| `nvidia-smi` 保留显存观察 | 约 14625MiB，14.28GiB；不同于活跃分配口径 |
| 导出一致性 | 适配器与第 4 步检查点参数逐位一致 |
| 固定 seed 图像变化 | 八张图像全部变化；逐图像素 MAE 见 JSON 报告 |

四次更新均有有限、非零 LoRA 梯度及参数变化。固定八张图片的平均奖励为：step 0 `0.2326493`、step 2 `0.2327555`、step 4 `0.2371746`。这是四个示例提示词上的短验证，不能外推到一般画质收益。

直接产物：

- [latest.json](../outputs/moss_siglip_4gpu/latest.json)。
- 第 4 步完整检查点：`outputs/moss_siglip_4gpu/checkpoints/step_00000004`。
- [adapter.safetensors](../outputs/moss_siglip_4gpu/adapter.safetensors)，16,572,304 字节；SHA256 `13dc08e4eb1cf07b3e1f3fc377d12b0932f2113915d3f9c09a1d55fc555c4acb`。
- [训练前画廊](../outputs/moss_siglip_4gpu/evaluation/step_00000000/index.html) 与 [第 4 步画廊](../outputs/moss_siglip_4gpu/evaluation/step_00000004/index.html)。

四张卡可用时的现成续训命令：

```bash
cd $CODE_ROOT/moss-art-jax
bash scripts/train_4gpu.sh --max-updates 20 \
  --output outputs/moss_siglip_4gpu --resume outputs/moss_siglip_4gpu
```

该命令继续同一配置、同一资产与同一优化器状态；它不会重新从基模开始。代码、资产、配置或提示词改变后，身份检查可能拒绝严格续训，应按 README 的新实验适配器入口处理。

## CPU 与辅助验证

最新冻结代码的 57 项单元测试和 7 项端到端测试通过，共 64 项。其中包括 ART 代数和梯度、LoRA、数据与资产身份、奖励协议、检查点、分布式窗口规约和恢复等覆盖。CPU tiny 的成功不替代真实模型/GPU 验证。

另外已完成原生 JAX 文本预编码、单独 CPU SigLIP2 评分和真实 MOSS 单卡 toy latent 两次更新。toy 奖励的分数不具有图像质量含义。

VAE 初次默认 cuDNN autotune 曾选择约 73GiB workspace 的候选算法并陷入重试；运行时默认禁用该自动调优探索后，独立 GPU 验证和上述三卡真实训练均成功完成原生 VAE 解码。这是候选 workspace 问题，不是 VAE 本体需要 73GiB 的证据。
