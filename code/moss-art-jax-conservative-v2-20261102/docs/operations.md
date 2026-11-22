# 运行、资源与排错

## 配置与资源

配置相对路径相对于工程工作目录解析，`scripts/run.sh` 会先切换到工程根目录。根配置会与 `moss_art/config.py` 默认值递归合并；未知字段直接报错，避免拼写错误被忽略。输出保存合并后的完整配置。

主干为约 5B 参数的 BF16 完整副本，单卡权重约 9.5GiB；FP32 全参数训练不在这个实现范围。LoRA 与优化器为 FP32，优化器只在 CPU coordinator 保存一份。中间激活在块边界卸载到 host，反向时逐块重算，所以还需要足够的系统内存、内存带宽与 PCIe 带宽。

同时满足以下条件才适合启动四卡配置：

1. 物理 GPU 0、1、2、3 均可用，且不在运行其他 agent 的任务。
2. 文本缓存覆盖本次所有正面和负面提示词。
3. MOSS、SigLIP2 和相应 Python 依赖都在配置的本地路径。
4. `prompts_per_rollout` 可被卡数整除，每组 `group_size` 为偶数。

GPU 预检按 `nvidia-smi` 报告的已用显存判断，默认阈值 1500MiB。它不会认领 GPU，也不能阻止另一任务随后启动；仍需与其他任务协调。工程不会自动杀死任何外部任务。CPU 线程预算也应结合并发任务调整。

## 显存不足

优先减小分辨率、提示词长度、LoRA rank 或训练目标范围，保持 `attn` 起步。减少 `train_timesteps` 可减少训练计算与 host rollout 缓存；因为逐条回放，它不必然按比例降低一次 block VJP 的显存峰值。减小 K 会减少 rollout 数量，但 K 必须为偶数且至少为 2。

默认 VAE 在各自 worker 所在 GPU 上解码，此时冻结 DiT 仍驻留。逐图解码减小峰值，但高分辨率时可能不足。可在新实验配置中设置：

```yaml
reward:
  cpu_decode: true
```

这会把原生 JAX VAE 放到 coordinator 的 CPU backend 执行，训练仍在 GPU。CPU 解码通常更慢；不是 PyTorch 回退。修改此字段影响实验配置指纹，不能直接对旧检查点使用严格 `run.resume`，可用适配器开始新实验。

`XLA_PYTHON_CLIENT_PREALLOCATE=false` 避免启动时预留整块显存，不保证分配器缓存立即归还操作系统，也不能保证任意分辨率都不会 OOM。初次 JIT 比稳定运行更慢，必须以进程完成、日志和产物判断结果。

运行时和诊断入口不注入或覆写 `XLA_FLAGS`、`TF_FP16_CONV_USE_FP32_COMPUTE`，不禁用 JIT、优化 pass 或全局 autotune；未设置时使用 XLA 默认，调用者显式设置的值原样保留。此前为处理 VAE 调优中的超大分配而全局关闭 autotune，影响范围过大，现已移除。GPU 可见性、分配器预分配/显存预算、CPU 线程与亲和性继续用于资源调度，`JAX_ENABLE_X64=true` 保留数值类型支持；它们不关闭 XLA 优化 pass。

原生 XLA 默认调优已在 1024 测试中完成：调优/编译期间超大临时分配被拒绝后恢复，最终 executable 临时内存约 1.875GiB。原生 `xla` 和显式 workspace 上限的 `cudnn_bounded` 均可选；后者的上限只作用于该 FFI 卷积，不约束其他 XLA 图。当前结果与历史基线区别见 [恢复 XLA 默认优化](xla_defaults.md)。

## Attention 和矩阵乘法后端

生产 CUDA 配置明确选择：

```yaml
model:
  attention_backend: cudnn
  dit_autotune_level: 4
```

`cudnn` 要求 CUDA、BF16 主干；无法支持的形状或依赖会直接报错。`native` 是原 XLA attention 对照，并带 query chunk 重计算；不会自动在二者之间切换。DiT 的 forward/VJP 可显式使用局部 autotune 4。配置值 0 表示不传局部 compiler_options、继承 XLA 默认或调用者环境，并不关闭调优；项目没有全局 autotune 覆写。首次遇到新的文本长度会有额外编译和调优成本，后续复用 JIT/persistent cache。

已完成单张 GB203(SM120) 上 1024×1024、10 步、CFG=3、rank=4 的真实 rollout、VAE、SigLIP 和有效 ART timestep 的反向容量测试和完整两窗口训练。JAX 活跃分配峰值约 11.66GiB；分配器在 `nvidia-smi` 中仍会保留约 14.28GiB。这个结果不覆盖任意长提示词、高 rank 或长时间 1024 训练。原始 native attention 反向曾接近分配上限，加入 chunk 重计算后通过，因此本次无需两卡 FSDP。具体时间和新 FlashAttention 结果分别见 [1024 容量](rollout_1024.md)、[速度与后端证据](cudnn_attention.md)。

更换 attention backend、autotune level 或相关源码会改变实验身份；使用旧适配器作为新实验起点，不能修改旧 checkpoint 身份来绕过检查。

## 检查点与资产身份

本工程保存的是 LoRA、优化器状态与数据/随机流计数，不复制完整 MOSS 权重。完整 checkpoint 使用 SHA256 检查 payload 和 manifest，并用 `COMMITTED` 标记原子提交；未完成或被篡改的文件不作为有效恢复点。

每个实验身份包含：

- 完整配置指纹；仅排除输出路径、允许占用开关和总更新预算。
- 提示词 JSONL 的完整字节 SHA256。
- JAX 模型、ART、LoRA、coordinator、缓存、奖励与 checkpoint 相关实现源码 SHA256。
- transformer/VAE 配置、索引文件和各 safetensors 权重身份。
- 文本缓存 manifest 的完整 SHA256；manifest 再绑定 encoder/tokenizer 与每条特征文件。
- 内置 SigLIP 的配置、processor、tokenizer 及权重身份。

大权重默认记录的是“完整 safetensors header 的 SHA256 + 文件字节数 + 纳秒修改时间 + 解析后的路径”，**不是完整权重 payload 的密码学哈希**。它检测正常的同路径替换、修改、路径变化，不宣称检测刻意保留上述元数据的恶意改写。文本 prepare 另有 `--full-weight-hash`，用于需要完整 encoder 权重校验的场景。

资产或源文件改变后不会自动更新旧 checkpoint 的身份，也不会自动复制正在被其他 agent 修改的源码。需要精确恢复就保留对应代码与资产；需要改变实验则导入独立 LoRA 并重新开始优化器。

同一个输出目录只有一个写入者，使用 Linux advisory lock。`.run.lock` 文件在正常退出后仍可能存在，这是有意保留 inode；文件存在本身不代表任务仍在运行。不要通过删除锁文件绕过正在工作的进程。

## 固定评估与训练日志

评估使用固定提示词和 seed，便于在同一后端、相同设置下比较检查点。它并不自动提供独立训练集划分：示例提示词同时用于训练与固定评估。正式实验应另备未训练提示词，使用 `--adapter` 和新的评估配置评估泛化。

`reward_mean` 上升只能说明选定奖励器分数上升。检查 PNG，记录失败样例，不要从几次优化器更新推断全面画质提升。mask 比例长期接近 1 表示当前更新受漂移限制；奖励标准差长期为零可能表示奖励器饱和、提示词/评分不匹配或重复样本。不要只看 loss 下降判断训练有效。

短测试可以前台运行；较长实验可由使用者显式启动后台任务：

```bash
mkdir -p logs
nohup bash scripts/train_4gpu.sh --output outputs/my_long_run \
  > logs/my_long_run.log 2>&1 &
```

记录 shell 返回的 PID，并只管理自己启动的任务。重用非空运行目录需要 `run.resume`；否则使用新的输出目录。`complete.json` 仅标识正常完成，一次失败后保留的完整 checkpoint 仍可用于恢复。

## 常见错误

| 现象 | 原因与处理 |
| --- | --- |
| `GPU ... is busy` | 有其他任务或显示占用超过阈值；选择空闲设备，或等待已有任务结束。 |
| `prompts_per_rollout must divide evenly` | 每轮 prompt 数不能被 worker 数整除；复制配置并调整 prompt 数。 |
| `Prompt not present in cache` | 缺少正面或负面文本；用同一配置重新运行 `prepare-text` 补齐。 |
| `fingerprint mismatch` / `incompatible resume field` | 模型、源码、缓存、数据或配置发生变化；不要篡改身份绕过校验。 |
| `Output already has an active writer` | 另一进程在写同一输出目录；使用不同目录或等待。 |
| `Output history is newer than checkpoint` | 不能把旧 checkpoint 写回较新的历史；在新输出目录恢复。 |
| `Selected training step is absent` | sigma 网格吸附去重使选择索引越界；缩小 train_timesteps。 |
| `Reward subprocess failed` | 检查异常中的独立奖励进程 stderr、解释器、模型路径与 CPU 依赖。 |
| 连续 `no_signal` 后停止 | 奖励无法区分当前组，或 endpoint 协方差为零；检查真实分数及图像。 |

## 测试命令

```bash
bash scripts/setup.sh --check-only
bash scripts/test_cpu.sh
```

测试脚本隐藏 GPU，使用 CPU backend。它不代表真实 5B/真实 GPU 验证。查看 `reports/` 中具体运行报告、时间、设备、奖励类型及是否确实完成更新；没有对应报告的配置按未实测处理。
