# GPU4 Qwen 与四张GB203(SM120)的双缓冲流水线

维护中的 ART 训练配置（例如 `configs/moss_lora_4gpu.json`、`configs/moss_http_4gpu.json`）已开启：

```json
{
  "preparation": {
    "device": 4,
    "async_prefetch": true
  },
  "runtime": {"devices": [0, 1, 2, 3]}
}
```

命令不变：

```bash
cd $CODE_ROOT/moss-art-jax
$CONDA_PREFIX/envs/jax_env/bin/python run.py train --config configs/moss_lora_4gpu.json
```

这是 ART 在线 rollout 的路径。SFT 已准备的条件缓存继续直接复用。
`async_prefetch=false` 保留离线整库预计算行为；此时 `auto_prepare` 决定是否在训练前准备缓存。
在线模式先做 CPU tokenizer 长度扫描、过滤超长条目并固定 short/full，然后启动独立 Qwen
进程。GPU4 与 DiT devices 必须互不重叠。物理GPU编号仍通过 UUID 绑定。

## 两个槽位的生命周期

```text
槽 A：Qwen 填入 batch 0 → READY → 计算卡复制并确认 → 可复用
槽 B：                           Qwen 填入 batch 1 → READY → 交接
GB203(SM120)：                       rollout 0 → reward → update 0/1 → rollout 1
GB206(SM120)：                       prefill 1 ────────────────────→ prefill 2
```

特征数据放在两个固定容量的 POSIX 共享内存区，没有 multiprocessing.Queue、任务队列
或通过 Pipe 序列化 embedding。控制只使用共享索引/长度/状态、Condition/Event。
DiT 原有的模型指令和梯度控制 Pipe 保留，传给它的条件消息只有共享内存描述符和小元数据。

这是 **CPU 共享内存双缓冲**，不是宣称跨GPU直接 P2P/CUDA IPC 零拷贝：GPU4 输出到 host
缓冲，计算卡获得自己的条件副本，再上传到本卡。host缓冲大小固定为：

`2 slots × prompts_per_rollout × 2 CFG branches × full × hidden_size × 4 bytes`

例如4个prompt、full4096、hidden4096，总预留约1GiB；只写有效 token 区域。
启动时检查 `/dev/shm` 空间，关闭后 unlink 两个共享内存区。

状态为 `FREE → REQUESTED → WRITING → READY → READING → FREE`，每次附加递增批次编号。
所有计算卡确认私有副本复制完成后，槽位才会释放。后续 rollout 和两次梯度回放使用
自己的数组，不受生产者下一次写入影响。旧票据、重复释放、批次不匹配、生产者异常或
超时都显式报错，不会读取未完成或过期的特征。

## 缓存与恢复

同一正/负提示词按内容哈希缓存。Qwen 仅编码尚未出现过的条件；空负提示词也只编码一次。
一份特征供同组所有生成seed、后续轮次以及两次ART更新复用。Qwen子进程在整个运行期间
存活，模型权重仍沿用host mmap/逐层streaming；这里的常驻进程不代表全部8B权重驻留显存。

在线缓存会逐步增长，因此 checkpoint 绑定编码器权重/tokenizer/实现、训练配置和数据集，
不绑定不断变化的cache manifest字节。缓存每个特征仍校验checksum，编码器或桶配置改变
必须使用兼容的新缓存。缓存被删除后也可按需重建，训练恢复不依赖已预取到哪一批。

预取不推进训练游标、随机种子或optimizer step。游标仅在当前真实rollout完成后推进。
无信号rollout、周期评估、断点恢复都经过同一交接逻辑。评估占用一个空闲槽位，不覆盖
已经预取的下一批训练条件。退出时取消未消费的预取并回收进程/共享内存。

## 延迟与验证

第一批必须等待 ready，可以与DiT模型加载重叠。之后只有当下一批prefill比当前计算窗口
更慢时才发生交接等待，因此异步不保证任意长度/负载下都完全隐藏延迟。
Qwen日志输出 `qwen_buffer_ready`（包含真实编码数、缓存命中和开始/结束时间）；
训练目录 `prefetch.jsonl` 记录 acquire 等待与 DiT rollout 区间，可直接核对重叠。

CPU测试覆盖双缓冲所有权、槽位复用与旧代拒绝、生产者崩溃/超时、copy ACK、缓存命中、
在线/离线训练数值一致、两卡分发、含评估的精确恢复以及无信号分支。
单卡 `moss_rollout*` / `moss_gradient*` 诊断配置仍采用离线条件，避免把异步工作混入单卡性能测量。
真实四卡1024/50-step集成配置：`configs/examples/async_prefetch_1024.json`。
该配置使用 toy_latent 奖励检验执行与并行，不代表图像质量提升。
