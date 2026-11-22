# vLLM 离线 Qwen prefill

服务器工程为 `$CODE_ROOT/moss-art-jax`，通过 SSH 配置 `5070Ti` 连接。
预处理使用现有 `$CONDA_PREFIX/envs/vllm_env/bin/python`，版本为
vLLM **0.24.0**。JAX 训练继续使用已有 `jax_env`。预计算是独立任务，完成后退出；
训练进程只读取已经完成的条件缓存。

## MOSS 条件特征契约

MOSS-Image-Base 的文本编码器是 Qwen3.5-9B：32 层、hidden size 4096，
原始权重为 BF16。它需要 HF `output.last_hidden_state`，即最后一个 decoder
block 经过模型自身最终 RMSNorm 后的结果。HF `hidden_states[-1]` 的完整索引是
32；这里没有多个中间层拼接，也不把一个序列池化成单个向量。

每条条件保存为未 padding 的 **`[1, S, 4096]`**。`S` 包含整个提示词序列，
包括 Qwen 聊天壳、`<think>` 前缀和用户正文。若通过图文 helper 编码参考图，
视觉 token、视觉边界 token 与文本 token 也全部保留。默认缓存将模型输出按
BF16 位模式保存；读取后恢复为 FP32 数组，数值仍是原 BF16 特征。

纯文本模板与原始 MOSS 一致：

```text
<|im_start|>user
{prompt.strip()}<|im_end|>
<|im_start|>assistant
<think>
```

用户正文透传；本层不拆解 MOSS 自身的 `Prompt:` / `Negative Prompt:` 内容。
ART JSONL 另有 `negative_prompt` 字段时，仍将它作为独立条件编码。相同正/负文本
去重后只编码一次，所有 rollout 复用其缓存。超过 `data.context_length` 的记录
输出丢弃日志并写入 `dropped.jsonl`，不会截断正文。

## 并行与原生 token pooling

此前验证配置是 **TP2 × PP1 × DP1**，两卡做 tensor parallel。新独立预计算示例为
**DP2 × PP2 × TP1**：两个独立副本各占两卡，在每个副本内按前 16 / 后 16 个
语言层进行 pipeline parallel，每个 stage 不再做 tensor parallel。DP 调度按
待编码 token 数平衡分片；副本各写自己的任务、缓存和日志，完成后由父进程
校验、合并根 manifest，不让多个副本直接竞争写根 manifest。

`topology_aware=true` 通过 Linux sysfs 的真实 PCI 路径选择配对。当前服务器的
两组为 **[1,2]** 与 **[0,3]**，执行顺序为 `[1,2,0,3]`。这些是 nvidia-smi
物理编号，不是某个子进程内的 CUDA 编号。启动前将每组转换为 UUID 写入其
`CUDA_VISIBLE_DEVICES`，避免索引重排影响放置；`topology.json` 保存发现结果。
2026-10-24 的对应关系为：

| DP 副本 | PP stage | 物理 GPU | UUID |
|---|---:|---:|---|
| 0 | 0 | 1 | `GPU-01d02def-ca29-8064-8e1a-5f4b8b1af4a2` |
| 0 | 1 | 2 | `GPU-1c999d9f-7a7a-302b-4c34-e02fc600505d` |
| 1 | 0 | 0 | `GPU-578e9e73-3c51-0641-3377-d11103202567` |
| 1 | 1 | 3 | `GPU-41ed5a10-8cc7-1fa8-dc2e-d072b9717c33` |

PP>1 使用 vLLM 原生 pooling runner：

```python
LLM(
    runner="pooling",
    convert="embed",
    pooler_config=PoolerConfig(
        task="token_embed", tok_pooling_type="ALL", use_activation=False,
    ),
    hf_overrides={"head_dtype": "model"},
    dtype="bfloat16",
    tensor_parallel_size=1,
    pipeline_parallel_size=2,
    # 其余资源、长度与模型参数由 JSON 配置给出
)
llm.encode(
    requests,
    pooling_task="token_embed",
    pooling_params=PoolingParams(use_activation=False),
)
```

这里的 `ALL` 保留每个输入 token 的特征，不是取均值或末 token。
`use_activation=false` 关闭 embedding pooler 的额外 L2 normalize；Qwen 自身的
最终 RMSNorm 仍然执行。`head_dtype=model` 保持模型 BF16 输出。非末 stage
返回 `IntermediateTensors`，末 stage 完成原生 final norm 后执行 token pooling。
该路径没有占位采样、没有自回归 decode、没有 hidden-state connector 临时文件，
也不启用 extractor 的 PP AST 兼容 patch。

`integrations/vllm_plugin/vllm_moss_plugin/model.py` 仍注册
`MossQwen3_5ForConditionalGeneration`，复用 vLLM 原生 Qwen3.5 计算。原始 MOSS
权重的 `language_model.*` 映射为 vLLM 的 `language_model.model.*`。适配器创建
metadata 与符号链接组成的 model view，引用原 safetensors 与 tokenizer，
**不复制、不改写权重文件**。原编码器没有 `lm_head`，view 将其绑定到已有 token
embedding；原生 pooling 不调用该采样 head。

PP1 保留旧 `extract_hidden_states` 路径供兼容与对照。插件将 auxiliary layer 32
替换为 vLLM forward 已算出的最终归一化输出，再交给官方 connector 导出；
不在 CPU 补算 RMSNorm，也不重新对分别舍入过的 BF16 residual 求和/归一化。
该旧路径使用 `max_tokens=1` 做一次占位采样，丢弃该 token，不进行后续 decode
forward；`include_output_tokens=false` 保证导出只有原 prompt 的 `S` 个 token。

旧 connector 的物理 block 修复也保留：Qwen3.5 hybrid cache 会把逻辑 272-token
block 拆为实际 68-token kernel block。`MossHiddenStatesConnector` 从实际分配的
`[blocks, block_tokens, layers, hidden]` 张量取 block size，避免按逻辑 block 展开
实际 block ID。既有 478-token 单条测试中，模型输出、缓存写入与导出 BF16 字节
哈希完全一致。此项只涉及旧 PP1 extractor，原生 pooling 不依赖它。

两条路径均关闭 prefix caching 与 chunked prefill，保留完整 prompt 特征。
当前 adapter 仍统一预留 `context_length + 1` 的容量，要求
`max_num_batched_tokens >= context_length + 1`；PP>1 实际不产生额外输出 token。

## 独立 JSON 配置与入口

新示例：`configs/preparation/vllm_qwen_dp2_pp2.json`。关键字段如下：

```json
{
  "preparation": {
    "backend": "vllm",
    "auto_prepare": false,
    "async_prefetch": false,
    "vllm": {
      "python": "$CONDA_PREFIX/envs/vllm_env/bin/python",
      "devices": [0, 1, 2, 3],
      "tensor_parallel_size": 1,
      "pipeline_parallel_size": 2,
      "data_parallel_size": 2,
      "topology_aware": true,
      "storage_dtype": "bfloat16",
      "batch_size": 32,
      "max_num_seqs": 8,
      "max_num_batched_tokens": 8192,
      "gpu_memory_utilization": 0.68,
      "enforce_eager": false
    }
  },
  "data": {
    "cache": "$MOSS_ROOT/cache/qwen_dp2_pp2_bf16_20261024",
    "context_length": 512,
    "short_context_length": 288
  }
}
```

`batch_size` 是每个副本一次提交的条件数，`max_num_seqs` 是该副本内 vLLM 的并发
序列上限；它们不是 DiT 训练 batch size。`devices` 总数必须等于 DP × PP × TP。
`model_path` 留空时在每个副本缓存的 `.vllm-work/model-view` 创建 view；填写时
必须指向空目录或与当前 MOSS 源匹配的已有 view。`processor_dir` 默认使用
`$CODE_ROOT/moss-image/reference-lab/processor`。
`preparation.device/qwen_gpu_layers/qwen_gpu_embedding/qwen_gpu_vision` 属于旧 JAX
预处理；vLLM 路径由以上并行参数决定放置，不采用 CPU/GPU 奇偶层加载。

先只查看分组和资源计划，不读取提示词数据、不加载权重或初始化模型：

```bash
cd $CODE_ROOT/moss-art-jax
$CONDA_PREFIX/envs/vllm_env/bin/python scripts/prefill_qwen.py \
  --config configs/preparation/vllm_qwen_dp2_pp2.json --plan
```

实际执行离线预计算：

```bash
cd $CODE_ROOT/moss-art-jax
$CONDA_PREFIX/envs/vllm_env/bin/python scripts/prefill_qwen.py \
  --config configs/preparation/vllm_qwen_dp2_pp2.json
```

脚本会使用配置指定的独立 vLLM 解释器，外层不初始化 JAX GPU，**不会启动 ART
训练或 DiT/VAE**。旧合成 smoke 配置 `configs/preparation/vllm_qwen_prefill.json`
仍可用于 TP2/PP1 extractor 对照。当前独立 ART 缓存入口读取文本 JSONL；图文
helper 仍位于 `moss_art.vllm_qwen.VllmPrefill.encode_requests()`，其精度限制见下文。
SFT 目标图 VAE 编码入口保持独立。

## 缓存、断点与训练生命周期

缓存 encoder 标记为 `moss-vllm-qwen3.5`。`conditioning_contract` 记录 final
normalized、layer 32、hidden 4096、全部 prompt token、固定模板与 BF16 来源。
`storage_dtype=bfloat16` 使用 version 2 契约 `bfloat16_uint16`：NPZ 的
`embedding` 数组实际 dtype 为 `uint16`，保存 BF16 的原始位模式，**不是 INT16
数值量化**；训练 reader 用移位和 reinterpret 恢复 FP32 数组。该转换不再次运行
Qwen，不改变原 BF16 值，未压缩 payload 为 FP32 存储的一半。
`storage_dtype=float32` 继续支持旧 version 1 契约。

reader 严格校验契约、权重/tokenizer 身份、文件 SHA256、`[1,S,4096]` 形状、
对应的 FP32/uint16 存储类型、INT32 input IDs、有效 token 数及全 true attention
mask；非有限或全零特征会拒绝。旧 JAX 缓存仍可读取，但不会将 vLLM 输出标为旧
JAX encoder。

NPZ 原子写入，每批完成后更新 manifest。中断后重复同一个独立脚本命令，校验并
复用已提交条目，只编码缺失条件。DP 子缓存也支持校验恢复，父进程校验其契约、
producer 与条目后再合并。模型身份、模板、长度、存储契约或 producer 源码/版本
不一致会明确报错，应选择新的缓存目录。缓存齐备时，不再加载 Qwen。
全部任务完成后写入 `complete=true` 和 `preparation_report.json`。

**训练端绝不自动启动 vLLM 预处理。** 当 `preparation.backend=vllm` 时，训练
只接受已存在且已完成的缓存；缺少 manifest 或 `complete=false` 会立即报错，
要求先运行 `scripts/prefill_qwen.py`。即使误配 `auto_prepare=true`，此 backend
也不会因此启动 Qwen。`async_prefetch` 必须为 false。训练期间不持有 Qwen 权重、
Qwen KV cache、PP workers 或双缓冲 Qwen 进程；只读缓存特征，再按 short/full
策略与有效序列长度送入 DiT。

配套训练配置 `configs/art/silva205_P8_K16_T4_AB_mxfp8_vllm_offline_20261024.json`
已指向上述 BF16 缓存，并关闭自动预处理；其 rollout、训练与奖励参数保持原值。
本轮没有启动该训练。预处理完成后再用该配置启动训练即可。

纯采样示例使用两步入口；预处理脚本接受 preparation/ART schema，不能直接读取
rollout 专用 schema：

```bash
python scripts/prefill_qwen.py --config configs/preparation/vllm_qwen_dp2_pp2_smoke.json
$CONDA_PREFIX/envs/jax_env/bin/python -m moss_art.rollout_only --config configs/rollout/vllm_offline_example.json
```

第二步会启动 DiT 生图，本轮未运行。两份配置引用相同的 Base、合成数据、
BF16 缓存及 short128/full512 长度。

## 数值与速度记录

现有全量数据集的实际离线导出也已完成：205 条记录、402 份去重正负条件，
无丢弃；77,036 tokens，长度 15–428，BF16 NPZ 合计 504,096,725 字节。
冷启动、编译、写盘和最终校验共 267.73 秒。两副本各处理 201 份条件，
各 7 次 API batch；API 调用累计各约 12 秒（含首批额外编译，不含最终 NPZ
写盘），不能把它等同于全流程耗时。所有 205 条记录已通过训练 reader 校验，
且该 reader 路径未导入 JAX、Torch 或 vLLM。审计报告为上述 audit 目录中的
`full205_validation.json`。本轮最终检查为 87 项 host 测试及 4 项 Torch adapter
测试通过；其中真实 Linux 测试覆盖父进程退出后所属孙进程的清理。

2026-10-24 的四卡 DP2 × PP2 实机检查已通过：两个副本同时启动，2 条合成记录
导出 4 份去重条件，共 72 tokens；全量 IDs、形状、BF16 位模式文件与训练 reader
校验通过。每卡模型权重约 9.33 GiB。冷启动、编译及导出共 237.55 秒，不能据此
计算稳态吞吐；重复执行耗时 0.40 秒、`encoded_now=0`，没有加载模型。

单副本 PP2 原生 pooling 与旧 TP2 extractor 的这 4 份条件比较：全特征 cosine
0.9998383、相对 L2 0.017994；新 DP2 与单副本 PP2 比较 cosine 0.9998716、
相对 L2 0.016025。均非逐位一致；不同并行布局、batch 形状及 kernel 的影响未
单独隔离。短合成样本不能替代下述长文本及图文精度检查。报告目录：

```text
$MOSS_ROOT/audits/vllm_dp_pp_20261024/
```


以下保留此前 **TP2 × PP1 × DP1、extractor、FP32 存储** 的 HF 对齐与速度记录。
它们不能直接视为新 DP2 × PP2 原生 pooling 的验收结果。

此前验证仅使用独立合成输入，未执行用户全量数据预处理或启动训练。
正常自然语言和 Tag 串均逐 token 核对 IDs，保留全部 `[1,S,4096]` 特征。
下表比较整个特征张量；不是只比较末 token。

| 条件 | 有效 token | 对原始 HF 5.20 BF16 cosine | 相对 L2 | 对 HF FP32 cosine |
|---|---:|---:|---:|---:|
| 不重复自然语言 | 340 | 0.99987576 | 0.01576311 | 0.99994687 |
| Tag 串 | 267 | 0.99986245 | 0.01658543 | 0.99994245 |

两例通过事先设置的全局 cosine ≥ 0.999、相对 L2 ≤ 0.05 条件。
FP32 oracle 逐层加载原始 BF16 权重并提升为 FP32，关闭 TF32，使用 math SDPA。
报告目录：

```text
$MOSS_ROOT/audits/vllm_prefill_20261024/representative/numeric_validation/reference.json
$MOSS_ROOT/audits/vllm_prefill_20261024/representative/hf_fp32/reference.json
```

压力样本“同一句话重复 36 次”共 478 tokens，没有通过上述精度条件：
vLLM compiled 对 FP32 cosine 0.86871、相对 L2 0.51225；原始 HF BF16
也只有 0.86575 / 0.51814。逐层检查显示两条 BF16 路径均从约 0.35% 的
首层相对误差平滑累积，并非导出错位或某层突然异常。该结果提示重复输入的
BF16 数值敏感性，不能推广为所有文本精度保证，也不能将此样本标记为通过。

TP2、BF16、正常编译配置下，预热后的 478-token 测速：B1 约 0.072 秒，
B8 约 0.51 秒（约 15.6 条/秒、7.5k token/秒）。这是整个 `encode_token_lists`
调用，含 connector 临时落盘、读回及 FP32 NumPy 转换，不含最终 NPZ 压缩写入，
也不含 DiT/VAE。冷启动/编译另需约 93–102 秒，不能算入预热吞吐。

实际 CLI 已验证：2 条合成记录生成 4 份去重正/负条件；重复执行
`encoded_now=0` 且不创建 vLLM 引擎；`doctor` 确认训练 reader 可读取缓存。
87 项 host 测试、11 个 subtest 和 3 项 Torch adapter 单元测试通过。

图文 helper 的 token/shape 对齐已通过，**数值未通过上述验收**。带纹理合成图
共 111 tokens，与原始 HF BF16 的全特征 cosine 0.97085、相对 L2 0.24132。
5.13/5.20 processor 的 pixel values 和 grid 逐值一致，vLLM/HF 的三轴
M-RoPE positions 也逐值一致。单独视觉编码器输出 vLLM/HF BF16 cosine
0.997896、相对 L2 0.065064；两者各自对视觉 FP32 的相对误差分别为
0.085437 / 0.070266。逐层没有突然跳变，这支持 BF16 误差继续被语言模型
放大的解释，但不是完整正确性证明，不能据此把图文验收改为通过。

当前可用范围是本文的 **ART 文本 JSONL 离线缓存**；图文 helper 保留供实验，
SFT 的目标图 VAE 编码及原有图文预处理入口未切换到该 helper。
