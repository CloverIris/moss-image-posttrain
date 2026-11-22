# MOSS Image SFT (JAX)

Supervised LoRA flow matching, separate from ART. Supports mixed text-to-image
and reference-image-plus-text-to-image datasets. Frozen BF16 DiT, FP32 LoRA and
Adam state, cuDNN BF16 attention, and block recomputation with GPU-resident
activation checkpoints by default. Optional CPU offload and sparse checkpoints
are available. No FP8 or process-wide XLA optimization restrictions.

## Data

Each JSONL record has a target image and condition. Paths are relative to the
JSONL file or absolute. IDs are unique. Dimensions are positive multiples of 16;
examples may have different resolutions and aspect ratios.

    {"id":"text-001","prompt":"A cat wearing a red hat.","target_image":"targets/cat.png","width":1024,"height":1024}
    {"id":"edit-001","prompt":"Change the hat to blue; preserve the cat.","conditioning_image":"references/red_hat.png","target_image":"targets/blue_hat.png","width":1024,"height":1024}

Multiple references use "conditioning_images":["ref1.png","ref2.png"].
Do not specify both reference fields. Text-only records omit both fields.

The target goes only to the VAE, never the condition encoder unless explicitly
supplied as a reference. Editing needs real paired reference/instruction/target
examples. Our reconstruction smoke examples validate execution, not edit quality.

## Frozen offline preprocessing

Offline preprocessing uses the existing CPU Transformers processor and a native JAX
Qwen vision/language prefill child on physical GPU4. The vision tower is FP32,
language layers BF16. The child exits before the Torch FP32 VAE target encoder
loads. No Torch model is resident during DiT training; no download is needed.

    cd $CODE_ROOT/moss-art-jax
    $CONDA_PREFIX/envs/torch_env/bin/python \
      -m moss_art.sft_prepare --config configs/sft/prepare.json

Set manifest, cache, model, processor_dir and device in that JSON file.

References precede text in the Qwen chat template (enable_thinking=True).
We retain EVERY final hidden state: image positions, text, vision boundaries
and chat framing. This matches mm_all. No caption substitution or token pooling.
Multimodal positions and preprocessing are handled by the Qwen processor.

Default reference budget: 262144 pixels per image, max 4096 condition tokens.
Overlength rows are logged and dropped before prefill. Adjust
max_reference_pixels or max_tokens in the preparation JSON with memory cost in mind.

Targets are EXIF-corrected, RGB-converted, center-cropped/resized, and scaled to
[-1,1]. Flux2 VAE output is packed 2x2 and normalized with running BN statistics,
yielding [1,128,H/16,W/16]. Default posterior is deterministic mode.
posterior="sample", seed=42 in the preparation JSON caches one seeded posterior draw per row.
No online random crop or fresh posterior draw per epoch.

Caches hash image contents, configs, tokenizer, processor and preparation code.
Weight identity uses full header + path/size/mtime, not full payload hashes.
NPZ checksums, visual-token counts and complete sequence lengths are validated.
Completed caches are reusable. Changed inputs or incomplete preparation require
a new cache directory. Keep manifest.json together with its NPZ files.

## Training

Set data.cache in configs/sft/train_1024.json to your cache and choose a fresh output.
The production config expects cache/sft. The separate smoke configs use a two-example fixture, not real training data.

    $CONDA_PREFIX/envs/jax_env/bin/python -m moss_art.sft doctor --config configs/sft/train_1024.json
    $CONDA_PREFIX/envs/jax_env/bin/python -m moss_art.sft train --config configs/sft/train_1024.json

Equivalent: python run.py sft train ...

Global batch_size includes all workers/microbatches and must divide evenly over
data-parallel groups. A group contains one GPU by default, or two with TP enabled.
Each group defaults to microbatch_size=1; larger true microbatches are optional. Image resolutions may vary; context
lengths select short/full buffers: data.short_context_length (0 = auto P70)
and data.context_length (full, default 4096). See PROMPT_BUCKETS.md.
Layout: [target image tokens][real condition tokens][tail padding].
cuDNN query_seq_lengths/key_value_seq_lengths receive image_count + actual
condition length. The two condition preprocessing blocks receive actual condition
length alone. No dense GPU attention mask is allocated. RoPE preserves the real
condition coordinates, including the image f-coordinate based on actual length.
Text, multimodal, and dropped/empty conditions use the SAME buffer capacity;
their token lengths within a bucket do not create new DiT/condition/RoPE/VJP compilations.

For fixed model settings and context capacity, changing image resolution creates
a new shape specialization; prompt contents/lengths do not. Changing capacity,
dtype, backend, LoRA rank etc. can also trigger compilation. Conditions longer
than full are logged and removed before scheduling. Frozen JAX Qwen prefill also
uses short/full buffers; only valid hidden states are stored.

Frozen weights upload one tensor at a time, completing each transfer before
enqueuing the next, to bound CUDA host staging across workers. This avoids
retaining unnecessary pinned host memory. Compiler settings are unchanged.

Padding still consumes space/work in projections, residuals, MLPs and saved
activations. Choose a smaller fixed capacity for datasets known to fit it if
memory/throughput is more important than accommodating long conditions.
The old bounded-executable mechanism now tracks resolution/capacity pairs,
not each real prompt length. Four GPUs with batch_size=16 accumulate four examples
each. Gradient sums are divided once by global sample count, clipped, then
updated by AdamW on the coordinator. Only LoRA arrays cross the host boundary.
This is data parallelism, not FSDP.

    noise ~ N(0,I)
    sigma ~ sigmoid(N(logit_mean, logit_std))
    sigma = shift * sigma / (1 + (shift - 1) * sigma)
    noisy = (1 - sigma) * clean + sigma * noise
    target = noise - clean
    loss = mean((dit(noisy, sigma, condition) - target) ** 2)

Uniform sigma sampling is also available. Loss is unweighted velocity MSE,
averaged per example; large images do not receive extra sample weight.
No ART covariance, reward, CFG amplification, negative-layer skipping or
multi-step sampling is involved in training.
conditioning_dropout replaces the ENTIRE image+text condition with empty text;
default 0.1, or 0 for strict conditional smoke tests.

LoRA presets: attn or attn-mlp. Full base-model finetuning and encoder/VAE training
are not implemented.

## Resume and export

Set run.resume="/absolute/path/run/checkpoints/step_00000100" in the training JSON:

    $CONDA_PREFIX/envs/jax_env/bin/python -m moss_art.sft train --config configs/sft/train_1024.json

A run directory also resolves via latest.json. Checkpoints store optimizer and
sample cursor. Shuffle/noise/sigma/dropout are deterministic from global sample
position. Worker count may change if batch_size still divides. Changes to
model/data/objective/optimizer/batch-size/seed or numerical implementation
invalidate exact resume. max_updates and checkpoint cadence may change.

Every checkpoint exports adapter.safetensors in the existing Kohaku/LyCORIS
diffusion_model LoRA format. Start with an adapter and fresh optimizer:

Set run.init_adapter="/absolute/path/adapter.safetensors", run.resume="", and
runtime.output_dir="outputs/new-sft" in the JSON, then use the same train command.

## Inference

Prepare an inference JSONL containing prompt and optional references. Omit
target_image and set conditioning_only=true in the preparation JSON. Specify output
width/height per record and set data.cache to this inference cache.

    $CONDA_PREFIX/envs/jax_env/bin/python -m moss_art.sft sample --config configs/sft/sample_1024.json

Set runtime.devices, runtime.output_dir, run.sample_index, run.seed and optional
run.adapter in JSON. run.checkpoint can replace run.adapter. Leaving both paths
empty uses the unchanged base model.
Generation starts from random noise, uses all image/text context tokens, and
defaults to 50-step res_multistep/beta. Native JAX VAE decoding starts after DiT
workers are released. This is reference conditioning, not strength-based
img2img with a noised reference latent.

Outputs: latent NPY, PNG, metadata JSON. Sample index is cache order. Multiple
data-parallel groups generate successive seeds; the devices inside one TP group
cooperate on the same sample. Existing samples are not overwritten.

## Verification

tests/test_sft.py checks manifests, full-token preservation, corrupt caches,
flow targets, deterministic noise, condition-dependent gradients, checkpoints,
adapter export, exact resume and one-vs-two-worker reduction using a tiny real
DiT. Real 1024 smoke runs exercise both conditions. Successful updates are not
evidence of editing quality; evaluate that on held-out real paired data.


## Fixed prompt padding verification

tests/test_prompt_padding.py verifies valid-token RoPE, padded/unpadded outputs
and LoRA gradients, independent CFG branch lengths, rejection of truncation,
cuDNN prefix-length dispatch without dense masks, and no extra JIT entries when
only true prompt length changes. The native CPU reference uses an explicit mask
for verification; production cuDNN receives sequence lengths.

scripts/diagnostics/check_prompt_padding.py runs the same GPU output/VJP and
compilation-reuse checks and inspects compiled HLO for cuDNN calls in the DiT
forward, backward, and condition preprocessing blocks.

Real-model validation is recorded in reports/prompt_padding_validation.json.
With fixed capacity 4096, BF16 base, attention+MLP rank-4 LoRA, and a mixed
27-token text / 283-token image+text fixture at 1024x1024, four GPUs completed
two global-batch-4 updates. The second update took 8.39 s; maximum JAX-reported
peak allocation was 11.99 GiB per GPU (driver reservation is larger). This is
a two-update execution smoke test, not a sustained throughput benchmark.
The four-GPU checkpoint identity and exported adapter roundtrip were verified.

## GPU activation checkpoints and recomputation

The SFT default is now:

    model:
      activation_storage: gpu
      checkpoint_interval: 1

Each block input remains on GPU. Its internal attention/MLP intermediates are
recomputed by the block VJP, so this still uses activation recomputation. No
block checkpoint is transferred to CPU. Input conditions and LoRA gradient /
optimizer synchronization still cross the host boundary. The legacy ART model
constructor keeps its CPU-checkpoint default; this change selects GPU storage
in the SFT worker.

For less GPU memory, set checkpoint_interval: 6. Forward retains every sixth
executed block input; backward reconstructs one segment on GPU, then performs
its block VJPs and releases temporary segment inputs. Skipped CFG layers do
not count toward segment length. This costs additional forward computation.
To restore the former storage policy, use activation_storage: cpu with interval 1.
The interval is positive, and checkpoint policies are covered by resume identity.

At 1024x1024, fixed 4096 prompt capacity, BF16 base, rank-4 attention+MLP LoRA:

| Storage / interval | Warm single-sample gradient call | Peak JAX allocation |
| --- | ---: | ---: |
| CPU / 1 | 7.68 s | 11.99 GiB |
| GPU / 1 | 6.57 s | 13.91 GiB |
| GPU / 6 | 7.84 s | 12.54 GiB |

These are fresh-process tests on three GB203(SM120) cards, averaging the second
and third calls with the same initial adapter and sample seeds. They exclude
optimizer updates. Four-GPU training with GPU / 1 completed three global-batch-4
updates: warm update times 7.27 and 7.21 s, maximum allocation 13.91 GiB per card.
Driver reservation is larger. These short smoke runs are not sustained workload
benchmarks. Larger resolutions, contexts, ranks, or simultaneous CFG tapes can
need sparse checkpoints or CPU storage. Per-GPU microbatch remains 1; increasing
global batch uses accumulation. This is data parallelism, not FSDP; no FSDP
implementation or speed claim is made by these results.

See reports/gpu_activation_checkpoints.json and
scripts/diagnostics/benchmark_activation_storage.py. The --parity option repeats
one real-model job across checkpoint modes in the same process, including a
CPU/CPU repeat to measure the existing BF16/cuDNN backward variability. CPU
FP32 tests independently compare sparse-checkpoint gradients with full autodiff.

## Two-device hidden tensor parallelism (TP2)

TP2 is available in SFT training and sampling. This partitions hidden channels,
attention heads, MLP intermediates, and the main DiT matrices across two GPUs.
It is tensor parallelism, not FSDP parameter gathering and not sequence-axis
parallelism. The 42 main blocks retain GPU checkpoints with half of the hidden
channels per device. Text preprocessing, time embedding and final-head parameters
are replicated; FP32 LoRA masters and their complete reduced gradients are also
replicated within a pair. The coordinator retains the CPU optimizer.

    runtime:
      devices: [0, 3, 1, 2]
      tensor_parallel_size: 2
    model:
      activation_storage: gpu
      checkpoint_interval: 1
    training:
      batch_size: 4

This creates physical groups [0,3] and [1,2] on 5070Ti_REMOTE. Each group handles
two samples sequentially per update for global batch 4. A single pair can use
devices: [0,3] or [1,2]. Config indices refer to nvidia-smi indices; workers resolve
them to GPU UUIDs before CUDA initializes, preserving physical order independently
of CUDA_DEVICE_ORDER. Physical device order determines pairing. TP size must be 1 or 2; device count
must be divisible by TP size; global batch must be divisible by group count.
The local microbatch defaults to one sample per pair; training.microbatch_size
can batch multiple same-resolution examples in a single forward/backward.
Training uses the same global
mean gradient as before; it does NOT divide again by the number of TP devices.

In MOSS, each card stores 1536 of 3072 checkpoint channels and executes 12 of
24 attention heads. Head dimension remains 128, and cuDNN Flash Attention is
retained in forward/backward. The Q/K/V projections share one activation gather;
normalization and row-parallel linear layers use XLA collective communication.
FFN up/gate weights shard their input axis to preserve the fused weight layout;
the resulting hidden and gate activations shard the FFN axis. No process-wide
XLA optimization flags or deterministic-kernel restrictions are added.

Start with the dedicated config (set data.cache and runtime.output_dir first):

    $CONDA_PREFIX/envs/jax_env/bin/python -m moss_art.sft train --config configs/sft/train_1024_tp2.json

Execution fixtures are configs/sft/smoke_1024_tp2.json (global batch 2) and
configs/sft/smoke_1024_tp2_b4.json (global batch 4). They use synthetic paired
examples and are not an image-quality training set. The regular train_1024.json
keeps TP disabled. On these PCIe-connected GB203(SM120) cards, TP reduces per-device
memory but did not improve throughput over four independent GPUs. Measurements
and validation are in reports/tensor_parallel_validation.json.

Full-shape LoRA export stays compatible with existing single-device inference.
Changing TP size invalidates exact training resume (numerical reduction order
changes); initialize from adapter.safetensors for a fresh optimizer instead.
An unchanged TP configuration can resume normally. In sampling, each pair gets
one seed/result, and VAE decoding uses that pair's first physical GPU after DiT
workers exit. Existing fixed prompt capacity and dynamic cuDNN lengths apply
unchanged, including independent CFG positive/negative lengths.

Tests cover TP vs single-device outputs/LoRA gradients, CFG/skip layers,
segmented GPU checkpoints, per-device weight and hidden shapes, compilation
reuse across prompt lengths, complete training and adapter export, and batch /
group validation. scripts/diagnostics/check_tensor_parallel.py --gpu inspects
actual partitioned cuDNN calls in both forward and backward. CPU test mode uses
two virtual CPU devices, solely for numerical correctness tests.

### Real-model numerical limits

TP2 remains opt-in. The tiny FP32 model matches full autodiff, and the tiny BF16
GPU check has approximately 2e-7 relative LoRA gradient error. These results do
not establish full-model numerical equivalence. For three fixed real 1024 MOSS
jobs at the same initial LoRA, the retained mixed row/column TP implementation
had loss absolute differences of about 6.2e-6, 2.35e-5 and 9.11e-5, but gradient
relative L2 differences of 7.0%, 13.0% and 31.9% (cosines 0.9976, 0.9916, 0.9481).
TP changes BF16 kernel shapes and reduction order. The complete gradient vectors,
not only their norms, were compared. No held-out training-quality equivalence
has been established, and these differences must not be described as bitwise or
near-exact full-model parity. The standard TP1 config remains available.

The original cross-switch TP2 groups [0,1] / [2,3] measured global-batch-4 warm
updates of 16.32 and 16.24 s with peak JAX
allocation about 7.65 GiB per card, versus the earlier TP1 four-worker measurements
of 7.27 and 7.21 s / 13.91 GiB. TP2 is a memory option on this hardware, not a
throughput improvement. These are short smoke measurements. A faster all-column
layout was investigated and rejected because its real-model gradient deviations
were larger. The shipped implementation retains the normal autodiff path and
mixed row/column layout; it does not install an experimental custom VJP.


### PCIe pairing on 5070Ti_REMOTE

The checked-in TP2 configs use [0,3,1,2]. PCI sysfs ancestry and lspci show
GPU 0 (08:00.0) and GPU 3 (1d:00.0) on the first PEX880xx switch, and
GPU 1 (10:00.0) and GPU 2 (14:00.0) on the cascaded second switch. Internal
PEX bridges are omitted from this simplified view:

    CPU -- PCIe 4.0 x8 -- switch 1 -- GPU 0 / GPU 3
                            |
                      PCIe 4.0 x16
                            |
                         switch 2 -- GPU 1 / GPU 2

All four GPU endpoint links negotiate PCIe 4.0 x16 under load. The common
CPU-facing bridge 01:00.0 currently negotiates x8, despite an x16 capability.
The switch-to-switch link at 07:10.0 / 09:00.0 negotiates x16. Read/write P2P
capability is reported OK by nvidia-smi. Its topo -m output labels all these
GPU pairs PXB, which does not distinguish the two switch domains.

Pairing [0,1] and [2,3] places both TP groups across the inter-switch uplink.
Pairing [0,3] and [1,2] keeps each group's repeated hidden-state collectives
within its switch. Host optimizer/gradient transfers still use the common CPU
uplink. This is a host-specific configuration, not a universal ordinal rule.
After hardware or enumeration changes, inspect the actual PCI ancestry again:

    PYTHONPATH=. $CONDA_PREFIX/envs/jax_env/bin/python scripts/diagnostics/inspect_gpu_topology.py

The diagnostic saves PCI ancestry/link state, nvidia-smi topology, P2P capability,
UUIDs, and a CUDA-driver verification of the requested physical order in
reports/pcie_topology.json. It is read-only and does not change PCIe settings.
Comparative 1024 training results are in reports/pcie_topology_training.json.

Matched four-update 1024/global-batch-4 smoke runs, excluding each first update:

| Physical TP pairs | Warm seconds/update (3 updates) | Images/s | Peak JAX GiB/card |
| --- | --- | --- | --- |
| [0,1] / [2,3], cross-switch | 16.10 | 0.248 | 7.65 |
| [0,3] / [1,2], within-switch | 13.49 | 0.296 | 7.65 |

Within-switch pairing reduced warm update time by 16.2% and increased
throughput by 19.4%. An earlier local-pair run averaged
13.50 s/update. NCCL logs verify the actual 08/1d and 10/14 GPU
pairs using P2P/direct pointer transport. These are short synthetic performance
checks; they do not establish training-quality equivalence. TP1 remains faster
in the previous four-worker benchmark, while TP2 uses much less memory.


### True SFT microbatches

`training.batch_size` counts all examples in an optimizer update across GPU
pairs. `training.microbatch_size` caps examples processed together by each pair.
For two TP2 pairs, batch_size=4 and microbatch_size=2 processes two examples
per pair in one forward/backward, instead of two sequential calls.

Each example retains its seed, independently sampled noise, sigma, conditioning
dropout, valid prompt length, and original RoPE coordinates. Text-only and
image+text examples can share a microbatch. Loss and gradients sum per-example
pixel means; the coordinator divides exactly once by the global example count.
Partial microbatches have the same weighting as full microbatches. Consecutive
jobs with different target resolutions split a batch without reordering samples.

The maximum defaults to 1. Fixed microbatch size, resolution, and context capacity
reuse compiled kernels across dynamic timesteps and prompt lengths. Changing
microbatch size (including a partial tail) requires another compiled shape.
Larger global batch alone increases accumulation and does not imply concurrent
batch execution. GPU activation checkpoints, optional segment recomputation,
and explicit cuDNN attention apply to every supported microbatch size.

Measured 1024 results on four GB203(SM120) cards (same-switch TP2 groups). Each
successful run used four updates; averages below exclude the first update.

| Microbatch per pair | Global batch | Checkpoint interval | Seconds/update | Images/s | Peak JAX GiB/card |
| --- | --- | --- | --- | --- | --- |
| 1 | 4 | 1 | 13.50 | 0.296 | 7.65 |
| 2 | 4 | 1 | 13.28 | 0.301 | 10.04 |
| 1 | 6 | 1 | 20.03 | 0.300 | 7.66 |
| 3 | 6 | 1 | 19.60 | 0.306 | 12.65 |
| 4 | 8 | 2 | 29.50 | 0.271 | 13.20 |

At the same global batch 6, microbatch 3 improves throughput by only 2.2%
while raising peak JAX allocation from about 7.66 to 12.65 GiB per card.
Microbatch 4 with interval 1 fails in backward with OOM; interval 2 fits but
extra replay makes it slower per image. Total nvidia-smi memory also includes
allocator reservations and runtime memory, and can approach 15 GiB with
microbatch 2. These short synthetic measurements do not establish training
quality. BF16 kernel/reduction differences still apply.

The regular configs keep microbatch_size=1. To opt into the modest gain, use
configs/sft/train_1024_tp2_mb3.json (set data.cache and output_dir first):

    $CONDA_PREFIX/envs/jax_env/bin/python -m moss_art.sft train --config configs/sft/train_1024_tp2_mb3.json

The essential settings are:

    runtime: {devices: [0, 3, 1, 2], tensor_parallel_size: 2}
    model: {activation_storage: gpu, checkpoint_interval: 1}
    training: {batch_size: 6, microbatch_size: 3}

Details, exact configs, all update metrics, OOM result and regression-test
summary are in reports/sft_microbatch_1024.json.


## MXFP8 冻结底座 + LoRA

本实现参考 KohakuMossTrainer 的 MXFP8 冻结权重与独立转置副本设计，使用原生 JAX
`jax.nn.scaled_matmul`，CUDA 上实际执行 cuDNN `blockScaledDot`。
仅转换 42 个 DiT block 的六种投影，共 252 个线性层；时间/文本嵌入、归一化、最终输出层
保持原有精度。attention 仍是 BF16 cuDNN，LoRA 因子与优化器主参数仍是 FP32。

在 SFT JSON 的 model 中设置：

```json
{
  "base_precision": "mxfp8",
  "mxfp8_dgrad": "bf16",
  "dtype": "bfloat16",
  "attention_backend": "cudnn",
  "activation_storage": "gpu",
  "checkpoint_interval": 1
}
```

冻结投影的权重和输入在前向中均使用 E4M3 + 每 32 个 contraction 元素一个 E8M0 scale。
权重加载时量化一次，不保留完整 BF16 投影副本。默认 `mxfp8_dgrad=bf16` 在反向时仅临时
解量化当前层计算输入梯度，不计算冻结底座的权重梯度。这使冻结模型存储从约 9.47 GiB
降为约 5.18 GiB。没有设置限制 XLA 优化的全局标志。

可选 `mxfp8_dgrad=mxfp8` 使用独立量化的 W 转置和动态量化的输入梯度，也走 cuDNN
block-scaled GEMM。转置副本必须按新的 contraction 方向重新分组，不能直接转置前向
scale。保存两个方向会使底座存储略高于 BF16，适合专门比较速度/数值，不是默认省显存方案。

当前支持每卡完整模型的 DP（`tensor_parallel_size=1`）；MXFP8 + TP2 显式拒绝。
CPU 路径仅作为 tiny-model 数值参考，生产 MOSS MXFP8 要求 CUDA Blackwell。
ART 的在线 rollout 和训练默认不变；MXFP8 选项在 SFT 和独立 rollout-only 配置中开放。

### 完整链路验证配置

```bash
cd $CODE_ROOT/moss-art-jax
$CONDA_PREFIX/envs/torch_env/bin/python -m moss_art.sft_prepare --config configs/diagnostics/mxfp8_e2e_prepare.json
$CONDA_PREFIX/envs/jax_env/bin/python run.py sft train --config configs/sft/mxfp8_e2e_train.json
$CONDA_PREFIX/envs/jax_env/bin/python run.py sft train --config configs/sft/mxfp8_e2e_resume.json
$CONDA_PREFIX/envs/jax_env/bin/python run.py sft sample --config configs/sft/mxfp8_e2e_sample_0.json
```

示例是服务器上的验证夹具：文生图、单参考图、双参考图三条记录，目标图 1024×1024；
预处理在 GPU4 使用 JAX Qwen 保存全部视觉和文本 tokens，再用 FP32 VAE 预编码目标 latent。
short/full 为 512/1024，超长过滤规则沿用现有流程。生产数据应替换 JSONL 路径、缓存与
输出目录，并根据 tokenizer 扫描结果选择桶长度。

训练示例在四张 GB203(SM120) 上 DP，先两步，再从 checkpoint 继续到第四步。sample_0/1/2
分别读取三种条件，使用 BF16 底模加载导出的 safetensors LoRA，以 50-step
res_multistep + beta 采样，验证适配器兼容性。样本少、步数少，只证明链路可用，不能据此
认定图生图能力或训练质量已收敛。输出已存在时需使用新的输出目录。

配置、缓存身份和代码版本参与 checkpoint 校验；更改训练精度后不能当作同一次精确续训。
导出 LoRA 使用标准 diffusion_model.*.lora_down/up.weight 键，可加载到相同底模的 BF16 路径。

验证报告：`reports/mxfp8_sft_e2e_validation.json`。MX 量化与 Kohaku 逐字节一致，
GPU 前向及输入梯度 HLO 证据保存在 `reports/mxfp8_*forward.hlo.txt` / `*backward.hlo.txt`。
可独立复现核心数值检查：

```bash
$CONDA_PREFIX/envs/jax_env/bin/python scripts/diagnostics/check_mxfp8_kernels.py --config configs/diagnostics/mxfp8_kernels.json
```

该诊断配置使用空闲 GPU3，并通过本地 Kohaku 项目生成参考数据；运行前可修改 JSON 的 device。


## Muon 优化器（SFT 与 DiffusionART 共用）

在训练 JSON 中增加 `optimizer`。不指定 `name` 时仍用 AdamW，已有 JSON 无需修改。

```json
{
  "optimizer": {
    "name": "muon",
    "learning_rate": 0.00001,
    "momentum": 0.95,
    "nesterov": true,
    "ns_steps": 5,
    "rms_match": 0.2,
    "weight_decay": 0.0,
    "max_grad_norm": 1.0,
    "betas": [0.9, 0.999],
    "epsilon": 1e-8
  }
}
```

`betas` / `epsilon` 仅作用于 AdamW 分组；Muon 正交化的归一化 epsilon 固定为 1e-7。
参考 `$HOME/KohakuMossTrainer/src/kohakumoss/training/optim/optimizers/muon.py`
及 `adapters/train/groups.py`：二维且两边均大于 1 的 LoRA down/up 因子走 Muon；
rank=1 因子、向量和 scale/alpha 参数走 AdamW。独立更新 down/up，不构造完整 delta-W，
也不更新冻结的 BF16/MXFP8 底座。LoRA 因子不按底模 fused 投影拆块，与参考 adapter 分组一致。

四卡 DP 先对梯度求全局平均，然后全局裁剪，再执行一次优化器更新。
Muon 使用 `v = momentum*v + g`，Nesterov 为 `u = g + momentum*v`；
五次 Newton–Schulz 后乘 `rms_match * sqrt(max(rows, cols))`，采用解耦 weight decay。
正交化会先把长边放到列方向，LoRA 的 Gram 矩阵仅为 rank×rank。
参数、动量、正交化全部为 FP32，在 CPU coordinator 上运行。
参考项目仅正交化用 BF16，因此两者公式一致但不宣称逐位相同。
矩阵正交化按形状 JIT 缓存；没有添加任何限制 XLA 的全局标志。

配置示例的 1e-5 用于链路验证；底模采用 Muon 并不意味着它的完整权重学习率可直接用于 LoRA。
优化器类型及全部参数绑定 checkpoint 身份，续训恢复动量/Adam 状态。
从 AdamW 切换 Muon 应以 `run.init_adapter` 开始新输出目录，不能 `run.resume` 复用另一种优化器状态。

1024×1024、四卡 DP、MXFP8 底座、BF16 dgrad 的验证配置：

```bash
cd $CODE_ROOT/moss-art-jax
$CONDA_PREFIX/envs/jax_env/bin/python run.py sft train --config configs/sft/muon_mxfp8_e2e_train.json
$CONDA_PREFIX/envs/jax_env/bin/python run.py sft train --config configs/sft/muon_mxfp8_e2e_resume.json
```

这两个配置复用 MXFP8 完整链路测试的三条缓存（文生图、单图参考、双图参考），
先训练两步，再恢复到第四步。重跑请换 output_dir，同时调整 resume 路径。
`metrics.jsonl` 记录 `optimizer` 和 `optimizer_seconds`，`run_info.json` 保存完整优化器配置。

本次验证通过 412 个测试和 74 个子测试。四卡共训练 4 步并从第 2 步恢复，
504 个 LoRA 因子均使用 Muon，252 个 down 与 252 个 up 在恢复后均有更新。
每卡峰值约 7.87～8.13 GiB，进程内第二步 Muon 更新约 0.17～0.23 秒；
这是短测且同时运行 CPU 测试套件，不作为独立吞吐基准。
导出适配器已加载回 BF16 底模，完成 1024×1024、50-step res_multistep + beta 生图：

```bash
$CONDA_PREFIX/envs/jax_env/bin/python run.py sft sample --config configs/sft/muon_mxfp8_e2e_sample.json
```

详细记录：`reports/muon_sft_validation.json`。仅为链路验证，不据此判断收敛或画质提升。
