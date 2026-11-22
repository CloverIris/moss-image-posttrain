# SILVA aesthetic reward (native JAX)

This experiment replaces PixAI TAG coverage with the calibrated SILVA score.
The pinned v3.0.0+20261018 release uses SigLIP2 SO400M patch14/384 raw MAP-pooled
features, an FP32 ordinal MLP, and the original 512-knot latent calibration LUT.
Do not normalize embeddings or replace the calibrated score with the raw ordinal score.

The reference is $CODE_ROOT/moss-aesthetic-judge/runtime/score_silva_cpu.py.
Weights stay local; only vision_model.* tensors from the backbone are uploaded.
No Torch/Transformers imports, model download, text tower, or domain-reference tensor
is needed by native JAX scoring. The default precision is FP32/native attention;
BF16/cuDNN remains optional because the measured score drift is larger.

SigLIP preprocessing reproduces the installed Transformers 5.20.1 Torchvision
CPU uint8 bilinear antialias resize, including int16 coefficient quantization and
per-pass rounding, followed by (pixel-127.5)/127.5. PIL bilinear is slightly different.
The implementation follows the algorithm in PyTorch 2.11 UpSampleKernel.cpp:
https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cpu/UpSampleKernel.cpp
Tests use hashes of deterministic synthetic inputs scored by the CPU reference.

## Execution

runtime.execution_backend=mesh: one JAX controller for GPUs 0-3 throughout
DiT rollout -> release DiT -> VAE -> release VAE -> SILVA -> release SILVA -> DiT replay.
All three phases use a JAX Mesh and NamedSharding: weights replicated (P()),
activations sharded along batch (P("dp", ...)). There is no Python thread pool,
phase queue, or per-GPU worker process in this path. JAX/CUDA still own their
internal execution threads. GPU 4 retains the independent asynchronous Qwen
producer and the existing two-slot shared-memory buffer.

VAE uses shard_map so each local convolution sees batch one. Its bounded cuDNN
FFI plans are prebuilt on all devices; axis_index selects the matching plan inside
the graph. The native library does not allocate or build plans while executing.

reward.batch_size is PER GPU. SILVA's global scoring batch is batch_size*4,
with last-batch repeat padding removed from returned results. VAE uses one image
per GPU at a time. These phase microbatches do not change the logical ART batch.

## Current experiment

205 JSONL records, explicit prompt and negative_prompt preserved. Original dataset
and source images are never rewritten. P=8, K=8, T=8 => 64 images / 512 points per
optimizer update, accumulated with per-GPU rollout/training microbatch one.
48 Euler steps with the existing MOSS shift=5 schedule. Cached step indices remain
[0,6,13,20,26,33,40,47]; mapped sigmas are intentionally nonuniform. Resolution profile weights
are 768x768:1024x1024=2:1; prompt text fixes its resolution bucket. Update profile
selection is deterministic weighted sampling, so the ratio is statistical rather
than an exact three-update cycle. Qwen short/full capacities are 288/512 tokens;
all 205 records fit, short covers 73.66%.

A fresh Base + attention LoRA(rank=48, alpha=48) run is used. Old TAG RL
checkpoints/images are preserved. Checkpoints every update (keep two), images
saved for every rollout, TensorBoard rollout/eval calibrated reward and raw score.

Audit scripts and numerical results:
$MOSS_ROOT/audits/silva_reward/
