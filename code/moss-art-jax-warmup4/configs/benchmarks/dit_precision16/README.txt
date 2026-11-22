MOSS DiT resident precision benchmark, 2026-10-21
1024x1024,16 Euler/MOSS steps,CFG3,rank32/alpha64,microbatch1.
Cached Qwen features:positive27 tokens,negative10,padded512.
Two full warmup images excluded; measured seeds42-45.
PhaseA:BF16 GPU0/2,MXFP8 GPU1/3; phaseB swapped.
16 measured images per precision. All8 runs passed;32 endpoints finite.
MXFP8 quantizes252 block projections; attention stays BF16 cuDNN.

Mean seconds/image: BF16=24.2941356414; MXFP8=14.9464787997.
Speedup=1.6254086308x; latency reduction=38.4770%.
Interior GPU busy=100% for both; GPU busy is not Tensor Core utilization or MFU.
JAX peak10.3232GiB vs6.17674GiB covers entire process including loading/warmup.
No VAE/Qwen/reward/backward is instantiated.
Timer includes sampler/host dispatch,cached condition pad/upload,and final latent D2H/finiteness checks.

Run one selected config on the server:
cd $CODE_ROOT/moss-art-jax
$CONDA_PREFIX/envs/jax_env/bin/python -u scripts/benchmark_dit_precision.py --config configs/benchmarks/dit_precision16/a_bf16_gpu0.json
Use an unused output_dir to repeat. Ensure device is free.
Raw NVML CSV and per-run JSON are bundled; latents excluded from ZIP.
