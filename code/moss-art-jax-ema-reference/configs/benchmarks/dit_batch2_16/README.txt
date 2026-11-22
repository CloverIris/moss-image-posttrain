MXFP8 real microbatch1 vs2 benchmark,2026-10-21
1024x1024,16 Euler/MOSS steps,CFG3,rank32/alpha64,prompt512.
True simultaneous B2 tensors:[2,128,64,64],not serial group_size2.
No VAE/Qwen/reward/backward; cold compile plus2 full warmup batches excluded.
Each physical5070Ti measured with both batch sizes;16 measured images per size.

B1:14.9559613154 seconds/batch=image;K4=59.8238452615 seconds.
B2:32.9039466394 seconds/batch(two images),16.4519733197 seconds/image;K4=65.8078932787 seconds.
B2 throughput0.9090679291x B1,down9.0932%;image latency up10.0028%.
JAX lifetime active allocation peak:B1 6.17963GiB,B2 6.73694GiB.
Interior GPU busy100% both;this is not MFU or TensorCore occupancy.

Same-seed final latents differ substantially between batch shapes(relativeL2 .280-.357).
No seed/prompt/time/RoPE mapping error found.
Short probes locate tiny differences in the low-precision linear/norm path that grow through the model.
Image quality equivalence was not evaluated.Do not interpret finite outputs as bitwise equivalence.
Production NativeWorker rollout remains serial;new script is diagnostic,not production batching integration.

Run on a free server GPU,using an unused output_dir:
cd $CODE_ROOT/moss-art-jax
$CONDA_PREFIX/envs/jax_env/bin/python -u scripts/benchmark_dit_batch.py --config configs/benchmarks/dit_batch2_16/a_b2_gpu1.json
