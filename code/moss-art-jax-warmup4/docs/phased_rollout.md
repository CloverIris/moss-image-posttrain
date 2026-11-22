# Phased GPU lifecycle

Rollout-only now completes **all** DiT waves for the input file before starting VAE. Final FP32 latents are transferred to host per image. ART selected-step latents and old velocities transfer to host at each selected step; no full trajectory accumulates on GPU.

For ART: DiT rollout -> export host replay/LoRA/text state -> terminate DiT GPU processes -> VAE-only decode -> release VAE processes -> reward -> restore DiT from the shared CPU BF16 base -> two LoRA update windows. Optimizer state stays in the coordinator. The base is one read-only shared-memory allocation for all DiT workers, not four CPU copies. It remains allocated across ART phase changes. Process exit releases the JAX GPU allocator; deleting Python references alone is insufficient.

Rollout configuration additions:

```json
{"rollout": {"cache_intermediates": true, "host_cache_limit_gib": 16.0}}
```

Final latents always queue in CPU RAM until decoding. By default `cache_intermediates: true` also caches **all actual sampler steps**, in FP32, and write per-image `*-trajectory.npz` containing `states`, `old` (raw CFG velocity), `selected`, `sigmas`. This is sampler state, not DiT layer activations. Set `save_latents: true` to additionally retain final `.npy` latents. Set `cache_intermediates: false` for final-latent-only inference. ART always keeps its configured training timesteps.

The cache limit checks payload size before starting DiT, and raises rather than silently spilling to disk or truncating records. The limit does not include shared base weights, prompt embeddings, IPC copies, or temporary arrays: allow additional RAM headroom. For 1792x768, 8 images, 50 steps with states and velocities: payload 2,224,029,696 bytes (~2.07 GiB). Shared CPU BF16 weights need ~9.5 GiB in `/dev/shm`.

VAE loads once in each VAE worker, reuses weights and JIT across waves, then exits. No process-wide XLA compiler/precision flags are introduced. The phase boundary changes memory ownership only; seeds, sampler, arithmetic precision and prompt forwarding are unchanged. Switching phases has process/load overhead; benchmark complete rounds instead of attributing this overhead to denoising speed.

Validation (2026-10-21): 4x5070Ti, MXFP8 DiT/cuDNN attention, FP32 bounded-cuDNN VAE, 1792x768, 8 images in two waves, Euler/MOSS shift5, 50 steps. Finished in 152.73s; DiT waves 66.96s + 66.49s. All 8 trajectory archives finite. Seeds42-45 PNG pixels exactly equal to the previous interleaved implementation. Separate 4-step memory probe saw GPU0-3 return to 266/15/15/15 MiB before VAE started. VAE peak live JAX arrays 2,857,726,976 bytes (~2.66 GiB) per worker; allocator pool ~4.25 GiB, not whole-process memory. ART restart/replay gradients verified bitwise on the CPU tiny-model integration test, not a full 5B GPU training benchmark.
