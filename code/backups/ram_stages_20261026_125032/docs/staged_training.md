# Two-stage on-policy ART

The ready-to-use configuration is `configs/art/silva_staged.json`:

```bash
cd $CODE_ROOT/moss-art-jax
$CONDA_PREFIX/envs/jax_env/bin/python run.py train \
  --config configs/art/silva_staged.json
```

`staging.enabled=true` and `staging.phase=auto` run a lightweight controller
which launches exactly one collection or training process at a time. Each
compute process retains the existing single-controller JAX Mesh. There are
no extra per-GPU training workers or an inter-stage message queue.

1. The collection process computes one normal on-policy batch using the
   current LoRA. Qwen prefill, DiT rollout, VAE decoding and reward scoring
   belong to this phase. It saves rollout images as before. It atomically
   persists selected states, raw old velocities, scheduler coefficients,
   shared conditioning, rewards, full-group covariance statistics, the two
   disjoint windows, current LoRA and optimizer state. It then exits.
2. A fresh training process reads only this committed batch. It loads the
   BF16 DiT, adapters and optimizer state without starting Qwen, VAE, reward
   models or their prefetch service. In calibrated mode it prepares the
   moving BF16 theta0 references before window A. It applies the original
   A/B updates and the independent frozen-base regularizer. It commits a
   checkpoint, publishes metrics, then exits.
3. The controller starts the next collection from the completed checkpoint.

This is not offline RL over a whole dataset of stale trajectories. Every
collection is consumed exactly once by the same two updates as the original
implementation. Group statistics, timestep selection, padding masks, loss
denominators, LoRA factors and optimizer momentum are preserved. The CPU
parity test compares complete adapter and optimizer trees after both windows.

The stage cache uses JSON plus uncompressed numeric NPY files and the existing
pickle-free learner checkpoint. Shared/equal conditioning arrays are stored
once, verified by checksums, and mapped read-only in the learner. A final
`COMMITTED` marker is required; partial writes cannot be trained. Cache and
learner snapshots bind source, model, dataset and configuration identities.
The learner hashes the original data for identity verification but does not
parse prompts, tokenize, prefill or load image-reward weights.

Both windows must finish before committing the final checkpoint and publishing
their metrics. A failure during window B can restart from the unchanged batch
and learner snapshot. A completed checkpoint identifies its stage cache, so
repeating the same training request does not apply the updates twice. The
auto controller also supports continuing additional collections from latest.

Execution options (excluded from the algorithm fingerprint) are:

```json
"staging": {
  "enabled": true,
  "phase": "auto",
  "bundle": "",
  "keep_consumed_cache": false
}
```

`phase=collect` stops after producing one committed batch. `phase=train`
requires `bundle` to name that cache directory and consumes it in a fresh
process. The controller writes concrete phase JSON files into
`<output>/staging/collect.json` and `train.json`. They can also be launched
manually with the same `run.py train --config ...` interface.

The recommended config uses its own output and TensorBoard directories. Its
sampler, rollout count, selected points and weak base-MSE regularizer match
the previous recommended configuration. Periodic extra evaluation is disabled
(`evaluate_every=0`); collection rewards and exported images remain enabled.
Completed collections always checkpoint to support the next process. The
configured checkpoint retention still applies. With `keep_consumed_cache=false`,
only a verified, successfully consumed cache directly under this run's
`stage_cache/` is retired. Images, reward records, metrics and checkpoints are
not part of that cache. Failed or unconsumed batches remain recoverable.

Timing records are written to:

- `<output>/staging/process_times.jsonl`: complete child-process wall times,
  including interpreter startup and shutdown.
- `<output>/staging/last_train.json`: import/setup, verified cache reads,
  learner-state reads, BF16 model loading, calibration, first optimizer update
  and checkpoint times.
- TensorBoard `staging/*`: these component timings, alongside existing
  `train/*`, `rollout/*` and base-reference diagnostics.

`startup_ready_seconds` ends once the BF16 learner and optimizer are loaded.
It excludes the subsequent calibration and gradient compilation/computation.
`ready_for_gradient_seconds` additionally includes the B-window calibration.
`first_update_seconds` also includes the entire first update window, whose
duration depends on prompt count, K, T, resolution and microbatch size.
These are not interchangeable startup measurements.

Persistent XLA executable cache behavior and optimizer/autotune policy are
unchanged. OS file cache helps read the same base weights after collection;
the persistent JAX cache can reuse matching compiled shapes in new processes.
First-use compilation for a new resolution/conditioning shape still costs
time and is reported separately from cached restarts.

The diagnostic `scripts/diagnostics/benchmark_staged.py --config ...` uses
real MOSS weights, a 48-step MXFP8 rollout, synthetic cached conditioning and
synthetic numeric rewards. It never starts Qwen, VAE or a reward model and
does not export images. It is a performance/mechanics measurement, not an
assessment of generation quality or reward improvement.

## Measured cached restarts (2026-10-26)

Four GB203(SM120) devices, real MOSS base, MXFP8 48-step rollout, BF16 training,
LoRA rank/alpha 48/48, base-reference beta 0.01, 288-token positive condition,
one prompt, four trajectories, four selected points. Each training stage runs
both eight-item windows. Weight file pages are warm after collection. The
"first" runs are first use of the training kernels for these tested shapes;
they are not a purge of every pre-existing persistent cache entry.

| Resolution | Learner ready, cached process | Ready for gradients including calibration | First window complete, cached | Complete training process, cached |
| --- | ---: | ---: | ---: | ---: |
| 768x768 | 6.90 s | 10.32 s | 25.83 s | 40.46 s |
| 1024x1024 | 6.92 s | 11.85 s | 35.45 s | 58.14 s |

Verified replay reads cost 0.019/0.049 s; learner snapshot reads cost
0.647/0.648 s; BF16 model loading costs 5.884/5.875 s for 768/1024. First-use
training processes take 98.37/133.03 s in total, versus 40.46/58.14 s when
reusing compiled shapes. Startup does not mean completing an optimizer step.
Calibration and update duration scale with the full batch's training points;
the small diagnostic is not a throughput forecast for P8 x K16 x T4.
