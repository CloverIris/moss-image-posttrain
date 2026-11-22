# RAM-backed infer / eval / train

Run on `5070Ti_REMOTE`:

```bash
cd $CODE_ROOT/moss-art-jax
bash scripts/run_stages.sh --config configs/art/silva_staged_mxfp8_offline_fast.json
```

The shell runs one process at a time: `scripts/infer.py`, `scripts/eval.py`,
then `scripts/train.py`. Each takes only `--config experiment.json`. Phase
JSONs, phase durations and the controller lock live in the durable output's
`staging/controller/`. `MOSS_PYTHON` can select another interpreter; the default
is `$CONDA_PREFIX/envs/jax_env/bin/python`.

1. Infer runs Qwen, DiT and VAE, saves images on NVMe, commits the selected
   FP32 states, raw rollout velocities, endpoints and deduplicated prompt
   conditions on tmpfs, then exits. It does not load a reward model.
2. Eval reads a small checksummed request with ordered image paths and prompt
   metadata. It does not read trajectory arrays, construct a DiT, run Qwen,
   or initialize the learner. It writes scalar scores on tmpfs and a durable
   copy under `<output>/rewards/`. The synthetic `toy_latent` fixture is the
   sole exception: its reward uses endpoints instead of rendered images.
3. Train loads the replay and scores, rebuilds ART covariance and the same
   two prompt-balanced windows, then runs the existing BF16 learner. In
   `mxfp8_calibrated` mode B-window BF16 theta0 references are still computed
   before A. Train commits a durable LoRA/Adam checkpoint and metrics before
   removing the consumed tmpfs bundle. The next infer uses that checkpoint.

The evaluator currently supports the existing reward backends. `command`
receives `{version:1,items:[{index,prompt,image,...}],config:{...}}`, with
absolute paths to the already saved PNGs, and returns `{scores:[...]}`.
This is the extension point for future multi-model evaluation; this change
does not silently select or combine the four planned reward models.

The staged configs now specify:

```json
{
  "staging": {
    "enabled": true,
    "phase": "auto",
    "bundle": "",
    "cache_dir": "/dev/shm/moss-art-jax/<user>",
    "min_free_cache_gib": 1.0,
    "keep_consumed_cache": false
  }
}
```

`cache_dir` is the scratch root. A hash of the absolute output directory
isolates each run. The serializer checks shape-based capacity before writing;
an atomic rename and `COMMITTED` marker publish complete bundles. Checksums
bind the evaluation request and scores to the exact replay. A crash leaves a
committed bundle available for retry without applying updates twice. If the
machine reboots or tmpfs contents disappear, restart the same shell command:
the missing handoff is recollected from the latest durable checkpoint. No
mid-window checkpoint is published. Temporary files left by SIGKILL before
commit are not treated as a valid handoff; they can be removed when no stage
is running. No tmpfs remount or global VM/XLA tuning is performed.

`/dev/shm` is Linux tmpfs: it grows on demand and shares physical RAM with
the processes. Its advertised 23.3 GiB capacity is not preallocated. Linux
may swap tmpfs pages under memory pressure. Model weights, exported images,
checkpoints, TensorBoard data, the persistent Qwen/XLA/MXFP8 caches and logs
remain on their configured durable filesystems.

At P8/K16/T4, replay plus full-size independent CFG conditions is at most
about 1.39 GiB (768) / 2.38 GiB (1024), before small headers. The current
implementation additionally includes a ~567 MiB immutable LoRA/Adam snapshot
in the bundle, about 1.94 / 2.93 GiB total. It deliberately keeps this snapshot
for robust restart isolation; it does not copy the frozen base model. Train
may allocate temporary reshards and derived statistics in process memory.
Images are separate on NVMe, with an RGB8 planning budget of 216 / 384 MiB
per 128-image collection; actual PNG sizes depend on content.

Legacy `run.py train` with `staging.phase=auto` remains a two-stage
collection+reward / train controller, and also respects `cache_dir`. Use the
shell above for three separate stages.
