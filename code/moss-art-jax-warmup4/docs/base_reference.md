# Frozen base output regularization

`art.reference_mse_beta` adds a fixed-base constraint alongside ART. A value of
zero disables it completely; old configuration files and checkpoint fingerprints
remain compatible. The recommended opt-in configuration starts with `0.01` as a
weak trial coefficient. Its strength has not been tuned on a full RL run.

```json
"art": {
  "noise_level": 0.8,
  "kl_threshold": 0.00001,
  "eps": 0.00001,
  "feedback": "scalar",
  "reference_mse_beta": 0.01
}
```

The ready-to-use training configuration is
`configs/art/silva_base_reference_mse.json`. It keeps the previous calibrated
MXFP8 rollout and sampler settings, with a separate run name, output directory
and TensorBoard directory so previous data is not mixed into the new run.

```bash
cd $CODE_ROOT/moss-art-jax
$CONDA_PREFIX/envs/jax_env/bin/python run.py train \
  --config configs/art/silva_base_reference_mse.json
```

For each saved training point `(state, sigma, conditioning)`, compute the frozen
base velocity with LoRA disabled and with the same CFG branches, skip layers and
padding lengths as the learner. Both use the BF16 learner base, even when rollout
uses MXFP8. A tiny CPU model can retain its configured float32 test weights.

The objective is `ART + beta * mean((v_current - stop_gradient(v_base))**2)`.
The feature mean includes latent pixels and channels. Sum the per-item losses
and gradients and divide by the original global window size, excluding padding
but retaining genuine ART-masked and zero-reward-signal samples. The base term
is independent of ART's directional mask. A collection with no reward signal
still performs the base regularization updates when beta is positive.

This is a velocity-MSE proxy for preserving the original model's behavior, not
an exact KL between final image distributions. The reference is always the
loaded base without any training or initial LoRA. It does not mean a frozen
copy of a nonzero initial adapter.

`MossModel.predict_base` shares the existing frozen weight buffers, passes an
empty adapter mapping to the original linear implementation, records no
backward tape, and detaches the result. Live LoRA factors are never cleared or
mutated. There is no second 5B model and no zero-filled LoRA copy. Base forwards
run before learner activation checkpoints are retained; each microbatch's base
prediction is consumed immediately rather than stored for a whole collection.
The two ART windows are disjoint, so every selected point needs one extra base
forward, not an extra full rollout. Base forwards include the same CFG cost.

The moving BF16 `theta0` reference in `mxfp8_calibrated` remains independent:
it is still prepared for window B before the window A update. The fixed base
term does not replace ART's target, guidance, corrective mask or rollout reward.

TensorBoard and `metrics.jsonl` now contain:

| Tag (under `train/`) | Meaning |
| --- | --- |
| `loss` | ART plus weighted base regularizer |
| `art_loss` | ART contribution after its own directional masking |
| `base_reference_mse` | Unweighted prediction drift from the frozen base |
| `base_reference_rms` | Square root of the mean drift |
| `base_reference_loss` | `reference_mse_beta * base_reference_mse` |
| `base_reference_beta` | Configured coefficient |
| `base_reference_seconds` | Extra base-forward time, including cold compilation |
| `policy_active_fraction` | Fraction retained by ART alone |
| `active_fraction` | Fraction retained by either ART or base regularization |

Mesh timing is elapsed time for base forwards; the legacy worker backend sums
worker base-forward times, so that sum is not the total job wall time. Loss and
gradient reductions retain per-item vectors inside the Mesh JIT and do not add
a scalar cross-device collective at that boundary.

Validation uses synthetic prompts and tiny models, without reading user prompt
files or generated images. Run CPU integration checks with:

```bash
cd $CODE_ROOT/moss-art-jax
JAX_PLATFORMS=cpu JAX_NUM_CPU_DEVICES=4 \
  $CONDA_PREFIX/envs/jax_env/bin/python -m pytest -q \
  tests/test_base_reference.py tests/test_mesh_loss_stability.py \
  tests/test_anchor_loss.py tests/test_anchor_integration.py
```

Existing TensorBoard servers discover the new scalar tags automatically when
a new regularized training run writes events; no TensorBoard package upgrade
is needed. Past events are not rewritten or assigned invented base-drift values.
