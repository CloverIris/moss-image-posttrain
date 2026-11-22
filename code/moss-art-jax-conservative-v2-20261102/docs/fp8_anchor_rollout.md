# Three rollout modes and same-version ART anchors

Configuration entry: `sampling.rollout_mode`.

| Mode | Rollout | Learner | ART old reference |
|---|---|---|---|
| `bf16` | BF16 base + FP32 additive LoRA | BF16 | saved BF16 rollout velocity |
| `mxfp8` | MXFP8 eligible frozen block projections + FP32 additive LoRA | BF16 | saved MXFP8 rollout velocity |
| `mxfp8_calibrated` | same MXFP8 rollout | BF16 | same-version learner-compute velocity |

MXFP8 modes currently require native CUDA Mesh. There is no silent kernel fallback. Qwen, VAE, SILVA reward and LoRA optimizer precision are unchanged. Calibration is inspired by removing computation-path drift; it does not restore a BF16 trajectory distribution or establish a quality improvement.

## Two ART windows

The previous logical-batch planner forced `updates_per_rollout=1` and merged A/B. That forced override has been removed. The three matched experiment configs explicitly use `updates_per_rollout=2`:

- 8 prompts, 8 trajectories each, 8 selected training points = 512 points per collection.
- Complete K=8 endpoint/reward groups produce the existing ART direction before any split.
- Split each prompt into 4 A trajectories and 4 B trajectories. All points of a trajectory follow it.
- A accumulates 256 points globally, then applies at most one optimizer step.
- B accumulates the other 256 points globally, then applies at most one optimizer step.
- Same learning rate, optimizer, sigma weighting, group aggregation, sampler and corrective movement metric in all modes.

`prompts_per_update` remains the legacy configuration name for prompts per rollout COLLECTION. It does not mean fresh prompts for each A/B window. `max_updates`, checkpoint/eval intervals and the `update` axis count scheduled window attempts; `optimizer_updates` counts actual optimizer calls. Normally both counts agree. A globally empty/masked window advances the schedule but preserves both parameters and AdamW state, and logs `optimizer_step_applied=false`. Real valid rows remain in the loss denominator even if masked/inactive; finite padding does not.

A legacy one-window experiment is still available explicitly for raw modes. Calibrated mode requires two windows. Checkpoint and evaluation intervals must be even for two-window training. Changing the mode/source identity is a new experiment, not an in-place resume of old frozen code.

## Calibrated execution order

1. With theta0 LoRA, load original base weights into the configured rollout precision. Quantization is stateless, E4M3/E8M0 in blocks of32; weight scales depend on frozen base and activation scales on each call.
2. Generate all MXFP8 trajectories, keeping actual model-input state, original model velocity, selected sigma and scheduler coefficient. Preserve final endpoint separately.
3. Release DiT for VAE/reward as before. Aggregate complete prompt groups and split A/B.
4. Rebuild BF16 learner from ORIGINAL shared host BF16 weights, never by dequantizing MXFP8. Verify replay source parameter version.
5. Before any optimizer update, cache all B old predictions using `predict_reference`, which shares the exact learner forward, CFG, skip layers, padding lengths and microbatch layout.
6. A uses `old = stop_gradient(current(theta0))`. Current remains differentiable. One update produces theta1.
7. B uses frozen cached old(theta0) and current(theta1). Its old reference is not recomputed after A. One update produces theta2.
8. Next collection synchronizes MXFP8 sampler from theta2 LoRA.

For velocity convention `v=dz/dsigma`, `x0=z-sigma*v`, the target is `old-G` and residual `(current-old)+G`. Target and movement mask share the same old. In calibrated paths, stale `old_ode_mean` cannot override it. Existing sampler coefficients are retained, including ResMultistep history coefficients; the movement quantity is a scheduler-reference proxy, not a measured true transition KL.

No second full model is retained on GPU. B references live on CPU as immutable FP32 arrays. For P8/K8/T8, 256 outputs take288MiB at768 and512MiB at1024. A reuses learner forward; B adds one learner forward per B point (4 extra CFG prediction evaluations per trajectory averaged over8 selected points). This is not 4 extra full diffusion trajectories.

## Presets

`configs/art/silva_anchor_bf16.json`
`configs/art/silva_anchor_mxfp8.json`
`configs/art/silva_anchor_mxfp8_calibrated.json`

All use48 Euler steps, MOSS schedule shift5, training indices0,6,13,20,26,33,40,47; P8/K8/T8, rank48/alpha48, AdamW1e-4, SILVA reward, 768:1024 weights2:1. They start from the same Base initialization. Outputs are separate under `$MOSS_ROOT/experiments/fp8_anchor_20261023/`. These production-length presets are prepared, not automatically launched.

Run from an NVMe working directory so JAX disk cache is also on NVMe. Example:

```bash
cd $MOSS_ROOT/experiments/fp8_anchor_20261023
MOSS_CUDNN_CONV_BUILD=$CODE_ROOT/moss-art-jax/outputs/native/build.json \
$CONDA_PREFIX/envs/jax_env/bin/python $CODE_ROOT/moss-art-jax/run.py train \
  --config $CODE_ROOT/moss-art-jax/configs/art/silva_anchor_mxfp8_calibrated.json
```

## Diagnostics and verification

Per-collection `anchor_diagnostics/rollout_*.json` records reference rows/calls, host bytes, seconds, source version and per-sigma `e/G` RMS ratio and cosine on the common pre-mask active set. TensorBoard `anchor/*` contains summary metrics. New training runs also write existing `reward_norm/*` metrics and per-image normalization records inline in the controller, without a separate recorder process. All three preset logs use the existing TensorBoard server root. These diagnose direct output-path mismatch and are not a quality score.

CPU tests cover self-anchor nonzero gradient, frozen B and restoring force, target/mask consistency, original denominator, all-empty AdamW state, metadata version checks, two-window Mesh execution and mixed-resolution prefetch. Real GPU diagnostic command/config and measured results are documented in the audit report after execution.

Online quality comparison must fork the same checkpoint and optimizer state, keep prompts/seeds and budgets aligned, allow each branch its own subsequent replay, and evaluate fixed held-out prompts with actual FP8 inference. Compare both equal rollout/update budgets and equal GPU time. CPU toy tests and one frozen-replay gradient probe cannot establish superior reward or image quality.

## Measured 768 diagnostic

The four-GPU checkpoint6 fixed-replay diagnostic completed successfully. B32 references took15.772s and36MiBhostmemory;Agradient differed17.65% withcosine0.984426. Bothbranches performedtwofiniteupdates;anchorchecksumsremainedidentical. First-usecompilationwasincluded. Noheld-out qualitygainhasbeenmeasured. Fullreport: `$MOSS_ROOT/audits/fp8_anchor/real768_probe/report.json`;validation: `$MOSS_ROOT/audits/fp8_anchor/VALIDATION.md`.
