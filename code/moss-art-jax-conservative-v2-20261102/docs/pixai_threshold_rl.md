# Thresholded PixAI TAG reward and ART pilot

Run on the server in `$CODE_ROOT/moss-art-jax`, using `jax_env`. Runtime settings stay in JSON; no global XLA flags are added.

## Preprocess the file

```json
{"input":"inputs/pixai_threshold_rl_20261021/prompts.txt","output":"data/pixai_threshold_rl_20261021.jsonl","report":"inputs/pixai_threshold_rl_20261021/preprocess_report.json","metadata_dir":"vendor/pixai_tagger/metadata"}
```

```bash
python scripts/preprocess_pixai_prompts.py --config configs/reward/preprocess_pixai_threshold_rl_20261021.json
```

Input may be JSONL or TXT. TXT supports one escaped prompt per physical line (`prompt:\n1girl, long hair, ...`), or multiple actual multiline `prompt:` blocks. The established newline/backslash escape pass runs once. Output JSONL contains real newlines inside the prompt string, preserves `prompt:` and paragraph layout, retains supported weighted syntax such as `(hair_spread_out:1.2)`, and preserves an optional multiline `negative prompt:` body. Only positive TAGs are cleaned. Unknown tags are deleted and logged; aliases/fuzzy spelling corrections are not invented. Spaces map to underscores for vocabulary lookup. Duplicate targets count once; the first supported token spelling/weight is retained. Source files are never modified. All-unknown prompts are rejected individually and logged; they do not abort the entire file.

The output metadata contains the explicit canonical `target_tags`; training never reparses semantic labels or guesses reward targets from prose. A numeric prompt weight does not increase a tag's reward contribution. The original MOSS prompt body still goes into Qwen's official template. This script is explicitly a TAG preprocessor; natural-language paragraphs are not a supported source of TAG targets.

## Thresholds and reward

PixAI currently publishes `tags_best_threshold: null`, so its official category thresholds are used: general0.17, character0.27, copyright0.24, style0.15, meta0.17, rating0.41. If valid per-tag thresholds are supplied in a future metadata file they take precedence. Prediction uses strict `probability > threshold`, matching the official postprocessor.

Let T be the unique target tags and P the predicted tags above their own thresholds. P is restricted to categories present in T, so unrequested rating/character categories do not inflate a general-only IoU denominator.

- `reward.metric: "coverage"` (pilot objective): |T intersect P| / |T|.
- `reward.metric: "iou"` (optional objective): |T intersect P| / |T union P|.

Both values, predicted tags, target probabilities/thresholds, matched and missing tags are saved. Coverage is recall; it does not penalize additional predictions. IoU can penalize legitimate unmentioned content, so it is a separate experiment rather than silently mixed into the reward.

The provided 54-item example keeps25 unique supported tags, deletes28 unknown terms and1 duplicate. Every target is general, threshold0.17; per-image reward has increments of0.04.

## Train and inspect

```bash
python run.py train --config configs/pixai_threshold_rl_20261021.json
```

The pilot starts from official MOSS-Image-Base, BF16, zero-up rank8 attention LoRA, AdamW1e-4, cuDNN attention;1024x1024,50 Euler/MOSS steps, CFG3, negative skip22. Selected training steps10/30; GPU activation recomputation every6 blocks. New reward and TensorBoard settings are validated configuration fields.

`sampling.split_group_across_workers: true` is an opt-in mode for one prompt with K divisible by the worker count. For K4/four GPUs, each worker generates one image. Qwen's two-buffer condition is broadcast once to every worker. Covariance and two ART windows are computed over the complete global group, not separate one-sample local groups. Unit integration verifies split and unsplit endpoint/gradient equivalence.

The lifecycle is DiT rollout -> CPU replay cache -> DiT processes exit -> VAE decode -> VAE processes exit -> PixAI on all configured GPUs -> PixAI processes exit -> restore DiT from one shared CPU base -> two LoRA updates. Reward gradients never enter the scorer. Fixed-seed evaluation uses a distinct seed set from training rollout. This is a single-prompt overfitting/pipeline pilot, not evidence of prompt-generalization or aesthetic quality gains.

TensorBoard scalars are written without Torch or a TensorFlow runtime in the JAX training coordinator. Event files are standard TensorBoard TFRecord/protobuf files. Set `logging.tensorboard: true`; optional `logging.log_dir` defaults to `<output_dir>/tensorboard`. Logged namespaces: `rollout` (new random seeds each rollout), `train` (loss/gradient diagnostics), `eval` (fixed seeds, suitable for before/after reward comparison). Every point is flushed immediately.

```bash
tensorboard --logdir outputs/pixai_threshold_rl_20261021/tensorboard --host 127.0.0.1 --port 6006
```

The desktop task forwards server6006 to local6006. Open http://127.0.0.1:6006/#scalars and select `eval/coverage`, `eval/iou`, and `rollout/reward_mean`. A point may fluctuate or remain flat; monotonic improvement is not assumed. Training0 baseline coverage was0.56 and IoU0.32215 on four fixed-seed images. Final experiment results are in the report generated after training completes.

To use the next complete file, change the preprocessor JSON input/output/report paths, run preprocessing, then change training `data.prompts`, `data.cache`, and `runtime.output_dir` to new paths. For multiple different prompts, turn off `split_group_across_workers` and set `prompts_per_rollout` divisible by GPU count; full K-image groups then stay on individual workers. The pilot assumes one TAG prompt.

## Completed pilot result (2026-10-21)

The six-update pilot finished in800.95s (three fresh-seed ART rollouts and four fixed-seed evaluations). Fixed-seed coverage at updates0/2/4/6 was0.56/0.56/0.59/0.60. IoU was0.322150/0.316620/0.322374/0.319390. Matched tags per image changed14,15,11,16 ->15,14,16,15: two images improved and two worsened, with aggregate +4 matches over100 target decisions. Extra predicted general tags averaged18.5 ->22.0. This establishes an observed increase of the chosen coverage objective in this one-prompt pilot; it does not establish a statistical/generalization/aesthetic improvement, and IoU did not improve. IoU-objective RL has been implemented and unit-tested but was not trained in this experiment.

TensorBoard events were reread and verified against all four JSON evaluation points.83 related tests passed. The report ZIP contains the final adapter, original/cleaned input, target-score details, all evaluation images, TensorBoard events, configurations and the preprocessing implementation. See `reports/pixai_threshold_rl_20261021.zip` and `inputs/pixai_threshold_rl_20261021/result.json`.

## Prompt filtering and automatic shell

Use `python scripts/preprocess_pixai_prompts.py --config configs/reward/clean_prompts.json` with these JSON fields:

```json
{"input":"inputs/prompts.txt","output":"data/prompts_cleaned.jsonl","report":"outputs/prompt_cleaning/report.json","metadata_dir":"vendor/pixai_tagger/metadata","max_removed_ratio":0.5,"min_tags":5,"min_chars":0,"auto_add_prompt_header":true}
```

- `max_removed_ratio`: discarded positive TAG occurrences divided by original nonempty positive TAG occurrences, counting both unknown TAGs and duplicates. Reject only if the ratio is strictly greater than this value. The shell, whitespace, empty comma fields, and negative prompt do not count as TAGs.
- `min_tags`: minimum retained unique recognized positive TAG count; reject if strictly below it. Numeric weighting does not change this count.
- `min_chars`: optional minimum cleaned positive-body character count (excluding shell and negative section);0 disables it.
- `auto_add_prompt_header`: add `prompt:` followed by a real newline if no shell was present; preserve existing shells. This occurs before Qwen's own chat template is applied; the user never has to supply Qwen special tokens.

The example uses50% removal / minimum5 TAGs. Omitted filter fields preserve the previous permissive behavior (`max_removed_ratio=1.0,min_tags=1,min_chars=0`), with automatic shell insertion enabled. Each rejected record is excluded from the cleaned JSONL and retained in the report with its original/cleaned text, counts, ratio and rejection reasons. Empty/all-unknown records are always rejected. If every record is rejected, output JSONL is empty and the report still records all reasons. Input and output/report paths must differ; existing outputs are never overwritten. No model inference is needed.

Unwrapped TXT input uses one nonempty physical line per prompt, each receiving its own automatic shell. For a single prompt spanning real physical lines, provide a `prompt:` block or use JSONL; escaped newline strings can remain on one TXT line. Empty JSONL prompt strings are logged/rejected as too short instead of aborting the file; malformed field types/duplicate IDs still fail explicitly.
