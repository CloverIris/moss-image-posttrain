# JSON configuration

Run from `$CODE_ROOT/moss-art-jax`. Public commands take an operation and
`--config path.json` only. No CLI overrides. Missing fields inherit defaults,
unknown fields and wrong types fail before workers start. JSON duplicates,
nonfinite numbers and trailing commas are rejected. YAML files are legacy
read compatibility; new presets are JSON. Paths are relative to the working
directory, not the configuration file. `scripts/run.sh` selects the project root.

| Operation | Command |
| --- | --- |
| ART train | `bash scripts/run.sh train --config configs/moss_lora_4gpu.json` |
| ART evaluate | `bash scripts/run.sh evaluate --config my_eval.json` |
| ART preflight | `bash scripts/run.sh doctor --config configs/moss_lora_4gpu.json` |
| ART text cache | `bash scripts/run.sh prepare-text --config configs/moss_lora_4gpu.json` |
| SFT train | `bash scripts/run.sh sft train --config configs/sft/train_1024.json` |
| SFT sample | `bash scripts/run.sh sft sample --config configs/sft/sample_1024.json` |
| SFT cache | `torch_env/bin/python -m moss_art.sft_prepare --config configs/sft/prepare.json` |
| Decode saved latents | `$CONDA_PREFIX/envs/jax_env/bin/python -m moss_art.evaluate --config configs/decode.json` |

The SFT cache command needs the installed Torch environment; `sft prepare` is an
alias but must also run under that environment. ART's direct `moss_art.prepare`
module is an alias for `prepare-text` using the same ART config schema.

## ART / SFT training and sampling

model, lora, optimizer, sampling, runtime and training retain their existing
nested fields. ART uses data.prompts/data.cache, reward and art. SFT uses
data.cache/data.context_length and sft; reward/art fields are not valid in SFT.
SFT-only TP2 and microbatch options do not enable them in the ART worker.

Operation-only settings live in run:

```json
{
  "run": {
    "resume": "",
    "init_adapter": "",
    "checkpoint": "",
    "adapter": "",
    "sample_index": 0,
    "seed": 42,
    "decode": false
  }
}
```

Train allows resume OR init_adapter. ART evaluate requires checkpoint OR adapter.
SFT sample permits checkpoint OR adapter, or neither for the base model.
Paths are mutually exclusive. sample_index/seed are SFT sampling settings;
training uses training.seed. decode forces image decoding in ART evaluation.
Nondefault operation fields invalid for the selected command fail explicitly.
run does not change checkpoint algorithm fingerprints, allowing exact resume
when all algorithm/data/source identities still match. A source change still
invalidates old exact-resume identities; use an exported adapter for a new run.

Move old flags into these JSON fields:

| Old flag | JSON field |
| --- | --- |
| --devices | runtime.devices (integer array) |
| --output | runtime.output_dir |
| --allow-busy | runtime.allow_busy |
| --max-updates | training.max_updates |
| --reward-url | reward.url, with kind=http and decode=true |
| --resume / --init-adapter | run.resume / run.init_adapter |
| --checkpoint / --adapter | run.checkpoint / run.adapter |
| --sample-index / --seed | run.sample_index / run.seed |
| prepare-text --device | preparation.device |

ART preprocessing additionally uses preparation.max_sequence_length,
preparation.full_weight_hash and preparation.limit (0 means all records).
SFT preprocessing and standalone latent decoding use flat JSON schemas,
as shown in configs/sft/prepare.json and configs/decode.json. Their keys are
underscore-separated versions of the original argument names. SFT preparation
uses a physical GPU index in device; standalone decoding uses a JAX device
string such as cpu:0 or gpu:0. vae_stages is a JSON array of stage names.

## Samplers and schedulers

| sampling.sampler | Meaning |
| --- | --- |
| euler | Deterministic first-order flow Euler |
| res_multistep | Deterministic RES multistep, eta=0, cfg_pp=false; first/final steps use Euler |

| sampling.scheduler | Meaning |
| --- | --- |
| moss | Original vendored MOSS shifted sigma grid with legacy grid snapping |
| beta | Beta-percentile selection on the shifted flow training grid |

All four combinations are supported in ART rollout and SFT sampling. Neither
sampler is used for SFT's direct supervised flow-matching target construction.
Canonical name is res_multistep, not resmutistep. No stochastic variant, CFG++,
Heun, DPM++, Karras or exponential scheduler is currently exposed.

```json
{
  "sampling": {
    "steps": 50,
    "sampler": "res_multistep",
    "scheduler": "beta",
    "shift": 5.0,
    "training_shift": 1.0,
    "beta_alpha": 0.6,
    "beta_beta": 0.6
  }
}
```

beta requires training_shift=1; flow shifting is controlled by shift. Alpha and
beta must be positive. Quantized grid indices are deduplicated, so extreme
settings can yield fewer actual steps than requested. ART train_timesteps are
indices in the resulting trajectory and are bounds checked. RES uses its
actual local velocity coefficient in ART's movement metric, with rollout
history frozen during replay. This is not a hardcoded Euler delta.


## short/full 与 Qwen GPU

见 [PROMPT_BUCKETS.md](PROMPT_BUCKETS.md)。ART/SFT 的 `data.short_context_length` 为 short，
`data.context_length` 为 full；short=0 自动按有效请求 P70 选择。`preparation.device=4`
和 `preparation.auto_prepare=true` 控制 ART 的独立 JAX 缓存预处理。旧字段
`preparation.max_sequence_length` 仅在未指定 `data.context_length` 时作为 full 的兼容别名。
SFT 预处理使用 `short_tokens`/`max_tokens`/`device=4`。


## 异步 Qwen 双缓冲

`preparation.async_prefetch=true` 启用 ART 在线两槽共享内存预取，维护中的 ART 训练 JSON
已开启；schema默认值保留false以兼容旧配置。`preparation.device` 必须与 DiT GPU 分离。
启用时无需提前 prepare-text，也不执行整库 ensure_cache；缓存按实际访问逐步填充。
细节和等待指标见 [ASYNC_PREFETCH.md](ASYNC_PREFETCH.md)。
