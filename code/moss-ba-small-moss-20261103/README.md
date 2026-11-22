# MOSS-shaped 101.8M BA student

This is a learned velocity correction network used alongside the frozen 5.08B MOSS teacher during few-step BA sampling. It does not replace MOSS with a standalone 0.1B image generator.

The trainable student has 101,796,864 FP32 master parameters: 11 MOSS joint image/text blocks, width 768, six 128-dimensional attention heads, FFN width 2048, QK RMSNorm, MOSS RoPE and SiTU, and two text refinement blocks. The shared cached positive/negative Qwen features are linearly projected from 4096 to 768 with explicit text-role embeddings and length masks. Inputs are latent x and teacher velocity v, with time t and signed interval h. A zero-initialized output projection multiplied by h makes step-zero correction exactly zero. Student compute and the frozen teacher use BF16.

Four separate processes on physical GPUs 0–3 compute distinct prompt trajectories concurrently. They synchronously average valid trajectory gradients, clip once, and apply one AdamW update. Gradients travel through host memory; this implementation does not claim NCCL all-reduce. Each chain averages four segment losses, keeping student parameters fixed throughout the chain. Three-prompt final batches use a denominator of three. Each of the 1971 prompts gets four trajectories, giving 7884 trajectories and 1972 optimizer updates. GPU4 is excluded.

`side.py` implements the network, `dp_replica.py` computes per-card trajectories and gradients, `dp_train.py` owns Adam and atomic checkpoint publication, and `control.py` starts an identity-checked coordinator. `prepare.py` audits and links the immutable completed teacher reference. The three `verify_*.py` scripts cover masks/full parameter gradients, official solver parity, and four-process versus serial update/tail/resume equivalence. Tests and GPU acceptance results are stored under the new run's `reports` directory.

New code: `$CODE_ROOT/moss-ba-small`.

New run: `$MOSS_ROOT/distillation/ba_smallmoss_102m_dp4_bf16_1024_20261103`.

Checkpoint `latest.json` is the only published learner cursor. It points to one of two full checkpoint slots, each containing student, Adam, warmup counter, prompt cursor and frozen configuration SHA. Four updates are committed together. A discarded preflight update is restored from the published checkpoint before training. Resume never applies a previously published update again. Fixed evaluations at 0/128/512/1024/1972 retain original latent arrays, metrics and full PNGs, with 48-step teacher, equal-NFE Euler, zero-student BA and learned BA8/12 comparisons.

The previous student and old ART runs remain stopped. The web image service has been stopped by user instruction. All original teacher references and historical results are retained.
