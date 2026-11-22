"""Real-weight synchronized Spectrum smoke/throughput probe (no Qwen or VAE).

Uses synthetic fixed BF16 text embeddings to test implementation and latency.
This does not measure image quality or reward. All settings use --config JSON.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from moss_art.config import load_config
from moss_art.runtime import preflight, configure_process


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    c = load_config(args.config)
    preflight(c)
    configure_process(c, c["runtime"]["devices"])
    import jax
    import jax.numpy as jnp
    import numpy as np
    from moss_art.model import MossModel
    from moss_art.sampling import schedule, step_coefficients, sample_step
    from moss_art.spectrum import SpectrumRollout
    output = Path(c["runtime"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    report = dict(config=c, quality_measured=False, conditioning="synthetic BF16, 128 padded tokens",
                  timings={}, status="running")
    def write():
        (output/"report.json").write_text(json.dumps(report, indent=2) + "\n")
    def measured(name, fn):
        jax.effects_barrier()
        start = time.perf_counter()
        value = fn()
        for x in jax.tree.leaves(value):
            if isinstance(x, jax.Array): x.block_until_ready()
        jax.effects_barrier()
        report["timings"][name] = time.perf_counter() - start
        write()
        print(json.dumps(dict(event="spectrum_benchmark", stage=name,
                              seconds=report["timings"][name])), flush=True)
        return value
    size = len(c["runtime"]["devices"])
    model = measured("load", lambda: MossModel.from_pretrained(c["model"]["path"],
        rank=c["lora"]["rank"], alpha=c["lora"]["alpha"], targets="attention",
        base_precision="bf16" if c["sampling"]["rollout_mode"] == "bf16" else "mxfp8",
        attention_backend=c["model"]["attention_backend"], offload_activations=False,
        dit_autotune_level=c["model"]["dit_autotune_level"], data_parallel_size=size))
    adapters = model.init_lora(42)
    batch = size * c["sampling"]["rollout_batch_size"]
    s = c["sampling"]
    shape = (batch, model.latent_channels, s["height"]//16, s["width"]//16)
    rng = np.random.default_rng(20261025)
    initial = model._put(rng.normal(size=shape).astype(np.float32))
    positive = model._put(jnp.asarray(rng.normal(0, .2, (batch, 128, model.text_dim)), jnp.bfloat16))
    negative = jnp.zeros_like(positive)
    kwargs = dict(cfg_scale=s["guidance_scale"], negative_context=negative,
        negative_skip_layers=tuple(s["skip_layers"]), context_lengths=np.full(batch, 96, np.int32),
        negative_context_lengths=np.full(batch, 12, np.int32))
    _, sigmas = schedule(s["steps"], scheduler=s["scheduler"], shift=s["shift"],
        training_shift=s["training_shift"], beta_alpha=s["beta_alpha"], beta_beta=s["beta_beta"])
    coefficients = step_coefficients(sigmas, sampler=s["sampler"])
    velocity, features = measured("warm_full_with_features", lambda:
        model.predict_with_features(adapters, initial, sigmas[0], positive, **kwargs))
    head = measured("warm_head", lambda:
        model.predict_from_features(features, sigmas[0], shape, cfg_scale=s["guidance_scale"]))
    report["exact_head_reentry"] = bool(np.array_equal(np.asarray(head), np.asarray(velocity)))
    if not report["exact_head_reentry"]:
        raise AssertionError("Head re-entry changed the model prediction")
    del velocity, features, head
    def rollout(enabled):
        forecast = SpectrumRollout(len(coefficients), s["spectrum"]) if enabled else None
        state, previous = initial, None
        for i, coeff in enumerate(coefficients):
            velocity = (forecast.predict(model, adapters, state, sigmas[i], positive, index=i, **kwargs)
                        if forecast else model.predict(adapters, state, sigmas[i], positive, **kwargs))
            state, previous = sample_step(state, velocity, coeff, previous)
        state.block_until_ready()
        if forecast: report["spectrum"] = forecast.stats()
        return state
    warm = measured("warm_spectrum", lambda: rollout(True))
    del warm
    baseline = measured("baseline", lambda: rollout(False))
    accelerated = measured("spectrum", lambda: rollout(True))
    report["finite"] = all(bool(np.isfinite(np.asarray(v)).all()) for v in (baseline, accelerated))
    if not report["finite"]:
        raise FloatingPointError("Nonfinite endpoint in real-weight Spectrum smoke")
    report["speedup"] = report["timings"]["baseline"] / report["timings"]["spectrum"]
    report["images_per_wave"] = batch
    report["scheduler_steps"] = len(coefficients)
    report["throughput_images_per_second"] = batch / report["timings"]["spectrum"]
    report["device_memory"] = [dict(device=str(d), bytes_in_use=d.memory_stats().get("bytes_in_use"),
        peak_bytes_in_use=d.memory_stats().get("peak_bytes_in_use")) for d in jax.local_devices()]
    report["status"] = "complete"
    write()
    print(json.dumps(dict(event="spectrum_benchmark_complete", speedup=report["speedup"],
        finite=report["finite"], exact_head_reentry=report["exact_head_reentry"], report=str(output/"report.json"))), flush=True)
    model.release_base()


if __name__ == "__main__": main()
