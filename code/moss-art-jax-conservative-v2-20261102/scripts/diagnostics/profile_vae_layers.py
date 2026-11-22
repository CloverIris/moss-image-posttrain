#!/usr/bin/env python3
"""Independent whole-layer and decomposed-op VAE timings, never end-to-end attribution.

Whole layers consume real preceding whole-layer outputs. Atomic inputs are
reconstructed from each real layer input; JIT partitioning changes fusion and
possibly rounding. Attention QK/softmax/PV use one real query chunk with all keys;
their chunk-count extrapolations are estimates, not measured full-loop times.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from moss_art.locking import output_lock
from scripts.benchmark_rollout import GPUMonitor
from scripts.benchmark_vae_precision import atomic_json, load_latents, sha256
from scripts.diagnostics.profile_vae_stages import configure


def parameters(params, prefix):
    return {key: value for key, value in params.items() if key.startswith(prefix)}


def conv_macs(params, prefix, shape):
    n, _, h, w = shape
    out_c, in_c, kh, kw = params[prefix + ".weight"].shape
    return int(n * h * w * out_c * in_c * kh * kw)


def linear_macs(params, prefix, shape):
    out_c, in_c = params[prefix + ".weight"].shape
    return int(np.prod(shape[:-1]) * in_c * out_c)


def layer_specs(config):
    from vendor.moss_jax import vae
    result = [{"name": "bn_unpack", "kind": "bn", "prefix": "bn."}]
    if vae._get(config, "use_post_quant_conv"):
        result.append({"name": "post_quant_conv", "kind": "conv", "prefix": "post_quant_conv"})
    result.append({"name": "decoder.conv_in", "kind": "conv", "prefix": "decoder.conv_in"})
    result.append({"name": "decoder.mid_block.resnets.0", "kind": "resnet", "prefix": "decoder.mid_block.resnets.0"})
    if vae._get(config, "mid_block_add_attention"):
        result.append({"name": "decoder.mid_block.attentions.0", "kind": "attention", "prefix": "decoder.mid_block.attentions.0"})
    result.append({"name": "decoder.mid_block.resnets.1", "kind": "resnet", "prefix": "decoder.mid_block.resnets.1"})
    count = len(vae._get(config, "up_block_types"))
    for block in range(count):
        for resnet in range(int(vae._get(config, "layers_per_block")) + 1):
            prefix = f"decoder.up_blocks.{block}.resnets.{resnet}"
            result.append({"name": prefix, "kind": "resnet", "prefix": prefix})
        if block < count - 1:
            prefix = f"decoder.up_blocks.{block}.upsamplers.0.conv"
            result.append({"name": prefix.removesuffix(".conv"), "kind": "upsample", "prefix": prefix})
    result.append({"name": "decoder.output", "kind": "output", "prefix": "decoder."})
    return result


def whole_function(spec, decoder, config, chunk):
    import jax.numpy as jnp
    from vendor.moss_jax import vae
    kind, prefix = spec["kind"], spec["prefix"]
    groups = int(vae._get(config, "norm_num_groups"))

    def function(params, x):
        if kind == "bn":
            return vae.unpack_latents(params, x, config).astype(decoder.dtype)
        if kind == "conv":
            return decoder._conv2d(params, x, prefix)
        if kind == "resnet":
            return decoder._resnet(params, x, prefix, groups)
        if kind == "attention":
            return decoder._attention(params, x, prefix, groups, chunk)
        if kind == "upsample":
            return decoder._conv2d(params, jnp.repeat(jnp.repeat(x, 2, axis=2), 2, axis=3), prefix)
        if kind == "output":
            return decoder._conv2d(params, decoder._silu(
                decoder._group_norm(params, x, "decoder.conv_norm_out", groups)), "decoder.conv_out")
        raise ValueError(kind)
    return function


def layer_macs(spec, params, shape):
    kind, prefix = spec["kind"], spec["prefix"]
    n, c, h, w = shape
    if kind == "bn":
        return 0
    if kind == "conv":
        return conv_macs(params, prefix, shape)
    if kind == "upsample":
        return conv_macs(params, prefix, (n, c, h * 2, w * 2))
    if kind == "output":
        return conv_macs(params, "decoder.conv_out", shape)
    if kind == "attention":
        tokens = h * w
        return int(4 * n * tokens * c * c + 2 * n * tokens * tokens * c)
    if kind == "resnet":
        first = conv_macs(params, prefix + ".conv1", shape)
        hidden = params[prefix + ".conv1.weight"].shape[0]
        total = first + conv_macs(params, prefix + ".conv2", (n, hidden, h, w))
        if prefix + ".conv_shortcut.weight" in params:
            total += conv_macs(params, prefix + ".conv_shortcut", shape)
        return total
    raise ValueError(kind)


def difference(actual, reference):
    """Bounded host scratch even for 800 MiB intermediate activations."""
    actual, reference = np.asarray(actual), np.asarray(reference)
    if actual.shape != reference.shape:
        raise ValueError("Comparison shapes differ")
    a, b = actual.reshape(-1), reference.reshape(-1)
    max_abs, squared_error, squared_ref = 0.0, 0.0, 0.0
    equal = True
    for start in range(0, len(a), 262144):
        av, bv = a[start:start+262144].astype(np.float64), b[start:start+262144].astype(np.float64)
        delta = av - bv
        max_abs = max(max_abs, float(np.max(np.abs(delta))))
        squared_error += float(np.dot(delta, delta))
        squared_ref += float(np.dot(bv, bv))
        equal = equal and bool(np.array_equal(av, bv))
    return {"max_abs": max_abs, "relative_l2": math.sqrt(squared_error / squared_ref) if squared_ref else None,
            "bitwise_equal": equal}


def atomic_plan(spec, decoder, config, x_shape):
    """Named DAG preserves residual lifetimes while discarding consumed buffers."""
    import jax.numpy as jnp
    from vendor.moss_jax import vae
    groups = int(vae._get(config, "norm_num_groups"))
    kind, prefix = spec["kind"], spec["prefix"]
    plan = []

    def add(name, inputs, output, fn, parameter_prefix=None, mac_prefix=None):
        plan.append(dict(name=name, inputs=inputs, output=output, fn=fn,
                         parameter_prefix=parameter_prefix, mac_prefix=mac_prefix))

    def norm(name, inp, out, key):
        add(name, (inp,), out, lambda p, a: decoder._group_norm(p, a, key, groups), key + ".")

    def silu(name, inp, out):
        add(name, (inp,), out, lambda p, a: decoder._silu(a))

    def conv(name, inp, out, key):
        add(name, (inp,), out, lambda p, a: decoder._conv2d(p, a, key), key + ".", key)

    if kind == "resnet":
        norm("norm1", "x", "n1", prefix + ".norm1")
        silu("silu1", "n1", "s1")
        conv("conv1_including_bias", "s1", "c1", prefix + ".conv1")
        norm("norm2", "c1", "n2", prefix + ".norm2")
        silu("silu2", "n2", "s2")
        conv("conv2_including_bias", "s2", "c2", prefix + ".conv2")
        # Caller removes a missing optional shortcut without executing a kernel.
        conv("shortcut", "x", "shortcut", prefix + ".conv_shortcut")
        add("residual_add", ("shortcut", "c2"), "out", lambda p, a, b: a + b)
    elif kind == "upsample":
        add("nearest_repeat", ("x",), "repeat", lambda p, a: jnp.repeat(jnp.repeat(a, 2, axis=2), 2, axis=3))
        conv("upsample_conv_including_bias", "repeat", "out", prefix)
    elif kind == "output":
        norm("norm_out", "x", "norm", "decoder.conv_norm_out")
        silu("silu_out", "norm", "silu")
        conv("conv_out_including_bias", "silu", "out", "decoder.conv_out")
    elif kind == "bn":
        eps = float(vae._get(config, "batch_norm_eps"))
        add("bn_denormalization_fp32", ("x",), "denorm", lambda p, a:
            a.astype(jnp.float32) * jnp.sqrt(p["bn.running_var"].astype(jnp.float32) + eps)[None, :, None, None]
            + p["bn.running_mean"].astype(jnp.float32)[None, :, None, None], "bn.")
        patch_h, patch_w = vae._get(config, "patch_size")
        channels = int(vae._get(config, "latent_channels"))

        def unpack(p, a):
            n, _, h, w = a.shape
            return a.reshape(n, channels, patch_h, patch_w, h, w).transpose(0, 1, 4, 2, 5, 3).reshape(
                n, channels, h * patch_h, w * patch_w).astype(decoder.dtype)
        add("unpack_and_decoder_cast", ("denorm",), "out", unpack)
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--precision", choices=("fp32", "float16"), required=True)
    parser.add_argument("--latents", required=True)
    parser.add_argument("--sample", type=int, default=3)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference-raw", required=True)
    parser.add_argument("--reserve-dit-gib", type=float, default=9.6)
    parser.add_argument("--warm-repeats", type=int, default=3)
    parser.add_argument("--attention-chunk-size", type=int, default=256)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--model-dir", default="$MODEL_ROOT/MOSS-Image-SFT")
    args = parser.parse_args(argv)
    if args.device < 0 or args.sample < 0 or min(args.warm_repeats, args.attention_chunk_size, args.cpu_threads) < 1:
        parser.error("Invalid device/sample/count")
    if not math.isfinite(args.reserve_dit_gib) or args.reserve_dit_gib < 0:
        parser.error("reserve-dit-gib must be finite and nonnegative")
    latents = load_latents(args.latents)
    if args.sample >= len(latents):
        parser.error("Sample index out of range")
    inventory = configure(args)  # Preflight GPU; preserve caller compiler/precision environment.
    import jax
    import jax.numpy as jnp
    from jax import lax
    from vendor.moss_jax import vae
    from vendor.moss_jax.weights import load_config, load_host_params
    from scripts.diagnostics.vae_full_fp16 import Decoder
    from scripts.diagnostics.profile_vae_stages import hlo_evidence

    output = Path(args.output).resolve()
    with output_lock(output):
        if any(path.name != ".run.lock" for path in output.iterdir()):
            raise FileExistsError("Select an empty output directory")
        (output / "hlo").mkdir()
        report = {"status": "running", "recorded_utc": datetime.now(timezone.utc).isoformat(),
                  "arguments": vars(args), "physical_device": args.device, "inventory": inventory,
                  "input_sha256": sha256(args.latents), "sample_index": args.sample, "layers": [],
                  "source_hashes": {name: sha256(ROOT / name) for name in (
                      "scripts/diagnostics/profile_vae_layers.py", "scripts/diagnostics/profile_vae_stages.py",
                      "scripts/diagnostics/vae_full_fp16.py", "vendor/moss_jax/vae.py")},
                  "runtime_environment": {k: os.environ.get(k) for k in (
                      "CUDA_VISIBLE_DEVICES", "JAX_PLATFORMS", "XLA_FLAGS",
                      "TF_FP16_CONV_USE_FP32_COMPUTE", "XLA_PYTHON_CLIENT_PREALLOCATE",
                      "XLA_PYTHON_CLIENT_MEM_FRACTION")},
                  "policy": {"compiler_options": {}, "global_compiler_policy": "Caller environment preserved; no global defaults injected",
                             "native_convolution_accumulator": "XLA/backend default unless caller supplied a precision environment override",
                             "warm_repeats": args.warm_repeats,
                             "whole_layers": "Real sequential whole-layer outputs; synchronized host latency includes dispatch and block_until_ready synchronization overhead, excludes compile/statistics/transfers",
                             "atomic": "Reconstruct subop inputs from the real layer input; changed JIT fusion/rounding means sums are NOT whole-layer or end-to-end attribution",
                             "attention_atomic": "QK/softmax/PV measure first real query chunk with all keys; multiply by chunk count only as labeled estimates",
                             "whole_output_offload": "Whole-layer output copied to host and freed before atomic profiling to avoid artificial extra GPU residency",
                             "MAC_definition": "Multiply-accumulate pairs in conv/GEMM only; excludes bias/norm/SiLU/softmax/repeat/add; FLOPs approximately 2*MAC for these contractions",
                             "precision": "FP32 throughout" if args.precision == "fp32" else
                                 "Decoder weights and cross-layer activations FP16; BN outside decoder FP32; GN statistics/affine and attention score/softmax/PV accumulation FP32 as diagnostic helper"}}
        atomic_json(output / "summary.json", report)
        monitor = GPUMonitor(args.device, True)
        monitor.start()
        started = time.monotonic()
        reserve = []
        try:
            device = jax.devices()[0]
            remaining = round(args.reserve_dit_gib * 1024**3)
            monitor.phase = "reservation"
            while remaining:
                count = min(remaining, 256 * 1024**2)
                reserve.append(jax.device_put(np.zeros(count, np.uint8), device))
                reserve[-1].block_until_ready()
                remaining -= count
            report["reservation_bytes"] = sum(value.nbytes for value in reserve)
            config = load_config(args.model_dir, "vae")
            sample_size = vae._get(config, "sample_size")
            sample_size = sample_size[0] if isinstance(sample_size, (list, tuple)) else sample_size
            scale = 2**(len(vae._get(config, "block_out_channels")) - 1)
            patch = tuple(vae._get(config, "patch_size"))
            if max(latents.shape[2] * patch[0], latents.shape[3] * patch[1]) > int(sample_size) // scale:
                raise ValueError("This layer profiler only supports direct-path inputs at/below the native tiling threshold")
            host = load_host_params(args.model_dir, "vae")
            params = {key: jax.device_put(np.asarray(value, np.float16 if args.precision == "float16" and not key.startswith("bn.") else np.float32), device)
                      for key, value in host.items() if key.startswith(("bn.", "decoder.", "post_quant_conv."))}
            del host
            jax.block_until_ready(params)
            decoder = Decoder(args.precision)
            report.update(vae_config=config, parameter_bytes=sum(p.nbytes for p in params.values()))
            x = jax.device_put(latents[args.sample:args.sample+1], device)
            x.block_until_ready()

            @jax.jit
            def reduce_stats(a):
                return jnp.isfinite(a).all(), jnp.min(a.astype(jnp.float32)), jnp.max(a.astype(jnp.float32))

            def stats(value):
                leaves = jax.tree.leaves(value)
                result = []
                for a in leaves:
                    finite, minimum, maximum = jax.device_get(reduce_stats(a))
                    result.append({"shape": list(a.shape), "dtype": str(a.dtype), "bytes": int(a.nbytes),
                                   "finite": bool(finite), "min": float(minimum) if finite else None,
                                   "max": float(maximum) if finite else None})
                return result

            def measure(label, function, local, values, macs=0, scope="actual complete operator"):
                monitor.phase = label
                begin = time.monotonic()
                compiled = jax.jit(function).lower(local, *values).compile()
                compile_seconds = time.monotonic() - begin
                result, durations = None, []
                for _ in range(args.warm_repeats + 1):
                    if result is not None:
                        jax.tree.map(lambda a: a.delete(), result)
                    begin = time.monotonic()
                    result = compiled(local, *values)
                    jax.block_until_ready(result)
                    durations.append(time.monotonic() - begin)
                safe = re.sub("[^A-Za-z0-9_-]", "_", label)
                entry = {"label": label, "scope": scope, "compile_seconds": compile_seconds,
                         "first_seconds": durations[0], "warm_seconds": durations[1:],
                         "warm_median_seconds": float(np.median(durations[1:])),
                         "input_shapes": [list(a.shape) for a in values],
                         "input_dtypes": [str(a.dtype) for a in values],
                         "parameter_dtypes": sorted({str(p.dtype) for p in local.values()}),
                         "output": stats(result), "MAC": int(macs),
                         "hlo": hlo_evidence(compiled, output / "hlo" / (safe + ".txt"))}
                entry["hlo"]["path"] = "hlo/" + entry["hlo"]["path"]
                entry["effective_TMAC_per_second"] = macs / entry["warm_median_seconds"] / 1e12
                if not all(leaf["finite"] for leaf in entry["output"]):
                    raise FloatingPointError(f"Nonfinite output in {label}")
                return result, entry

            def profile_atomic(spec, value, whole_entry):
                if spec["kind"] == "conv":
                    return None, [{"name": "conv_including_bias", "reuse_of_whole": whole_entry["label"],
                                   "warm_median_seconds": whole_entry["warm_median_seconds"],
                                   "MAC": whole_entry["MAC"], "scope": "Identical complete convolution; reused measurement"}]
                if spec["kind"] == "attention":
                    return attention_atomic(spec, value)
                plan = atomic_plan(spec, decoder, config, value.shape)
                if spec["kind"] == "resnet" and spec["prefix"] + ".conv_shortcut.weight" not in params:
                    plan = [op for op in plan if op["name"] != "shortcut"]
                    plan[-1]["inputs"] = ("x", "c2")
                uses = Counter(key for op in plan for key in op["inputs"])
                buffers, entries = {"x": value}, []
                for op in plan:
                    values = tuple(buffers[key] for key in op["inputs"])
                    local = parameters(params, op["parameter_prefix"]) if op["parameter_prefix"] else {}
                    macs = conv_macs(local, op["mac_prefix"], values[0].shape) if op["mac_prefix"] else 0
                    result, entry = measure(spec["name"] + "/" + op["name"], op["fn"], local, values, macs)
                    entry["name"] = op["name"]
                    entries.append(entry)
                    buffers[op["output"]] = result
                    for key in op["inputs"]:
                        uses[key] -= 1
                        if uses[key] == 0 and key != "x":
                            buffers.pop(key).delete()
                    del values
                return buffers["out"], entries

            def attention_atomic(spec, value):
                prefix = spec["prefix"]
                groups = int(vae._get(config, "norm_num_groups"))
                n, c, h, w = value.shape
                tokens, chunk = h * w, min(args.attention_chunk_size, h * w)
                count = (tokens + chunk - 1) // chunk
                entries = []

                def run(name, fn, local, values, macs=0, chunk_only=False):
                    scope = f"actual first {chunk}-query chunk against all {tokens} keys; full-loop estimate only" if chunk_only else "actual complete operator"
                    out, entry = measure(spec["name"] + "/" + name, fn, local, values, macs, scope)
                    entry["name"] = name
                    if chunk_only:
                        entry.update(query_chunk_index=0, query_chunk_count=count,
                                     estimated_all_chunks_seconds=entry["warm_median_seconds"] * count,
                                     estimated_all_chunks_MAC=entry["MAC"] * count)
                    entries.append(entry)
                    return out

                norm = run("group_norm", lambda p, a: decoder._group_norm(p, a, prefix + ".group_norm", groups),
                           parameters(params, prefix + ".group_norm."), (value,))
                flat = norm.reshape(n, c, tokens).transpose(0, 2, 1)
                qkv_params = {key: val for key, val in parameters(params, prefix + ".").items()
                              if any(tag in key for tag in (".to_q.", ".to_k.", ".to_v."))}
                def qkv(p, a):
                    return tuple(decoder._linear(p, a, prefix + "." + suffix) for suffix in ("to_q", "to_k", "to_v"))
                query, key, val = run("qkv_projections", qkv, qkv_params, (flat,), 3 * n * tokens * c * c)
                del flat
                norm.delete()
                scale = jnp.float32(c ** -.5)
                scores = run("QK_representative_chunk", lambda p, q, k:
                    jnp.matmul(q[:, :chunk], k.transpose(0, 2, 1), precision=lax.Precision.HIGHEST,
                               preferred_element_type=jnp.float32) * scale,
                    {}, (query, key), n * chunk * tokens * c, True)
                probability = run("softmax_representative_chunk", lambda p, s: jax.nn.softmax(s, -1).astype(decoder.dtype),
                                  {}, (scores,), 0, True)
                scores.delete()
                attended_chunk = run("PV_representative_chunk", lambda p, a, v:
                    jnp.matmul(a, v, precision=lax.Precision.HIGHEST, preferred_element_type=jnp.float32).astype(decoder.dtype),
                    {}, (probability, val), n * chunk * tokens * c, True)
                probability.delete()
                attended_chunk.delete()

                # Reconstruct the real full preprojection operand without
                # materializing all T*T scores; this work is excluded from times.
                @jax.jit
                def construct(p, q, k, v):
                    padding = (-tokens) % chunk
                    chunks = jnp.pad(q, ((0, 0), (0, padding), (0, 0))).reshape(n, -1, chunk, c).transpose(1, 0, 2, 3)
                    kt = k.transpose(0, 2, 1)
                    def attend(qblock):
                        s = jnp.matmul(qblock, kt, precision=lax.Precision.HIGHEST,
                                       preferred_element_type=jnp.float32) * scale
                        probabilities = jax.nn.softmax(s, -1).astype(decoder.dtype)
                        return jnp.matmul(probabilities, v, precision=lax.Precision.HIGHEST,
                                          preferred_element_type=jnp.float32).astype(decoder.dtype)
                    result = lax.map(attend, chunks)
                    return result.transpose(1, 0, 2, 3).reshape(n, -1, c)[:, :tokens]
                attended = construct({}, query, key, val)
                attended.block_until_ready()
                query.delete(); key.delete(); val.delete()
                projection = run("output_projection", lambda p, a: decoder._linear(p, a, prefix + ".to_out.0"),
                                 parameters(params, prefix + ".to_out.0."), (attended,), n * tokens * c * c)
                attended.delete()
                result = run("residual_add", lambda p, a, b: a.transpose(0, 2, 1).reshape(n, c, h, w) + b,
                             {}, (projection, value))
                projection.delete()
                return result, entries

            for spec in layer_specs(config):
                name, kind = spec["name"], spec["kind"]
                print(json.dumps({"event": "layer_start", "name": name, "precision": args.precision}), flush=True)
                local = parameters(params, spec["prefix"] if kind == "bn" else spec["prefix"] + ("" if kind == "output" else "."))
                whole, whole_entry = measure(name + "/whole", whole_function(spec, decoder, config, args.attention_chunk_size),
                                             local, (x,), layer_macs(spec, params, x.shape))
                # Retain real whole-layer output on host while atomic kernels run.
                whole_host = np.array(jax.device_get(whole), copy=True)
                whole.delete()
                del whole
                atomic, subops = profile_atomic(spec, x, whole_entry)
                entry = {**spec, "whole": whole_entry, "subops": subops,
                         "atomic_subop_sum_seconds": sum(item.get("estimated_all_chunks_seconds", item["warm_median_seconds"]) for item in subops),
                         "atomic_sum_scope": "Sum of isolated subop medians, with attention chunk extrapolations; not measured whole-layer latency"}
                if atomic is not None:
                    entry["atomic_vs_whole"] = difference(jax.device_get(atomic), whole_host)
                    atomic.delete()
                denominator = entry["atomic_subop_sum_seconds"]
                for item in subops:
                    item["share_of_decomposed_sum"] = item.get("estimated_all_chunks_seconds", item["warm_median_seconds"]) / denominator if denominator else 0
                x.delete()
                x = jax.device_put(whole_host, device)
                x.block_until_ready()
                del whole_host
                entry["jax_memory_after"] = device.memory_stats()
                report["layers"].append(entry)
                atomic_json(output / "summary.json", report)
                print(json.dumps({"event": "layer_done", "name": name, "whole_seconds": whole_entry["warm_median_seconds"],
                                  "atomic_sum_seconds": entry["atomic_subop_sum_seconds"]}), flush=True)
                gc.collect()
            raw = np.asarray(x, np.float32)
            np.save(output / "layerwise.raw.npy", raw, allow_pickle=False)
            reference = np.load(args.reference_raw, allow_pickle=False)
            report["reference_comparison"] = difference(raw, reference)
            report["reference_raw_sha256"] = sha256(args.reference_raw)
            total = sum(layer["whole"]["warm_median_seconds"] for layer in report["layers"])
            report["whole_layer_sum_seconds_not_end_to_end"] = total
            report["total_MAC"] = sum(layer["whole"]["MAC"] for layer in report["layers"])
            for layer in report["layers"]:
                layer["whole_share_of_layer_sum"] = layer["whole"]["warm_median_seconds"] / total
            report["status"] = "passed"
        except BaseException as error:
            report.update(status="failed", failure_phase=monitor.phase, error_type=type(error).__name__,
                          error=str(error), traceback=traceback.format_exc())
            raise
        finally:
            report["total_seconds"] = time.monotonic() - started
            report["nvidia_smi"] = monitor.finish(output / "gpu_memory.csv")
            atomic_json(output / "summary.json", report)
            print(json.dumps({"event": "profile_finished", "status": report["status"],
                              "total_seconds": report["total_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
