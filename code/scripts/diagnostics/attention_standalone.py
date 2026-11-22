#!/usr/bin/env python3
"""Reproduce cuDNN attention timing, padding constraints, and actual HLO calls.

Examples (run from $CODE_ROOT/moss-art-jax):
  $CONDA_PREFIX/envs/jax_env/bin/python scripts/diagnostics/attention_standalone.py --device 2 \
    --mode padded --output-prefix reports/attention_repro
  $CONDA_PREFIX/envs/jax_env/bin/python scripts/diagnostics/attention_standalone.py --device 2 \
    --mode lengths --output-prefix reports/attention_lengths_repro

The unpadded mode deliberately reproduces the JAX 0.10.2 odd-length VJP
failure. No model weights are loaded. Imports alone never initialize a GPU.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import time
import traceback

from _common import add_common_arguments, save_hlo, save_report, setup


def custom_calls(executable):
    return sorted(set(re.findall(r'custom_call_target="([^"]+)"', executable.as_text())))


def require_cudnn_calls(targets, backward):
    expected = {"__cudnn$fmhaSoftmax"}
    if backward:
        expected.add("__cudnn$fmhaSoftmaxBackward")
    if not expected.issubset(targets):
        raise AssertionError(f"Missing actual compiled cuDNN calls: {expected-set(targets)}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--mode", choices=("padded", "unpadded", "lengths"), default="padded")
    parser.add_argument("--image-height", type=int, default=64, help="Packed latent height")
    parser.add_argument("--image-width", type=int, default=64, help="Packed latent width")
    parser.add_argument("--text-tokens", type=int, default=27)
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=81024)
    parser.add_argument("--lengths", default="10,26,27,4106,4122")
    args = parser.parse_args(argv)
    root, path = setup(args)
    sys.path.insert(0, str(root))
    import numpy as np
    import jax
    import jax.numpy as jnp
    import jaxlib
    from jax._src.cudnn.fused_attention_stablehlo import cuda_versions
    from moss_art.attention import attention as native_attention
    from vendor.moss_jax import dit

    if min(args.image_height, args.image_width, args.heads, args.head_dim) < 1 or args.text_tokens < 0:
        raise ValueError("Invalid attention dimensions")
    length = args.image_height * args.image_width + args.text_tokens
    report = {
        "status": "running", "physical_gpu": args.device, "device": str(jax.devices()[0]),
        "compute_capability": jax.devices()[0].compute_capability, "mode": args.mode,
        "versions": {"jax": jax.__version__, "jaxlib": jaxlib.__version__},
        "shape": [1, length, args.heads, args.head_dim], "dtype": "bfloat16",
        "warm_repeats": args.repeats, "argv": sys.argv, "implementations": {},
        "cache_policy": "Shared project persistent cache may be reused; compile timings are not cold-cache guarantees.",
        "scope": "Standalone attention only, not whole-model throughput or quality."}
    for function in ("cuda_runtime_get_version", "cuda_driver_get_version", "cudnn_get_version"):
        try:
            report["versions"][function] = getattr(cuda_versions, function)()
        except Exception as error:
            report["versions"][function] = str(error)

    def fused(q, k, v, valid_length, pad_to_even):
        padding = (-valid_length) % 2 if pad_to_even else 0
        sizes = {}
        if padding:
            pads = ((0, 0), (0, padding), (0, 0), (0, 0))
            q, k, v = (jnp.pad(x, pads) for x in (q, k, v))
            lengths = jnp.full((q.shape[0],), valid_length, jnp.int32)
            sizes = dict(query_seq_lengths=lengths, key_value_seq_lengths=lengths)
        out = jax.nn.dot_product_attention(
            q, k, v, scale=float(args.head_dim ** -.5), implementation="cudnn", **sizes)
        return out[:, :valid_length]

    def compare(actual, reference):
        a, b = np.asarray(actual, np.float64), np.asarray(reference, np.float64)
        finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
        if not finite:
            raise FloatingPointError("Nonfinite attention output or VJP")
        delta = a - b
        return {"finite": finite,
                "max_abs": float(np.max(np.abs(delta))),
                "relative_l2": float(np.linalg.norm(delta.reshape(-1)) / np.linalg.norm(b.reshape(-1))),
                "reference_l2": float(np.linalg.norm(b.reshape(-1))),
                "candidate_l2": float(np.linalg.norm(a.reshape(-1)))}

    save_report(path, report)
    try:
        if args.mode == "lengths":
            report["cases"] = []
            for count in [int(x) for x in args.lengths.split(",")]:
                if count < 1:
                    raise ValueError("Sequence lengths must be positive")
                shape = (1, count, args.heads, args.head_dim)
                rng = np.random.default_rng(count)
                q, k, v, cot = [jax.device_put(rng.normal(size=shape).astype(np.float32)).astype(jnp.bfloat16)
                                for _ in range(4)]
                def compute(q, k, v, cot):
                    out, pullback = jax.vjp(lambda q, k, v: fused(q, k, v, count, True), q, k, v)
                    return out, pullback(cot)
                lowered = jax.jit(compute).lower(q, k, v, cot)
                exe = lowered.compile()
                out, gradients = exe(q, k, v, cot)
                jax.block_until_ready((out, gradients))
                targets = custom_calls(exe)
                require_cudnn_calls(targets, True)
                finite = all(bool(np.isfinite(np.asarray(x, np.float32)).all()) for x in (out, *gradients))
                if not finite:
                    raise FloatingPointError("Nonfinite cuDNN output or gradient")
                row = {"length": count, "padded_tokens": (-count) % 2,
                       "sequence_length_mask": bool(count % 2),
                       "forward_and_qkv_vjp_finite": finite, "custom_call_targets": targets,
                       **save_hlo(args, f"length_{count}", exe, lowered)}
                report["cases"].append(row)
                save_report(path, report)
                print(json.dumps(row), flush=True)
        else:
            shape = (1, length, args.heads * args.head_dim)
            rng = np.random.default_rng(args.seed)
            q, k, v = [jax.device_put(rng.normal(size=shape).astype(np.float32)).astype(jnp.bfloat16)
                       for _ in range(3)]
            cot = jax.device_put(rng.normal(0, .2, shape).astype(np.float32)).astype(jnp.bfloat16)
            rope = dit.rope_embeddings(args.image_height, args.image_width, args.text_tokens,
                                       args.head_dim, dtype=jnp.bfloat16)
            jax.block_until_ready((q, k, v, cot, rope))
            def native(q, k, v):
                return native_attention(q, k, v, rope, args.heads, backend="native")
            def cudnn(q, k, v):
                layout = (*q.shape[:2], args.heads, args.head_dim)
                q = dit.rotate(q.reshape(layout), rope).astype(v.dtype)
                k = dit.rotate(k.reshape(layout), rope).astype(v.dtype)
                v = v.reshape(layout)
                output = fused(q, k, v, length, args.mode == "padded")
                return dit._cast(output, v.dtype).reshape(shape)
            results = {}
            for name, function in (("native_remat_xla", native), ("cudnn", cudnn)):
                report["implementations"][name] = {}
                for mode in ("forward", "forward_vjp"):
                    if mode == "forward":
                        compute, arguments = function, (q, k, v)
                    else:
                        def compute(q, k, v, cot):
                            out, pullback = jax.vjp(function, q, k, v)
                            return out, pullback(cot)
                        arguments = (q, k, v, cot)
                    began = time.monotonic()
                    lowered = jax.jit(compute).lower(*arguments)
                    exe = lowered.compile()
                    compile_seconds = time.monotonic() - began
                    targets = custom_calls(exe)
                    if name == "cudnn":
                        require_cudnn_calls(targets, mode == "forward_vjp")
                    jax.block_until_ready(exe(*arguments))
                    timings = []
                    for _ in range(args.repeats):
                        began = time.perf_counter()
                        result = exe(*arguments)
                        jax.block_until_ready(result)
                        timings.append(time.perf_counter() - began)
                    results[name, mode] = result
                    memory = exe.memory_analysis()
                    item = {"compile_seconds": compile_seconds, "median_seconds": float(np.median(timings)),
                            "min_seconds": min(timings), "all_seconds": timings, "custom_call_targets": targets,
                            "compiled_memory": {key: getattr(memory, key, None) for key in
                                                ("argument_size_in_bytes", "output_size_in_bytes",
                                                 "temp_size_in_bytes", "alias_size_in_bytes")},
                            **save_hlo(args, name + "_" + mode, exe, lowered)}
                    report["implementations"][name][mode] = item
                    save_report(path, report)
                    print(json.dumps({"event": "measured", "implementation": name, "mode": mode, **item}), flush=True)
            report["numeric_comparison"] = {
                "forward_vs_native": compare(results["cudnn", "forward"], results["native_remat_xla", "forward"]),
                "qkv_vjp_vs_native": {label: compare(actual, reference) for label, actual, reference in
                    zip(("q", "k", "v"), results["cudnn", "forward_vjp"][1],
                        results["native_remat_xla", "forward_vjp"][1])}}
            report["speedup"] = {mode: report["implementations"]["native_remat_xla"][mode]["median_seconds"] /
                                report["implementations"]["cudnn"][mode]["median_seconds"]
                                for mode in ("forward", "forward_vjp")}
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
        raise
    finally:
        save_report(path, report)
        print(json.dumps({"event": "finished", "status": report["status"], "report": str(path)}), flush=True)


if __name__ == "__main__":
    main()
