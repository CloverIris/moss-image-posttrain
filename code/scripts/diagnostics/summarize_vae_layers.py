"""Combine VAE isolated-layer/subop profiles without treating sums as full-graph time."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compact_operator(op, directory, fallback=None):
    measured = fallback if op.get("reuse_of_whole") else op
    hlo = measured.get("hlo", {})
    return {
        "name": op.get("name", "whole"),
        "measured_seconds": op["warm_median_seconds"],
        "warm_seconds": measured.get("warm_seconds"),
        "scope": "Synchronized host call including dispatch/sync: " + op["scope"],
        "estimated_all_chunks_seconds": op.get("estimated_all_chunks_seconds"),
        "query_chunk_count": op.get("query_chunk_count"),
        "share_of_decomposed_sum": op.get("share_of_decomposed_sum"),
        "reuse_of_whole": op.get("reuse_of_whole"),
        "MAC": op["MAC"],
        "estimated_all_chunks_MAC": op.get("estimated_all_chunks_MAC"),
        "input_shapes": measured.get("input_shapes"),
        "input_dtypes": measured.get("input_dtypes"),
        "parameter_dtypes": measured.get("parameter_dtypes"),
        "output": measured.get("output"),
        "compile_seconds_excluded": measured.get("compile_seconds"),
        "hlo": {
            "path": str(directory / hlo["path"]) if "path" in hlo else None,
            "sha256": hlo.get("sha256"),
            "custom_call_counts": hlo.get("custom_call_counts"),
            "operation_source_lines": [item["line"] for item in hlo.get("operation_instructions", [])],
            "compiled_memory": hlo.get("compiled_memory"),
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp32", default="outputs/vae_layer_profile/fp32/summary.json")
    parser.add_argument("--fp16", default="outputs/vae_layer_profile/full_fp16/summary.json")
    parser.add_argument("--output", default="reports/vae_layer_profile.json")
    args = parser.parse_args(argv)
    report = {
        "status": "passed",
        "scope": "Independent whole layers and isolated subops measured as synchronized host latency, including dispatch and block_until_ready synchronization overhead, excluding compilation; these sums are not full-graph end-to-end time or CUPTI GPU-kernel attribution",
        "timing_correction": {
            "measured_boundary": "time.monotonic around compiled(params, *inputs) followed by jax.block_until_ready(output)",
            "includes_host_dispatch_and_sync": True,
            "small_operator_caution": "For sub-millisecond operators host dispatch/synchronization overhead can materially inflate latency and decomposed shares; use complete-graph CUPTI for device kernel attribution.",
            "source_metadata_only_correction": json.loads((ROOT / "reports/vae_layer_profile_measured_sources.json").read_text()),
            "original_summaries_unchanged": True,
        },
        "attention_scope": "Whole attention processes all 16384 queries/keys; isolated QK/softmax/PV use the first real 256-query chunk with every key and label 64x extrapolation",
        "precision_scope": "FP16 decoder uses FP16 stored weights/activations, FP32 GroupNorm reductions/affine on stored FP16 weights, FP32 QK/PV accumulation and softmax before FP16 probability/output conversion",
        "attention_comparability": {
            "warning": "The isolated FP32 attention and complete decoder graph select different compiled GEMM kernels despite identical FP32 JAXPR; shared persistent autotune caches are enabled. The measured 17.85ms isolated kernel must not replace the complete-graph attention time or imply that FP32 attention is generally faster than FP16.",
            "evidence_path": str(ROOT / "reports/vae_fp32_attention_semantics.json"),
            "isolated_fp32_QK_PV_tile": "64x64 stages3",
            "prior_complete_graph_and_fp16_tile": "16x8 stages1",
        },
        "modes": {},
    }
    for mode, filename in (("fp32", args.fp32), ("full_fp16", args.fp16)):
        path = (ROOT / filename).resolve()
        original = json.loads(path.read_text())
        if original["status"] != "passed":
            raise ValueError(f"{path}: status is {original['status']}")
        if len(original["layers"]) != 22 or original["total_MAC"] != 2_998_675_701_760:
            raise AssertionError("Unexpected real MOSS decoder layer/MAC count")
        if original["sample_index"] != 3:
            raise AssertionError("Expected real sample 3")
        layers = []
        for layer in original["layers"]:
            whole = compact_operator(layer["whole"], path.parent)
            if not all(item["finite"] for item in whole["output"]):
                raise AssertionError("Non-finite whole-layer output")
            layers.append({
                "name": layer["name"], "kind": layer["kind"],
                "whole": whole,
                "whole_share_of_layer_sum": layer["whole_share_of_layer_sum"],
                "atomic_subop_sum_seconds": layer["atomic_subop_sum_seconds"],
                "atomic_vs_whole": layer.get("atomic_vs_whole"),
                "subops": [compact_operator(op, path.parent, layer["whole"]) for op in layer["subops"]],
                "jax_peak_bytes": layer["jax_memory_after"]["peak_bytes_in_use"],
            })
        report["modes"][mode] = {
            "original_summary": str(path), "original_summary_sha256": sha256(path),
            "physical_device": original["physical_device"], "arguments": original["arguments"],
            "source_hashes": original["source_hashes"], "input_sha256": original["input_sha256"],
            "reference_raw_sha256": original["reference_raw_sha256"],
            "reference_comparison": original["reference_comparison"],
            "policy": {**original["policy"],
                "whole_layers": report["timing_correction"]["source_metadata_only_correction"]["corrected_policy"],
                "original_whole_layers_policy_corrected": original["policy"]["whole_layers"]},
            "reservation_bytes": original["reservation_bytes"],
            "whole_layer_sum_seconds_not_end_to_end": original["whole_layer_sum_seconds_not_end_to_end"],
            "total_MAC": original["total_MAC"], "parameter_bytes": original["parameter_bytes"],
            "max_jax_peak_bytes_in_use_including_reservation": max(x["jax_peak_bytes"] for x in layers),
            "nvidia_smi": original["nvidia_smi"], "total_profile_wall_seconds": original["total_seconds"],
            "raw_output": str(path.parent / "layerwise.raw.npy"), "layers": layers,
        }
    f32, f16 = report["modes"]["fp32"], report["modes"]["full_fp16"]
    if f32["input_sha256"] != f16["input_sha256"]:
        raise AssertionError("Inputs differ")
    report["comparison"] = {
        "isolated_layer_sum_speedup_fp32_over_fp16": f32["whole_layer_sum_seconds_not_end_to_end"] / f16["whole_layer_sum_seconds_not_end_to_end"],
        "layers": [{
            "name": a["name"], "fp32_seconds": a["whole"]["measured_seconds"],
            "fp16_seconds": b["whole"]["measured_seconds"],
            "speedup_fp32_over_fp16": a["whole"]["measured_seconds"] / b["whole"]["measured_seconds"],
            "fp32_share": a["whole_share_of_layer_sum"], "fp16_share": b["whole_share_of_layer_sum"],
        } for a, b in zip(f32["layers"], f16["layers"], strict=True)],
    }
    output = (ROOT / args.output).resolve()
    output.write_text(json.dumps(report, indent=2) + "\n")
    columns = [
        "mode", "layer", "kind", "operator", "scope", "measured_seconds",
        "estimated_all_chunks_seconds", "whole_layer_share", "decomposed_subop_share",
        "MAC", "estimated_all_chunks_MAC", "input_shapes", "input_dtypes", "output_shapes",
        "output_dtypes", "output_finite", "compile_seconds_excluded", "hlo_path", "hlo_sha256",
        "hlo_source_lines", "argument_bytes", "output_bytes", "temp_bytes", "jax_peak_bytes",
        "atomic_vs_whole_max_abs", "atomic_vs_whole_relative_l2",
    ]
    with output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for mode, result in report["modes"].items():
            for layer in result["layers"]:
                for op in (layer["whole"], *layer["subops"]):
                    memory = op["hlo"].get("compiled_memory") or {}
                    error = layer.get("atomic_vs_whole") or {}
                    writer.writerow({
                        "mode": mode, "layer": layer["name"], "kind": layer["kind"],
                        "operator": op["name"], "scope": op["scope"],
                        "measured_seconds": op["measured_seconds"],
                        "estimated_all_chunks_seconds": op["estimated_all_chunks_seconds"],
                        "whole_layer_share": layer["whole_share_of_layer_sum"] if op["name"] == "whole" else "",
                        "decomposed_subop_share": op["share_of_decomposed_sum"],
                        "MAC": op["MAC"], "estimated_all_chunks_MAC": op["estimated_all_chunks_MAC"],
                        "input_shapes": json.dumps(op["input_shapes"]),
                        "input_dtypes": json.dumps(op["input_dtypes"]),
                        "output_shapes": json.dumps([a["shape"] for a in op["output"]]),
                        "output_dtypes": json.dumps([a["dtype"] for a in op["output"]]),
                        "output_finite": all(a["finite"] for a in op["output"]),
                        "compile_seconds_excluded": op["compile_seconds_excluded"],
                        "hlo_path": op["hlo"]["path"], "hlo_sha256": op["hlo"]["sha256"],
                        "hlo_source_lines": json.dumps(op["hlo"]["operation_source_lines"]),
                        "argument_bytes": memory.get("argument_size_in_bytes"),
                        "output_bytes": memory.get("output_size_in_bytes"),
                        "temp_bytes": memory.get("temp_size_in_bytes"),
                        "jax_peak_bytes": layer["jax_peak_bytes"],
                        "atomic_vs_whole_max_abs": error.get("max_abs"),
                        "atomic_vs_whole_relative_l2": error.get("relative_l2"),
                    })
    print(json.dumps({
        "output": str(output), "csv": str(output.with_suffix(".csv")),
        "modes": {k: {field: v[field] for field in (
            "whole_layer_sum_seconds_not_end_to_end", "reference_comparison",
            "max_jax_peak_bytes_in_use_including_reservation",
        )} for k, v in report["modes"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
