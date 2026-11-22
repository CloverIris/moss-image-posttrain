#!/usr/bin/env python3
"""Summarize four completed bounded tiled-convolution experiments; CPU only."""
from __future__ import annotations
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.benchmark_vae_precision import atomic_json,sha256


def main():
    output=ROOT/"reports/vae_tiled_conv_fallback.json"
    csv_path=ROOT/"reports/vae_tiled_conv_fallback.csv"
    if output.exists() or csv_path.exists():
        raise FileExistsError("Refusing to overwrite existing fallback report")
    rows=[]
    source_hashes=set()
    for dtype in ("fp32","fp16"):
        for shape in ("1024_192","512_384"):
            directory=ROOT/"outputs/tiled_conv"/("bounded_"+dtype+"_"+shape)
            report=json.loads((directory/"summary.json").read_text())
            assert report["status"]=="passed"
            source_hashes.add(report["source_sha256"])
            direct=report["methods"]["direct_cudnn"]
            tiled=report["methods"]["tiled_im2col_gemm"]
            assert direct["finite"] and tiled["finite"]
            assert tiled["compiled_memory"]["temp_size_in_bytes"]<256*1024**2
            hlo=(directory/"tiled_im2col_gemm.hlo.txt").read_text()
            assert "__triton_nested_gemm_fusion" in hlo
            assert "operand_precision={highest,highest}" in hlo
            assert not any("conv" in target.lower() for target in tiled["custom_call_targets"])
            evidence=[{"line":n,"text":line[:2500]} for n,line in enumerate(hlo.splitlines(),1)
                      if "__triton_nested_gemm_fusion" in line or " dot(" in line or line.startswith("%gemm_fusion")]
            rows.append({
                "label":dtype+"_"+shape,"shape":report["case"],"source_sha256":report["source_sha256"],
                "summary_path":str(directory/"summary.json"),"summary_sha256":sha256(directory/"summary.json"),
                "direct_seconds":direct["times"],"tiled_seconds":tiled["times"],
                "direct_mean_seconds":direct["mean_seconds"],"tiled_mean_seconds":tiled["mean_seconds"],
                "observed_speedup":report["observed_speedup"],
                "direct_compiled_temp_bytes":direct["compiled_memory"]["temp_size_in_bytes"],
                "tiled_compiled_temp_bytes":tiled["compiled_memory"]["temp_size_in_bytes"],
                "argument_bytes":tiled["compiled_memory"]["argument_size_in_bytes"],
                "output_bytes":tiled["compiled_memory"]["output_size_in_bytes"],
                "logical_patch_tile_bytes":report["logical_patch_tile_bytes"],
                "full_im2col_bytes_avoided":report["full_im2col_bytes_avoided"],
                "numeric_error":tiled["comparison_to_direct"],
                "direct_custom_call_targets":direct["custom_call_targets"],
                "tiled_custom_call_targets":tiled["custom_call_targets"],
                "tiled_hlo_path":str(directory/"tiled_im2col_gemm.hlo.txt"),
                "tiled_hlo_sha256":tiled["hlo_sha256"],"tiled_hlo_evidence":evidence,
                "compile_seconds":{"direct":direct["compile_seconds"],"tiled":tiled["compile_seconds"]},
                "process_peak_including_baseline_bytes":report["jax_memory_after_tiled_im2col_gemm"]["peak_bytes_in_use"],
                "runtime_environment":report["runtime_environment"],"versions":report["versions"],
            })
    assert len(source_hashes)==1
    result={
        "status":"completed","recorded_utc":datetime.now(timezone.utc).isoformat(),
        "summary_script_sha256":sha256(Path(__file__)),
        "scope":"Fallback prototype only: two N=1 3x3 same conv+bias shapes, FP32/FP16, synthetic input/weights, no production/vendor changes.",
        "method":"Gather a tile directly from NCHW with zero-padding masks; concatenate 9 patches; HIGHEST GEMM with preferred FP32 result; cast to input dtype and bias; update NCHW output in fori_loop. No whole-image im2col, padding or layout copies.",
        "workspace":"Compiler-reported temporary bytes exclude necessary full input, weights and output. Every tiled executable temp_size is below 256MiB, asserted before execution.",
        "timing":"One unmeasured synchronized warmup then two synchronized already-device calls; compile/H2D/D2H excluded. Same physical GPU2, same inputs per direct/tiled pair.",
        "precision":"Source requests HIGHEST, no low-precision TF32 setting. Optimized HLO preserves operand_precision highest and FP32 dot result. Actual hardware instruction precision is not inferred solely from HLO; no silent dtype change.",
        "data":"Seed 20261020; synthetic activations std .25, weights std 1/sqrt(9*Cin), bias std .01; each dtype receives rounded versions of same generated arrays.",
        "limitations":["Only two shapes, batch1,3x3,stride1,padding1 and tile4096 dividing output size; no generic operator promise.",
                       "FP16 may differ through accumulation order and conv/bias rounding boundaries; model images/reward/full rollout not tested here.",
                       "No 9.6GiB DiT reservation and no real model activations; not a full training memory validation.",
                       "Compiler settings belong to each source report: inspect runtime_environment and recorded local method options. Historical runs used global autotune0; current diagnostic uses explicit local level0/4 without injecting global policy.",
                       "Existing engine0 direct convolution is exceptionally slow; speedup is against that measured baseline only.",
                       "Process memory peaks include the direct baseline and are not an isolated tiled-method peak.",
                       "Initial layout-copy prototype is excluded; its 809MB temporary workspace did not meet the final bound."],
        "runs":rows,
    }
    fields=("label","direct_mean_seconds","tiled_mean_seconds","observed_speedup","tiled_compiled_temp_bytes",
            "argument_bytes","output_bytes","relative_l2","max_abs","mae")
    with csv_path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key:row.get(key,row["numeric_error"].get(key)) for key in fields})
    result["csv"]={"path":str(csv_path),"sha256":sha256(csv_path)}
    atomic_json(output,result)
    print(json.dumps({"report":str(output),"results":[{key:row[key] for key in
        ("label","direct_mean_seconds","tiled_mean_seconds","observed_speedup","tiled_compiled_temp_bytes","numeric_error")} for row in rows]}))


if __name__=="__main__":
    main()
