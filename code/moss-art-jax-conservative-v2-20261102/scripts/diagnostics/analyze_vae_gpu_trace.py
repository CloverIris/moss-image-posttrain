#!/usr/bin/env python3
"""CPU-only whole-graph CUPTI event accounting with explicit unattributed work."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.benchmark_vae_precision import atomic_json,sha256


def union_us(intervals):
    total=0.0
    end=None
    for start,stop in sorted(intervals):
        if end is None or start>end:
            total+=stop-start
        elif stop>end:
            total+=stop-end
        end=max(end if end is not None else stop,stop)
    return total


def hlo_convolutions(text,macs):
    result={}
    for line_no,line in enumerate(text.splitlines(),1):
        if 'custom_call_target="__cudnn$conv' not in line:
            continue
        op=re.match(r"\s*%([^ ]+) =",line).group(1)
        matches=[row for row in macs if "%p__"+row["prefix"].replace(".","_")+"_bias__" in line]
        if len(matches)!=1:
            raise ValueError(f"Expected one parameter prefix for {op}; got {len(matches)}")
        target=re.search(r'custom_call_target="([^"]+)"',line).group(1)
        dtype=re.search(r"= \((f16|f32|bf16)\[",line).group(1)
        result[op]={"prefix":matches[0]["prefix"],"stage":matches[0]["stage"],"hlo_line":line_no,
                    "custom_call_target":target,"output_dtype":dtype,"mac_entry":matches[0],
                    "fuses_side_input":'"side_input_scale":1' in line}
    return result


def category(event,conv_map):
    name=event["name"].lower()
    if event.get("args",{}).get("hlo_op") in conv_map or "cudnn::cnn::conv" in name:
        return "convolution_including_fused_bias_and_optional_residual"
    if any(word in name for word in ("gemm","gemv","matmul","mma")):
        return "matmul_including_attention_and_projections"
    if name.startswith("memcpy32"):
        return "device_copy_helper_kernel"
    if "softmax" in name:
        return "softmax_kernel"
    if "reduce" in name:
        return "reduction_kernel_including_norm_and_softmax"
    if "transpose" in name or "copy" in name:
        return "layout_transpose_or_copy"
    if "fusion" in name:
        return "other_fusion_including_elementwise_norm_silu_residual"
    return "unclassified_kernel"


def one_run(path,macs):
    capture=json.loads((path/"capture.json").read_text())
    if capture["status"]!="captured":
        raise ValueError("Capture incomplete")
    trace_path=next((path/"trace").rglob("*.trace.json.gz"))
    trace=json.loads(gzip.decompress(trace_path.read_bytes()))
    events=trace["traceEvents"]
    processes={e["pid"]:e.get("args",{}).get("name","") for e in events if e.get("ph")=="M" and e.get("name")=="process_name"}
    gpu_pids={pid for pid,name in processes.items() if "/device:GPU:" in name}
    if not gpu_pids:
        raise RuntimeError("No real GPU process in trace; host timing is insufficient")
    scopes=[e for e in events if e.get("ph")=="X" and e.get("name")=="vae_decode_traced"]
    if len(scopes)!=len(capture["traced_seconds"]):
        raise ValueError("Trace/repetition scope mismatch")
    ranges=[(e["ts"],e["ts"]+e["dur"]) for e in scopes]
    all_gpu=[e for e in events if e.get("pid") in gpu_pids and e.get("ph")=="X" and e.get("dur",0)>0]
    gpu=[e for e in all_gpu if any(e["ts"]>=a and e["ts"]+e["dur"]<=b for a,b in ranges)]
    if not gpu or not all("kernel_details" in e.get("args",{}) for e in gpu):
        raise RuntimeError("Expected real CUPTI GPU kernel events with kernel_details")
    conv_map=hlo_convolutions((path/"optimized.hlo.txt").read_text(),macs)
    if len(conv_map)!=36:
        raise ValueError("Expected 36 actual decoder convolutions")
    by_category=defaultdict(lambda:{"events":0,"gpu_seconds":0.})
    by_kernel=defaultdict(lambda:{"events":0,"gpu_seconds":0.})
    by_conv=defaultdict(list)
    intervals=[]
    for event in gpu:
        seconds=event["dur"]/1e6
        intervals.append((event["ts"],event["ts"]+event["dur"]))
        for group,key in ((by_category,category(event,conv_map)),(by_kernel,event["name"])):
            group[key]["events"]+=1
            group[key]["gpu_seconds"]+=seconds
        if event.get("args",{}).get("hlo_op") in conv_map:
            by_conv[event["args"]["hlo_op"]].append(event)
    total=sum(event["dur"] for event in gpu)/1e6
    union=union_us(intervals)/1e6
    span=(max(b for a,b in intervals)-min(a for a,b in intervals))/1e6
    repeats=len(scopes)
    conv_rows=[]
    for op,meta in conv_map.items():
        rows=by_conv.get(op,[])
        if len(rows)!=repeats:
            raise ValueError(f"Expected one kernel per traced invocation for {op}: {len(rows)}")
        seconds=sum(e["dur"] for e in rows)/1e6/repeats
        mac_entry=meta.pop("mac_entry")
        conv_rows.append(dict(meta,hlo_op=op,gpu_seconds_per_decode=seconds,event_count=len(rows),
                              percent_all_gpu_time=100*seconds/(total/repeats),
                              kernel_names=sorted(set(e["name"] for e in rows)),
                              nominal_mac=mac_entry["mac"],effective_tflop_s=2*mac_entry["mac"]/seconds/1e12))
    stage=defaultdict(lambda:{"convolution_count":0,"convolution_gpu_seconds_per_decode":0.})
    for row in conv_rows:
        stage[row["stage"]]["convolution_count"]+=1
        stage[row["stage"]]["convolution_gpu_seconds_per_decode"]+=row["gpu_seconds_per_decode"]
    for label in ("convolution_including_fused_bias_and_optional_residual",
                  "matmul_including_attention_and_projections",
                  "reduction_kernel_including_norm_and_softmax","layout_transpose_or_copy",
                  "other_fusion_including_elementwise_norm_silu_residual",
                  "device_copy_helper_kernel","softmax_kernel","unclassified_kernel"):
        by_category[label]
    for row in by_category.values():
        row["percent_all_gpu_time"]=100*row["gpu_seconds"]/total
        row["gpu_seconds_per_decode"]=row["gpu_seconds"]/repeats
    conv_total=sum(row["gpu_seconds_per_decode"] for row in conv_rows)*repeats
    return {
        "variant":capture["variant"],"physical_device":capture["physical_device"],
        "capture_json":str(path/"capture.json"),"capture_sha256":sha256(path/"capture.json"),
        "input_sha256":capture["input_sha256"],"source_hashes":capture["sources"],
        "trace_file":str(trace_path),"trace_sha256":sha256(trace_path),"hlo_sha256":capture["hlo_sha256"],
        "reserved_bytes":capture["reserved_bytes"],"warmup_seconds":capture["warmup_seconds"],
        "traced_synchronized_host_seconds":capture["traced_seconds"],
        "debug_logging_enabled":capture["runtime_environment"].get("CUDNN_LOGLEVEL_DBG")=="3",
        "gpu_event_evidence":{"processes":{str(k):v for k,v in processes.items() if k in gpu_pids},
                              "stream_ids":sorted(set(e["tid"] for e in gpu)),
                              "kernel_details_present":True,"all_gpu_events":len(all_gpu),"within_decode_scopes":len(gpu),
                              "excluded_gpu_events_outside_scopes":len(all_gpu)-len(gpu),
                              "non_kernel_gpu_events":sum("kernel_details" not in e.get("args",{}) for e in all_gpu),
                              "cupti_memcpy_or_memset_engine_events":sum(
                                  any("memcpy_details" in k or "memset_details" in k for k in e.get("args",{}))
                                  for e in all_gpu),
                              "memcpy32_post_kernel_events":sum(e["name"].startswith("memcpy32") for e in gpu),
                              "copy_scope_note":"memcpy32_post has kernel_details/register/grid/block on compute stream; it is a GPU device-copy helper kernel, not a CUPTI transfer-engine H2D/D2H event. It is retained in the kernel-only denominator."},
        "time_accounting":{"repeats":repeats,"sum_kernel_seconds":total,"union_kernel_seconds":union,
                           "overlap_sum_minus_union_seconds":total-union,"gpu_first_to_last_span_seconds":span,
                           "gap_span_minus_union_seconds":span-union,
                           "sum_kernel_seconds_per_decode":total/repeats,
                           "nonconv_time_unattributed_to_individual_layers_seconds_per_decode":(total-conv_total)/repeats},
        "kernel_categories":dict(by_category),
        "convolution_stage_subtotals":dict(stage),
        "convolutions":conv_rows,
        "kernel_name_totals":[dict(name=name,**row) for name,row in sorted(by_kernel.items(),key=lambda pair:-pair[1]["gpu_seconds"])],
        "output_finite":capture["traced_output_finite"],
        "output_matches_warmup_bitwise":capture["traced_matches_warmup_bitwise"],
        "jax_peak_active_bytes":capture["final_memory"]["peak_bytes_in_use"],
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",default=str(ROOT/"reports/vae_gpu_trace.json"))
    parser.add_argument("--overwrite",action="store_true")
    args=parser.parse_args()
    output=Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(output)
    mac_path=ROOT/"reports/vae_layer_macs.json"
    macs=[row for row in json.loads(mac_path.read_text())["all_entries"] if row["kind"]=="conv"]
    runs=[one_run(ROOT/"outputs/vae_gpu_trace"/name,macs) for name in ("fp32","full_fp16")]
    assert runs[0]["input_sha256"]==runs[1]["input_sha256"]
    report={
        "status":"completed","recorded_utc":datetime.now(timezone.utc).isoformat(),
        "scope":"Actual CUPTI GPU kernel events from each unchanged complete 1024 VAE JIT graph after a synchronized warmup; explicit params and latent arguments; sample3",
        "parser_sha256":sha256(Path(__file__)),"mac_report_sha256":sha256(mac_path),
        "units":"Chrome trace ts/dur are microseconds; seconds here divide by 1e6 and are reconciled with separately synchronized wall-clock decode.",
        "attribution":"Convolution event args.hlo_op matches actual optimized HLO custom-call; its explicit bias parameter operand uniquely identifies decoder prefix. All 36 convs mapped. Conv timings include fused bias and optional residual; cannot decompose these internal operations. Remaining GPU kernels categorized by kernel name but not assigned to individual layers.",
        "overlap":"Kernel totals sum durations and can double-count overlapping kernels. Union merges timestamp intervals across streams; both are reported with span and gaps. These captures use one compute stream; sub-microsecond overlap can reflect timestamp granularity.",
        "limitations":["One traced decode after one warmup for each variant; no latency distribution.",
            "FP16 cuDNN debug logging enabled for algorithm evidence, FP32 logging disabled; traced times are diagnostics, not a controlled precision speed claim.",
            "No H2D, D2H, parameter load, compile, file writing or reward inside traced region.",
            "9.6GiB live device buffers approximate memory pressure but are not actual DiT.",
            "No vendor edits or named-scope monkeypatch; CUDA graph command-buffer fusion events are not forcibly assigned to GN/SiLU individually.",
            "Static MAC-derived throughput is nominal 2*MAC/kernel_seconds, not a measured hardware instruction or Tensor Core rate."],
        "runs":runs,
    }
    csv_path=output.with_name("vae_gpu_trace_convolutions.csv")
    if csv_path.exists() and not args.overwrite:
        raise FileExistsError(csv_path)
    fields=["variant","physical_device","prefix","stage","hlo_op","hlo_line","custom_call_target","output_dtype",
            "fuses_side_input","gpu_seconds_per_decode","percent_all_gpu_time","event_count",
            "nominal_mac","effective_tflop_s","kernel_names","debug_logging_enabled"]
    with csv_path.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for run in runs:
            for row in run["convolutions"]:
                writer.writerow({**row,"variant":run["variant"],"physical_device":run["physical_device"],
                                 "kernel_names":" | ".join(row["kernel_names"]),"debug_logging_enabled":run["debug_logging_enabled"]})
    report["convolution_csv"]={"path":str(csv_path),"sha256":sha256(csv_path),"rows":sum(len(r["convolutions"]) for r in runs)}
    report["denominator"]="Sum of actual CUPTI GPU kernel durations only; excludes CPU API/host scopes and nonkernel transfer-engine events. Device-copy helper kernels remain included."
    atomic_json(output,report)
    print(json.dumps({"report":str(output),"results":[{"variant":r["variant"],"categories":r["kernel_categories"],
         "time_accounting":r["time_accounting"],"top5_convolutions":sorted(r["convolutions"],key=lambda row:-row["gpu_seconds_per_decode"])[:5]} for r in runs]}))


if __name__=="__main__":
    main()
