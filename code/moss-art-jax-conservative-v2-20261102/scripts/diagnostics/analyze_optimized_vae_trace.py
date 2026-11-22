"""Attribute optimized VAE CUPTI kernels to 36 explicit cuDNN FFI operations."""
from __future__ import annotations
import argparse,csv,gzip,json,re,sys
from collections import defaultdict,Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.benchmark_vae_precision import atomic_json,sha256
from scripts.diagnostics.analyze_vae_gpu_trace import union_us,category
from scripts.diagnostics.profile_vae_layers import difference

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--capture",default=str(ROOT/"outputs/vae_conv_optimized_trace"))
 p.add_argument("--output",default=str(ROOT/"reports/vae_conv_optimized_trace.json"))
 a=p.parse_args();folder=Path(a.capture);capture=json.loads((folder/"capture.json").read_text())
 if capture["status"]!="captured":raise RuntimeError("Capture is incomplete")
 tracefile=next((folder/"trace").rglob("*.trace.json.gz"))
 events=json.loads(gzip.decompress(tracefile.read_bytes()))["traceEvents"]
 processes={e["pid"]:e.get("args",{}).get("name","") for e in events if e.get("ph")=="M" and e.get("name")=="process_name"}
 gpuids={pid for pid,name in processes.items() if "/device:GPU:" in name}
 scopes=[e for e in events if e.get("ph")=="X" and e.get("name")=="vae_decode_traced"]
 intervals=[(e["ts"],e["ts"]+e["dur"]) for e in scopes]
 gpu=[e for e in events if e.get("pid") in gpuids and e.get("ph")=="X" and e.get("dur",0)>0 and any(e["ts"]>=start and e["ts"]+e["dur"]<=end for start,end in intervals)]
 if not gpu or not all("kernel_details" in e.get("args",{}) for e in gpu):raise RuntimeError("Actual CUPTI kernel events required")
 text=(folder/"optimized.hlo.txt").read_text();definitions={}
 for lineno,line in enumerate(text.splitlines(),1):
  match=re.match(r"\s*(?:ROOT )?%([^ ]+) =",line)
  if match:definitions[match.group(1)]=(lineno,line)
 macs=[x for x in json.loads((ROOT/"reports/vae_layer_macs.json").read_text())["all_entries"] if x["kind"]=="conv"]
 plans={x["ffi_target"]:x for x in capture["plan_inventory"]}
 convmap={}
 for lineno,line in enumerate(text.splitlines(),1):
  if 'custom_call_target="moss_cudnn' not in line:continue
  opname=re.match(r"\s*%([^ ]+) =",line).group(1)
  target=re.search(r'custom_call_target="([^"]+)"',line).group(1)
  weight=re.search(r"custom-call\((.*?)\)",line).group(1).split(",")[1].strip().lstrip("%")
  seen,found=set(),set()
  def visit(name):
   if name in seen or name not in definitions:return
   seen.add(name);_,definition=definitions[name]
   for row in macs:
    if row["prefix"].replace(".","_")+"_weight__" in definition:found.add(row["prefix"])
   for operand in re.findall(r"%([A-Za-z0-9_.-]+)",definition)[1:]:visit(operand)
  visit(weight)
  if len(found)!=1:raise RuntimeError(f"Ambiguous weight identity for {opname}: {found}")
  prefix=found.pop();mac=next(x for x in macs if x["prefix"]==prefix)
  convmap[opname]={"prefix":prefix,"hlo_line":lineno,"ffi_target":target,"plan":plans[target],"MAC":mac["mac"],"stage":mac["stage"]}
 if len(convmap)!=36:raise RuntimeError(f"Expected 36 FFI convolutions, found {len(convmap)}")
 byop=defaultdict(list);categories=defaultdict(lambda:{"event_count":0,"gpu_seconds":0.})
 kernel_totals=defaultdict(lambda:{"event_count":0,"gpu_seconds":0.})
 ffi_categories=defaultdict(lambda:{"event_count":0,"gpu_seconds":0.})
 for e in gpu:
  op=e.get("args",{}).get("hlo_op");seconds=e["dur"]/1e6
  label="bounded_cudnn_convolution_including_internal_layout_helpers" if op in convmap else category(e,{})
  categories[label]["event_count"]+=1;categories[label]["gpu_seconds"]+=seconds
  kernel_totals[e["name"]]["event_count"]+=1;kernel_totals[e["name"]]["gpu_seconds"]+=seconds
  if op in convmap:
   byop[op].append(e);name=e["name"].lower()
   if any(word in name for word in ("nhwctonchw","nchwtonhwc","transpose","copy")):part="internal_layout_or_copy"
   elif any(word in name for word in ("convolve","sgemm")):part="convolution_math"
   else:part="other_cudnn_helper"
   ffi_categories[part]["event_count"]+=1;ffi_categories[part]["gpu_seconds"]+=seconds
 repeats=len(scopes);total=sum(e["dur"] for e in gpu)/1e6
 occupied=[(e["ts"],e["ts"]+e["dur"]) for e in gpu]
 union=union_us(occupied)/1e6;span=(max(b for _,b in occupied)-min(a for a,_ in occupied))/1e6
 rows=[];engine_counts=Counter();allnames=set();stage=defaultdict(float)
 for op,meta in convmap.items():
  kernels=byop[op]
  if not kernels:raise RuntimeError(f"No GPU kernels for {op}")
  names=sorted({e["name"] for e in kernels});allnames.update(names)
  seconds=sum(e["dur"] for e in kernels)/1e6/repeats
  plan=meta["plan"]["selected_plan"];engine_counts[plan["engine"]]+=1
  if any(n in plan["numerical_notes"] for n in ("TENSOR_CORE","DOWN_CONVERT_INPUTS","REDUCED_PRECISION_REDUCTION")):raise RuntimeError("Forbidden precision note in strict FP32 trace")
  rows.append({"hlo_op":op,"hlo_line":meta["hlo_line"],"prefix":meta["prefix"],"stage":meta["stage"],"ffi_target":meta["ffi_target"],"input_shape":meta["plan"]["input_shape"],"weight_shape":meta["plan"]["weight_shape"],"engine":plan["engine"],"knobs":plan["knobs"],"numerical_notes":plan["numerical_notes"],"workspace_bytes":plan["workspace_bytes"],"gpu_seconds_per_decode":seconds,"percent_all_gpu_time":100*seconds/(total/repeats),"kernel_event_count":len(kernels),"kernel_names":names,"MAC":meta["MAC"],"effective_TFLOP_s":2*meta["MAC"]/seconds/1e12})
  stage[meta["stage"]]+=seconds
 forbidden_names=[n for n in allnames if any(token in n.lower() for token in ("tf32","hmma","hgmma","xmma","tensorop"))]
 if forbidden_names:raise RuntimeError("Unexpected Tensor Core/TF32 name in strict FP32 convolution trace")
 for row in categories.values():row["percent_all_gpu_time"]=row["gpu_seconds"]/total*100;row["gpu_seconds_per_decode"]=row["gpu_seconds"]/repeats
 for row in ffi_categories.values():row["gpu_seconds_per_decode"]=row["gpu_seconds"]/repeats
 import numpy as np
 reference=ROOT/"outputs/vae_precision/fp32_reference/repeat_00/sample-000003.raw.npy"
 report={"status":"passed","capture":str(folder/"capture.json"),"capture_sha256":sha256(folder/"capture.json"),"trace_file":str(tracefile),"trace_sha256":sha256(tracefile),"hlo":str(folder/"optimized.hlo.txt"),"hlo_sha256":capture["hlo_sha256"],"source_hashes":capture["sources"],"backend_info":capture["backend_info"],"policy":{"timing":"Real CUPTI GPU kernel durations; distinct from synchronized host latency; includes only events inside one warm complete-decode annotation","convolution_attribution":"Kernel args.hlo_op joined to typed FFI custom-call HLO; weight operand dataflow uniquely identifies each of 36 VAE convolutions","internal_helpers":"Per-convolution time includes cuDNN's internal layouts/precompute helpers; external JAX transposes/bias/residual remain in separate categories","not_an_unprofiled_benchmark":True},"sample_index":capture["sample_index"],"reserved_bytes":capture["reserved_bytes"],"warmup_synchronized_host_seconds":capture["warmup_seconds"],"traced_synchronized_host_seconds":capture["traced_seconds"],"time_accounting":{"repeats":repeats,"kernel_sum_seconds":total,"kernel_union_seconds":union,"gpu_span_seconds":span,"overlap_seconds":total-union,"gap_seconds":span-union},"gpu_evidence":{"processes":{str(k):v for k,v in processes.items() if k in gpuids},"kernel_event_count":len(gpu),"streams":sorted({e["tid"] for e in gpu}),"all_have_kernel_details":True,"ffi_call_count":len(convmap),"ffi_kernel_event_count":sum(len(v) for v in byop.values())},"strict_fp32_evidence":{"engine_call_counts":dict(engine_counts),"all_compute_float32":all(p["compute"]=="float32" for p in capture["plan_inventory"]),"no_tensor_core_or_downconvert_or_reduced_accum_notes":True,"no_tf32_tensorcore_kernel_name_markers":True,"actual_convolution_kernel_names":sorted(allnames)},"kernel_categories":dict(categories),"within_cudnn_convolution_categories":dict(ffi_categories),"convolution_stage_seconds":dict(stage),"convolutions":rows,"all_kernel_name_totals":[{"name":name,**v} for name,v in sorted(kernel_totals.items(),key=lambda item:-item[1]["gpu_seconds"])],"compiled_memory":capture["compiled_memory"],"jax_peak_bytes_including_reservation":capture["final_memory"]["peak_bytes_in_use"],"finite":capture["traced_output_finite"],"trace_matches_warmup_bitwise":capture["traced_matches_warmup_bitwise"],"reference_comparison":difference(np.load(folder/"warmup.raw.npy"),np.load(reference))}
 out=Path(a.output);atomic_json(out,report)
 with out.with_suffix(".csv").open("w",newline="") as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader()
  for row in rows:writer.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()})
 print(json.dumps({k:report[k] for k in ("time_accounting","gpu_evidence","kernel_categories","within_cudnn_convolution_categories","convolution_stage_seconds","reference_comparison")},indent=2))
if __name__=="__main__":main()
