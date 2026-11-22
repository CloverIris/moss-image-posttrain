"""Parse existing cuDNN API logs without loading CUDA, JAX, or model weights."""
from __future__ import annotations
import argparse,collections,hashlib,json,re
from pathlib import Path

REV="5a9e73cbd92530cac2ac36f4736a774b2412afe2"
GITHUB="https://github.com/openxla/xla/blob/"+REV+"/"
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def counts(xs):return dict(collections.Counter(str(x) for x in xs))
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--log",default="outputs/vae_gpu_trace/full_fp16_cudnn.log")
    ap.add_argument("--output",default="reports/vae_cudnn_engine_audit.json")
    a=ap.parse_args();root=Path(__file__).resolve().parents[2];path=(root/a.log).resolve()
    text=path.read_text();lines=text.splitlines()
    starts=[i for i,s in enumerate(lines) if re.search(r"function \w+\(\) called:",s)]
    blocks=[(start+1,"\n".join(lines[start:end])) for start,end in zip(starts,starts[1:]+[len(lines)])]
    attrs=[];executions=[];internals=[]
    for line,b in blocks:
        function=re.search(r"function (\w+)\(\) called:",b).group(1)
        if function=="cudnnBackendSetAttribute":
            attr=re.search(r"attributeName:.*?val=(\w+)",b)
            val=re.search(r"\[0\]:.*?val=([^;\n]+)",b)
            if attr and val and attr.group(1) in ["CUDNN_ATTR_ENGINE_GLOBAL_INDEX","CUDNN_ATTR_CONVOLUTION_COMP_TYPE"]:
                attrs.append(dict(line=line,attribute=attr.group(1),value=val.group(1)))
        if function=="cudnnBackendExecute" and "executionPlan:" in b:
            engine=re.search(r"engine_id: type=int; val=(-?\d+);",b)
            compute=re.search(r"computeType:.*?val=([^;\n]+)",b)
            tensors=[]
            for m in re.finditer(r"(xDesc|wDesc|yDesc|bDesc): type=CUDNN_BACKEND_TENSOR_DESCRIPTOR:\n(.*?)(?=\n\w! [^\n]*?(?:xDesc|wDesc|yDesc|bDesc|convDesc|operation|pwDesc|alpha|engine_id):|\Z)",b,re.S):
                body=m.group(2)
                dtype=re.search(r"type: type=cudnnDataType_t; val=(\w+)",body)
                shape=re.search(r"dimA: type=int; val=(\[[\d,]+\])",body)
                virtual=re.search(r"isVirtual: type=bool; val=(true|false)",body)
                if dtype and shape:
                    tensors.append(dict(role=m.group(1),dtype=dtype.group(1),shape=json.loads(shape.group(1)),
                        virtual=virtual.group(1)=="true" if virtual else None))
            notes=sorted(set(re.findall(r"CUDNN_NUMERICAL_NOTE_[A-Z_]+",b)))
            relevant=[]
            for offset,s in enumerate(b.splitlines()):
                if any(k in s for k in ["computeType:","engine_id:","CUDNN_NUMERICAL_NOTE_"]):
                    relevant.append(dict(line=line+offset,text=s.strip()))
            executions.append(dict(line=line,engine_id=int(engine.group(1)) if engine else None,
                convolution_compute_type=compute.group(1) if compute else None,
                numerical_notes=notes,tensors=tensors,evidence_lines=relevant))
        if function=="cudnnBackendExecuteInternal":
            engine=re.search(r"engine_id: type=int; val=(-?\d+);",b)
            math=re.search(r"mathType:.*?val=([^;\n]+)",b)
            internals.append(dict(line=line,engine_id=int(engine.group(1)) if engine else None,
                math_type=math.group(1) if math else None,contains_empty_legacy_descriptors="nbDims: type=int; val=0;" in b))
    assert len(executions)==72 and len(attrs)==72,(len(executions),len(attrs))
    assert all(e["engine_id"]==0 for e in executions)
    assert all(e["convolution_compute_type"]=="CUDNN_DATA_FLOAT (0)" for e in executions)
    assert all(e["tensors"][0]["dtype"]=="CUDNN_DATA_HALF" and e["tensors"][1]["dtype"]=="CUDNN_DATA_HALF" for e in executions)
    sources=[
        ("xla/service/gpu/autotuning/autotuner_pass.cc",124,133,"Autotune decision skips cuDNN custom calls when disabled."),
        ("xla/service/gpu/autotuning/autotuner_pass.cc",346,352,"Global autotune level zero disables CUDA cuDNN custom-call autotuning."),
        ("xla/service/gpu/gpu_conv_runner.cc",323,337,"Runtime config constructs AlgorithmDesc from backend algorithm proto."),
        ("xla/stream_executor/lazy_op_runner.h",103,110,"Lazy runner passes descriptor unchanged to runner construction."),
        ("xla/stream_executor/lazy_op_runner.h",225,233,"Fused runner dispatches to FusedConvolveRunnerFromDesc."),
        ("xla/stream_executor/cuda/cuda_dnn.cc",3906,3947,"RebuildExecutionPlan uses descriptor algorithm id as frontend GlobalEngineIdx and applies tuning knobs."),
        ("xla/stream_executor/cuda/cuda_dnn.cc",5520,5551,"Fused convolution reconstructs operation graph and execution plan."),
        ("xla/stream_executor/cuda/cuda_dnn.cc",2547,2562,"Half and BF16 convolution accumulator type requests float when FP32 compute flag is enabled."),
        ("xla/xla.proto",1708,1717,"Former redzone_scratch_max_megabytes and strict_conv_algorithm_picker names are reserved, not active supported flags.")]
    src=[]
    for rel,start,end,claim in sources:
        p=root/"outputs/vae_profile_source_audit"/Path(rel).name
        src.append(dict(path=rel,line_start=start,line_end=end,url=GITHUB+rel+f"#L{start}-L{end}",
            description=claim,downloaded_source_sha256=digest(p) if p.exists() else None))
    report=dict(status="completed",scope="Read-only parse of one full-FP16 whole-graph cuDNN diagnostic log; no new GPU execution",
        log_path=str(path),log_sha256=digest(path),log_bytes=path.stat().st_size,
        script_sha256=digest(Path(__file__)),cudnn_version_log=sorted(set(re.findall(r"CuDNN \((v\d+ \d+)\)",text))),
        observed_execution_count=len(executions),expected_convolutions_per_decode=36,
        execute_engine_id_counts=counts(e["engine_id"] for e in executions),
        execute_compute_type_counts=counts(e["convolution_compute_type"] for e in executions),
        execute_numerical_note_counts=counts(n for e in executions for n in e["numerical_notes"]),
        tensor_core_note_occurrences_in_entire_log=text.count("CUDNN_NUMERICAL_NOTE_TENSOR_CORE"),
        attribute_set_counts={name:counts(e["value"] for e in attrs if e["attribute"]==name) for name in ["CUDNN_ATTR_ENGINE_GLOBAL_INDEX","CUDNN_ATTR_CONVOLUTION_COMP_TYPE"]},
        attributes=attrs,executions=executions,
        internal_legacy_math_type_counts=counts(e["math_type"] for e in internals),
        internal_legacy_empty_descriptor_count=sum(e["contains_empty_legacy_descriptors"] for e in internals),
        conclusion=[
            "All36 convolution descriptors are configured with frontend engine0 and FP32 compute; all72 logged actual executions retain engine0.",
            "Real operation-graph input/filter tensors are FP16. Fused convolution intermediate output can be virtual FP32 before bias and final FP16 output; do not infer FP32 storage across decoder layers.",
            "Engine numerical notes shown in this log contain STRICT_NAN_PROP only, with no TENSOR_CORE note. Combine this API metadata with the independent CUPTI grouped_direct kernel evidence; note absence alone is not proof of executed hardware instructions.",
            "The internal legacy DEFAULT_MATH field is accompanied by empty legacy tensor/convolution descriptors and is not sufficient standalone hardware evidence.",
            "Frontend engine index0 is not the legacy cudnnConvolutionFwdAlgo_t enum0; do not call it IMPLICIT_GEMM based on numeric equality.",
            "Original full-FP32 and full-FP16 HLO custom-call scratch output is u8[0]; this supports zero declared XLA scratch, not a claim about all cuDNN internal memory."],
        upstream_sources=src,
        official_logging_docs="https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/troubleshooting.html",
        official_engine_note_docs="https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-graph-library.html#cudnnbackendnumericalnote-t",
        profiling_cautions=[
            "CUDNN_LOGLEVEL_DBG=3 adds logging overhead; logged wall time must not replace unlogged benchmark timing.",
            "Physical GPU3 is visible as logical CUDA device0 in this subprocess; do not mistake this for use of physical GPU0.",
            "72 detailed execute records reflect two executions of36 convolutions, not72 distinct model layers.",
            "Do not claim Tensor Core instruction use or absence from precision dtype, throughput, missing HLO algorithm field, or one enum alone."],
        workspace_limit_audit=dict(status="no_supported_conv_workspace_cap_identified_in_pinned_debug_flags",
            searched_source_files=["xla/debug_options_flags.cc","xla/xla.proto","xla/backends/gpu/autotuner/cudnn.cc"],
            absent_or_irrelevant_flags={
                "xla_gpu_redzone_scratch_max_megabytes":"reserved removed flag in pinned xla.proto",
                "xla_gpu_memory_limit_slop_factor":"compiler scheduling/rematerialization threshold, not a per-cuDNN-plan workspace cap",
                "xla_gpu_autotune_max_solutions":"deprecated GEMM solution count, not conv workspace bytes",
                "xla_gpu_cudnn_gemm_max_plans":"cuDNN GEMM-fusion plan count, not convolution workspace bytes"},
            next_step="A bounded-workspace conv plan search needs verified plan filtering or vetted cached algorithm descriptors; unrestricted global autotune4 is not justified by these read-only findings.",
            experiments_run=False))
    out=root/a.output;out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:report[k] for k in ["status","observed_execution_count","execute_engine_id_counts","execute_compute_type_counts","execute_numerical_note_counts","attribute_set_counts"]}))
    print("Wrote",out)
if __name__=="__main__":main()
