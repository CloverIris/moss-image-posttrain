#!/usr/bin/env python3
"""CPU-only two-update training audit for native XLA or bounded cuDNN VAE.

Select completed artifacts with --run and a new report path with --output.
The historical default paths are retained; existing reports are never overwritten.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ.update(CUDA_VISIBLE_DEVICES="",JAX_PLATFORMS="cpu",JAX_PLATFORM_NAME="cpu",
                  OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2")
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import jax
import numpy as np
from safetensors.numpy import load_file
from moss_art.checkpoint import load_checkpoint,CheckpointError
from moss_art.config import fingerprint
from moss_art.lora import init_lora,import_safetensors,DOWN,UP
from moss_art.trainer import identity_for
from scripts.benchmark_vae_precision import atomic_json,sha256


def require(condition,message):
    if not condition:
        raise AssertionError(message)


def all_finite(tree):
    return all(np.isfinite(np.asarray(value)).all() for value in jax.tree_util.tree_leaves(tree))


def bitwise_tree(a,b):
    aa,sa=jax.tree_util.tree_flatten(a)
    bb,sb=jax.tree_util.tree_flatten(b)
    return sa==sb and len(aa)==len(bb) and all(
        np.asarray(x).dtype==np.asarray(y).dtype and np.array_equal(np.asarray(x),np.asarray(y))
        for x,y in zip(aa,bb))


def optimizer_counts(tree):
    results=[]
    def visit(node):
        if isinstance(node,tuple) and hasattr(node,"_fields") and "count" in node._fields:
            results.append(int(np.asarray(node.count)))
        if isinstance(node,dict):
            for value in node.values():visit(value)
        elif isinstance(node,(tuple,list)):
            for value in node:visit(value)
    visit(tree)
    return results


def arguments(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",default=str(ROOT/"outputs/train_1024_fast_vae"))
    parser.add_argument("--output",default=str(ROOT/"reports/train_1024_fast_vae_validation.json"))
    return parser.parse_args(argv)


def main(argv=None):
    args=arguments(argv)
    directory=Path(args.run).resolve()
    output=Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    os.chdir(ROOT)
    require(all(d.platform=="cpu" for d in jax.devices()),"Audit must be CPU-only")
    config=json.loads((directory/"config.json").read_text())
    complete=json.loads((directory/"complete.json").read_text())
    run_info=json.loads((directory/"run_info.json").read_text())
    latest=json.loads((directory/"latest.json").read_text())
    metrics=[json.loads(line) for line in (directory/"metrics.jsonl").read_text().splitlines() if line.strip()]
    updates=[m for m in metrics if m.get("event")=="update"]
    require([m["update"] for m in updates]==[1,2],"Expected exactly updates 1,2")
    require([m["window"] for m in updates]==[0,1],"Expected ART windows 0,1")
    require(complete["update"]==latest["update"]==2,"Completion/update pointer mismatch")
    require(complete["rollout"]==1,"Expected one rollout")
    expected_items=len(config["sampling"]["train_timesteps"])*config["sampling"]["prompts_per_rollout"]
    for m in updates:
        require(m["rollout"]==1 and m["items"]==expected_items,"Wrong update work")
        require(m["masked_fraction"]==0 and m["active_fraction"]==1,"Unexpected masked/inactive items")
        for key in ("loss","grad_norm","adapter_delta_norm","reward_mean","reward_std","reward_min","reward_max"):
            require(math.isfinite(m[key]),"Nonfinite metric "+key)
        require(m["grad_norm"]>0 and m["adapter_delta_norm"]>0,"Zero gradient/update")
    expected=identity_for(config)
    require(run_info["identity"]==expected,"Current identity differs from recorded run identity")
    checkpoint=Path(latest["checkpoint"])
    lora,optimizer,metadata=load_checkpoint(checkpoint,expected_metadata=expected)
    lora_again,optimizer_again,metadata_again=load_checkpoint(checkpoint,expected_metadata=expected)
    require(bitwise_tree(lora,lora_again) and bitwise_tree(optimizer,optimizer_again) and metadata==metadata_again,
            "Checkpoint repeat load differs")
    require(metadata["config_fingerprint"]==fingerprint(config),"Config fingerprint mismatch")
    require(metadata["update"]==2 and metadata["rollout"]==1 and metadata["prompt_cursor"]==1,"Resume counters mismatch")
    require(metadata["seed"]==config["training"]["seed"],"Resume seed mismatch")
    require(metadata["lora_config"]==config["lora"],"LoRA config mismatch")
    require(metadata["model_config"]==run_info["model_config"],"Model shape config mismatch")
    counts=optimizer_counts(optimizer)
    require(counts==[2],"Adam update counter must be 2")
    require(all_finite((lora,optimizer)),"Nonfinite adapter/optimizer leaf")

    adapter=Path(complete["adapter"])
    raw_export=load_file(str(adapter))
    imported=import_safetensors(adapter,lora,alpha=config["lora"]["alpha"],device=jax.devices()[0])
    require(bitwise_tree(lora,imported),"Export differs from saved checkpoint")
    require(all(np.isfinite(v).all() for v in raw_export.values()),"Nonfinite export tensor")
    expected_modules=[f"layers.{layer}.proj.{proj}" for layer in range(run_info["model_config"]["num_layers"])
                      for proj in ("q","k","v","o")]
    require(config["lora"]["preset"]=="attn","This audit expects attention LoRA")
    require(set(lora)=={module+suffix for module in expected_modules for suffix in (DOWN,UP)},"Adapter targets differ")
    dim=run_info["model_config"]["dim"]
    rank=config["lora"]["rank"]
    for module in expected_modules:
        require(lora[module+DOWN].shape==(rank,dim) and lora[module+UP].shape==(dim,rank),"Adapter shape mismatch")
    if run_info["initial_adapter"] is None:
        base={module+".weight":SimpleNamespace(shape=(dim,dim)) for module in expected_modules}
        initial=init_lora(base,rank,seed=config["lora"]["seed"],targets="attention",device=jax.devices()[0])
        initial_scope="Project init_lora on verified shape-only placeholders, seed from config; no base tensors loaded."
    else:
        initial_path=Path(run_info["initial_adapter"]["path"])
        require(sha256(initial_path)==run_info["initial_adapter"]["sha256"],"Initial adapter replaced")
        initial=import_safetensors(initial_path,lora,alpha=config["lora"]["alpha"],device=jax.devices()[0])
        initial_scope="Re-read initialized adapter with recorded SHA."
    delta_sq=0.
    changed=0
    for key,value in lora.items():
        delta=np.asarray(value,np.float64)-np.asarray(initial[key],np.float64)
        delta_sq+=float(np.sum(delta**2))
        changed+=int(np.any(delta!=0))
    require(delta_sq>0,"No final adapter change")

    experiment=metadata["experiment"]
    backend=config["model"].get("vae_conv_backend","xla")
    require(backend in ("xla","cudnn_bounded","tiled_gemm"),"Unknown VAE backend")
    workspace_mib=config["model"].get("vae_conv_workspace_mib",256)
    required_sources=["moss_art/vae_precision.py","vendor/moss_jax/vae.py"]
    build=experiment.get("vae_convolution_build")
    build_source_sha256=None
    if backend=="cudnn_bounded":
        require(build is not None,"Bounded cuDNN backend lacks native build identity")
        required_sources.extend(("moss_art/cudnn_conv.py",
                                 "moss_art/native/cudnn_conv_ffi.cc","scripts/build_cudnn_conv.py"))
        manifest_path=Path(build["manifest"]["resolved_path"])
        binary_path=Path(build["binary"]["resolved_path"])
        build_manifest=json.loads(manifest_path.read_text())
        require(sha256(manifest_path)==build["manifest"]["sha256"],"Build manifest binding mismatch")
        require(sha256(binary_path)==build["binary"]["sha256"]==build_manifest["binary_sha256"],
                "Binary binding mismatch")
        build_source_sha256=build_manifest["source_sha256"]
        require(build_source_sha256==experiment["source_files"]["moss_art/native/cudnn_conv_ffi.cc"]["sha256"],
                "Build source differs from checkpoint source")
    else:
        require(build is None,"Non-FFI backend unexpectedly depends on a native build")
        if backend=="tiled_gemm":
            required_sources.append("moss_art/tiled_conv.py")
    for name in required_sources:
        require(experiment["source_files"][name]["sha256"]==sha256(ROOT/name),"Source binding changed "+name)
    negative_checks={}
    def rejected(name,wanted):
        try:
            load_checkpoint(checkpoint,expected_metadata=wanted)
        except CheckpointError as error:
            negative_checks[name]={"rejected":True,"error":str(error)}
        else:
            raise AssertionError("Incompatible resume accepted: "+name)
    changed_config=copy.deepcopy(config)
    changed_config["model"]["vae_conv_backend"]="xla" if backend=="cudnn_bounded" else "cudnn_bounded"
    rejected("different_backend",{"config_fingerprint":fingerprint(changed_config)})
    changed_config=copy.deepcopy(config)
    changed_config["model"]["vae_conv_workspace_mib"]=128 if workspace_mib!=128 else 256
    rejected("different_workspace_cap",{"config_fingerprint":fingerprint(changed_config)})
    source_to_mutate="moss_art/native/cudnn_conv_ffi.cc" if backend=="cudnn_bounded" else "vendor/moss_jax/vae.py"
    rejected("different_backend_source",{"experiment":{"source_files":{
        source_to_mutate:{"sha256":"0"*64}}}})
    if backend=="cudnn_bounded":
        rejected("different_native_binary",{"experiment":{"vae_convolution_build":{"binary":{"sha256":"0"*64}}}})
    else:
        negative_checks["different_native_binary"]={
            "applicable":False,"reason":"Native XLA/tiled backend has no FFI native-build dependency"}
    output.parent.mkdir(parents=True,exist_ok=True)
    result={
        "status":"passed","recorded_utc":datetime.now(timezone.utc).isoformat(),"cpu_only":True,
        "validation_script_sha256":sha256(Path(__file__)),"complete":complete,
        "run_directory":str(directory),"argv":list(sys.argv if argv is None else [sys.argv[0],*argv]),
        "validation_devices":[str(device) for device in jax.devices()],
        "config":config,"metadata_update":metadata["update"],"metadata_rollout":metadata["rollout"],
        "metadata_prompt_cursor":metadata["prompt_cursor"],"metadata_seed":metadata["seed"],
        "adam_counts":counts,"adapter_tensor_count":len(lora),
        "adapter_parameters":sum(v.size for v in lora.values()),
        "adapter_l2_delta_from_initial":math.sqrt(delta_sq),"changed_adapter_tensors":changed,
        "initialization_check":initial_scope,"export_equals_checkpoint_bitwise":True,
        "checkpoint_repeat_load_bitwise":True,"all_adapter_and_optimizer_leaves_finite":True,
        "optimizer_leaf_count":len(jax.tree_util.tree_leaves(optimizer)),
        "checkpoint_integrity_and_config_fingerprint_verified":True,
        "run_current_and_checkpoint_experiment_identity_match":True,
        "config_fingerprint":metadata["config_fingerprint"],"experiment_fingerprint":experiment["fingerprint"],
        "vae_backend_binding":{"backend":backend,"workspace_mib":workspace_mib,
            "source_files":{name:experiment["source_files"][name] for name in required_sources},
            "native_build":build,"native_build_required":backend=="cudnn_bounded","build_source_sha256":build_source_sha256,
            "negative_resume_checks":negative_checks},
        "metrics":updates,"peak_active_gib":max(v for m in updates for v in m["worker_peak_bytes_in_use"])/1024**3,
        "hashes":{str(path.relative_to(directory)):sha256(path) for path in
                  [directory/"config.json",directory/"complete.json",directory/"metrics.jsonl",directory/"run_info.json",
                   checkpoint/"manifest.json",checkpoint/"arrays.npz",checkpoint/"COMMITTED",adapter]},
        "scope":"One real 1024 prompt group (K=2), two ART windows each containing all five selected timesteps, two optimizer updates. CPU integrity/reload/export and resume-identity rejection audit; no additional resumed GPU step or post-update image-quality claim.",
        "weight_identity_limit":"Base/reward weights are bound by safetensors header+size+mtime+path, not full payload hashes; numerical source/build manifest/native binary use full SHA256.",
    }
    atomic_json(output,result)
    print(json.dumps({key:result[key] for key in ("status","complete","adam_counts","adapter_l2_delta_from_initial",
        "changed_adapter_tensors","export_equals_checkpoint_bitwise","peak_active_gib")}))
    print(json.dumps({"report":str(output),"backend_binding":result["vae_backend_binding"]}))


if __name__=="__main__":
    main()
