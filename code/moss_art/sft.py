"""Supervised flow-matching LoRA training and cached-condition generation.

python -m moss_art.sft train --config configs/sft/train_1024.json
python -m moss_art.sft prepare --help
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from .config_io import read_mapping, RUN_DEFAULTS, validate_run
from .config import DEFAULTS as ART_DEFAULTS, _merge, _validate_types
from .sft_data import SFTCache, digest
from .data import atomic_json, sha256_file
from .runtime import WorkerPool, configure_process, preflight, device_groups
from .locking import output_lock

DEFAULTS = {key:copy.deepcopy(ART_DEFAULTS[key]) for key in ("model","lora","optimizer","runtime","sampling")}
DEFAULTS["run"]=copy.deepcopy(RUN_DEFAULTS)
DEFAULTS["model"].update(attention_backend="cudnn",dit_autotune_level=4,
                         activation_storage="gpu",checkpoint_interval=1,base_precision="bf16",mxfp8_dgrad="bf16")
DEFAULTS["data"]={"cache":"cache/sft","context_length":4096,"short_context_length":0}
DEFAULTS["training"]={"max_updates":1000,"seed":42,"batch_size":4,"microbatch_size":1,"checkpoint_every":100}
DEFAULTS["sft"]={"sigma_distribution":"logit_normal","logit_mean":0.0,"logit_std":1.0,
                 "sigma_shift":1.0,"conditioning_dropout":0.1}
DEFAULTS["runtime"]["output_dir"]="outputs/sft"
DEFAULTS["runtime"]["tensor_parallel_size"]=1

def load_config(path):
    c=_merge(copy.deepcopy(DEFAULTS),read_mapping(path))
    return validate(c)

def validate(c):
    _validate_types(c,DEFAULTS)
    validate_run(c["run"])
    t,s,m,r=c["training"],c["sft"],c["model"],c["runtime"]
    if m["mxfp8_weight_mode"] not in ("online","offline"):
        raise ValueError("mxfp8_weight_mode must be online or offline")
    if "\x00" in m["mxfp8_cache_dir"] or (m["mxfp8_weight_mode"]=="offline" and not m["mxfp8_cache_dir"].strip()):
        raise ValueError("Offline MXFP8 requires a valid mxfp8_cache_dir")
    if m["activation_storage"] not in ("cpu","gpu") or m["checkpoint_interval"] < 1:
        raise ValueError("Invalid activation_storage/checkpoint_interval")
    if m["base_precision"] not in ("bf16", "mxfp8") or m["mxfp8_dgrad"] not in ("bf16", "mxfp8"):
        raise ValueError("Invalid base_precision / mxfp8_dgrad")
    if m["base_precision"] == "mxfp8":
        if m["dtype"] != "bfloat16" or r["tensor_parallel_size"] != 1:
            raise ValueError("MXFP8 requires BF16 activations and tensor_parallel_size=1 (DP)")
        if m["kind"] == "moss" and r["platform"] != "cuda":
            raise ValueError("Production MXFP8 requires CUDA; CPU is a tiny-model test reference")
    from .prompt_buckets import validate_lengths
    validate_lengths(c["data"]["short_context_length"],c["data"]["context_length"])
    if c["data"]["context_length"] < 1:
        raise ValueError("data.context_length must be positive")
    if t["microbatch_size"]<1:
        raise ValueError("microbatch_size must be positive")
    if t["max_updates"]<1 or t["batch_size"]<1 or t["checkpoint_every"]<0 or not 0<=t["seed"]<2**32:
        raise ValueError("Invalid SFT training budget or seed")
    if not r["devices"] or len(set(r["devices"]))!=len(r["devices"]) or any(type(x) is not int or x<0 for x in r["devices"]):
        raise ValueError("devices must be unique nonnegative integers")
    groups=device_groups(c)
    if t["batch_size"]%len(groups):
        raise ValueError("global batch_size must divide evenly over data-parallel group count")
    if s["sigma_distribution"] not in ("uniform","logit_normal") or s["sigma_shift"]<=0 or s["logit_std"]<=0 or not 0<=s["conditioning_dropout"]<=1:
        raise ValueError("Invalid SFT timestep/dropout options")
    if m["kind"] not in ("moss","tiny") or m["dtype"] not in ("bfloat16","float32"):
        raise ValueError("Invalid model kind/dtype")
    if m["kind"]=="moss" and m["dtype"]!="bfloat16":
        raise ValueError("Pretrained SFT requires BF16 base weights")
    if m["attention_backend"] not in ("cudnn","native") or m["dit_autotune_level"] not in (0,4):
        raise ValueError("Invalid attention/autotuning settings")
    if m["attention_backend"]=="cudnn" and (r["platform"]!="cuda" or m["dtype"]!="bfloat16"):
        raise ValueError("cuDNN attention needs CUDA and BF16")
    if r["busy_memory_mib"] < 0 or not 0 <= m["tiny_seed"] < 2**32:
        raise ValueError("Invalid memory threshold or model seed")
    if m["dit_autotune_level"] and r["platform"] != "cuda":
        raise ValueError("DiT autotuning needs CUDA")
    if r["platform"] not in ("cuda","cpu") or r["cpu_threads"]<1 or not 0<r["memory_fraction"]<=1 or r["worker_timeout"]<=0:
        raise ValueError("Invalid runtime")
    l,o=c["lora"],c["optimizer"]
    if l["rank"]<1 or l["alpha"]<=0 or l["preset"] not in ("attn","attn-mlp") or not 0<=l["seed"]<2**32:
        raise ValueError("Invalid LoRA configuration")
    from .optimizers import validate_optimizer
    validate_optimizer(o)
    for value in (m["path"],c["data"]["cache"],r["output_dir"]):
        if not value.strip():
            raise ValueError("Model/cache/output paths must be nonempty")
    from .sampling import schedule
    sampling=c["sampling"]
    schedule(sampling["steps"],scheduler=sampling["scheduler"],shift=sampling["shift"],
             training_shift=sampling["training_shift"],
             beta_alpha=sampling["beta_alpha"],beta_beta=sampling["beta_beta"])
    if sampling["sampler"] not in ("euler","res_multistep") or sampling["guidance_scale"]<0:
        raise ValueError("Invalid sampling options")
    layers=2 if m["kind"]=="tiny" else 42
    if any(type(x) is not int or x<0 or x>=layers for x in sampling["skip_layers"]):
        raise ValueError("Invalid sampling skip_layers")
    return c

def model_binding(c,cache):
    from .assets import component_identity
    current=component_identity(c["model"]["path"],"transformer") if c["model"]["kind"]=="moss" else {"tiny_seed":c["model"]["tiny_seed"]}
    if c["model"]["kind"]=="moss" and current!=cache.manifest["identity"]["models"]["transformer"]:
        raise ValueError("SFT cache and training model are different")
    return digest(current)

def identity(c,cache):
    from .assets import component_identity
    model_identity = component_identity(c["model"]["path"],"transformer") if c["model"]["kind"]=="moss" else {"tiny_seed":c["model"]["tiny_seed"]}
    if c["model"]["kind"]=="moss" and model_identity!=cache.manifest["identity"]["models"]["transformer"]:
        raise ValueError("SFT cache and training model are different")
    fields={key:copy.deepcopy(c[key]) for key in ("model","lora","optimizer","sft")}
    for key in ("mxfp8_weight_mode","mxfp8_cache_dir","mxfp8_cache_verify"):
        fields["model"].pop(key,None)
    fields["context_length"]=c["data"]["context_length"]
    fields["short_context_length"]=c["data"]["short_context_length"]
    fields["tensor_parallel_size"]=c["runtime"]["tensor_parallel_size"]
    fields["training"]={k:v for k,v in c["training"].items() if k not in ("max_updates","checkpoint_every")}
    files=("config_io.py","optimizers.py","sft.py","sft_worker.py","sft_data.py","model.py","lora.py","mxfp8.py","mxfp8_cache.py","attention.py","conditioning.py","prompt_buckets.py","tensor_parallel.py","runtime.py")
    source={p:sha256_file(Path(__file__).parent/p) for p in files}
    source["dit.py"]=sha256_file(Path(__file__).parents[1]/"vendor/moss_jax/dit.py")
    return dict(task="moss-flow-sft",transformer_identity_hash=model_binding(c,cache),identity_hash=digest(dict(config=fields,
        cache=sha256_file(cache.directory/"manifest.json"),model=model_identity,source=source)))

def jobs_for(cache,cursor,count,seed):
    jobs=[]
    for pos in range(cursor,cursor+count):
        epoch,offset=divmod(pos,len(cache.rows))
        order=np.random.default_rng(np.random.SeedSequence([seed,epoch])).permutation(len(cache.rows))
        item_seed=int(np.random.SeedSequence([seed,pos,9187]).generate_state(1)[0])
        jobs.append({"index":int(order[offset]),"seed":item_seed})
    return jobs

def train(c,resume=None,init_adapter=None):
    validate(c)
    preflight(c)
    cache=SFTCache(c["data"]["cache"])
    from .prompt_buckets import configure_sft_cache
    configure_sft_cache(cache,c["data"])
    expected=identity(c,cache)
    configure_process(c,None)  # optimizer on CPU; GPU workers select their devices
    import jax
    import optax
    from .trainer import _optimizer,resolve_checkpoint
    from .checkpoint import load_checkpoint,save_checkpoint
    from .lora import export_safetensors,import_safetensors
    from .sft_worker import worker_main
    output=Path(c["runtime"]["output_dir"]).resolve()
    with output_lock(output):
        if not resume and (output/"metrics.jsonl").exists():
            raise ValueError("Existing run: use --resume or a new output directory")
        restored=load_checkpoint(resolve_checkpoint(resume),expected_metadata=expected) if resume else None
        with WorkerPool(c,worker_main=worker_main) as pool:
            lora=pool.info[0]["lora"]
            tx=_optimizer(c)
            state=tx.init(lora)
            step,cursor=0,0
            if init_adapter:
                lora=import_safetensors(init_adapter,lora,alpha=c["lora"]["alpha"])
            if restored:
                lora,state,metadata=restored
                step,cursor=metadata["update"],metadata["sample_cursor"]
                if (output/"metrics.jsonl").exists():
                    lines=(output/"metrics.jsonl").read_text().splitlines()
                    if lines and json.loads(lines[-1])["update"]>step:
                        raise ValueError("Output has newer updates than resume checkpoint; use another output")
            if step>=c["training"]["max_updates"]:
                raise ValueError("Resume step must be below max_updates")
            lora=jax.tree.map(lambda a:np.asarray(a,np.float32),lora)
            pool.call("set_lora",[{"lora":lora} for _ in pool.workers])
            atomic_json(output/"config.json",c)
            atomic_json(output/"run_info.json",dict(expected,records=len(cache.rows),
                precision={k:pool.info[0][k] for k in ("base_precision","mxfp8_dgrad","base_storage_bytes","mxfp8_modules")},
                modes={"text_only":sum(not r["conditioning_images"] for r in cache.rows),
                       "image_text":sum(bool(r["conditioning_images"]) for r in cache.rows)},
                worker_devices=[x["devices"] for x in pool.info],
                physical_device_groups=device_groups(c),parallelism=pool.info[0]["parallelism"],
                optimizer=c["optimizer"],adapter_parameters=sum(x.size for x in lora.values())))
            while step<c["training"]["max_updates"]:
                started=time.monotonic()
                count=c["training"]["batch_size"]
                jobs=jobs_for(cache,cursor,count,c["training"]["seed"])
                width=count//len(pool.workers)
                results=pool.call("gradient",[{"jobs":jobs[i*width:(i+1)*width]} for i in range(len(pool.workers))])
                denominator=sum(x["denominator"] for x in results)
                if denominator!=count:
                    raise RuntimeError("Worker sample count mismatch")
                grads={key:sum((x["gradients"][key] for x in results),start=np.zeros_like(value))/np.float32(count)
                       for key,value in lora.items()}
                norm=float(optax.global_norm(grads))
                if not np.isfinite(norm):
                    raise FloatingPointError("Nonfinite global gradient")
                optimizer_started=time.monotonic()
                updates,state=tx.update(grads,state,lora)
                updated=optax.apply_updates(lora,updates)
                delta=float(optax.global_norm(jax.tree.map(lambda a,b:a-b,updated,lora)))
                lora=jax.tree.map(lambda a:np.asarray(a,np.float32),updated)
                if any(not np.isfinite(a).all() for a in lora.values()):
                    raise FloatingPointError("Nonfinite adapter")
                optimizer_seconds=time.monotonic()-optimizer_started
                pool.call("set_lora",[{"lora":lora} for _ in pool.workers])
                step+=1
                cursor+=count
                metrics=dict(update=step,sample_cursor=cursor,optimizer=c["optimizer"]["name"],optimizer_seconds=optimizer_seconds,loss=sum(x["loss_sum"] for x in results)/count,
                    grad_norm=norm,adapter_delta_norm=delta,seconds=time.monotonic()-started,
                    conditioning_dropped=sum(x["dropped"] for x in results),
                    sigmas=sum((x["sigmas"] for x in results),[]),
                    worker_peak_bytes=[stats.get("peak_bytes_in_use") for x in results for stats in x["device_memory_stats"]],
                    sample_ids=[cache.rows[x["index"]]["id"] for x in jobs])
                with (output/"metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(metrics,allow_nan=False)+"\n")
                print(json.dumps(metrics,allow_nan=False),flush=True)
                interval=c["training"]["checkpoint_every"]
                if step==c["training"]["max_updates"] or interval and step%interval==0:
                    path=output/"checkpoints"/f"step_{step:08d}"
                    save_checkpoint(path,lora,state,dict(expected,update=step,sample_cursor=cursor,
                        lora_config=c["lora"],model_config=pool.info[0]["model_config"]))
                    export_safetensors(path/"adapter.safetensors",lora,alpha=c["lora"]["alpha"],
                        metadata=dict(task="flow-matching-sft",conditioning="visual+text-all",update=step,transformer_identity_hash=expected["transformer_identity_hash"],
                                      base_precision=c["model"]["base_precision"],mxfp8_dgrad=c["model"]["mxfp8_dgrad"]))
                    atomic_json(output/"latest.json",{"checkpoint":str(path),"update":step})

def sample(c,index,seed,adapter=None,checkpoint=None):
    validate(c)
    if not 0<=seed<2**32:
        raise ValueError("seed must be uint32")
    cache=SFTCache(c["data"]["cache"],require_targets=False)
    from .prompt_buckets import configure_sft_cache
    configure_sft_cache(cache,c["data"])
    binding=model_binding(c,cache)
    if not 0<=index<len(cache.rows):
        raise ValueError("sample-index outside cache")
    from .sft_worker import worker_main
    from .trainer import resolve_checkpoint
    preflight(c)
    configure_process(c,None)
    from .lora import import_safetensors
    from .checkpoint import load_checkpoint
    with WorkerPool(c,worker_main=worker_main) as pool:
        lora=pool.info[0]["lora"]
        if checkpoint:
            lora,_,metadata=load_checkpoint(resolve_checkpoint(checkpoint))
            if metadata.get("task")!="moss-flow-sft" or metadata.get("transformer_identity_hash")!=binding or metadata.get("lora_config")!=c["lora"] or metadata.get("model_config")!=pool.info[0]["model_config"]:
                raise ValueError("Checkpoint model/LoRA mismatch")
        elif adapter:
            lora=import_safetensors(adapter,lora,alpha=c["lora"]["alpha"])
        pool.call("set_lora",[{"lora":lora} for _ in pool.workers])
        results=pool.call("sample",[{"index":index,"seed":seed+i} for i in range(len(pool.workers))])
    # Release DiT workers before loading VAE; decode in a separate GPU process.
    out=Path(c["runtime"]["output_dir"]).resolve()
    out.mkdir(parents=True,exist_ok=True)
    for i,result in enumerate(results):
        path=out/f"sample-{index:04d}-seed-{seed+i}.npy"
        if path.exists():
            raise FileExistsError(path)
        np.save(path,result["latent"])
        if c["model"]["kind"]=="moss":
            code=("import json,sys; from moss_art.sft import decode_sample; "
                  "decode_sample(json.loads(sys.argv[1]),sys.argv[2],sys.argv[3],int(sys.argv[4]))")
            subprocess.run([sys.executable,"-c",code,json.dumps(c),str(path),str(path.with_suffix(".png")),
                            str(c["runtime"]["devices"][i*c["runtime"]["tensor_parallel_size"]])],check=True)
        atomic_json(path.with_suffix(".json"),{"id":result["id"],"steps":result["steps"],"seed":seed+i,
                    "adapter":adapter,"checkpoint":checkpoint,"visual_tokens":cache.rows[index]["visual_tokens"]})
        print(json.dumps({"event":"sft_sample","latent":str(path)}),flush=True)

def decode_sample(c,latent_path,image_path,device):
    """Isolated VAE process: honor JSON precision/backend and UUID GPU binding."""
    configure_process(c,device)
    from .evaluate import decode_latents
    from PIL import Image
    m=c["model"]
    pixels=decode_latents(m["path"],np.load(latent_path,allow_pickle=False),
        precision=m["vae_precision"],low_precision_stages=m["vae_low_precision_stages"],
        conv_backend=m["vae_conv_backend"],conv_workspace_mib=m["vae_conv_workspace_mib"])
    Image.fromarray(pixels[0]).save(image_path)

def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0]=="prepare":
        from .sft_prepare import main as prepare
        return prepare(argv[1:])
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("train","sample","doctor"))
    parser.add_argument("--config",required=True)
    args=parser.parse_args(argv)
    c=load_config(args.config)
    run=validate_run(c["run"],args.command,sft=True)
    if args.command=="train":
        train(c,run["resume"] or None,run["init_adapter"] or None)
    elif args.command=="sample":
        sample(c,run["sample_index"],run["seed"],run["adapter"] or None,run["checkpoint"] or None)
    else:
        cache=SFTCache(c["data"]["cache"])
        from .prompt_buckets import configure_sft_cache
        configure_sft_cache(cache,c["data"])
        print(json.dumps(dict(identity(c,cache),devices=preflight(c),records=len(cache.rows)),indent=2))

if __name__=="__main__":
    main()
