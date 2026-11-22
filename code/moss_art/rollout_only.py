"""Inference-only prompt-file generation, with reusable Qwen double buffers."""
import argparse
import copy
from contextlib import ExitStack
import itertools
import json
from pathlib import Path
import time
import traceback

import numpy as np
from .config import DEFAULTS as ART_DEFAULTS, _merge, _validate_types, validate as validate_art
from .config_io import read_mapping

DEFAULTS = {k:copy.deepcopy(ART_DEFAULTS[k]) for k in ("model","lora","data","runtime")}
DEFAULTS["model"].update(base_precision="bf16",attention_backend="cudnn",dit_autotune_level=4,vae_conv_backend="tiled_gemm")
DEFAULTS["runtime"]["output_dir"]="outputs/rollout_only"
DEFAULTS["data"].update(prompts="configs/rollout/prompts.txt",cache="cache/rollout_only")
DEFAULTS["sampling"]={k:v for k,v in ART_DEFAULTS["sampling"].items()
    if k not in ("group_size","prompts_per_rollout","train_timesteps")}
DEFAULTS["sampling"].update(height=1024,width=1024)
DEFAULTS["preparation"]=copy.deepcopy(ART_DEFAULTS["preparation"])
DEFAULTS["preparation"]["async_prefetch"]=True
DEFAULTS["run"]={"adapter":""}
DEFAULTS["rollout"]={"count":4,"seed":42,"prompts_per_batch":4,"decode":True,"save_latents":False,"cache_intermediates":True,"host_cache_limit_gib":16.0}


def engine_config(c):
    """Reuse validated model/sampler/runtime settings, with neutral ART fields.

    The shared worker's group_size becomes one only when starting inference.
    No training, reward, optimizer, or trajectory-replay operation is called.
    """
    engine=copy.deepcopy(ART_DEFAULTS)
    for k in ("model","lora","data","runtime","sampling","preparation"):
        engine[k].update(copy.deepcopy(c[k]))
    engine["sampling"].update(group_size=2,train_timesteps=[0],
        prompts_per_rollout=max(len(c["runtime"]["devices"]),1))
    precision=engine["model"].pop("base_precision")
    if precision not in ("bf16","mxfp8"):
        raise ValueError("model.base_precision must be bf16 or mxfp8")
    if precision=="mxfp8" and (engine["model"]["dtype"]!="bfloat16" or
            engine["model"]["kind"]=="moss" and engine["runtime"]["platform"]!="cuda"):
        raise ValueError("MXFP8 requires BF16 activations and CUDA for production MOSS")
    resident=engine["sampling"].get("vae_resident_rollout",False)
    if resident:
        if precision!="mxfp8":
            raise ValueError("Resident rollout VAE requires model.base_precision=mxfp8")
        if not c["rollout"]["decode"]:
            raise ValueError("Resident rollout VAE requires rollout.decode=true")
        size=len(engine["runtime"]["devices"])
        if size<2 or size%2:
            raise ValueError("Resident rollout Mesh requires an even device count of at least two")
        engine["runtime"]["execution_backend"]="mesh"
        engine["model"]["activation_storage"]="gpu"
        engine["sampling"].update(rollout_mode="mxfp8",prompts_per_rollout=1,
            split_group_across_workers=True,group_size=size)
    validate_art(engine)
    engine["model"]["base_precision"]=precision
    # Legacy worker mode uses this field as prompt-buffer capacity. Mesh uses
    # a separate producer config so its one-prompt collective stays valid.
    if not resident:
        engine["sampling"]["prompts_per_rollout"]=c["rollout"]["prompts_per_batch"]
    return engine


def load_config(path):
    c=_merge(copy.deepcopy(DEFAULTS),read_mapping(path))
    _validate_types(c,DEFAULTS)
    r=c["rollout"]
    if r["host_cache_limit_gib"]<=0:
        raise ValueError("rollout.host_cache_limit_gib must be positive")
    if r["count"]<1 or r["prompts_per_batch"]<1 or not 0<=r["seed"]<2**32:
        raise ValueError("rollout.count/prompts_per_batch must be positive; seed must be uint32")
    if not r["decode"] and not r["save_latents"]:
        raise ValueError("Enable rollout.decode or rollout.save_latents")
    if c["model"]["kind"]=="tiny" and r["decode"]:
        raise ValueError("Tiny model has no image VAE; use decode=false and save_latents=true")
    engine_config(c)
    return c


def unescape_prompt(text):
    """Decode newline/backslash escapes once; leave other escapes unchanged."""
    result=[]
    i=0
    while i<len(text):
        if text[i]=="\\" and i+1<len(text):
            following=text[i+1]
            if following in ("\\","n"):
                result.append("\\" if following=="\\" else "\n")
                i+=2
                continue
        result.append(text[i])
        i+=1
    return "".join(result)


def read_prompt_file(path,negative=""):
    """Unescape prompt text after record parsing; labels remain content."""
    from dataclasses import replace
    from .data import PromptRecord,load_prompts
    path=Path(path)
    if path.suffix.lower()==".txt":
        records=[PromptRecord(f"line-{i:06d}",line.strip(),negative)
                 for i,line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(),1) if line.strip()]
        if not records:raise ValueError("Prompt file is empty")
    elif path.suffix.lower() in (".jsonl",".ndjson"):
        records=load_prompts(path,default_negative_prompt=negative)
    else:
        raise ValueError("Prompt file must be .txt, .jsonl or .ndjson")
    # Do not use unicode_escape: Chinese text and other backslashes are literal.
    records=[replace(r,prompt=unescape_prompt(r.prompt),
                     negative_prompt=unescape_prompt(r.negative_prompt)) for r in records]
    if any(not r.prompt.strip() for r in records):
        raise ValueError("Prompt is empty after newline conversion")
    return records


def jobs_for(records,source_positions,count,seed):
    for record in records:
        position=source_positions[record["sample_id"]]
        for repeat in range(count):
            yield dict(record=record,prompt_index=position,repeat=repeat,
                       seed=(seed+position*count+repeat)%(2**32))


def worker_main(conn,c,rank,device):
    from .runtime import configure_process
    configure_process(c,device)
    try:
        from .worker import NativeWorker
        worker=NativeWorker(c,rank,base_precision=c["model"]["base_precision"])
        info=worker.info()
        info.update(base_precision=worker.model.base_precision,mxfp8_modules=worker.model.mxfp8_modules,
                    base_storage_bytes=worker.model.base_storage_bytes)
        conn.send({"result":info})
        while True:
            request=conn.recv();op=request.pop("op")
            if op=="close":break
            if op=="generate":
                job=request["job"]
                keep = c.get("_cache_intermediates", False)
                result=None if job is None else worker.rollout([job["record"]],[job["seed"]],keep_cache=keep)
                if result is not None and keep:
                    result["trajectory"] = {key: worker.cache[key] for key in ("states", "old", "selected", "sigmas")}
                    worker.cache = None
            elif op=="decode_generated":
                endpoints=request["endpoints"]
                result=None if endpoints is None else worker.decode(endpoints)
            elif op in ("set_lora","bind_conditions"):
                result=getattr(worker,op)(**request)
            else:raise ValueError(f"Unsupported inference operation: {op}")
            conn.send({"result":result})
    except EOFError:pass
    except BaseException as e:
        try:conn.send({"error":f"{type(e).__name__}: {e}","traceback":traceback.format_exc()})
        except (BrokenPipeError,OSError):pass
    finally:conn.close()


def bind_all(prefetch,pool,records,ticket):
    descriptor,event=prefetch.acquire(ticket)
    acks=pool.call("bind_conditions",[dict(descriptor=descriptor,records=records,
        buckets=[prefetch.buckets.short,prefetch.buckets.full]) for _ in pool.workers])
    if any(a["generation"]!=ticket[1] or a["copied"]!=len(records) for a in acks):
        raise RuntimeError("Incomplete condition-buffer acknowledgement")
    prefetch.release(ticket)
    return event


def _run_resident(c,engine,source,records,source_positions,output):
    """Generate and decode one device wave at a time in one Mesh controller."""
    from .runtime import configure_process
    configure_process(engine,engine["runtime"]["devices"])
    from .mesh_runtime import MeshPool
    from .prefetch import DoubleBuffer
    from .data import atomic_json,sha256_file
    from .evaluate import save_images,write_gallery
    from .sampling import schedule
    keep=c["rollout"]["cache_intermediates"]
    if keep:
        _,sigmas=schedule(c["sampling"]["steps"],scheduler=c["sampling"]["scheduler"],
            shift=c["sampling"]["shift"],training_shift=c["sampling"]["training_shift"],
            beta_alpha=c["sampling"]["beta_alpha"],beta_beta=c["sampling"]["beta_beta"])
        engine["sampling"]["train_timesteps"]=list(range(len(sigmas)-1))
    size=len(engine["runtime"]["devices"])
    # A single wave includes padding to the device count. It is saved and freed
    # before sampling the next wave, independent of the dataset image count.
    latent_bytes=128*(c["sampling"]["height"]//16)*(c["sampling"]["width"]//16)*4
    copies=1+(2*len(engine["sampling"]["train_timesteps"]) if keep else 0)
    cache_bytes=size*latent_bytes*copies
    if cache_bytes>c["rollout"]["host_cache_limit_gib"]*2**30:
        raise MemoryError(f"One rollout wave needs {cache_bytes} host bytes; reduce trajectory caching or increase host_cache_limit_gib")
    print(json.dumps(dict(event="host_cache_plan",bytes=cache_bytes,intermediates=keep,
                          streaming=True,images_per_wave=size)),flush=True)
    batch_size=c["rollout"]["prompts_per_batch"]
    batches=[records[i:i+batch_size] for i in range(0,len(records),batch_size)]
    producer=copy.deepcopy(engine)
    producer["sampling"]["prompts_per_rollout"]=batch_size
    gallery=[];finished=0;started=time.monotonic()
    adapter=c["run"]["adapter"]
    adapter_info={"path":str(Path(adapter).resolve()),"sha256":sha256_file(adapter)} if adapter else None
    effective=dict(engine["sampling"])
    atomic_json(output/"run_info.json",dict(input_prompts=len(source),retained_prompts=len(records),
        dropped_ids=[r.id for r in source if r.id not in {x["sample_id"] for x in records}],
        images_per_prompt=c["rollout"]["count"],expected_images=len(records)*c["rollout"]["count"],
        prompt_file_sha256=sha256_file(c["data"]["prompts"]),adapter=adapter_info,
        training=False,reward=False,trajectory_cache=keep,model_lifecycle="mxfp8_resident_vae",
        execution_backend="mesh",effective_sampling=effective,peak_wave_cache_bytes=cache_bytes))
    with ExitStack() as stack:
        prefetch=stack.enter_context(DoubleBuffer(producer,records)) if c["preparation"]["async_prefetch"] else None
        ticket=prefetch.submit(batches[0]) if prefetch else None
        pool=stack.enter_context(MeshPool(engine))
        pool._ensure_precision("mxfp8")
        pool.ensure_rollout_vae()
        model=pool.worker.model
        precision_info=[dict(rank=rank,device=str(device),base_precision=model.base_precision,
            mxfp8_modules=model.mxfp8_modules,base_storage_bytes=model.base_storage_bytes,
            attention_backend=model.attention_backend) for rank,device in enumerate(pool.workers)]
        atomic_json(output/"worker_info.json",precision_info)
        print(json.dumps(dict(event="rollout_workers_ready",workers=precision_info,
                              execution_backend="mesh",vae_resident=True)),flush=True)
        if adapter:
            from .lora import import_safetensors
            lora=import_safetensors(adapter,pool.worker.lora,alpha=c["lora"]["alpha"])
            pool.call("set_lora",[{"lora":lora} for _ in pool.workers])
        for batch_index,batch in enumerate(batches):
            if prefetch:
                bind_all(prefetch,pool,batch,ticket)
                if batch_index+1<len(batches):ticket=prefetch.submit(batches[batch_index+1])
            # A mesh collective broadcasts one complete prompt across its cards.
            for record in batch:
                jobs=iter(jobs_for([record],source_positions,c["rollout"]["count"],c["rollout"]["seed"]))
                while wave:=list(itertools.islice(jobs,size)):
                    begin=time.monotonic()
                    padded=wave+[wave[-1]]*(size-len(wave))
                    results=pool.call("rollout",[dict(records=[record],seeds=[job["seed"]],keep_cache=keep)
                                                  for job in padded])
                    generation_seconds=time.monotonic()-begin
                    decode_started=time.monotonic()
                    images=pool.decode([r["endpoints"] for r in results])
                    decode_seconds=time.monotonic()-decode_started
                    # Mesh owns the one CPU replay. Slice views only; do not
                    # export/copy the complete replay once for every device.
                    cache=pool.worker.cache if keep else None
                    for index,(job,result,image) in enumerate(zip(wave,results,images)):
                        stem=f"p{job['prompt_index']:06d}-r{job['repeat']:06d}-s{job['seed']}"
                        latent_path=f"{stem}.npy" if c["rollout"]["save_latents"] else None
                        trajectory_path=f"{stem}-trajectory.npz" if keep else None
                        if keep:
                            np.savez(output/trajectory_path,states=cache["states"][index:index+1],
                                old=cache["old"][index:index+1],selected=cache["selected"],sigmas=cache["sigmas"])
                        if latent_path:np.save(output/latent_path,result["endpoints"][0],allow_pickle=False)
                        image_path=f"{stem}.png"
                        save_images(image["images"],output,stems=[stem])
                        gallery.append(dict(image=str(output/image_path),prompt=record["prompt"]+f"\nseed: {job['seed']}"))
                        metadata=dict(prompt_id=record["sample_id"],prompt=record["prompt"],
                            negative_prompt=record["negative_prompt"],prompt_index=job["prompt_index"],
                            repeat=job["repeat"],seed=job["seed"],image=image_path,latent=latent_path,
                            sampling=c["sampling"],effective_sampling=effective,actual_steps=result["actual_steps"],
                            adapter=adapter_info,base_precision=model.base_precision,trajectory=trajectory_path,
                            **({"spectrum": result["spectrum"]} if "spectrum" in result else {}))
                        atomic_json(output/f"{stem}.json",metadata)
                        with (output/"samples.jsonl").open("a",encoding="utf-8") as f:
                            f.write(json.dumps(metadata,ensure_ascii=False,allow_nan=False)+"\n")
                        finished+=1
                    cache=None
                    pool.call("clear_replay")
                    del results,images
                    print(json.dumps(dict(event="rollout_only_progress",completed=finished,
                        expected=len(records)*c["rollout"]["count"],generation_seconds=generation_seconds,
                        decode_seconds=decode_seconds,wave_seconds=time.monotonic()-begin,
                        vae_resident=True,padded_images=size-len(wave))),flush=True)
    if gallery:write_gallery(gallery,output,title="MOSS rollout only")
    complete=dict(images=finished,prompts=len(records),elapsed_seconds=time.monotonic()-started,
                  output_dir=str(output),training=False,reward=False,model_lifecycle="mxfp8_resident_vae")
    atomic_json(output/"complete.json",complete)
    print(json.dumps(dict(event="rollout_only_complete",**complete)),flush=True)
    return output


def run(c):
    from .runtime import configure_process,preflight,WorkerPool
    from .locking import output_lock
    from .data import atomic_json,sha256_file
    from .trainer import read_records
    from .prefetch import DoubleBuffer
    from .evaluate import save_images,write_gallery
    source=read_prompt_file(c["data"]["prompts"],c["data"]["negative_prompt"])
    if len(source)*c["rollout"]["count"]>2**32:raise ValueError("Requested image count exceeds unique uint32 seed space")
    source_positions={r.id:i for i,r in enumerate(source)}
    engine=engine_config(c)
    preflight(engine)
    with output_lock(c["runtime"]["output_dir"]) as output:
        if any(p.name!=".run.lock" for p in output.iterdir()):
            raise FileExistsError("Rollout output is not empty; choose a new output_dir")
        atomic_json(output/"config.json",c)
        normalized=output/"prompts.jsonl"
        normalized.write_text("".join(json.dumps(dict(id=r.id,prompt=r.prompt,negative_prompt=r.negative_prompt,
            metadata=r.metadata),ensure_ascii=False)+"\n" for r in source),encoding="utf-8")
        engine["data"]["prompts"]=str(normalized)
        if not c["preparation"]["async_prefetch"]:
            from .prepare import ensure_cache
            prepare=copy.deepcopy(engine)
            if not prepare["sampling"].get("vae_resident_rollout",False):
                prepare["sampling"]["prompts_per_rollout"]=len(prepare["runtime"]["devices"])
            prepare["model"].pop("base_precision")
            ensure_cache(prepare)
        records=read_records(engine)
        if engine["sampling"].get("vae_resident_rollout",False):
            return _run_resident(c,engine,source,records,source_positions,output)
        configure_process(engine)
        engine["sampling"]["group_size"]=1
        batch_size=c["rollout"]["prompts_per_batch"]
        batches=[records[i:i+batch_size] for i in range(0,len(records),batch_size)]
        gallery=[];finished=0;started=time.monotonic()
        adapter=c["run"]["adapter"]
        adapter_info={"path":str(Path(adapter).resolve()),"sha256":sha256_file(adapter)} if adapter else None
        atomic_json(output/"run_info.json",dict(input_prompts=len(source),retained_prompts=len(records),
            dropped_ids=[r.id for r in source if r.id not in {x["sample_id"] for x in records}],
            images_per_prompt=c["rollout"]["count"],expected_images=len(records)*c["rollout"]["count"],
            prompt_file_sha256=sha256_file(c["data"]["prompts"]),adapter=adapter_info,
            training=False,reward=False,trajectory_cache=c["rollout"]["cache_intermediates"],model_lifecycle="dit_then_vae"))
        pending_waves = []
        engine["_cache_intermediates"] = c["rollout"]["cache_intermediates"]
        if c["rollout"]["cache_intermediates"]:
            from .sampling import schedule
            _, sigmas = schedule(c["sampling"]["steps"], scheduler=c["sampling"]["scheduler"],
                shift=c["sampling"]["shift"], training_shift=c["sampling"]["training_shift"],
                beta_alpha=c["sampling"]["beta_alpha"], beta_beta=c["sampling"]["beta_beta"])
            engine["sampling"]["train_timesteps"] = list(range(len(sigmas)-1))
        channels = 8 if c["model"]["kind"] == "tiny" else 128
        latent_bytes = channels*(c["sampling"]["height"]//16)*(c["sampling"]["width"]//16)*4
        copies = 1 + (2*len(engine["sampling"]["train_timesteps"]) if c["rollout"]["cache_intermediates"] else 0)
        cache_bytes = len(records)*c["rollout"]["count"]*latent_bytes*copies
        if cache_bytes > c["rollout"]["host_cache_limit_gib"]*2**30:
            raise MemoryError(f"Host trajectory cache needs {cache_bytes} bytes; increase rollout.host_cache_limit_gib or reduce count")
        print(json.dumps(dict(event="host_cache_plan",bytes=cache_bytes,intermediates=c["rollout"]["cache_intermediates"])),flush=True)
        with ExitStack() as stack:
            prefetch=stack.enter_context(DoubleBuffer(engine,records)) if c["preparation"]["async_prefetch"] else None
            ticket=prefetch.submit(batches[0]) if prefetch else None
            pool=stack.enter_context(WorkerPool(engine,worker_main=worker_main))
            precision_info=[{k:info[k] for k in ("rank","device","base_precision","mxfp8_modules","base_storage_bytes","attention_backend")} for info in pool.info]
            atomic_json(output/"worker_info.json",precision_info)
            print(json.dumps(dict(event="rollout_workers_ready",workers=precision_info)),flush=True)
            if adapter:
                from .lora import import_safetensors
                lora=import_safetensors(adapter,pool.info[0]["lora"],alpha=c["lora"]["alpha"])
                pool.call("set_lora",[{"lora":lora} for _ in pool.workers])
            for batch_index,batch in enumerate(batches):
                if prefetch:
                    bind_all(prefetch,pool,batch,ticket)
                    if batch_index+1<len(batches):ticket=prefetch.submit(batches[batch_index+1])
                jobs=iter(jobs_for(batch,source_positions,c["rollout"]["count"],c["rollout"]["seed"]))
                while wave:=list(itertools.islice(jobs,len(pool.workers))):
                    begin=time.monotonic()
                    padded=wave+[None]*(len(pool.workers)-len(wave))
                    results=pool.call("generate",[{"job":job} for job in padded])
                    generation_seconds=time.monotonic()-begin
                    pending_waves.append((wave, results, generation_seconds))
                    print(json.dumps(dict(event="dit_host_cache_progress",
                        cached=sum(len(w) for w,_,_ in pending_waves),
                        expected=len(records)*c["rollout"]["count"], generation_seconds=generation_seconds)),flush=True)
        # ExitStack joins all DiT/Qwen processes before constructing any VAE.
        from .runtime import gpu_inventory
        inventory = gpu_inventory() if c["runtime"]["platform"] == "cuda" else {}
        released_memory = {i: inventory[i]["used_mib"] for i in c["runtime"]["devices"]} if inventory else {}
        print(json.dumps(dict(event="dit_released", cached_images=sum(len(w) for w,_,_ in pending_waves),
            phase="vae", gpu_memory_mib=released_memory)),flush=True)
        from .phases import vae_worker_main
        from contextlib import nullcontext
        with (WorkerPool(engine, worker_main=vae_worker_main, cache_base=False)
              if c["rollout"]["decode"] else nullcontext(None)) as decoder:
            for wave,results,generation_seconds in pending_waves:
                begin=time.monotonic()
                images=decoder.call("decode",[{"endpoints":r["endpoints"] if r else None}
                    for r in results]) if decoder else [None]*len(results)
                for job,result,image in zip(wave,results,images):
                    stem=f"p{job['prompt_index']:06d}-r{job['repeat']:06d}-s{job['seed']}"
                    latent_path=f"{stem}.npy" if c["rollout"]["save_latents"] else None
                    if "trajectory" in result:
                        np.savez(output/f"{stem}-trajectory.npz", **result["trajectory"])
                    if latent_path:np.save(output/latent_path,result["endpoints"][0],allow_pickle=False)
                    image_path=f"{stem}.png" if image else None
                    if image:
                        save_images(image["images"],output,stems=[stem])
                        gallery.append(dict(image=str(output/image_path),prompt=job["record"]["prompt"]+f"\nseed: {job['seed']}"))
                    metadata=dict(prompt_id=job["record"]["sample_id"],prompt=job["record"]["prompt"],
                        negative_prompt=job["record"]["negative_prompt"],prompt_index=job["prompt_index"],
                        repeat=job["repeat"],seed=job["seed"],image=image_path,latent=latent_path,
                        sampling=c["sampling"],actual_steps=result["actual_steps"],adapter=adapter_info,
                        base_precision=c["model"]["base_precision"],trajectory=f"{stem}-trajectory.npz" if "trajectory" in result else None,
                        **({"spectrum": result["spectrum"]} if "spectrum" in result else {}))
                    atomic_json(output/f"{stem}.json",metadata)
                    with (output/"samples.jsonl").open("a",encoding="utf-8") as f:
                        f.write(json.dumps(metadata,ensure_ascii=False,allow_nan=False)+"\n")
                    finished+=1
                if decoder:
                    print(json.dumps(dict(event="vae_memory",workers=[x["memory_stats"] if x else {} for x in images])),flush=True)
                # Drop decoded latent/trajectory buffers as each wave is consumed.
                results.clear()
                print(json.dumps(dict(event="rollout_only_progress",completed=finished,
                    expected=len(records)*c["rollout"]["count"],generation_seconds=generation_seconds,
                    wave_seconds=time.monotonic()-begin)),flush=True)
        if gallery:write_gallery(gallery,output,title="MOSS rollout only")
        complete=dict(images=finished,prompts=len(records),elapsed_seconds=time.monotonic()-started,
                      output_dir=str(output),training=False,reward=False)
        atomic_json(output/"complete.json",complete)
        print(json.dumps(dict(event="rollout_only_complete",**complete)),flush=True)
        return output


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True)
    args=parser.parse_args(argv)
    return run(load_config(args.config))

if __name__=="__main__":main()
