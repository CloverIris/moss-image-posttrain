"""Run a saved SideNet on an idle GPU, optionally benchmark paired Euler controls."""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from common import RUN, NATIVE, arrays, read, save, sha

def render(folder, model_path, latents):
    from PIL import Image,ImageDraw
    from moss_art.evaluate import LatentDecoder
    sheet=Image.new("RGB",(len(latents)*320,350),"white")
    stages=("input","mid","up0","up1","up2","up3","output")
    with LatentDecoder(model_path,precision="float16",low_precision_stages=stages,conv_backend="cudnn_bounded") as decoder:
        for i,(name,latent) in enumerate(latents.items()):
            image=Image.fromarray(decoder(latent)[0])
            image.save(folder/(name+".png"))
            sheet.paste(image.resize((320,320),Image.Resampling.LANCZOS),(320*i,0))
            ImageDraw.Draw(sheet).text((320*i+5,327),name,fill="black")
    sheet.save(folder/"comparison.jpg",quality=95)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--prompt-id",required=True)
    parser.add_argument("--seed",type=int,required=True)
    parser.add_argument("--steps",type=int,choices=(8,12),default=8)
    parser.add_argument("--gpu",type=int,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--compare",action="store_true")
    parser.add_argument("--decode-only",action="store_true",help="Finish decoding a matching completed latent benchmark")
    args=parser.parse_args()
    if args.output.exists() and not args.decode_only:raise FileExistsError(args.output)
    uuid,used=subprocess.check_output(["nvidia-smi",f"--id={args.gpu}","--query-gpu=uuid,memory.used","--format=csv,noheader,nounits"],text=True).strip().split(",")
    if int(used)>300:raise RuntimeError("Selected GPU is busy")
    os.environ.update(CUDA_VISIBLE_DEVICES=uuid.strip(),JAX_PLATFORMS="cuda",XLA_PYTHON_CLIENT_PREALLOCATE="false",
                      JAX_ENABLE_X64="true",JAX_COMPILATION_CACHE_DIR=str(RUN/"jax_cache"))
    from teacher import Teacher
    from side import SideNet
    from solver import sample
    from flax import serialization
    import jax
    import jax.numpy as jnp
    cfg=read(RUN/"config.json")
    if args.decode_only:
        meta=read(args.output/"result.json")
        if (meta["prompt_id"],meta["seed"],meta["steps"],meta["checkpoint_sha256"])!=(args.prompt_id,args.seed,args.steps,sha(args.checkpoint)):
            raise ValueError("Stored benchmark does not match this request")
        with np.load(args.output/"latents.npz") as saved:
            render(args.output,cfg["model"],{k:saved[k] for k in saved.files})
        return
    oracle=Teacher(cfg)
    c=oracle.condition(args.prompt_id)
    x=oracle.noise(args.seed)
    net=SideNet(cfg["width"])
    template=net.init(jax.random.key(0),x,x,.5,.1,c)["params"]
    p=serialization.from_bytes(template,args.checkpoint.read_bytes())
    probe=jax.jit(jax.vmap(lambda p,x,v,t,h,c:net.apply({"params":p},x,v,t,h,c),in_axes=(None,None,None,None,0,None)))
    velocity=lambda x,t:oracle.velocity(c,x,t)
    query=lambda x,v,t,h:probe(p,x,v,t,h,c)
    # Compile the two probe batch shapes and both CFG branches before timing.
    v=velocity(x,np.float32(1))
    for count in (1,3):
        jax.block_until_ready(query(x,v,np.float32(1),jnp.full(count,.1,jnp.float32)))
    names=[f"ba{args.steps}"]
    if args.compare:names=["euler48",f"euler{args.steps}",*names]
    latents={};times={}
    for name in names:
        start=time.perf_counter()
        if name.startswith("euler"):result=oracle.euler(c,args.seed,int(name[5:]))
        else:result=sample(velocity,query,x,oracle.grid(args.steps))
        latents[name]=np.asarray(result)
        times[name]=time.perf_counter()-start
        if not np.isfinite(latents[name]).all():raise FloatingPointError(name)
    args.output.mkdir(parents=True)
    arrays(args.output/"latents.npz",**latents)
    summary=dict(prompt_id=args.prompt_id,seed=args.seed,steps=args.steps,checkpoint=str(args.checkpoint),
                 checkpoint_sha256=sha(args.checkpoint),cfg=cfg["sampling"]["guidance_scale"],
                 precision="bf16",device=uuid.strip(),denoising_seconds=times)
    if args.compare:
        summary["mse_to_teacher48"]={name:float(np.mean((value-latents["euler48"])**2)) for name,value in latents.items()}
        summary["observed_denoising_speedup"]=times["euler48"]/times[names[-1]]
    save(args.output/"result.json",summary)
    oracle.model.release_base()
    del oracle,v
    import gc
    gc.collect()
    render(args.output,cfg["model"],latents)
    print(summary,flush=True)

if __name__=="__main__":main()

