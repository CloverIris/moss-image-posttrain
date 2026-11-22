"""Offline frozen JAX Qwen prefill and Torch VAE target encoding.

Run this module with the installed torch_env Python. No external model code is
executed and no downloads are attempted. All multimodal hidden states survive.
"""
import argparse
import gc
import json
from pathlib import Path
import numpy as np
from .data import atomic_json
from .sft_data import VERSION, load_manifest, cache_identity, digest, write_npz, SFTCache

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=4)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--model", default="$MODEL_ROOT/MOSS-Image-SFT")
    parser.add_argument("--processor-dir", default="$CODE_ROOT/moss-image/reference-lab/processor")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--max-reference-pixels", type=int, default=262144)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--short-tokens",type=int,default=0)
    parser.add_argument("--jax-python",default='$CONDA_PREFIX/envs/jax_env/bin/python')
    parser.add_argument("--encoder-memory", default="11GiB")
    parser.add_argument("--posterior", choices=("mode","sample"), default="mode")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--conditioning-only", action="store_true")
    from .config_io import parse_file_arguments
    args = parse_file_arguments(parser,argv)
    from .prompt_buckets import validate_lengths
    validate_lengths(args.short_tokens,args.max_tokens)
    if args.max_reference_pixels < 65536 or args.max_tokens < 1 or not 0 <= args.seed < 2**32:
        raise ValueError("Invalid reference pixel/token budget or seed")
    import copy
    from .config import DEFAULTS
    from .runtime import preflight,configure_process
    c=copy.deepcopy(DEFAULTS)
    c["runtime"]["devices"]=[args.device]
    preflight(c)
    configure_process(c,args.device)
    from .locking import output_lock
    directory = Path(args.cache).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with output_lock(directory):
        return prepare(args, directory)

def prepare(args, directory):
    rows = load_manifest(args.manifest, args.width, args.height, args.conditioning_only)
    settings = {k:v for k,v in vars(args).items() if k not in ("manifest","cache","device")}
    identity = cache_identity(rows, settings, args.model)
    fingerprint = digest(identity)
    manifest_path = directory/"manifest.json"
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if saved["identity_hash"] != fingerprint:
            raise ValueError("Cache inputs/settings changed; choose a new cache directory")
        if saved.get("complete"):
            cache = SFTCache(directory, require_targets=not args.conditioning_only)
            print(json.dumps({"event":"sft_cache_ready","records":len(cache.rows),"reused":True}),flush=True)
            return
        raise ValueError("Previous preparation was incomplete; use a new cache directory")
    manifest = dict(version=VERSION, complete=False, conditioning="all_multimodal_hidden_states",
                    identity=identity, identity_hash=fingerprint, entries=[], latent_channels=128)
    atomic_json(manifest_path, manifest)
    from .prompt_lengths import LengthScanner
    from .prompt_buckets import PromptBuckets,log_drop
    scanner=LengthScanner(args.model,args.max_reference_pixels)
    scanned=[scanner.tokens(row["prompt"],row["conditioning_images"]) for row in rows]
    kept=[];dropped=[]
    for row,info in zip(rows,scanned):
        if info["tokens"]>args.max_tokens:
            dropped.append(log_drop(row["id"],info["tokens"],args.max_tokens,"sft_prepare"))
        else:kept.append((row,info))
    buckets=PromptBuckets.resolve([i["tokens"] for i in scanned],args.short_tokens,args.max_tokens,
                                  minimum=scanner.tokens("")["tokens"])
    manifest["prefill_buckets"]={"short":buckets.short,"full":buckets.full}
    manifest["dropped_records"]=dropped
    (directory/"dropped.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in dropped))
    import torch
    from PIL import Image, ImageOps
    from transformers import AutoTokenizer, AutoImageProcessor, AutoVideoProcessor, Qwen3VLProcessor
    torch.set_num_threads(4)
    model_dir=Path(args.model)
    tokenizer=AutoTokenizer.from_pretrained(model_dir/"tokenizer",local_files_only=True)
    processor=Qwen3VLProcessor(
        image_processor=AutoImageProcessor.from_pretrained(args.processor_dir,local_files_only=True),
        video_processor=AutoVideoProcessor.from_pretrained(args.processor_dir,local_files_only=True),
        tokenizer=tokenizer,chat_template=tokenizer.chat_template)
    manifest["image_token_id"]=scanner.image_token_id
    manifest["encoder"]="native-jax-qwen3.5-multimodal"
    manifest["encoder_precision"]={"vision":"float32","language":"bfloat16"}
    from .qwen_service import PrefillClient
    # Fixed vision patch capacity derived from pixel budget; only Qwen owns GPU
    # until this child exits. Torch processor stays entirely on CPU.
    vision_capacity=max([4,*[int(np.prod(g)) for _,info in kept for g in info["image_grid_thw"]]])
    with PrefillClient(args.jax_python,args.model,vision_capacity) as encoder:
        def encode(prompt,refs):
            if refs:
                images=[]
                for path in refs:
                    with Image.open(path) as image:
                        images.append(ImageOps.exif_transpose(image).convert("RGB").copy())
                inputs=processor.apply_chat_template(
                    [{"role":"user","content":[*[{"type":"image","image":im} for im in images],{"type":"text","text":prompt}]}],
                    tokenize=True,add_generation_prompt=True,enable_thinking=True,return_dict=True,return_tensors="pt",
                    processor_kwargs={"images_kwargs":{"size":{"shortest_edge":65536,"longest_edge":args.max_reference_pixels}}})
            else:
                rendered=tokenizer.apply_chat_template([{"role":"user","content":prompt.strip()}],tokenize=False,
                    add_generation_prompt=True,enable_thinking=True)
                inputs=tokenizer(rendered,return_tensors="pt",truncation=False)
            ids=inputs["input_ids"].cpu().numpy()
            grid=inputs["image_grid_thw"].cpu().tolist() if "image_grid_thw" in inputs else []
            predicted=scanner.tokens(prompt,refs)
            if ids.shape[1]!=predicted["tokens"] or grid!=predicted["image_grid_thw"]:
                raise ValueError("Tokenizer-only scanner disagrees with actual processor")
            pixels=inputs["pixel_values"].float().cpu().numpy() if "pixel_values" in inputs else None
            hidden=encoder.encode(ids,buckets.capacity(ids.shape[1]),pixels,grid)
            return hidden,ids,grid
        empty,ids,_=encode("",[])
        manifest["empty_context"]=write_npz(directory,"empty.npz",context=empty,input_ids=ids,attention_mask=np.ones(ids.shape,bool))
        shared={}
        for i,(row,info) in enumerate(kept):
            # Content-addressed condition, independent of target image or seed.
            key=digest(dict(prompt=row["prompt"],refs=[identity["source_hashes"][r] for r in row["conditioning_images"]]))
            reused=key in shared
            if not reused:
                context,ids,grid=encode(row["prompt"],row["conditioning_images"])
                shared[key]=write_npz(directory,f"context-{key}.npz",context=context,input_ids=ids,attention_mask=np.ones(ids.shape,bool))
            entry=dict(row,context=shared[key],visual_tokens=info["visual_tokens"],sequence_length=info["tokens"],
                       image_grid_thw=info["image_grid_thw"],latent=None)
            manifest["entries"].append(entry)
            atomic_json(manifest_path,manifest)
            print(json.dumps(dict(event="sft_context",id=row["id"],tokens=info["tokens"],
                capacity=buckets.capacity(info["tokens"]),visual_tokens=info["visual_tokens"],reused=reused)),flush=True)
    del processor,tokenizer,empty,shared
    gc.collect()

    if not args.conditioning_only:
        from diffusers import AutoencoderKLFlux2
        vae = AutoencoderKLFlux2.from_pretrained(model_dir/"vae",torch_dtype=torch.float32,
                                               local_files_only=True).eval().requires_grad_(False).to("cuda")
        # Exactly the standard Flux2 patchification and running-stat normalization.
        for i,row in enumerate(manifest["entries"]):
            with Image.open(row["target_image"]) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                image = ImageOps.fit(image,(row["width"],row["height"]),method=Image.Resampling.LANCZOS)
                pixels = np.asarray(image,dtype=np.float32)/127.5-1.
            tensor = torch.from_numpy(pixels.transpose(2,0,1)[None]).to("cuda")
            with torch.inference_mode():
                dist = vae.encode(tensor).latent_dist
                generator = torch.Generator(device="cuda").manual_seed(args.seed+i)
                z = (dist.mode() if args.posterior=="mode" else dist.sample(generator=generator)).float()
                n,c,h,w = z.shape
                z = z.reshape(n,c,h//2,2,w//2,2).permute(0,1,3,5,2,4).reshape(n,c*4,h//2,w//2)
                z = (z-vae.bn.running_mean[None,:,None,None])/torch.sqrt(
                    vae.bn.running_var[None,:,None,None]+vae.config.batch_norm_eps)
                latent = z.cpu().numpy()
            if not np.isfinite(latent).all():
                raise ValueError("Nonfinite VAE target")
            row["latent"] = write_npz(directory,f"latent-{i:06d}.npz",latent=latent)
            atomic_json(manifest_path,manifest)
            print(json.dumps({"event":"sft_latent","id":row["id"],"shape":list(latent.shape)}),flush=True)
        del vae
        gc.collect()
        torch.cuda.empty_cache()
    manifest["complete"] = True
    atomic_json(manifest_path,manifest)
    SFTCache(directory,require_targets=not args.conditioning_only)
    print(json.dumps({"event":"sft_cache_ready","records":len(kept),"dropped":len(dropped),"cache":str(directory)}),flush=True)

if __name__ == "__main__":
    main()
