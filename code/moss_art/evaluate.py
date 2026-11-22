"""Native-JAX VAE decoding with explicit device-state ownership."""
from __future__ import annotations
import argparse
import gc
import html
import json
from pathlib import Path

import numpy as np


class LatentDecoder:
    """Load once per VAE phase and release all owned device buffers on close."""
    def __init__(self, model_dir, *, device=None, tiling=True, attention_chunk_size=256,
                 precision="fp32", low_precision_stages=(), conv_backend="xla", conv_workspace_mib=256):
        import jax
        from .vae_precision import decode_latents as native_decode
        from vendor.moss_jax.weights import load_config, load_host_params, to_device
        if isinstance(device, str):
            platform, _, index = device.partition(":")
            device = jax.devices(platform)[int(index or 0)]
        self.device = device or jax.devices()[0]
        config = load_config(model_dir, "vae")
        host = load_host_params(model_dir, "vae")
        host = {k: v for k,v in host.items() if k.startswith(("decoder.", "post_quant_conv.", "bn."))}
        self.params = to_device(host, self.device)
        options = {"xla_gpu_autotune_level": 4} if conv_backend == "tiled_gemm" and self.device.platform == "gpu" else None
        self.decode = jax.jit(lambda p,z: native_decode(p,z,config,tiling=tiling,
            attention_chunk_size=attention_chunk_size,clamp=False,precision=precision,
            low_precision_stages=low_precision_stages,conv_backend=conv_backend,
            conv_workspace_mib=conv_workspace_mib),compiler_options=options)

    def __call__(self, packed_latents):
        import jax
        latents = np.asarray(packed_latents, dtype=np.float32)
        if latents.ndim != 4 or latents.shape[0] < 1 or latents.shape[1] != 128 or not np.isfinite(latents).all():
            raise ValueError("Expected finite packed MOSS latents [B,128,H,W]")
        images = []
        for row in latents:
            tensor = jax.device_put(row[None], self.device)
            output = None
            try:
                output = self.decode(self.params, tensor)
                pixels = np.asarray(output, dtype=np.float32)[0].transpose(1,2,0)
                if not np.isfinite(pixels).all(): raise ValueError("VAE produced nonfinite raw pixels before clipping")
                images.append(np.rint(np.clip((pixels+1)*127.5,0,255)).astype(np.uint8))
            finally:
                tensor.delete()
                if output is not None: output.delete()
        return np.stack(images)

    def close(self):
        import jax
        from vendor.moss_jax.weights import release_device
        params,decode=getattr(self,"params",None),getattr(self,"decode",None)
        self.params=None;self.decode=None
        if params is not None:
            jax.block_until_ready(params)
            release_device(params)
        # Clear this decoder's executables only; keep DiT/JAX global caches intact.
        if decode is not None and hasattr(decode,"clear_cache"):decode.clear_cache()
        del params,decode
        gc.collect()

    def __enter__(self): return self
    def __exit__(self, *exc): self.close()


class MeshLatentDecoder(LatentDecoder):
    """One decoder graph/state, replicated weights and batch-sharded latents."""
    def __init__(self,model_dir,parallel,*,tiling=True,attention_chunk_size=256,
                 precision="fp32",low_precision_stages=(),conv_backend="xla",conv_workspace_mib=256):
        import jax
        from jax.sharding import PartitionSpec as P
        from .vae_precision import decode_latents as native_decode
        from .cudnn_conv import mesh_convolution
        from vendor.moss_jax.weights import load_config,load_host_params,to_device
        self.parallel=parallel;self.device=jax.local_devices()[0]
        config=load_config(model_dir,"vae")
        host={k:v for k,v in load_host_params(model_dir,"vae").items() if k.startswith(("decoder.","post_quant_conv.","bn."))}
        self.params=to_device(host,parallel.replicated)
        def local_decode(p,z):
            with mesh_convolution(jax.local_devices(),"dp"):
                return native_decode(p,z,config,tiling=tiling,attention_chunk_size=attention_chunk_size,clamp=False,
                    precision=precision,low_precision_stages=low_precision_stages,conv_backend=conv_backend,conv_workspace_mib=conv_workspace_mib)
        graph=jax.shard_map(local_decode,mesh=parallel.mesh,in_specs=(P(),P("dp",None,None,None)),
                            out_specs=P("dp",None,None,None),check_vma=False)
        self.decode=jax.jit(graph,out_shardings=parallel.batch(4))

    def __call__(self,packed_latents):
        import jax
        latents=np.asarray(packed_latents,np.float32)
        if latents.ndim!=4 or latents.shape[0]<1 or latents.shape[1]!=128 or not np.isfinite(latents).all():
            raise ValueError("Expected finite packed MOSS latents [B,128,H,W]")
        images=[];batch=self.parallel.size
        for offset in range(0,len(latents),batch):
            chunk=latents[offset:offset+batch];count=len(chunk)
            if count<batch:chunk=np.concatenate((chunk,np.repeat(chunk[-1:],batch-count,axis=0)))
            tensor=self.parallel.put(chunk);output=None
            try:
                output=self.decode(self.params,tensor)
                pixels=np.asarray(output,np.float32)[:count].transpose(0,2,3,1)
                if not np.isfinite(pixels).all():raise ValueError("VAE produced nonfinite raw pixels before clipping")
                images.append(np.rint(np.clip((pixels+1)*127.5,0,255)).astype(np.uint8))
            finally:
                tensor.delete()
                if output is not None:output.delete()
        return np.concatenate(images)


def decode_latents(model_dir, packed_latents, **kwargs):
    with LatentDecoder(model_dir, **kwargs) as decoder:
        return decoder(packed_latents)


def save_images(images, output_dir, *, stems=None):
    from PIL import Image
    from .rewards import as_image_batch
    images = as_image_batch(images)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if stems is None:
        stems = [f"sample-{i:06d}" for i in range(len(images))]
    if len(stems) != len(images):
        raise ValueError("One filename stem is required per image")
    paths = []
    for pixels, stem in zip(images, stems):
        if not isinstance(stem, str) or not stem or Path(stem).name != stem or stem in {".", ".."}:
            raise ValueError("Image stems must be simple filenames")
        path = directory / f"{stem}.png"
        Image.fromarray(pixels).save(path)
        paths.append(str(path.resolve()))
    return paths


def write_gallery(rows, output_dir, *, title="MOSS ART evaluation"):
    """Write a static local gallery; rows contain image,prompt and optional score."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    parts = ["<!doctype html><meta charset='utf-8'><title>" + html.escape(title) + "</title>",
             "<style>body{font:16px system-ui;background:#14141a;color:#eee;margin:2rem}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:1rem}img{max-width:100%}figure{background:#242431;padding:1rem;margin:0}figcaption{white-space:pre-wrap;overflow-wrap:anywhere}</style>",
             "<h1>" + html.escape(title) + "</h1><main>"]
    for row in rows:
        path = Path(row["image"]).resolve()
        try:
            image_ref = path.relative_to(directory.resolve()).as_posix()
        except ValueError:
            # Copy into the report directory so the report stays portable.
            import shutil
            target = directory / f"gallery-{len(parts):06d}-{path.name}"
            shutil.copy2(path, target)
            image_ref = target.name
        score = "" if "score" not in row else f"\nreward: {float(row['score']):.6f}"
        parts.append("<figure><img loading='lazy' src='" + html.escape(image_ref, quote=True)
                     + "'><figcaption>" + html.escape(str(row.get("prompt", "")) + score) + "</figcaption></figure>")
    parts.append("</main>")
    file = directory / "index.html"
    file.write_text("\n".join(parts), encoding="utf-8")
    return str(file.resolve())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--latents", required=True, help=".npy or .npz with final_latents or latents")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default=None, help="JAX platform:index, e.g. cpu:0 or gpu:0")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--vae-precision", choices=("fp32", "bfloat16", "float16"), default="fp32")
    parser.add_argument("--vae-stages", nargs="*", default=[], choices=("input","mid","up0","up1","up2","up3","output"))
    parser.add_argument("--vae-conv-backend", choices=("xla", "cudnn_bounded", "tiled_gemm"), default="xla")
    parser.add_argument("--vae-conv-workspace-mib", type=int, default=256)
    parser.add_argument("--reward-config", help="JSON reward config file; optional")
    from .config_io import parse_file_arguments
    args = parse_file_arguments(parser,argv)
    stages = tuple(args.vae_stages)
    if (args.vae_precision == "fp32") != (not stages):
        parser.error("FP32 requires empty --vae-stages; low precision requires explicit --vae-stages")
    # Inherit XLA and precision environment; backend-specific compute is local.
    latent_file = np.load(args.latents, allow_pickle=False)
    if isinstance(latent_file, np.lib.npyio.NpzFile):
        with latent_file as archive:
            key = "final_latents" if "final_latents" in archive else "latents"
            latents = archive[key]
    else:
        latents = latent_file
    images = decode_latents(args.model_dir, latents, device=args.device,
                            precision=args.vae_precision,
                            low_precision_stages=stages,
                            conv_backend=args.vae_conv_backend,
                            conv_workspace_mib=args.vae_conv_workspace_mib)
    paths = save_images(images, args.output_dir)
    rows = [{"image": path, "prompt": args.prompt} for path in paths]
    if args.reward_config:
        from .rewards import score_images
        config = json.loads(Path(args.reward_config).read_text())
        for row, score in zip(rows, score_images(images, [args.prompt] * len(images), config)):
            row["score"] = float(score)
    from .data import atomic_json
    atomic_json(Path(args.output_dir) / "evaluation.json", rows)
    print(json.dumps({"images": paths, "gallery": write_gallery(rows, args.output_dir)}))


if __name__ == "__main__":
    main()
