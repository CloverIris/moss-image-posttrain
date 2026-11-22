"""Resume identities for local model assets and the actual text-cache manifest.

Weight fingerprints intentionally avoid rereading tens of GiB on every run:
complete safetensors header SHA256 + resolved path + byte size + mtime_ns.
They detect ordinary replacement/editing, not malicious edits that preserve
all of those properties. Configs, source code and cache manifests use full
file SHA256. No asset is copied, modified, refreshed, or downloaded here.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import struct

from .data import sha256_file

IDENTITY_VERSION = 1
SOURCE_FILES = (
    "moss_art/mxfp8_cache.py",
    "moss_art/staged.py", "moss_art/stage_cache.py",
    "moss_art/model.py", "moss_art/mxfp8.py", "moss_art/attention.py", "moss_art/conditioning.py", "moss_art/prompt_buckets.py", "moss_art/tensor_parallel.py", "moss_art/lora.py", "moss_art/art.py",
    "moss_art/worker.py", "moss_art/sampling.py", "moss_art/trainer.py", "moss_art/optimizers.py", "moss_art/config.py", "moss_art/config_io.py",
    "moss_art/reward_diagnostics.py", "moss_art/silva_jax.py", "moss_art/silva_reward.py", "moss_art/tag_reward.py", "moss_art/pixai_reward.py", "moss_art/pixai_tagger.py", "moss_art/tensorboard_log.py", "moss_art/phases.py", "moss_art/host_weights.py",
    "moss_art/runtime.py", "moss_art/mesh_runtime.py", "moss_art/memory.py", "moss_art/data_parallel.py", "moss_art/replay.py", "moss_art/checkpoint.py", "moss_art/data.py", "moss_art/resolution_mixing.py", "moss_art/batch_plan.py",
    "moss_art/prepare.py", "moss_art/prefetch.py", "moss_art/qwen_prefill.py", "moss_art/prompt_lengths.py", "moss_art/evaluate.py", "moss_art/vae_precision.py", "moss_art/rewards.py",
    "moss_art/tiled_conv.py", "moss_art/cudnn_conv.py", "moss_art/native/cudnn_conv_ffi.cc",
    "scripts/build_cudnn_conv.py",
    "moss_art/http_reward.py", "moss_art/reward_worker.py", "moss_art/assets.py", "moss_art/locking.py", "moss_art/cli.py",
    "vendor/moss_jax/weights.py", "vendor/moss_jax/dit.py", "vendor/moss_jax/scheduler.py",
    "vendor/moss_jax/vae.py", "vendor/moss_jax/text_encoder.py",
)


def _full_file(path):
    path = Path(path)
    stat = path.stat()
    if not path.is_file():
        raise ValueError(f"Expected a regular asset file: {path}")
    return {"resolved_path": str(path.resolve()), "size": stat.st_size,
            "sha256": sha256_file(path), "fingerprint_method": "full-file-sha256"}


def _weight_file(path):
    path = Path(path)
    stat = path.stat()
    if not path.is_file():
        raise ValueError(f"Expected a regular weight file: {path}")
    with path.open("rb") as source:
        prefix = source.read(8)
        if len(prefix) != 8:
            raise ValueError(f"Truncated safetensors prefix: {path}")
        header_size = struct.unpack("<Q", prefix)[0]
        if header_size > stat.st_size - 8 or header_size > 128 * 1024 * 1024:
            raise ValueError(f"Invalid safetensors header extent: {path}")
        header = source.read(header_size)
    try:
        metadata = json.loads(header)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"Invalid safetensors header JSON: {path}") from error
    if not isinstance(metadata, dict):
        raise ValueError(f"Invalid safetensors header mapping: {path}")
    return {"resolved_path": str(path.resolve()), "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "header_sha256": hashlib.sha256(header).hexdigest(),
            "fingerprint_method": "header-sha256+size+mtime_ns+resolved-path; not-full-payload-hash"}


def component_identity(model_dir, component):
    """Read a component's config/index hashes and every safetensors identity."""
    directory = Path(model_dir) / component
    result = {"config": _full_file(directory / "config.json")}
    indexes = sorted(directory.glob("*.safetensors.index.json"))
    if indexes:
        result["indexes"] = {file.name: _full_file(file) for file in indexes}
    files = sorted(directory.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors files in {directory}")
    result["weights"] = {file.name: _weight_file(file) for file in files}
    return result


def experiment_identity(config):
    """Return deterministic JSON metadata to save AND compare on resume/eval.

    For a real MOSS run, bind the current transformer/VAE weights, their JSON
    configs, vendored numerical implementations, and the exact text-cache
    manifest. The manifest itself binds text encoder/tokenizer identities and
    individual feature payload checksums. A tiny run needs no external model
    directory or prepared cache, but remains bound to its numerical source.
    Call before opening an existing run's outputs or spawning GPU workers.
    """
    root = Path(__file__).resolve().parents[1]
    result = {"identity_version": IDENTITY_VERSION,
              "model_kind": config["model"]["kind"],
              "source_files": {name: _full_file(root / name) for name in SOURCE_FILES},
              "weight_hash_scope": "safetensors header/size/mtime_ns/path, not full tensor payload"}
    if config["model"].get("vae_conv_backend") == "cudnn_bounded":
        manifest_path = Path(os.environ.get("MOSS_CUDNN_CONV_BUILD", str(root / "outputs/native/build.json")))
        manifest = json.loads(manifest_path.read_text())
        result["vae_convolution_build"] = {"manifest": _full_file(manifest_path),
                                           "binary": _full_file(manifest["binary"])}
    if config["model"]["kind"] == "moss":
        directory = Path(config["model"]["path"]).resolve()
        result["model_directory"] = str(directory)
        result["components"] = {name: component_identity(directory, name)
                                 for name in ("transformer", "vae")}
        if config.get("preparation",{}).get("async_prefetch",False):
            from .data import model_identity
            result["prompt_encoder_identity"]=model_identity(config["model"]["path"],full_weight_hash=config["preparation"]["full_weight_hash"])
            result["condition_cache_policy"]="content-addressed incremental cache; manifest growth excluded from resume identity"
        else:
            result["text_cache_manifest"] = _full_file(Path(config["data"]["cache"]) / "manifest.json")
    elif config["model"]["kind"] != "tiny":
        raise ValueError(f"Unknown model kind: {config['model']['kind']}")
    reward = config.get("reward", {})
    if reward.get("kind") == "pixai":
        result["reward_assets"]={"kind":"pixai","weights":_weight_file(reward["model_path"]),
            "metadata":_full_file(Path(reward["metadata_dir"])/"config.json")}
    if reward.get("kind") == "silva":
        head=Path(reward["model_path"]).resolve();backbone=Path(reward["backbone_path"]).resolve()
        result["reward_assets"]={"kind":"silva","head":component_identity(head,"."),"backbone":component_identity(backbone,"."),
            "preprocessor":_full_file(backbone/"preprocessor_config.json"),"calibration":"latent-to-score LUT in head model.safetensors"}
        if (head/"release.json").is_file():result["reward_assets"]["release"]=_full_file(head/"release.json")
    if reward.get("kind") == "siglip":
        reward_directory = Path(reward["model_path"]).resolve()
        reward_identity = component_identity(reward_directory, ".")
        # Config, preprocessing, tokenizer vocabulary and sentencepiece model
        # are small enough to checksum in full on every resume preflight.
        reward_identity["processor_tokenizer_files"] = {
            file.name: _full_file(file)
            for file in sorted(reward_directory.iterdir())
            if file.is_file() and file.suffix in {".json", ".model", ".txt", ".jinja"}
        }
        result["reward_assets"] = {"kind": "siglip", "model_directory": str(reward_directory),
                                   "files": reward_identity}
    elif reward.get("kind") == "http":
        result["reward_assets"] = {"kind": "http", "external_state_verification":
                                   "not available; URL and request settings are configuration-bound, remote scorer state is caller-owned"}
    elif reward.get("kind") == "command":
        result["reward_assets"] = {"kind": "command", "external_state_verification":
                                   "not available; argv is configuration-bound but external service/model state is caller-owned"}
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    result["fingerprint"] = hashlib.sha256(encoded).hexdigest()
    return result
