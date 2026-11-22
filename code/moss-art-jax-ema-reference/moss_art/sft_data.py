"""Strict image/text SFT manifests and immutable, verified feature caches."""
from pathlib import Path
import json
import hashlib
import numpy as np
from .data import sha256_file, atomic_json

VERSION = 1

def load_manifest(path, width=1024, height=1024, conditioning_only=False):
    path = Path(path).resolve()
    rows, seen = [], set()
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict):
            raise ValueError(f"Line {number}: expected object")
        allowed = {"id", "prompt", "target_image", "conditioning_image", "conditioning_images", "width", "height"}
        if set(obj)-allowed:
            raise ValueError(f"Line {number}: unknown fields {set(obj)-allowed}")
        identifier = obj.get("id", f"line-{number:06d}")
        if not isinstance(identifier, str) or not identifier.strip() or identifier in seen:
            raise ValueError("SFT ids must be unique nonempty strings")
        prompt = obj.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"{identifier}: prompt must be nonempty")
        if "conditioning_image" in obj and "conditioning_images" in obj:
            raise ValueError("Use conditioning_image OR conditioning_images")
        refs = obj.get("conditioning_images", [obj["conditioning_image"]] if "conditioning_image" in obj else [])
        if not isinstance(refs, list):
            raise ValueError("conditioning_images must be a list")
        def resolve(value):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("Image path must be a nonempty string")
            p = Path(value)
            p = (path.parent / p).resolve() if not p.is_absolute() else p.resolve()
            if not p.is_file():
                raise FileNotFoundError(p)
            return str(p)
        refs = [resolve(p) for p in refs]
        target = resolve(obj["target_image"]) if "target_image" in obj else None
        if target is None and not conditioning_only:
            raise ValueError(f"{identifier}: target_image required for SFT")
        w, h = obj.get("width", width), obj.get("height", height)
        if any(type(x) is not int or x <= 0 or x % 16 for x in (w,h)):
            raise ValueError("Target dimensions must be positive multiples of 16")
        seen.add(identifier)
        rows.append(dict(id=identifier, prompt=prompt, target_image=target,
                         conditioning_images=refs, width=w, height=h))
    if not rows:
        raise ValueError("Empty SFT manifest")
    return rows

def cache_identity(rows, settings, model_dir):
    from .assets import component_identity
    model_dir = Path(model_dir)
    sources = sorted({p for row in rows for p in [row["target_image"], *row["conditioning_images"]] if p})
    identity = dict(rows=rows, settings=settings, source_hashes={p:sha256_file(p) for p in sources},
                    models={part:component_identity(model_dir, part)
                            for part in ("transformer","text_encoder","vae")})
    identity["tokenizer"] = {str(p.relative_to(model_dir)):sha256_file(p)
        for p in sorted((model_dir/"tokenizer").glob("*")) if p.is_file()}
    processor = Path(settings["processor_dir"])
    identity["processor"] = {p.name:sha256_file(p) for p in sorted(processor.glob("*.json"))}
    identity["implementation"] = {name:sha256_file(Path(__file__).parent/name)
                                 for name in ("sft_data.py","sft_prepare.py","qwen_prefill.py","qwen_service.py","prompt_lengths.py","prompt_buckets.py")}
    identity["jax_text_encoder"]=sha256_file(Path(__file__).parents[1]/"vendor/moss_jax/text_encoder.py")
    return identity

def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def write_npz(directory, name, **arrays):
    import os, tempfile
    directory = Path(directory)
    descriptor, tmp = tempfile.mkstemp(prefix=".sft-", suffix=".npz", dir=directory)
    os.close(descriptor)
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, directory/name)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return {"file":name, "sha256":sha256_file(directory/name)}

class SFTCache:
    def __init__(self, directory, require_targets=True):
        self.directory = Path(directory).resolve()
        self.manifest = json.loads((self.directory/"manifest.json").read_text())
        if self.manifest.get("version") != VERSION or not self.manifest.get("complete"):
            raise ValueError("Incomplete or unsupported SFT cache")
        if self.manifest.get("conditioning") != "all_multimodal_hidden_states":
            raise ValueError("SFT cache must preserve every visual and text token")
        if self.manifest["identity_hash"] != digest(self.manifest["identity"]):
            raise ValueError("Invalid SFT cache identity")
        self.rows = self.manifest["entries"]
        if not self.rows or len({r["id"] for r in self.rows}) != len(self.rows):
            raise ValueError("Invalid cache entries")
        self.verified = set()
        for row in self.rows:
            self.load(row, require_target=require_targets)

    def _read(self, entry):
        p = (self.directory/entry["file"]).resolve()
        if self.directory not in p.parents:
            raise ValueError("Cache path escapes its directory")
        if str(p) not in self.verified:
            if sha256_file(p) != entry["sha256"]:
                raise ValueError(f"Corrupted cache file: {p}")
            self.verified.add(str(p))
        with np.load(p, allow_pickle=False) as f:
            return {key:f[key] for key in f.files}

    def load(self, row, require_target=True):
        data = self._read(row["context"])
        context = data["context"].astype(np.float32)
        ids = data["input_ids"]
        mask = data["attention_mask"]
        visual = int(row["visual_tokens"])
        if (context.ndim != 3 or context.shape[0] != 1 or context.shape[1] < 1
                or ids.shape != context.shape[:2] or mask.shape != ids.shape
                or not mask.all() or not np.isfinite(context).all()
                or context.shape[1] != row["sequence_length"]):
            raise ValueError("Invalid full-sequence conditioning")
        if bool(row["conditioning_images"]) != (visual > 0):
            raise ValueError("Missing or unexpected visual tokens")
        if visual != int(np.sum(ids == self.manifest["image_token_id"])):
            raise ValueError("Visual-token metadata mismatch")
        latent = None
        if row.get("latent"):
            latent = self._read(row["latent"])["latent"].astype(np.float32)
            channels = self.manifest.get("latent_channels", 128)
            if latent.shape != (1,channels,row["height"]//16,row["width"]//16) or not np.isfinite(latent).all():
                raise ValueError("Invalid normalized packed target latent")
        elif require_target:
            raise ValueError("Conditioning-only cache cannot train")
        return latent, context

    def empty_context(self):
        value = self._read(self.manifest["empty_context"])["context"].astype(np.float32)
        if value.ndim != 3 or value.shape[0] != 1 or value.shape[1] < 1 or not np.isfinite(value).all():
            raise ValueError("Invalid empty context")
        return value
