"""Validated JSONL prompts and content-addressed native-JAX text caches."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import hashlib
import json
import os
import struct
import tempfile

import numpy as np

CACHE_VERSION = 1


@dataclass(frozen=True)
class PromptRecord:
    id: str
    prompt: str
    negative_prompt: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def load_prompts(path, *, default_negative_prompt="") -> list[PromptRecord]:
    """Read JSONL objects with a nonempty prompt and unique optional string id.

    Blank lines are ignored. Unknown fields are retained in metadata. JSON is
    deliberately never evaluated as code, and there is no network access.
    """
    if not isinstance(default_negative_prompt, str):
        raise ValueError("default_negative_prompt must be a string")
    records, seen = [], set()
    with Path(path).open(encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            prompt = obj.get("prompt")
            negative = obj.get("negative_prompt", default_negative_prompt)
            if "id" in obj and "sample_id" in obj and obj["id"] != obj["sample_id"]:
                raise ValueError(f"{path}:{line_number}: id conflicts with sample_id")
            identifier = obj.get("id", obj.get("sample_id", f"line-{line_number:06d}"))
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"{path}:{line_number}: prompt must be a nonempty string")
            if not isinstance(negative, str):
                raise ValueError(f"{path}:{line_number}: negative_prompt must be a string")
            if not isinstance(identifier, str) or not identifier.strip() or identifier in seen:
                raise ValueError(f"{path}:{line_number}: id must be a unique nonempty string")
            metadata = obj.get("metadata", {})
            if not isinstance(metadata, dict):
                raise ValueError(f"{path}:{line_number}: metadata must be an object")
            metadata = dict(metadata)
            metadata.update({k: v for k, v in obj.items()
                             if k not in {"id", "sample_id", "prompt", "negative_prompt", "metadata"}})
            seen.add(identifier)
            records.append(PromptRecord(identifier, prompt, negative, metadata))
    if not records:
        raise ValueError(f"{path}: no prompt records")
    return records


def text_key(text: str) -> str:
    # Match the vendored MOSS chat template's whitespace normalization.
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def model_identity(model_dir, *, full_weight_hash=False) -> dict:
    """Fingerprint tokenizer/config, encoder implementation, and encoder weights.

    Default weight identity includes the complete safetensors header, byte
    size, nanosecond mtime and resolved path (fast, detects normal replacement).
    Set full_weight_hash for cryptographic verification of every weight byte.
    The fingerprint method is explicit in the manifest, never called a full
    weight checksum when only metadata was read.
    """
    directory = Path(model_dir).resolve()
    files = {}
    for name in ("text_encoder/config.json", "tokenizer/tokenizer.json",
                 "tokenizer/tokenizer_config.json", "tokenizer/chat_template.jinja",
                 "text_encoder/model.safetensors.index.json"):
        file = directory / name
        if file.is_file():
            files[name] = {"sha256": sha256_file(file)}
    for required in ("text_encoder/config.json", "tokenizer/tokenizer.json"):
        if required not in files:
            raise FileNotFoundError(directory / required)
    weights = sorted((directory / "text_encoder").glob("*.safetensors"))
    if not weights:
        raise FileNotFoundError(f"No text encoder weights in {directory}")
    for file in weights:
        stat = file.stat()
        with file.open("rb") as stream:
            raw_size = stream.read(8)
            if len(raw_size) != 8:
                raise ValueError(f"Truncated safetensors file: {file}")
            size = struct.unpack("<Q", raw_size)[0]
            if size > stat.st_size - 8:
                raise ValueError(f"Invalid safetensors header: {file}")
            header = stream.read(size)
        entry = {"resolved_path": str(file.resolve()), "size": stat.st_size,
                 "mtime_ns": stat.st_mtime_ns,
                 "header_sha256": hashlib.sha256(header).hexdigest()}
        if full_weight_hash:
            entry["sha256"] = sha256_file(file)
        files[str(file.relative_to(directory))] = entry
    implementation = Path(__file__).resolve().parents[1] / "vendor/moss_jax/text_encoder.py"
    identity = {"model_dir": str(directory), "files": files,
                "weight_fingerprint_method": "sha256" if full_weight_hash else "header+size+mtime_ns+path",
                "encoder_source_sha256": sha256_file(implementation)}
    identity["prefill_source_sha256"] = sha256_file(Path(__file__).parent/"qwen_prefill.py")
    identity["fingerprint"] = _canonical_digest(identity)
    return identity


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class PromptCache:
    """Read-only checked cache. Features remain unpadded [1, sequence, hidden]."""
    def __init__(self, cache_dir, model_dir=None, *, verify_files=True):
        self.directory = Path(cache_dir)
        self.manifest = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != CACHE_VERSION:
            raise ValueError("Unsupported prompt cache schema")
        encoder = self.manifest.get("encoder")
        if encoder not in ("moss-native-jax-qwen3.5", "moss-vllm-qwen3.5"):
            raise ValueError("Cache is not marked as supported MOSS text features")
        self.conditioning_contract = None
        if encoder == "moss-vllm-qwen3.5":
            from .vllm_contract import validate_conditioning_contract
            contract = self.manifest.get("conditioning_contract")
            self.storage_dtype = validate_conditioning_contract(contract)
            self.conditioning_contract = contract
        self.verify_files = verify_files
        self._verified = set()
        identity = self.manifest["model_identity"]
        if model_dir is not None:
            full = identity.get("weight_fingerprint_method") == "sha256"
            current = model_identity(model_dir, full_weight_hash=full)
            if current["fingerprint"] != identity["fingerprint"]:
                raise ValueError("Prompt cache model/tokenizer fingerprint mismatch; prepare a new cache")

    def load_text(self, text):
        key = text_key(text)
        if key not in self.manifest["entries"]:
            raise KeyError(f"Prompt not present in cache: {text[:100]!r}")
        entry = self.manifest["entries"][key]
        filename = (self.directory / entry["file"]).resolve()
        if self.directory.resolve() not in filename.parents:
            raise ValueError("Cache entry escapes its directory")
        if self.verify_files and key not in self._verified:
            if sha256_file(filename) != entry["sha256"]:
                raise ValueError(f"Corrupted text cache entry: {key}")
            self._verified.add(key)
        with np.load(filename, allow_pickle=False) as saved:
            raw_embedding = saved["embedding"]
            if self.conditioning_contract is not None:
                ids = saved["input_ids"] if "input_ids" in saved else None
                expected_dtype = np.uint16 if self.storage_dtype == "bfloat16" else np.float32
                if (raw_embedding.dtype != expected_dtype or ids is None or ids.ndim != 2 or ids.dtype != np.int32
                        or ids.shape != raw_embedding.shape[:2]
                        or entry.get("token_count") != ids.shape[1]
                        or (self.storage_dtype == "bfloat16" and entry.get("embedding_storage_dtype") != "bfloat16")):
                    raise ValueError(f"Invalid vLLM prompt cache storage/token metadata: {key}")
                from .vllm_contract import decode_embedding_arrays
                embedding = decode_embedding_arrays(raw_embedding, self.conditioning_contract)
            else:
                embedding = raw_embedding.astype(np.float32)
            mask = saved["attention_mask"].astype(bool)
        if (embedding.ndim != 3 or embedding.shape[0] != 1
                or mask.shape != embedding.shape[:2]
                or list(embedding.shape) != entry["shape"]
                or not np.isfinite(embedding).all() or not mask.all()):
            raise ValueError(f"Invalid unpadded text embedding: {key}")
        if self.conditioning_contract is not None and embedding.shape[-1] != self.conditioning_contract["hidden_size"]:
            raise ValueError(f"Invalid vLLM text hidden size: {key}")
        if self.conditioning_contract is not None and not np.any(embedding):
            raise ValueError(f"Invalid vLLM all-zero features cannot be used for training: {key}")
        return embedding, mask

    def load(self, record: PromptRecord) -> dict:
        positive, positive_mask = self.load_text(record.prompt)
        negative, negative_mask = self.load_text(record.negative_prompt)
        return {"positive": positive, "negative": negative,
                "positive_mask": positive_mask, "negative_mask": negative_mask}

    def validate_prompts(self, records):
        for record in records:
            self.load(record)
        return len(records)
