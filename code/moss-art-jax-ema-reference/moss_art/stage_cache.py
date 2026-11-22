"""Atomic, deduplicated, pickle-free handoff between sequential RL processes."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import tempfile
import numpy as np


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8*1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_bundle(path, payload, lora, optimizer, metadata, *, attachments=None, reserve_bytes=0,
                reference_state=None):
    from .checkpoint import save_checkpoint, _fsync_dir
    destination = Path(path).absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    # Shape-only upper bound: do not copy device parameters or hash replay twice.
    seen = set()
    def size(value):
        if id(value) in seen:
            return 0
        seen.add(id(value))
        if isinstance(value, dict):
            return sum(size(v) for v in value.values())
        if isinstance(value, (tuple, list)):
            return sum(size(v) for v in value)
        return int(getattr(value, "nbytes", 0))+256
    required = size((payload, lora, optimizer, reference_state))+int(reserve_bytes)
    if shutil.disk_usage(destination.parent).free < required:
        raise RuntimeError("Stage scratch filesystem cannot fit the replay and learner snapshot")
    temporary = Path(tempfile.mkdtemp(prefix=".stage-", dir=destination.parent))
    arrays, objects, contents = {}, {}, {}
    def encode(value):
        if isinstance(value, np.generic):
            return encode(value.item())
        if isinstance(value, dict):
            if any(not isinstance(k, str) for k in value):
                raise ValueError("Stage dictionaries require string keys")
            return {"kind": "dict", "items": [[k, encode(v)] for k, v in value.items()]}
        if isinstance(value, (tuple, list)):
            return {"kind": "tuple" if isinstance(value, tuple) else "list", "items": [encode(v) for v in value]}
        if value is None or isinstance(value, (str, bool, int, float)):
            return {"kind": "scalar", "value": value}
        identity = id(value)
        if identity in objects:
            return {"kind": "array", "key": objects[identity]}
        array = np.asarray(value)
        if array.dtype.hasobject or array.dtype.kind in "OUS":
            raise ValueError("Only numeric stage arrays are supported")
        dtype = str(array.dtype)
        stored = np.ascontiguousarray(array.view(np.uint16) if dtype == "bfloat16" else array).reshape(array.shape)
        content = (dtype, stored.shape, hashlib.sha256(memoryview(stored).cast("B")).hexdigest())
        key = contents.get(content)
        if key is None:
            key = "array_%06d" % len(arrays)
            filename = key+".npy"
            with (temporary/filename).open("wb") as stream:
                np.save(stream, stored, allow_pickle=False)
                stream.flush(); os.fsync(stream.fileno())
            arrays[key] = dict(file=filename, dtype=dtype, shape=list(array.shape), sha256=_hash(temporary/filename))
            contents[content] = key
        objects[identity] = key
        return {"kind": "array", "key": key}
    try:
        tree = encode(payload)
        save_checkpoint(temporary/"learner", lora, optimizer, metadata, reference_state=reference_state)
        auxiliary = {}
        for name, value in (attachments or {}).items():
            if Path(name).name != name or not name.endswith(".json") or name in ("manifest.json", "consumed.json"):
                raise ValueError("Invalid stage attachment filename")
            with (temporary/name).open("w", encoding="utf-8") as stream:
                json.dump(value, stream, sort_keys=True, allow_nan=False)
                stream.flush(); os.fsync(stream.fileno())
            auxiliary[name] = dict(sha256=_hash(temporary/name))
        manifest = dict(format_version=1, tree=tree, arrays=arrays, metadata=metadata,
                        attachments=auxiliary)
        content = json.dumps(manifest, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
        with (temporary/"manifest.json").open("wb") as stream:
            stream.write(content); stream.flush(); os.fsync(stream.fileno())
        with (temporary/"COMMITTED").open("w") as stream:
            stream.write(hashlib.sha256(content).hexdigest()+"\n"); stream.flush(); os.fsync(stream.fileno())
        _fsync_dir(temporary)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        os.rename(temporary, destination)
        _fsync_dir(destination.parent)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return destination


def load_attachment(path, name):
    """Read the evaluator's small request without paging in any replay arrays."""
    directory = Path(path).absolute()
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Stage bundle missing or symlinked")
    content = (directory/"manifest.json").read_bytes()
    if (directory/"COMMITTED").read_text().strip() != hashlib.sha256(content).hexdigest():
        raise ValueError("Stage manifest checksum mismatch")
    manifest = json.loads(content)
    if manifest["format_version"] != 1:
        raise ValueError("Unsupported stage cache version")
    if Path(name).name != name or name not in manifest.get("attachments", {}):
        raise ValueError("Unknown stage attachment")
    file = directory/name
    if file.is_symlink() or _hash(file) != manifest["attachments"][name]["sha256"]:
        raise ValueError("Stage attachment checksum mismatch")
    return json.loads(file.read_text(encoding="utf-8"))


def load_bundle(path):
    """Verify before use, mmap each unique array once, and preserve aliases."""
    directory = Path(path).absolute()
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("Stage bundle missing or symlinked")
    content = (directory/"manifest.json").read_bytes()
    if (directory/"COMMITTED").read_text().strip() != hashlib.sha256(content).hexdigest():
        raise ValueError("Stage manifest checksum mismatch")
    manifest = json.loads(content)
    if manifest["format_version"] != 1:
        raise ValueError("Unsupported stage cache version")
    arrays = {}
    for key, spec in manifest["arrays"].items():
        if spec["file"] != key+".npy" or not key.startswith("array_") or not key[6:].isdigit():
            raise ValueError("Invalid stage array filename")
        file = directory/spec["file"]
        if file.is_symlink() or _hash(file) != spec["sha256"]:
            raise ValueError("Stage array checksum mismatch")
        value = np.load(file, mmap_mode="r", allow_pickle=False)
        if spec["dtype"] == "bfloat16":
            import ml_dtypes
            if value.dtype != np.uint16:
                raise ValueError("Invalid BF16 stage storage")
            value = value.view(ml_dtypes.bfloat16)
        if list(value.shape) != spec["shape"] or str(value.dtype) != spec["dtype"]:
            raise ValueError("Stage array shape/dtype mismatch")
        arrays[key] = value
    def decode(spec):
        kind = spec["kind"]
        if kind == "scalar": return spec["value"]
        if kind == "array": return arrays[spec["key"]]
        if kind == "dict": return {k: decode(v) for k, v in spec["items"]}
        if kind == "list": return [decode(v) for v in spec["items"]]
        if kind == "tuple": return tuple(decode(v) for v in spec["items"])
        raise ValueError("Invalid stage tree node")
    return decode(manifest["tree"]), manifest["metadata"]
