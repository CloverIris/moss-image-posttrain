"""Atomic, pickle-free adapter/optimizer checkpoints (NPZ plus JSON).

Only the coordinator writes checkpoints. Pass adapter parameters, never the
frozen base model. Existing checkpoint directories are immutable: use a fresh
step directory. Save only at completed update boundaries, after worker results
have been synchronized. Metadata should carry config_fingerprint, step,
round_index, prompt_cursor, RNG state, model identity and LoRA specification.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import zipfile
import jax
import numpy as np
import optax

FORMAT_VERSION = 1


class CheckpointError(ValueError):
    """An incomplete, corrupt, unsupported or incompatible checkpoint."""


def _json_value(value):
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            raise TypeError("metadata object keys must be strings")
        return {k: _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (np.ndarray, jax.Array)):
        return np.asarray(value).tolist()
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"metadata value {type(value).__name__} is not JSON serializable")


def _canonical(value):
    return json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def config_fingerprint(config):
    """Stable SHA256 of JSON configuration; caller chooses runtime exclusions."""
    return hashlib.sha256(_canonical(config)).hexdigest()


def _hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _registry():
    # No module path from an untrusted checkpoint is dynamically imported.
    result = {}
    for name in dir(optax):
        cls = getattr(optax, name)
        if isinstance(cls, type) and issubclass(cls, tuple) and hasattr(cls, "_fields"):
            result[cls.__module__+"."+cls.__qualname__] = cls
    return result


def _encode(value, arrays):
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            raise TypeError("checkpoint pytree dictionary keys must be strings")
        return {"kind": "dict", "items": [[k, _encode(v, arrays)] for k, v in value.items()]}
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        name = type(value).__module__+"."+type(value).__qualname__
        if name not in _registry():
            raise TypeError(f"unsupported namedtuple {name}; flatten into a dict first")
        return {"kind": "namedtuple", "class": name, "items": [_encode(v, arrays) for v in value]}
    if isinstance(value, (tuple, list)):
        return {"kind": "tuple" if isinstance(value, tuple) else "list",
                "items": [_encode(v, arrays) for v in value]}
    if value is None or isinstance(value, (str, bool, int, float)):
        return {"kind": "scalar", "value": _json_value(value)}
    key_impl = None
    if isinstance(value, jax.Array) and jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key):
        key_impl = str(jax.random.key_impl(value))
        value = jax.random.key_data(value)
    on_device=isinstance(value,jax.Array)
    array=value if on_device else np.asarray(value)
    if array.dtype.hasobject or array.dtype.kind in "OUS":
        raise TypeError("only numeric/boolean checkpoint arrays are supported")
    if array.dtype.kind in "fc" or str(array.dtype)=="bfloat16":
        finite=(bool(np.asarray(jax.numpy.all(jax.numpy.isfinite(array)))) if on_device else np.isfinite(array).all())
        if not finite:raise ValueError("checkpoint arrays must be finite")
    key = f"array_{len(arrays):06d}"
    # bfloat16 NPZ is represented as raw void bytes; preserve logical dtype.
    dtype = str(array.dtype)
    if on_device:
        # Keep device references; the ZIP writer stages only one temporary copy.
        # Fetching the master directly would cache a persistent NumPy copy in JAX.
        arrays[key]=array
    else:
        host=np.asarray(array)
        stored=host.view(np.uint16) if dtype=="bfloat16" else host
        arrays[key]=np.ascontiguousarray(stored).reshape(array.shape)
    spec = {"kind": "array", "key": key, "dtype": dtype, "shape": list(array.shape)}
    if key_impl is not None:
        spec["prng_impl"] = key_impl
    return spec


def _decode(spec, arrays, used):
    if not isinstance(spec, dict):
        raise CheckpointError("invalid pytree descriptor")
    kind = spec.get("kind")
    if kind == "scalar":
        value = spec["value"]
        if value is not None and not isinstance(value, (str, bool, int, float)):
            raise CheckpointError("invalid scalar descriptor")
        return value
    if kind == "dict":
        pairs = spec["items"]
        if any(not isinstance(p, list) or len(p) != 2 or not isinstance(p[0], str) for p in pairs):
            raise CheckpointError("invalid dictionary descriptor")
        if len({p[0] for p in pairs}) != len(pairs):
            raise CheckpointError("duplicate dictionary key")
        return {key: _decode(child, arrays, used) for key, child in pairs}
    if kind in ("tuple", "list", "namedtuple"):
        children = [_decode(child, arrays, used) for child in spec["items"]]
        if kind == "tuple":
            return tuple(children)
        if kind == "list":
            return children
        cls = _registry().get(spec.get("class"))
        if cls is None or len(children) != len(cls._fields):
            raise CheckpointError("unknown optimizer state namedtuple")
        return cls(*children)
    if kind == "array":
        key = spec["key"]
        if key not in arrays or key in used:
            raise CheckpointError("missing or duplicate checkpoint array")
        used.add(key)
        value = arrays[key]
        if list(value.shape) != spec["shape"]:
            raise CheckpointError(f"array shape mismatch: {key}")
        if spec["dtype"] == "bfloat16":
            import ml_dtypes
            if value.dtype != np.uint16:
                raise CheckpointError("invalid bfloat16 representation")
            value = value.view(ml_dtypes.bfloat16)
        elif str(value.dtype) != spec["dtype"]:
            raise CheckpointError(f"array dtype mismatch: {key}")
        if value.dtype.hasobject or value.dtype.kind in "OUS":
            raise CheckpointError("unsupported array dtype")
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise CheckpointError("nonfinite checkpoint array")
        if "prng_impl" in spec:
            value = jax.random.wrap_key_data(value, impl=spec["prng_impl"])
        return value
    raise CheckpointError(f"unknown pytree descriptor kind: {kind}")


def _fsync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_npz(file,arrays):
    """Standard NPZ, bounded CPU staging and no host cache on master JAX arrays."""
    with zipfile.ZipFile(file,mode="w",compression=zipfile.ZIP_STORED,allowZip64=True) as archive:
        for key,value in arrays.items():
            temporary=value.copy() if isinstance(value,jax.Array) else None
            host=np.asarray(temporary if temporary is not None else value)
            if str(host.dtype)=="bfloat16":host=host.view(np.uint16)
            with archive.open(key+".npy",mode="w",force_zip64=True) as member:
                np.lib.format.write_array(member,host,allow_pickle=False)
            del host,temporary


def save_checkpoint(path, lora, optimizer_state, metadata, *, reference_state=None):
    """Atomically create an immutable checkpoint; return resolved directory.

    Arbitrary dict/list/tuple numeric pytrees and public Optax state namedtuples
    are supported. The JSON commit marker and payload hash detect partial or
    corrupt saves. Do not have multiple workers save the same checkpoint.
    """
    destination = Path(path).expanduser().absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"checkpoint already exists: {destination}")
    metadata = _json_value(metadata)
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be a JSON object")
    arrays = {}
    lora_spec = _encode(lora, arrays)
    optimizer_spec = _encode(optimizer_state, arrays)
    reference_spec = None
    if reference_state is not None:
        from .reference import validate_reference_state
        validate_reference_state({"art": {"reference_mode": "ema",
            "reference_ema_decay": reference_state.get("decay") if isinstance(reference_state, dict) else None}},
            reference_state, lora)
        reference_spec = _encode(reference_state, arrays)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))
    try:
        payload = temporary/"arrays.npz"
        with open(payload, "wb") as f:
            _write_npz(f,arrays)
            f.flush()
            os.fsync(f.fileno())
        manifest = dict(format_version=FORMAT_VERSION, metadata=metadata,
                        lora=lora_spec, optimizer_state=optimizer_spec,
                        array_count=len(arrays), payload_sha256=_hash(payload))
        if reference_spec is not None:
            manifest["reference_state"] = reference_spec
        manifest_bytes = _canonical(manifest)
        with open(temporary/"manifest.json", "wb") as f:
            f.write(manifest_bytes)
            f.flush()
            os.fsync(f.fileno())
        with open(temporary/"COMMITTED", "w", encoding="ascii") as f:
            f.write(hashlib.sha256(manifest_bytes).hexdigest()+"\n")
            f.flush()
            os.fsync(f.fileno())
        _fsync_dir(temporary)
        # Empty directories are replaceable on POSIX; do not overwrite even one.
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"checkpoint already exists: {destination}")
        os.rename(temporary, destination)
        _fsync_dir(destination.parent)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return destination


def _validate_expected(actual, expected, prefix="metadata"):
    for key, wanted in expected.items():
        if key not in actual:
            raise CheckpointError(f"missing resume field: {prefix}.{key}")
        if isinstance(wanted, dict):
            if not isinstance(actual[key], dict):
                raise CheckpointError(f"incompatible resume field: {prefix}.{key}")
            _validate_expected(actual[key], wanted, prefix+"."+key)
        elif actual[key] != _json_value(wanted):
            raise CheckpointError(f"incompatible resume field: {prefix}.{key}")


def load_checkpoint(path, *, expected_metadata=None, include_reference_state=False):
    """Validate fully before returning (numpy_lora, optimizer_state, metadata).

    include_reference_state=True appends the saved EMA state (or None for an
    old/base checkpoint). Reference arrays are validated even for legacy readers.

    expected_metadata is a recursive subset that must match, e.g. model identity,
    config_fingerprint and LoRA rank/targets. It is the caller's responsibility
    to include every configuration field whose change makes resume invalid.
    """
    directory = Path(path).expanduser().absolute()
    try:
        if not directory.is_dir() or directory.is_symlink():
            raise CheckpointError("checkpoint directory missing or symlinked")
        for name in ("arrays.npz", "manifest.json", "COMMITTED"):
            if not (directory/name).is_file() or (directory/name).is_symlink():
                raise CheckpointError(f"missing or symlinked checkpoint file: {name}")
        content = (directory/"manifest.json").read_bytes()
        if len(content) > 64*1024*1024:
            raise CheckpointError("checkpoint manifest is unreasonably large")
        if (directory/"COMMITTED").read_text("ascii").strip() != hashlib.sha256(content).hexdigest():
            raise CheckpointError("checkpoint manifest checksum mismatch")
        manifest = json.loads(content)
        if manifest.get("format_version") != FORMAT_VERSION:
            raise CheckpointError("unsupported checkpoint format version")
        if _hash(directory/"arrays.npz") != manifest.get("payload_sha256"):
            raise CheckpointError("checkpoint payload checksum mismatch")
        metadata = manifest["metadata"]
        if not isinstance(metadata, dict):
            raise CheckpointError("invalid checkpoint metadata")
        if expected_metadata is not None:
            _validate_expected(metadata, _json_value(expected_metadata))
        with np.load(directory/"arrays.npz", allow_pickle=False) as payload:
            if len(payload.files) != manifest["array_count"]:
                raise CheckpointError("checkpoint array count mismatch")
            arrays = {key: payload[key] for key in payload.files}
        used = set()
        lora = _decode(manifest["lora"], arrays, used)
        optimizer = _decode(manifest["optimizer_state"], arrays, used)
        reference = None
        if "reference_state" in manifest:
            reference = _decode(manifest["reference_state"], arrays, used)
            if not isinstance(reference, dict):
                raise CheckpointError("invalid reference state")
            from .reference import validate_reference_state
            validate_reference_state({"art": {"reference_mode": "ema",
                "reference_ema_decay": reference.get("decay")}}, reference, lora)
        if used != set(arrays):
            raise CheckpointError("unreferenced checkpoint arrays")
        return (lora, optimizer, metadata, reference) if include_reference_state else (lora, optimizer, metadata)
    except CheckpointError:
        raise
    except (OSError, ValueError, TypeError, KeyError, EOFError) as error:
        raise CheckpointError(f"cannot load checkpoint: {error}") from error
