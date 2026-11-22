"""Immutable offline MXFP8 weights; reuses the online quantizer unchanged.

No JAX import at module load. NPY files store FP8/BF16 as raw unsigned bits;
logical dtypes, shapes, source provenance and checksums are committed in JSON.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np


def _hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8*1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _source(model_dir):
    from .assets import component_identity
    return component_identity(model_dir, "transformer")


def _quantizer_identity():
    from .mxfp8 import quantization_contract
    return dict(contract=quantization_contract(), source_sha256=_hash(Path(__file__).with_name("mxfp8.py")),
                input_cast="all source parameters BF16 except rope.bands")


def _write_array(directory, key, value):
    value = np.asarray(value)
    dtype = str(value.dtype)
    storage = (value.view(np.uint8) if dtype.startswith("float8_") else
               value.view(np.uint16) if dtype == "bfloat16" else value)
    filename = hashlib.sha256(key.encode()).hexdigest()+".npy"
    path = directory/filename
    with path.open("wb") as stream:
        np.save(stream, storage, allow_pickle=False)
        stream.flush(); os.fsync(stream.fileno())
    return dict(file=filename, dtype=dtype, shape=list(value.shape),
                size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns, sha256=_hash(path))


def write_cache(directory, params, *, source_identity, include_transpose=False):
    """Quantize one tensor at a time on the selected JAX device, then commit."""
    import jax
    import ml_dtypes
    from .mxfp8 import eligible, quantize, SCALE_SUFFIX, TRANSPOSE_SUFFIX, TRANSPOSE_SCALE_SUFFIX
    target = Path(directory).expanduser().absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise FileExistsError("MXFP8 cache already exists: %s" % target)
    temporary = Path(tempfile.mkdtemp(prefix=".mxfp8-", dir=target.parent))
    entries, original, converted = {}, {}, 0
    started = time.monotonic()
    try:
        for index, (key, value) in enumerate(params.items()):
            host = np.asarray(value) if key == "rope.bands" else np.asarray(value, dtype=ml_dtypes.bfloat16)
            if not np.isfinite(host).all(): raise ValueError("Nonfinite source parameter: %s" % key)
            original[key] = dict(shape=list(host.shape), dtype=str(host.dtype))
            if eligible(key, host):
                dense = jax.device_put(host)
                q, scale = quantize(dense)
                jax.block_until_ready((q, scale))
                prefix = key[:-7]
                entries[key] = _write_array(temporary, key, np.asarray(q))
                entries[prefix+SCALE_SUFFIX] = _write_array(temporary, prefix+SCALE_SUFFIX, np.asarray(scale))
                if include_transpose:
                    qt, st = quantize(dense.T)
                    jax.block_until_ready((qt, st))
                    entries[prefix+TRANSPOSE_SUFFIX] = _write_array(temporary, prefix+TRANSPOSE_SUFFIX, np.asarray(qt))
                    entries[prefix+TRANSPOSE_SCALE_SUFFIX] = _write_array(temporary, prefix+TRANSPOSE_SCALE_SUFFIX, np.asarray(st))
                    del qt, st
                del dense, q, scale
                converted += 1
            else:
                entries[key] = _write_array(temporary, key, host)
            if (index+1) % 100 == 0:
                print(json.dumps(dict(event="mxfp8_export_progress", tensors=index+1, quantized=converted)), flush=True)
        manifest = dict(format="moss-art-mxfp8-v1", source=source_identity, quantizer=_quantizer_identity(),
                        include_transpose=bool(include_transpose), source_tensors=original,
                        entries=entries, quantized_modules=converted)
        content = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        with (temporary/"manifest.json").open("wb") as stream:
            stream.write(content); stream.flush(); os.fsync(stream.fileno())
        with (temporary/"COMMITTED").open("w") as stream:
            stream.write(hashlib.sha256(content).hexdigest()+"\n"); stream.flush(); os.fsync(stream.fileno())
        from .checkpoint import _fsync_dir
        _fsync_dir(temporary)
        if target.exists() or target.is_symlink(): raise FileExistsError(target)
        os.rename(temporary, target); _fsync_dir(target.parent)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return dict(cache_dir=str(target), seconds=time.monotonic()-started, quantized_modules=converted,
                cache_bytes=sum(p.stat().st_size for p in target.iterdir() if p.is_file()))


def load_cache(directory, *, source_identity, verify=True, require_transpose=False):
    """Mmap checked quantized tensors; never quantizes or dequantizes weights."""
    import ml_dtypes
    from .mxfp8 import eligible, BLOCK, SCALE_SUFFIX, TRANSPOSE_SUFFIX, TRANSPOSE_SCALE_SUFFIX
    directory = Path(directory).expanduser().absolute()
    if not directory.is_dir() or directory.is_symlink(): raise ValueError("Missing MXFP8 cache directory")
    content = (directory/"manifest.json").read_bytes()
    if (directory/"COMMITTED").read_text().strip() != hashlib.sha256(content).hexdigest():
        raise ValueError("MXFP8 manifest checksum mismatch")
    manifest = json.loads(content)
    if manifest.get("format") != "moss-art-mxfp8-v1": raise ValueError("Unsupported MXFP8 cache format")
    if manifest["source"] != source_identity: raise ValueError("MXFP8 cache source model differs")
    if manifest["quantizer"] != _quantizer_identity(): raise ValueError("MXFP8 quantizer contract/source differs")
    if require_transpose and not manifest["include_transpose"]:
        raise ValueError("MXFP8 dgrad requires a cache exported with include_transpose=true")
    arrays = {}
    logical = {"bfloat16": ml_dtypes.bfloat16, "float8_e4m3fn": ml_dtypes.float8_e4m3fn,
               "float8_e8m0fnu": ml_dtypes.float8_e8m0fnu}
    for key, spec in manifest["entries"].items():
        if spec["file"] != hashlib.sha256(key.encode()).hexdigest()+".npy":
            raise ValueError("Invalid MXFP8 cache filename")
        if not require_transpose and key.endswith((TRANSPOSE_SUFFIX, TRANSPOSE_SCALE_SUFFIX)):
            continue
        path = directory/spec["file"]
        stat = path.stat()
        if path.is_symlink() or stat.st_size != spec["size"] or (not verify and stat.st_mtime_ns != spec["mtime_ns"]):
            raise ValueError("MXFP8 cache tensor file changed")
        if verify and _hash(path) != spec["sha256"]: raise ValueError("MXFP8 tensor checksum mismatch")
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        dtype = spec["dtype"]
        if dtype in logical:
            expected = np.uint16 if dtype == "bfloat16" else np.uint8
            if value.dtype != expected: raise ValueError("Invalid MXFP8 raw storage dtype")
            value = value.view(logical[dtype])
        if value.dtype.hasobject or str(value.dtype) != dtype or list(value.shape) != spec["shape"]:
            raise ValueError("MXFP8 tensor dtype/shape differs")
        arrays[key] = value
    expected_keys = set()
    converted = 0
    for key, original in manifest["source_tensors"].items():
        expected_keys.add(key)
        if key not in arrays or list(arrays[key].shape) != original["shape"]:
            raise ValueError("MXFP8 cache is missing a source tensor/shape")
        if eligible(key, arrays[key]):
            converted += 1; prefix = key[:-7]
            expected_keys.add(prefix+SCALE_SUFFIX)
            if (str(arrays[key].dtype) != "float8_e4m3fn" or prefix+SCALE_SUFFIX not in arrays or
                    arrays[prefix+SCALE_SUFFIX].shape != (arrays[key].shape[0], arrays[key].shape[1]//BLOCK) or
                    str(arrays[prefix+SCALE_SUFFIX].dtype) != "float8_e8m0fnu"):
                raise ValueError("Invalid quantized MXFP8 weight/scale")
            if require_transpose:
                expected_keys.update((prefix+TRANSPOSE_SUFFIX, prefix+TRANSPOSE_SCALE_SUFFIX))
                if (arrays[prefix+TRANSPOSE_SUFFIX].shape != arrays[key].T.shape or
                        arrays[prefix+TRANSPOSE_SCALE_SUFFIX].shape != (arrays[key].shape[1], arrays[key].shape[0]//BLOCK) or
                        str(arrays[prefix+TRANSPOSE_SUFFIX].dtype) != "float8_e4m3fn" or
                        str(arrays[prefix+TRANSPOSE_SCALE_SUFFIX].dtype) != "float8_e8m0fnu"):
                    raise ValueError("Invalid transposed MXFP8 weight/scale")
        elif str(arrays[key].dtype) != original["dtype"]:
            raise ValueError("Nonquantized source dtype differs")
    if set(arrays) != expected_keys or converted != manifest["quantized_modules"]:
        raise ValueError("Unexpected MXFP8 cache tensors/count")
    return arrays


def load_pretrained_cache(model_dir, directory, *, verify=True, require_transpose=False):
    return load_cache(directory, source_identity=_source(model_dir), verify=verify, require_transpose=require_transpose)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    c = json.loads(Path(parser.parse_args(argv).config).read_text())
    if not isinstance(c,dict) or any(not isinstance(c.get(k),str) or not c[k].strip() or "\x00" in c[k]
                                     for k in ("model_dir","output_dir")):
        raise ValueError("model_dir and output_dir must be nonempty paths")
    if set(c)-{"model_dir", "output_dir", "platform", "device", "include_transpose"}:
        raise ValueError("Unknown MXFP8 export configuration field")
    if c.get("platform", "cpu") not in ("cpu", "cuda"): raise ValueError("platform must be cpu or cuda")
    if type(c.get("device", 0)) is not int or c.get("device", 0) < 0: raise ValueError("Invalid export device")
    if type(c.get("include_transpose", False)) is not bool: raise ValueError("include_transpose must be boolean")
    from .config import DEFAULTS
    from .runtime import configure_process, preflight
    import copy
    runtime = copy.deepcopy(DEFAULTS)
    runtime["runtime"].update(platform=c.get("platform", "cpu"), devices=[c.get("device", 0)])
    preflight(runtime); configure_process(runtime, runtime["runtime"]["devices"])
    from vendor.moss_jax.weights import load_host_params
    source = _source(c["model_dir"])
    if Path(c["output_dir"]).exists():
        load_cache(c["output_dir"], source_identity=source, require_transpose=c.get("include_transpose", False))
        print(json.dumps(dict(event="mxfp8_cache_ready", reused=True, cache_dir=c["output_dir"])), flush=True)
        return
    result = write_cache(c["output_dir"], load_host_params(c["model_dir"], "transformer"),
                         source_identity=source, include_transpose=c.get("include_transpose", False))
    print(json.dumps(dict(event="mxfp8_cache_ready", reused=False, **result)), flush=True)


if __name__ == "__main__": main()
