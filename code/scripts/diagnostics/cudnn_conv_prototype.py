"""Explicitly loaded diagnostic bounded-workspace cuDNN FFI wrapper; importing never builds."""
from __future__ import annotations
import ctypes
import hashlib
import json
from pathlib import Path

class CudnnPrototype:
    def __init__(self, library):
        import jax
        self.path = Path(library).resolve()
        self.binary_sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.lib = ctypes.CDLL(str(self.path))
        self.lib.MossBuildPlans.argtypes = [ctypes.c_int64]*5 + [ctypes.c_int, ctypes.c_int64]
        self.lib.MossBuildPlans.restype = ctypes.c_char_p
        self.targets = {}

    def prepare(self, h, w, channels, output_channels, kernel, dtype, workspace_mib=256):
        """Build all precision-compatible heuristic plans on initialized current GPU."""
        import numpy as np
        code = {np.dtype("float32"): 0, np.dtype("float16"): 1}[np.dtype(dtype)]
        raw = self.lib.MossBuildPlans(h, w, channels, output_channels, kernel, code, int(workspace_mib*1024**2))
        result = json.loads(raw)
        if "error" in result:
            raise RuntimeError(result["error"])
        result["binary_sha256"] = self.binary_sha256
        result["shape"] = [1, h, w, channels]
        result["weight_shape"] = [output_channels, kernel, kernel, channels]
        result["dtype"] = str(np.dtype(dtype))
        unique = {}
        for plan in result["candidates"]:
            if plan["status"] == "ready":
                unique.setdefault(plan["key"], plan)
        result["ready"] = list(unique.values())
        return result

    def operation(self, manifest, plan):
        """NHWC input / OHWI filter => NHWC output; no bias, FP32 accumulation."""
        import jax
        import numpy as np
        key = plan["key"]
        fingerprint = hashlib.sha256((self.binary_sha256 + key).encode()).hexdigest()[:32]
        target = "moss_cudnn_bounded_schema1_" + fingerprint
        if target not in self.targets:
            jax.ffi.register_ffi_target(target, jax.ffi.pycapsule(self.lib.MossCudnnConv), platform="CUDA")
            self.targets[target] = key
        n, h, w, c = manifest["shape"]
        k, r, s, wc = manifest["weight_shape"]
        dtype = np.dtype(manifest["dtype"])
        result = jax.ShapeDtypeStruct((n, h, w, k), dtype)
        scratch = jax.ShapeDtypeStruct((max(1, plan["workspace_bytes"]),), np.uint8)
        call = jax.ffi.ffi_call(
            target, (result, scratch), vmap_method="sequential",
            input_layouts=((0,1,2,3),(0,1,2,3)),
            output_layouts=((0,1,2,3),(0,)),
        )
        def operation(x, weight):
            if x.shape != (n,h,w,c) or weight.shape != (k,r,s,wc) or x.dtype != dtype or weight.dtype != dtype:
                raise ValueError("Operand shape/dtype differs from prepared manifest")
            y, _scratch = call(x, weight, plan_key=key)
            return y
        operation.ffi_target = target
        return operation
