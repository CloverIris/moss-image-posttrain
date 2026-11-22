"""Explicitly built, bounded-workspace cuDNN JAX FFI for sliced MOSS VAE convolution.

NCHW/OIHW 1x1/3x3, batch one, SAME stride/dilation one, FP32 or FP16 operands,
FP32 compute. No bias/residual/autodiff/fallback. Import never builds.
Run python scripts/build_cudnn_conv.py explicitly. Plans are prepared by the
host before FFI execution, including Python JIT tracing. Execute never builds
plans/handles or calls cudaMalloc. Multi-device controllers select the target with jax.default_device.
"""
from __future__ import annotations
import ctypes,hashlib,json,os,threading
from pathlib import Path
from numbers import Integral
from contextlib import contextmanager
from contextvars import ContextVar
_MESH_CONV=ContextVar("moss_mesh_conv",default=None)

@contextmanager
def mesh_convolution(devices,axis_name):
 token=_MESH_CONV.set((tuple(devices),axis_name))
 try:yield
 finally:_MESH_CONV.reset(token)

_ROOT=Path(__file__).resolve().parents[1]
_SOURCE=Path(__file__).with_name("native")/"cudnn_conv_ffi.cc"
_LOCK=threading.RLock()
_LIBRARY=None
_BUILD=None
_TARGETS={}
_PREPARED={}
def _validate(input_shape,weight_shape,dtype,workspace_mib):
 import numpy as np
 xs,ws=tuple(input_shape),tuple(weight_shape)
 if len(xs)!=4 or len(ws)!=4 or any(not isinstance(n,Integral) or n<1 for n in xs+ws):
  raise ValueError("cuDNN bounded convolution requires positive rank-four shapes")
 if xs[0]!=1:raise ValueError("cuDNN bounded convolution requires batch=1; enable VAE slicing")
 if ws[1]!=xs[1] or ws[2]!=ws[3] or ws[2] not in (1,3):
  raise ValueError("cuDNN bounded convolution requires matching channels and 1x1/3x3 OIHW weights")
 dt=np.dtype(dtype)
 if dt not in (np.dtype("float32"),np.dtype("float16")):raise ValueError("cuDNN bounded convolution supports float32 or float16 operands")
 if isinstance(workspace_mib,bool) or not isinstance(workspace_mib,Integral) or not 0<=workspace_mib<=512:
  raise ValueError("workspace_mib must be an integer in [0, 512]")
 return xs,ws,dt,int(workspace_mib)
def _load():
 global _LIBRARY,_BUILD
 if _LIBRARY is not None:return _LIBRARY
 location=Path(os.environ.get("MOSS_CUDNN_CONV_BUILD",str(_ROOT/"outputs/native/build.json")))
 if not location.exists():raise RuntimeError("Bounded cuDNN library is not built; run python scripts/build_cudnn_conv.py explicitly")
 build=json.loads(location.read_text())
 if build.get("schema")!=1:raise RuntimeError("Unsupported bounded cuDNN build manifest")
 if hashlib.sha256(_SOURCE.read_bytes()).hexdigest()!=build["source_sha256"]:
  raise RuntimeError("Bounded cuDNN source changed; explicitly rebuild the native library")
 binary=Path(build["binary"])
 if hashlib.sha256(binary.read_bytes()).hexdigest()!=build["binary_sha256"]:raise RuntimeError("Bounded cuDNN binary does not match build manifest")
 lib=ctypes.CDLL(str(binary));lib.MossBuildPlans.argtypes=[ctypes.c_int64]*5+[ctypes.c_int,ctypes.c_int64];lib.MossBuildPlans.restype=ctypes.c_char_p
 lib.MossCudnnVersion.argtypes=[];lib.MossCudnnVersion.restype=ctypes.c_size_t
 _LIBRARY,_BUILD=lib,build
 return lib
def prepare_conv(input_shape,weight_shape,dtype,*,workspace_mib=256):
 """Prebuild first precision-safe heuristic plan; NCHW/OIHW shapes; JSON metadata."""
 import jax
 xs,ws,dt,cap=_validate(input_shape,weight_shape,dtype,workspace_mib)
 try:devices=jax.local_devices(backend="gpu")
 except RuntimeError as exc:raise ValueError("cuDNN bounded convolution requires a CUDA GPU") from exc
 device=jax.config.jax_default_device or devices[0]
 if device not in devices:raise ValueError("cuDNN bounded convolution requires a GPU default device")
 cachekey=(str(device),xs,ws,dt.str,cap)
 with _LOCK:
  if cachekey in _PREPARED:return dict(_PREPARED[cachekey]["metadata"])
  lib=_load();n,c,h,w=xs;k,_,r,_=ws
  # cuDNN plan preparation uses the CUDA thread-local device, not a JAX array.
  # Scope it explicitly; execution still validates the stream's device in C++.
  lib.cudaGetDevice.argtypes=[ctypes.POINTER(ctypes.c_int)];lib.cudaGetDevice.restype=ctypes.c_int
  lib.cudaSetDevice.argtypes=[ctypes.c_int];lib.cudaSetDevice.restype=ctypes.c_int
  previous=ctypes.c_int()
  if lib.cudaGetDevice(ctypes.byref(previous))!=0:raise RuntimeError("cudaGetDevice failed")
  ordinal=int(device.local_hardware_id)
  if lib.cudaSetDevice(ordinal)!=0:raise RuntimeError("cudaSetDevice failed")
  try:manifest=json.loads(lib.MossBuildPlans(h,w,c,k,r,0 if dt.itemsize==4 else 1,cap*1024**2))
  finally:
   if lib.cudaSetDevice(previous.value)!=0:raise RuntimeError("Restoring CUDA device failed")
  if "error" in manifest:raise RuntimeError("cuDNN plan preparation failed: "+manifest["error"])
  ready=[p for p in manifest["candidates"] if p["status"]=="ready"]
  if not ready:raise RuntimeError("No precision-safe cuDNN plan satisfies workspace cap")
  plan=ready[0];fingerprint=hashlib.sha256((_BUILD["binary_sha256"]+plan["key"]).encode()).hexdigest()[:32]
  target="moss_cudnn_bounded_schema1_"+fingerprint
  if target not in _TARGETS:
   jax.ffi.register_ffi_target(target,jax.ffi.pycapsule(lib.MossCudnnConv),platform="CUDA");_TARGETS[target]=plan["key"]
  import numpy as np
  call=jax.ffi.ffi_call(target,(jax.ShapeDtypeStruct((n,h,w,k),dt),jax.ShapeDtypeStruct((max(1,plan["workspace_bytes"]),),np.uint8)),input_layouts=((0,1,2,3),(0,1,2,3)),output_layouts=((0,1,2,3),(0,)),vmap_method="sequential")
  metadata={"input_shape":list(xs),"weight_shape":list(ws),"dtype":str(dt),"workspace_cap_bytes":cap*1024**2,"compute":"float32","selection":"first precision-safe heuristic plan, no runtime autotune","ffi_target":target,"selected_plan":plan,"all_candidates":manifest["candidates"],"device_fingerprint":manifest["fingerprint"],"binary_sha256":_BUILD["binary_sha256"]}
  _PREPARED[cachekey]={"call":call,"metadata":metadata,"plan_key":plan["key"]}
  return dict(metadata)
def conv2d_nchw(x,weight,*,workspace_mib=256):
 """Same-dtype NCHW/OIHW SAME convolution; explicit failure for unsupported inputs."""
 if x.dtype!=weight.dtype:raise ValueError("cuDNN bounded convolution requires equal input/weight dtypes")
 import jax
 def prepare(device=None):
  if device is None:meta=prepare_conv(x.shape,weight.shape,x.dtype,workspace_mib=workspace_mib)
  else:
   with jax.default_device(device):meta=prepare_conv(x.shape,weight.shape,x.dtype,workspace_mib=workspace_mib)
  with _LOCK:return next(v for v in _PREPARED.values() if v["metadata"]["ffi_target"]==meta["ffi_target"])
 def execute(entry):
  y,_scratch=entry["call"](x.transpose(0,2,3,1),weight.transpose(0,2,3,1),plan_key=entry["plan_key"])
  return y.transpose(0,3,1,2)
 mesh=_MESH_CONV.get()
 if mesh is None:return execute(prepare())
 devices,axis=mesh
 entries=[prepare(device) for device in devices]
 branches=[lambda entry=entry:execute(entry) for entry in entries]
 # One SPMD graph; select the already prepared plan belonging to this shard.
 return jax.lax.switch(jax.lax.axis_index(axis),branches)

def backend_info():
 """Identity only: does not implicitly build or load a binary."""
 with _LOCK:
  return {"backend":"cudnn_bounded","schema":1,"loaded":_LIBRARY is not None,"source":str(_SOURCE),"source_sha256":hashlib.sha256(_SOURCE.read_bytes()).hexdigest(),"build":_BUILD,"cudnn_version":int(_LIBRARY.MossCudnnVersion()) if _LIBRARY is not None else None,"plan_count":len(_PREPARED),"workspace_policy":"explicit XLA U8 output; per-plan cap <=512MiB; zero-workspace plans use a 1-byte unused scratch sentinel","precision_policy":"FP32 compute; reject DOWN_CONVERT_INPUTS/REDUCED_PRECISION_REDUCTION; FP32 also rejects TENSOR_CORE to exclude unmarked TF32","device_policy":"per-device plans selected by jax.default_device","command_buffer_compatible":False}
def plan_inventory():
 with _LOCK:return [dict(v["metadata"]) for v in _PREPARED.values()]
