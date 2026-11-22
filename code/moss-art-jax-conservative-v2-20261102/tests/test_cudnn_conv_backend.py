"""CPU-only guards for the explicitly built bounded cuDNN bridge."""
import hashlib,json
from types import SimpleNamespace
import numpy as np
import pytest
from moss_art import cudnn_conv as backend

@pytest.mark.parametrize("xs,ws,dtype,cap,match",[
 ((2,16,8,8),(24,16,3,3),np.float32,256,"batch=1"),
 ((1,16,8,8),(24,15,3,3),np.float32,256,"matching channels"),
 ((1,16,8,8),(24,16,5,5),np.float32,256,"1x1/3x3"),
 ((1,16,8),(24,16,3,3),np.float32,256,"rank-four"),
 ((1,16,0,8),(24,16,3,3),np.float32,256,"positive"),
 ((1,16,8,8),(24,16,3,3),np.float64,256,"float32 or float16"),
 ((1,16,8,8),(24,16,3,3),np.float32,-1,"[0, 512]"),
 ((1,16,8,8),(24,16,3,3),np.float32,513,"[0, 512]"),
 ((1,16,8,8),(24,16,3,3),np.float32,True,"integer"),
])
def test_invalid_inputs_rejected_before_backend(xs,ws,dtype,cap,match):
 with pytest.raises(ValueError,match=__import__("re").escape(match)):
  backend._validate(xs,ws,dtype,cap)

def test_zero_workspace_is_valid():
 assert backend._validate((1,16,8,8),(24,16,3,3),np.float16,0)[-1]==0

def test_mixed_operand_dtype_fails_without_loading(monkeypatch):
 monkeypatch.setattr(backend,"prepare_conv",lambda *a,**k:pytest.fail("must reject before plan preparation"))
 with pytest.raises(ValueError,match="equal"):
  backend.conv2d_nchw(np.zeros((1,4,3,3),np.float16),np.zeros((4,4,1,1),np.float32))

def test_missing_build_never_compiles(monkeypatch,tmp_path):
 monkeypatch.setattr(backend,"_LIBRARY",None)
 monkeypatch.setenv("MOSS_CUDNN_CONV_BUILD",str(tmp_path/"absent.json"))
 with pytest.raises(RuntimeError,match="explicitly"):
  backend._load()
 assert not list(tmp_path.iterdir())

def test_stale_source_rejected(monkeypatch,tmp_path):
 monkeypatch.setattr(backend,"_LIBRARY",None)
 manifest=tmp_path/"build.json";manifest.write_text(json.dumps({"schema":1,"source_sha256":"bad"}))
 monkeypatch.setenv("MOSS_CUDNN_CONV_BUILD",str(manifest))
 with pytest.raises(RuntimeError,match="source changed"):backend._load()

def test_tampered_binary_rejected(monkeypatch,tmp_path):
 monkeypatch.setattr(backend,"_LIBRARY",None)
 binary=tmp_path/"invalid.so";binary.write_bytes(b"invalid")
 manifest=tmp_path/"build.json";manifest.write_text(json.dumps({"schema":1,
  "source_sha256":hashlib.sha256(backend._SOURCE.read_bytes()).hexdigest(),
  "binary":str(binary),"binary_sha256":"bad"}))
 monkeypatch.setenv("MOSS_CUDNN_CONV_BUILD",str(manifest))
 with pytest.raises(RuntimeError,match="binary does not match"):backend._load()

def test_cpu_rejected_before_load(monkeypatch):
 import jax
 def missing(**kwargs):raise RuntimeError("no gpu")
 monkeypatch.setattr(jax,"local_devices",missing)
 monkeypatch.setattr(backend,"_load",lambda:pytest.fail("must fail before loading"))
 with pytest.raises(ValueError,match="CUDA GPU"):backend.prepare_conv((1,4,3,3),(4,4,1,1),np.float32)

def test_multiple_visible_devices_rejected(monkeypatch):
 import jax
 monkeypatch.setattr(jax,"local_devices",lambda **kw:[object(),object()])
 monkeypatch.setattr(backend,"_load",lambda:pytest.fail("must fail before loading"))
 with pytest.raises(ValueError,match="exactly one visible"):backend.prepare_conv((1,4,3,3),(4,4,1,1),np.float32)

def test_no_plan_never_falls_back(monkeypatch):
 import jax
 monkeypatch.setattr(jax,"local_devices",lambda **kw:[SimpleNamespace(id=0)])
 monkeypatch.setattr(backend,"_PREPARED",{})
 monkeypatch.setattr(backend,"_load",lambda:SimpleNamespace(MossBuildPlans=lambda *a:b'{"candidates": []}'))
 with pytest.raises(RuntimeError,match="No precision-safe"):backend.prepare_conv((1,4,3,3),(4,4,1,1),np.float32)

def test_zero_workspace_registration_is_stable_and_preparation_reused(monkeypatch):
 import jax
 monkeypatch.setattr(jax,"local_devices",lambda **kw:[SimpleNamespace(id=0)])
 monkeypatch.setattr(backend,"_PREPARED",{})
 monkeypatch.setattr(backend,"_TARGETS",{})
 monkeypatch.setattr(backend,"_BUILD",{"binary_sha256":"a"*64})
 builds=[];targets=[];scratch=[]
 def build(*args):
  builds.append(args)
  return json.dumps({"fingerprint":"device-shape-key","candidates":[{"status":"ready","key":"stable_full_fingerprint","workspace_bytes":0,"engine":28}]}).encode()
 monkeypatch.setattr(backend,"_load",lambda:SimpleNamespace(MossBuildPlans=build,MossCudnnConv=object()))
 monkeypatch.setattr(jax.ffi,"pycapsule",lambda x:x)
 monkeypatch.setattr(jax.ffi,"register_ffi_target",lambda name,*a,**k:targets.append(name))
 def ffi_call(name,results,**kwargs):scratch.append(results[1].shape);return object()
 monkeypatch.setattr(jax.ffi,"ffi_call",ffi_call)
 a=backend.prepare_conv((1,4,3,3),(4,4,1,1),np.float32,workspace_mib=0)
 b=backend.prepare_conv((1,4,3,3),(4,4,1,1),np.float32,workspace_mib=0)
 assert a==b and len(builds)==len(targets)==1
 assert builds[0][-1]==0 and scratch==[(1,)]
 assert "stable_full_fingerprint" not in targets[0]
