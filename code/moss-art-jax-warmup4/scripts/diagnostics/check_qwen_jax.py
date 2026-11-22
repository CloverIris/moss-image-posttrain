import os
os.environ["CUDA_VISIBLE_DEVICES"]="GPU-2fbea160-cfc0-82dc-6002-d5ef4c5bf8f4"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
import sys
sys.path.insert(0,"$CODE_ROOT/moss-art-jax")
import numpy as np,json,time
from pathlib import Path
import moss_art.qwen_prefill as q
root=Path("$CODE_ROOT/moss-art-jax")
r=np.load(root/"outputs/qwen_reference_fp32_1.npz")
results={}
def stats(name,a,b):
 x=a.astype(float).ravel();y=b.astype(float).ravel()
 results[name]=dict(cosine=float(x@y/np.linalg.norm(x)/np.linalg.norm(y)),relative_l2=float(np.linalg.norm(x-y)/np.linalg.norm(x)),max_abs=float(abs(x-y).max()))
 print(name,results[name],flush=True)
encoder=q.JaxPrefill("$MODEL_ROOT/MOSS-Image-SFT")
original=q.vision_block;counter=0
def capture(*args,**kw):
 global counter
 x=original(*args,**kw)
 if counter in (0,1,26):stats(f"vision_block_{counter}",r[f"vision_block_{counter}"],np.asarray(x))
 counter+=1
 return x
q.vision_block=capture
start=time.monotonic()
vis=encoder.vision(r["pixel_values"],r["image_grid_thw"][0]);stats("vision",r["vision"],vis)
np.save(root/"outputs/qwen_jax_vision.npy",vis.astype(np.float32))
encoder.vision=lambda *a:vis
out=encoder.encode(r["input_ids"],320,r["pixel_values"],r["image_grid_thw"]);stats("full",r["context"],out)
encoder.vision=lambda *a:r["vision"].astype(vis.dtype)
out=encoder.encode(r["input_ids"],320,r["pixel_values"],r["image_grid_thw"]);stats("torch_vision_jax_language",r["context"],out)
print('seconds',time.monotonic()-start,flush=True)
(root/"reports/qwen_numerical_diagnostic.json").write_text(json.dumps(results,indent=2))
