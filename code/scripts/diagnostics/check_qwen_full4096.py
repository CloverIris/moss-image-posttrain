import os
os.environ["CUDA_VISIBLE_DEVICES"]="GPU-2fbea160-cfc0-82dc-6002-d5ef4c5bf8f4"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
import sys
sys.path.insert(0,"$CODE_ROOT/moss-art-jax")
from moss_art.qwen_prefill import JaxPrefill
from vendor.moss_jax.text_encoder import encode_prompt_tokens,decoder_layer
import time,json,numpy as np
from pathlib import Path
m="$MODEL_ROOT/MOSS-Image-SFT"
e=JaxPrefill(m);results=[]
for prompt,capacity in [("A red cat.",320),("A blue cat on a roof.",320),("A detailed landscape. "*20,4096),("A mountain.",4096)]:
 ids,_=encode_prompt_tokens(m,prompt,None);t=time.monotonic()
 out=e.encode(ids,capacity)
 results.append(dict(tokens=ids.shape[1],capacity=capacity,seconds=time.monotonic()-t,finite=bool(np.isfinite(out).all()),compiled_layer_shapes=decoder_layer._cache_size(),peak_bytes=e.device.memory_stats().get("peak_bytes_in_use")))
 print(json.dumps(results[-1]),flush=True)
Path("reports/qwen_full4096_check.json").write_text(json.dumps(results,indent=2))
