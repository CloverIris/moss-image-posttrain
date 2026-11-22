"""Verify quantization against local Kohaku and audit GPU forward/dgrad HLO."""
import argparse,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
p=argparse.ArgumentParser();p.add_argument('--config',required=True)
settings=json.loads(Path(p.parse_args().config).read_text())
Path(settings['reference_fixture']).parent.mkdir(parents=True,exist_ok=True)
REFERENCE_CODE = "import sys,numpy as np,torch\nimport json\nc=json.loads(sys.argv[1])\nsys.path.insert(0,c['reference_project']+'/src')\nfrom kohakumoss_kernels.fp8.formats import quantize\nrng=np.random.default_rng(123);w=rng.normal(size=(256,128)).astype(np.float32)\nx=rng.normal(size=(257,128)).astype(np.float32)\nq,s=quantize(torch.from_numpy(w).to(torch.bfloat16),'mx','weight')\nqt,st=quantize(torch.from_numpy(w).to(torch.bfloat16).T,'mx','weight')\nnp.savez(c['reference_fixture'],x=x,w=w,q=q.view(torch.uint8).numpy(),s=s.numpy(),qt=qt.view(torch.uint8).numpy(),st=st.numpy())\nprint('Kohaku reference fixture written')"
subprocess.run([settings['torch_python'],'-c',REFERENCE_CODE,json.dumps(settings)],check=True)
from pathlib import Path
from moss_art.config import load_config
from moss_art.runtime import configure_process
c=load_config('configs/moss_lora_4gpu.json');c['runtime']['devices']=[settings['device']]
from moss_art.runtime import preflight
preflight(c);configure_process(c,settings['device'])
import jax,jax.numpy as j
import numpy as np,json,hashlib
from moss_art import mxfp8 as mx
f=np.load(settings['reference_fixture'])
w=j.asarray(f['w'],j.bfloat16);x=j.asarray(f['x'],j.bfloat16)
q,s=mx.quantize(w);qt,st=mx.quantize(w.T)
checks={name:bool(np.array_equal(np.asarray(jax.lax.bitcast_convert_type(v,j.uint8)),f[name])) for name,v in [('q',q),('s',s),('qt',qt),('st',st)]}
assert all(checks.values()),checks
records=[]
for mode in ('bf16','mxfp8'):
 t,ts=(qt,st) if mode=='mxfp8' else (None,None)
 fn=jax.jit(lambda x,q,s,t,ts:mx.frozen_matmul(x,q,s,t,ts))
 compiled=fn.lower(x,q,s,t,ts).compile();y=compiled(x,q,s,t,ts);y.block_until_ready()
 xq,xs=mx.quantize(x)
 ref=mx.dequantize(xq,xs,j.float32)@mx.dequantize(q,s,j.float32).T
 fwd_rel=float(j.linalg.norm(y-ref)/j.linalg.norm(ref))
 grad_fn=jax.jit(jax.grad(lambda z,q,s,t,ts:j.sum(mx.frozen_matmul(z,q,s,t,ts)),argnums=0))
 cg=grad_fn.lower(x,q,s,t,ts).compile();g=cg(x,q,s,t,ts);g.block_until_ready()
 if mode=='bf16':gref=(j.ones(y.shape,j.bfloat16).astype(j.float32)@mx.dequantize(q,s,j.bfloat16).astype(j.float32)).astype(j.bfloat16)
 else:
  gq,gs=mx.quantize(j.ones(y.shape,j.bfloat16));gref=(mx.dequantize(gq,gs,j.float32)@mx.dequantize(qt,st,j.float32).T).astype(j.bfloat16)
 backward_rel=float(j.linalg.norm(g.astype(j.float32)-gref.astype(j.float32))/j.linalg.norm(gref.astype(j.float32)))
 fwd_text=compiled.as_text();back_text=cg.as_text()
 assert '__cudnn$blockScaledDot' in fwd_text
 assert fwd_rel<1e-5 and backward_rel<.01
 if mode=='mxfp8':assert '__cudnn$blockScaledDot' in back_text
 Path(f'reports/mxfp8_{mode}_forward.hlo.txt').write_text(fwd_text)
 Path(f'reports/mxfp8_{mode}_backward.hlo.txt').write_text(back_text)
 records.append(dict(dgrad=mode,fwd_relative_error=fwd_rel,dgrad_relative_error=backward_rel,cuDNN_scaled_forward=True,cuDNN_scaled_backward='__cudnn$blockScaledDot' in back_text))
r=dict(physical_device=settings['device'],device=str(jax.devices()[0]),jax=jax.__version__,kohaku_quantization_byte_exact=checks,records=records,reference_fixture_sha256=hashlib.sha256(Path(settings['reference_fixture']).read_bytes()).hexdigest())
Path('reports/mxfp8_kernel_validation.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))