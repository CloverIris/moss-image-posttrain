"""GPU proof: dynamic cuDNN lengths, padded/unpadded VJP parity and JIT reuse."""
import argparse
import json
from pathlib import Path
import time
import numpy as np

def relative(a,b):
    a,b=np.asarray(a,np.float32),np.asarray(b,np.float32)
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-12))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--device",type=int,default=1)
    parser.add_argument("--output",default="reports/prompt_padding_gpu.json")
    args=parser.parse_args()
    from moss_art.config import DEFAULTS
    from moss_art.runtime import configure_process,preflight
    import copy
    c=copy.deepcopy(DEFAULTS)
    c["runtime"]["devices"]=[args.device]
    preflight(c)
    configure_process(c,args.device)
    import jax
    import jax.numpy as jnp
    from moss_art.model import MossModel,_view
    from moss_art.conditioning import pad_context,padded_rope,condition_block
    from vendor.moss_jax import dit
    model=MossModel.tiny(seed=31,rank=2,dtype=jnp.bfloat16,attention_backend="cudnn",
        config={"dim":256,"ffn_dim":384,"num_heads":2},dit_autotune_level=4)
    adapters=model.init_lora(9)
    rng=np.random.default_rng(42)
    state=rng.normal(size=(1,8,2,2)).astype(np.float32)
    context=rng.normal(size=(1,9,24)).astype(np.float32)
    cotangent=rng.normal(size=state.shape).astype(np.float32)
    checks=[]
    for length in (3,9):
        ref,back=model.predict_with_pullback(adapters,state,.4,context[:,:length],
                                             context_lengths=np.asarray([length],np.int32))
        ref_grad=back(cotangent)
        padded,valid=pad_context(context[:,:length],32)
        padded[:,length:]=rng.normal(0,10,padded[:,length:].shape)
        actual,back=model.predict_with_pullback(adapters,state,.4,padded,context_lengths=valid)
        grad=back(cotangent)
        flat=lambda values:np.concatenate([np.asarray(values[k]).ravel() for k in sorted(values)])
        error=relative(actual,ref)
        grad_error=relative(flat(grad),flat(ref_grad))
        assert np.isfinite(actual).all() and error<.03 and grad_error<.03,(error,grad_error)
        checks.append({"length":length,"output_relative_rms":error,"gradient_relative_rms":grad_error})
    kernels=(model._block_forward,model._block_vjp,condition_block,padded_rope,dit._compiled_linear)
    for kernel in kernels:
        kernel.clear_cache()
    counts=[]
    times=[]
    for length in (3,9,3):
        padded,valid=pad_context(context[:,:length],32)
        start=time.monotonic()
        value,back=model.predict_with_pullback(adapters,state,.4,padded,context_lengths=valid)
        jax.block_until_ready(back(cotangent))
        times.append(time.monotonic()-start)
        counts.append([kernel._cache_size() for kernel in kernels])
    assert counts[0]==counts[1]==counts[2],counts
    # Inspect compiled text and DiT blocks for cuDNN custom calls in both passes.
    x=jnp.zeros((1,36,256),jnp.bfloat16)
    rope=padded_rope(2,2,32,128,10000.,jnp.asarray([3],jnp.int32),dtype=jnp.bfloat16)
    mod=jnp.zeros((1,6,256),jnp.float32)
    base=model._blocks[0]
    local=_view(adapters,"layers.0")
    valid=jnp.asarray([7],jnp.int32)
    forward=model._block_forward.lower(base,local,x,rope,mod,2,model.scale,"cudnn",valid).compile().as_text()
    backward=model._block_vjp.lower(base,local,x,rope,mod,jnp.ones_like(x),2,model.scale,"cudnn",valid).compile().as_text()
    text=condition_block.lower(_view(model.base,"text_embedder.blocks.0"),
        x[:,:32],rope[:,4:],2,jnp.asarray([3],jnp.int32),"cudnn").compile().as_text()
    calls={name:[line.strip() for line in hlo.splitlines() if "custom_call_target" in line and "cudnn" in line.lower()]
           for name,hlo in (("dit_forward",forward),("dit_backward",backward),("condition",text))}
    assert all(calls.values()),calls
    report={"device":str(jax.devices()[0]),"checks":checks,"cache_counts":counts,
        "cache_order":["dit_forward","dit_backward","condition","rope","linear"],
        "seconds":times,"cudnn_custom_calls":calls}
    Path(args.output).write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="cudnn_custom_calls"},indent=2))

if __name__=="__main__":
    main()
