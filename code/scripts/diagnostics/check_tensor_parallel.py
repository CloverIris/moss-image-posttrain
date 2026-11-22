"""Same-model TP=1/2 parity, hidden/weight shards, and cuDNN partition proof."""
import argparse,json,os,time
from pathlib import Path
import numpy as np

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--gpu",action="store_true")
    parser.add_argument("--output",default="/tmp/moss-tp-check.json")
    args=parser.parse_args()
    import jax
    import jax.numpy as jnp
    from moss_art.model import MossModel,_view
    from moss_art.conditioning import pad_context
    from vendor.moss_jax import dit
    assert len(jax.local_devices())==2
    cfg={"dim":256,"ffn_dim":384,"num_heads":2,"num_layers":3}
    kwargs=dict(seed=82,rank=2,config=cfg,dtype=jnp.bfloat16 if args.gpu else jnp.float32,
        attention_backend="cudnn" if args.gpu else "native",offload_activations=False,
        checkpoint_interval=2,dit_autotune_level=4 if args.gpu else 0)
    reference=MossModel.tiny(**kwargs)
    parallel=MossModel.tiny(**kwargs,tensor_parallel_size=2)
    rng=np.random.default_rng(19)
    adapters={k:np.asarray(v)+rng.normal(0,.01,v.shape).astype(np.float32) for k,v in reference.init_lora(7).items()}
    ar={k:reference._put(v) for k,v in adapters.items()}
    at={k:parallel._put(v) for k,v in adapters.items()}
    state=rng.normal(size=(1,8,2,2)).astype(np.float32)
    context=rng.normal(size=(1,9,24)).astype(np.float32)
    cot=rng.normal(size=state.shape).astype(np.float32)
    rel=lambda a,b:float(np.linalg.norm(np.asarray(a,np.float32)-np.asarray(b,np.float32))/max(np.linalg.norm(np.asarray(b,np.float32)),1e-12))
    rows=[]
    for length in (3,9):
        ctx,valid=pad_context(context[:,:length],16)
        opts=dict(context_lengths=valid)
        out,back=reference.predict_with_pullback(ar,state,.4,ctx,**opts); grad=back(cot)
        actual,back=parallel.predict_with_pullback(at,state,.4,ctx,**opts); ag=back(cot)
        flat=lambda d:np.concatenate([np.asarray(d[k]).ravel() for k in sorted(d)])
        error=rel(actual,out);gerror=rel(flat(ag),flat(grad))
        assert error < (.04 if args.gpu else 2e-5),(error,gerror)
        assert gerror < (.05 if args.gpu else 5e-5),(error,gerror)
        rows.append(dict(length=length,output_relative_rms=error,gradient_relative_rms=gerror,
            forward_cache=parallel._block_forward._cache_size(),backward_cache=parallel._block_vjp._cache_size()))
    assert rows[0]["forward_cache"]==rows[1]["forward_cache"]
    assert rows[0]["backward_cache"]==rows[1]["backward_cache"]
    neg,nl=pad_context(context[:,:1],16)
    cfgopts=dict(opts,cfg_scale=3.,negative_context=neg,negative_context_lengths=nl,negative_skip_layers=(1,))
    out,back=reference.predict_with_pullback(ar,state,.4,ctx,**cfgopts); grad=back(cot)
    actual,back=parallel.predict_with_pullback(at,state,.4,ctx,**cfgopts); ag=back(cot)
    cfg_error=rel(actual,out);cfg_grad_error=rel(flat(ag),flat(grad))
    assert cfg_error < (.04 if args.gpu else 2e-5)
    assert cfg_grad_error < (.05 if args.gpu else 5e-5)
    _,tape=parallel._forward(at,state,.4,ctx,record=True,**opts)
    checkpoints=list(tape["checkpoints"].values())
    assert all(v.sharding==parallel.parallel.hidden for v in checkpoints)
    weight=parallel.base['layers.0.proj.q.weight']
    assert len(weight.addressable_shards)==2
    assert all(s.data.size*2==weight.size for s in weight.addressable_shards)
    x=checkpoints[0]
    hlo=parallel._block_forward.lower(parallel._blocks[0],_view(at,'layers.0'),x,
        tape['rope'],tape['modulation'],2,parallel.scale,parallel.attention_backend,tape['valid_lengths']).compile().as_text()
    backward_hlo=parallel._block_vjp.lower(parallel._blocks[0],_view(at,'layers.0'),x,
        tape['rope'],tape['modulation'],jax.numpy.ones_like(x),2,parallel.scale,parallel.attention_backend,tape['valid_lengths']).compile().as_text()
    backward_calls=[line.strip() for line in backward_hlo.splitlines() if 'custom_call_target' in line and 'cudnn' in line.lower()]
    calls=[line.strip() for line in hlo.splitlines() if 'custom_call_target' in line and 'cudnn' in line.lower()]
    collectives=[line.strip() for line in hlo.splitlines() if any(t in line for t in ('all-gather', 'all-reduce', 'reduce-scatter', 'all-to-all'))]
    if args.gpu:
        assert calls,'Missing cuDNN fused attention in partitioned block'
        assert any('Backward' in line for line in backward_calls),'Missing cuDNN backward'
    assert collectives,'No TP communication found'
    result=dict(gpu=args.gpu,checks=rows,cfg_output_relative_rms=cfg_error,cfg_gradient_relative_rms=cfg_grad_error,checkpoint_shape=list(x.shape),
        checkpoint_local_shapes=[list(s.data.shape) for s in x.addressable_shards],
        q_weight_local_shapes=[list(s.data.shape) for s in weight.addressable_shards],
        cudnn_calls=calls,cudnn_backward_calls=backward_calls,collectives=collectives)
    Path(args.output).write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ('cudnn_calls','cudnn_backward_calls','collectives')}),flush=True)
if __name__=='__main__': main()
