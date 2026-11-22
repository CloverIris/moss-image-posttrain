"""Fresh-process per-replica DP capacity probe; all options come from JSON.
Synthetic inputs have the exact configured valid context length. This measures
kernel capacity, not image quality or complete ART optimizer semantics.
"""
import argparse,json,sys,time,traceback,gc,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    q=json.loads(Path(a.config).read_text());out=Path(q['output']);out.mkdir(parents=True,exist_ok=True)
    from moss_art.config import load_config
    from moss_art.runtime import configure_process,preflight
    c=load_config('configs/moss_lora_4gpu.json');c['runtime']['devices']=[q['device']]
    c['runtime']['memory_fraction']=q.get('memory_fraction',.9)
    preflight(c);configure_process(c,q['device'])
    import numpy as np
    import jax,jax.numpy as jnp
    from moss_art.model import MossModel
    from scripts.benchmark_rollout import GPUMonitor
    monitor=GPUMonitor(q['device'],True);monitor.start()
    report={'config':q,'status':'running','events':[],'xla_flags':os.environ.get('XLA_FLAGS'),
            'scope':'Per-card true batch, TP=1; synthetic full-valid context; no optimizer, no Qwen; VAE decodes B=1 sequentially.'}
    def save(): (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    def record(phase,**extra):
        event=dict(phase=phase,**extra,memory={k:int(v) for k,v in jax.devices()[0].memory_stats().items() if isinstance(v,(int,np.integer))})
        report['events'].append(event);save();print(json.dumps(event),flush=True)
    try:
        monitor.phase='load_dit';start=time.monotonic()
        model=MossModel.from_pretrained(c['model']['path'],rank=4,alpha=4,
            targets='all' if q['mode']=='sft' else 'attention',attention_backend='cudnn',dit_autotune_level=4,
            offload_activations=q['activation_storage']=='cpu',checkpoint_interval=q['checkpoint_interval'])
        lora=model.init_lora(42);jax.block_until_ready(lora)
        report['dit_weight_bytes']=sum(v.nbytes for v in model.base.values())
        record('dit_loaded',seconds=time.monotonic()-start)
        vae_params=None;decode=None
        if q.get('resident_vae',False):
            monitor.phase='load_vae';start=time.monotonic()
            from vendor.moss_jax.weights import load_config as load_component,load_host_params,to_device
            from moss_art.vae_precision import decode_latents
            vc=load_component(c['model']['path'],'vae')
            host={k:v for k,v in load_host_params(c['model']['path'],'vae').items() if k.startswith(('decoder.','post_quant_conv.','bn.'))}
            vae_params=to_device(host,model.device);del host
            report['vae_weight_bytes']=sum(v.nbytes for v in vae_params.values())
            decode=jax.jit(lambda p,z:decode_latents(p,z,vc,tiling=True,attention_chunk_size=256,clamp=False,
                precision='fp32',low_precision_stages=(),conv_backend='xla',conv_workspace_mib=256))
            record('vae_loaded_resident',seconds=time.monotonic()-start)
        b=q['batch'];length=q['context'];rng=np.random.default_rng(42)
        shape=(b,128,64,64)
        state=model._put(rng.standard_normal(shape,dtype=np.float32))
        context=model._put(rng.normal(0,.2,(b,length,model.text_dim)).astype(np.float32))
        valid=jnp.full((b,),length,jnp.int32);sigma=jnp.full((b,),.5,jnp.float32)
        kwargs={'context_lengths':valid}
        if q['mode']=='art':
            negative=model._put(np.zeros((b,length,model.text_dim),np.float32))
            kwargs.update(cfg_scale=3.,negative_context=negative,negative_context_lengths=jnp.full((b,),min(16,length),jnp.int32),negative_skip_layers=(22,))
        for repeat in range(q.get('repeats',2)):
            monitor.phase=f'forward_{repeat}';start=time.monotonic()
            prediction=model.predict(lora,state,sigma,context,**kwargs);jax.block_until_ready(prediction)
            elapsed=time.monotonic()-start
            assert np.isfinite(np.asarray(prediction)).all()
            record(monitor.phase,seconds=elapsed,images=b,images_per_second=b/elapsed);del prediction;gc.collect()
        if decode is not None:
            monitor.phase='decode_resident_dit';start=time.monotonic()
            pixels=decode(vae_params,state[:1]);pixels.block_until_ready()
            assert np.isfinite(np.asarray(pixels)).all()
            record(monitor.phase,seconds=time.monotonic()-start,output_shape=list(pixels.shape));del pixels;gc.collect()
        for repeat in range(q.get('repeats',2)):
            monitor.phase=f'backward_{repeat}';start=time.monotonic()
            prediction,pullback=model.predict_with_pullback(lora,state,sigma,context,**kwargs)
            grad=pullback(jnp.ones_like(prediction)/prediction.size);jax.block_until_ready(grad)
            elapsed=time.monotonic()-start
            norm=sum(float(np.square(np.asarray(v,dtype=np.float64)).sum()) for v in grad.values())**.5
            assert np.isfinite(norm) and norm>0
            record(monitor.phase,seconds=elapsed,gradient_norm=norm,images=b,images_per_second=b/elapsed)
            del prediction,pullback,grad;gc.collect()
        if decode is not None:
            monitor.phase='decode_after_backward';start=time.monotonic()
            pixels=decode(vae_params,state[:1]);pixels.block_until_ready()
            assert np.isfinite(np.asarray(pixels)).all()
            record(monitor.phase,seconds=time.monotonic()-start);del pixels
        report['status']='passed'
    except BaseException as e:
        report.update(status='failed',error_type=type(e).__name__,error=str(e),failed_phase=monitor.phase)
        traceback.print_exc()
    finally:
        report['gpu_monitor']=monitor.finish(out/'gpu_memory.csv');save()
    return 0 if report['status']=='passed' else 1
if __name__=='__main__':sys.exit(main())