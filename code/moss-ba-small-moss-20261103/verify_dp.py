"""Four real CPU student replicas vs serial Adam, tail weighting and exact resume."""
import os
from pathlib import Path
import tempfile
import numpy as np


class TinyTeacher:
    def grid(self,n):return np.linspace(1.,0.,n+1,dtype=np.float32)
    def noise(self,seed):
        import jax
        return jax.random.normal(jax.random.key(seed),(1,2,2,2))
    def condition(self,pid):
        import jax.numpy as jnp
        return (jnp.ones((1,4,8))*(int(pid)+1),jnp.ones((1,4,8)),jnp.int32([2]),jnp.int32([3]))
    def velocity(self,c,x,t):return .2*x+t*t+.03*c[0].mean()


def fixture_rank(rank,uuids,parent):
    import dp_replica
    from common import RUN,read
    dp_replica.replica=dp_replica.Replica(read(RUN/'config.json'),TinyTeacher(),32)
    return rank


def test():
    import jax
    import optax
    import dp_train,dp_replica
    from side import SideNet
    from common import RUN,save,read,sha
    cfg=dict(order=list(map(str,range(7))),seed=2701,nfe=[8,12],updates_per_prompt=4,
             eval_plan=[dict(prompt_id='7',seed=87)],lr=.0003,width=32,chain=4,qwen_width=8,
             sampling=dict(guidance_scale=3.,rollout_mode='bf16'),
             student=dict(ffn_dim=64,depth=1,heads=4,channels=2,freq_dim=16,backend='native'))
    net=SideNet(32,64,1,4,8,2,16,'native')
    cfg['student']['parameters']=int(sum(np.prod(s) for s in net.shapes().values()))
    save(RUN/'config.json',cfg);save(RUN/'producer_complete.json',dict(training=7,eval=1,verified=True))
    (RUN/'latents/train').mkdir(parents=True);(RUN/'latents/eval').mkdir()
    teacher=TinyTeacher()
    for i,pid in enumerate(cfg['order']):
        np.savez(RUN/f'latents/train/{i:05d}.npz',prompt_id=pid,seed=cfg['seed']+i,euler48=np.asarray(teacher.noise(i)).astype(np.float16))
    ev=np.asarray(teacher.noise(87));np.savez(RUN/'latents/eval/00000.npz',prompt_id='7',seed=87,euler48=ev,euler8=ev,euler12=ev)
    save(RUN/'plan.json',dict(run_sha=sha(RUN/'config.json'),uuids=list(range(4))))
    for step in (0,8):
        folder=RUN/'evaluation'/f'{step:06d}';folder.mkdir(parents=True);(folder/'gallery.jpg').touch()
    dp_train.run(fixture_rank)
    latest=read(RUN/'latest.json')
    opt=optax.chain(optax.clip_by_global_norm(1.),optax.adamw(optax.linear_schedule(cfg['lr']/32,cfg['lr'],31),weight_decay=.01))
    actual=dp_train.restore(Path(latest['checkpoint']),opt)
    assert actual[2:]==(8,7)
    p=net.init(jax.random.key(cfg['seed']))['params'];st=opt.init(p)
    seen=[]
    for group in ([0,1,2,3],[4,5,6]):
        record=read(RUN/'metrics'/f'{group[0]:05d}.json');assert record['indices']==group
        for repeat,logged in enumerate(record['updates']):
            assert [(v['index'],v['repeat']) for v in logged['trajectories']]==[(i,repeat) for i in group]
            seen.extend((i,repeat) for i in group)
            parts=[dp_replica.replica.trajectory(p,i,repeat)[0] for i in group]
            gradient=jax.tree.map(lambda *a:np.mean(np.stack(a),axis=0),*parts)
            delta,st=opt.update(gradient,st,p);p=optax.apply_updates(p,delta)
    assert len(seen)==len(set(seen))==28
    for a,b in zip(jax.tree.leaves((p,st)),jax.tree.leaves(actual[:2]),strict=True):
        np.testing.assert_allclose(a,b,rtol=3e-5,atol=3e-6)
    final_sha=latest['checkpoint_sha256']
    prior=RUN/'checkpoint_1.msgpack'
    save(RUN/'latest.json',dict(step=4,processed_prompts=4,checkpoint=str(prior),checkpoint_sha256=sha(prior)))
    dp_train.run(fixture_rank)
    assert read(RUN/'latest.json')['checkpoint_sha256']==final_sha
    print('PASS: four CPU replicas equal independent serial batch-mean Adam; 28 unique trajectories; 3-sample tail; exact checkpoint resume; counters; evaluation/export',flush=True)

if __name__=='__main__':
    assert os.environ.get('JAX_PLATFORMS')=='cpu'
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:8])
    # Before importing common, propagate the same isolated test root to spawned workers.
    with tempfile.TemporaryDirectory(prefix='ba102m-dp-',dir=os.environ.get('TMPDIR')) as directory:
        os.environ['MOSS_BA_RUN']=directory
        test()
