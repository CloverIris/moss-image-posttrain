"""CPU invariants and live GPU memory/gradient acceptance for the new SideNet."""
import argparse
import time
import jax
import jax.numpy as jnp
import numpy as np
from side import SideNet
from common import RUN,read,save

def cpu():
    full=SideNet()
    count=sum(np.prod(s) for s in full.shapes().values())
    assert count==101796864
    net=SideNet(32,64,2,4,24,8,16,"native")
    k1,k2,k3,k4=jax.random.split(jax.random.key(81),4)
    x=jax.random.normal(k1,(1,8,4,4));v=jax.random.normal(k2,x.shape)
    pos=jax.random.normal(k3,(1,6,24));neg=jax.random.normal(k4,(1,6,24))
    c=(pos,neg,jnp.int32([3]),jnp.int32([2]))
    p=net.init(jax.random.key(17))["params"]
    apply=jax.jit(net.apply)
    assert not np.count_nonzero(np.asarray(apply({"params":p},x,v,.6,.1,c)))
    assert all(str(z.dtype)=="float32" for z in p.values())
    p["final_linear_video.weight"]=jax.random.normal(k1,p["final_linear_video.weight"].shape)*.02
    reference=apply({"params":p},x,v,.6,.1,c)
    padded=(pos.at[:,3:].set(999),neg.at[:,2:].set(-999),c[2],c[3])
    np.testing.assert_array_equal(reference,apply({"params":p},x,v,.6,.1,padded))
    short=(pos[:,:3],neg[:,:2],c[2],c[3])
    np.testing.assert_allclose(reference,apply({"params":p},x,v,.6,.1,short),rtol=.03,atol=.002)
    assert not np.count_nonzero(np.asarray(apply({"params":p},x,v,.6,0.,c)))
    assert not np.allclose(reference,apply({"params":p},x,v,.6,-.1,c))
    for which in (0,1):
        values=list(c);values[which]=values[which]*0
        assert not np.allclose(reference,apply({"params":p},x,v,.6,.1,tuple(values)))
    def loss(weights):return jnp.mean(net.apply({"params":weights},x,v,.6,.1,c)**2)
    gradients=jax.jit(jax.grad(loss))(p)
    assert all(np.isfinite(np.asarray(z)).all() for z in gradients.values())
    assert np.linalg.norm(gradients["video_embedder.weight"])>0
    assert np.linalg.norm(gradients["layers.0.proj.q.weight"])>0
    assert np.linalg.norm(gradients["text_embedder.proj.weight"])>0
    assert not np.allclose(reference,apply({"params":p},x*0,v,.6,.1,c))
    assert not np.allclose(reference,apply({"params":p},x,v*0,.6,.1,c))
    print("PASS CPU: 101796864 parameter schema; FP32 masters; zero initialization; padding/length masks; both text branches; signed/zero h; full student parameter gradients",flush=True)

def gpu():
    from teacher import Teacher
    from dp_replica import Replica
    import optax
    cfg=read(RUN/"config.json")
    teacher=Teacher(cfg);r=Replica(cfg,teacher,cfg["width"])
    p=r.net.init(jax.random.key(cfg["seed"]))["params"]
    count=sum(z.size for z in p.values())
    assert count==101796864
    start=time.monotonic()
    g,m=r.trajectory(p,0,0)
    elapsed=time.monotonic()-start
    assert all(np.isfinite(np.asarray(a)).all() for a in g.values())
    norm=float(optax.global_norm(g))
    assert norm>0 and np.isfinite(norm)
    memory=jax.devices()[0].memory_stats()
    report=dict(parameters=count,metrics=m,gradient_norm=norm,chain_seconds_including_compile=elapsed,
                device=str(jax.devices()[0]),device_kind=jax.devices()[0].device_kind,memory=memory)
    save(RUN/"reports/gpu_acceptance.json",report)
    print(report,flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--gpu",action="store_true");args=parser.parse_args()
    gpu() if args.gpu else cpu()
