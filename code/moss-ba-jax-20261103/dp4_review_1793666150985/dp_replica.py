"""One isolated BF16 teacher and one student replica per physical GPU."""
import os
import time
import numpy as np
from common import RUN, read, save

DP = RUN / "dp4"
replica = None


class Replica:
    def __init__(self, cfg, teacher, width):
        import jax
        import jax.numpy as jnp
        from side import SideNet
        self.cfg, self.teacher = cfg, teacher
        self.net = SideNet(width)
        self.deploy = np.concatenate([-np.diff(teacher.grid(n)) for n in cfg["nfe"]])
        self.probes = jax.jit(jax.vmap(
            lambda p,x,v,t,h,c: self.net.apply({"params":p},x,v,t,h,c),
            in_axes=(None,None,None,None,0,None)))
        def objective(p,x,v,t,h,c,target):
            delta = self.net.apply({"params":p},x,v,t,h,c)
            truth = jax.lax.stop_gradient(target-v)
            return jnp.mean(jnp.square(delta-truth)), jnp.mean(jnp.square(truth))
        self.grad = jax.jit(jax.value_and_grad(objective,has_aux=True))
        self.index = None

    def load(self, index):
        import jax.numpy as jnp
        pid = self.cfg["order"][index]
        with np.load(RUN/"latents/train"/f"{index:05d}.npz",allow_pickle=False) as f:
            if str(f["prompt_id"]) != pid or int(f["seed"]) != self.cfg["seed"]+index:
                raise ValueError("Reference identity differs from the frozen ordering")
            self.clean = jnp.asarray(f["euler48"],jnp.float32)
        self.cond = self.teacher.condition(pid)
        self.index = index

    def trajectory(self, params, index, repeat):
        import jax
        import jax.numpy as jnp
        from solver import gl_step
        from train import sample_interval
        if self.index != index:
            self.load(index)
        rng = np.random.default_rng(np.random.SeedSequence([self.cfg["seed"],index,repeat]))
        t = np.float32(rng.uniform(.001,.999))
        x = (1-t)*self.clean+t*self.teacher.noise(int(rng.integers(0,2**31)))
        v = self.teacher.velocity(self.cond,x,t)
        grads = None
        losses, zeros, intervals = [], [], []
        for _ in range(self.cfg["chain"]):
            h = sample_interval(rng,t,self.deploy)
            xn = gl_step(lambda x,v,t,h:self.probes(params,x,v,t,h,self.cond),x,v,t,h)
            tn = np.float32(np.clip(t-h,0.,1.))
            vn = self.teacher.velocity(self.cond,xn,tn)
            (loss,zero),g = self.grad(params,x,v,t,h,self.cond,vn)
            grads = g if grads is None else jax.tree.map(jnp.add,grads,g)
            losses.append(float(loss)); zeros.append(float(zero)); intervals.append(float(h))
            x,v,t = xn,vn,tn
        return jax.tree.map(lambda g:g/len(losses),grads), dict(
            index=index, repeat=repeat, mse=float(np.mean(losses)),
            zero_offset_mse=float(np.mean(zeros)),signed_intervals=intervals)


def initialize(rank, uuids, parent_pid):
    # Workers are spawned with a fresh interpreter. Set the mask before importing JAX.
    global replica
    if rank:
        import ctypes, signal
        if ctypes.CDLL(None).prctl(1,signal.SIGTERM,0,0,0) or os.getppid()!=parent_pid:
            raise RuntimeError("Lost DP coordinator")
    os.environ["CUDA_VISIBLE_DEVICES"] = uuids[rank]
    import jax
    from teacher import Teacher
    from pool import identity
    cfg = read(RUN/"config.json")
    devices = jax.local_devices()
    if len(devices)!=1 or "nvidia" not in devices[0].device_kind.lower():
        raise RuntimeError(f"Expected one GB203(SM120) on rank {rank}: {devices}")
    oracle = Teacher(cfg)
    contract = oracle.model.precision_contract()
    if (contract["base_precision"]!="bf16" or contract["quantized_modules"] or
            any(str(v.dtype)!="bfloat16" for k,v in oracle.model.base.items() if k!="rope.bands")):
        raise ValueError("Expected unquantized BF16 DiT weights")
    replica = Replica(cfg,oracle,cfg["width"])
    save(DP/f"rank_{rank}.json",dict(rank=rank,pid=os.getpid(),birth=identity(os.getpid())[0],
         coordinator=parent_pid,uuid=uuids[rank],contract=contract,at=time.time()))
    return rank


def host_gradient(params,index,repeat):
    import jax
    grads,metrics=replica.trajectory(jax.device_put(params),index,repeat)
    return jax.device_get(grads),metrics


def mean_gradient(parts):
    import jax
    import jax.numpy as jnp
    if not parts:raise ValueError("Empty synchronous batch")
    # Reduction and the single optimizer step run on physical GPU 0.
    return jax.tree.map(lambda *g:sum(map(jnp.asarray,g[1:]),jnp.asarray(g[0]))/len(g),*parts)
