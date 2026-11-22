"""Numerical parity against the pinned PyTorch inference code and SideNet invariants."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent/"upstream"))
oracle_file = Path(__file__).parent/"parity_oracle.npz"

def make_oracle(x):
    import torch
    from BA_utils_side import BA_bidirectional_anchor_sampler
    class Base(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.unused = torch.nn.Parameter(torch.zeros(()))
        def forward(self,x,t,y):
            return .3*x+t[:,None,None,None]**2
    class Side(torch.nn.Module):
        def forward(self,x,v,t,h,y):
            return h[:,None,None,None]*(.2*x+.1*v+t[:,None,None,None])
    results = {str(n):BA_bidirectional_anchor_sampler(Base(),Side(),torch.tensor(x),torch.tensor([0]),num_steps=n,cfg_scale=1.).detach().numpy() for n in (1,5,8,12)}
    np.savez(oracle_file,**results)

def main():
    x = np.linspace(-1,1,32,dtype=np.float32).reshape(1,2,4,4)
    if "--oracle" in sys.argv:
        make_oracle(x)
        return
    import jax
    import jax.numpy as jnp
    from solver import sample,gl_step
    from side import SideNet
    for n in (1,5,8,12):
        with np.load(oracle_file) as f:
            reference = f[str(n)]
        calls=[]
        def teacher(z,t):
            calls.append(float(t))
            return .3*z+t**2
        def probe(z,v,t,h):
            return h[:,None,None,None,None]*(.2*z+.1*v+t)
        actual = sample(teacher,probe,jnp.asarray(x),np.linspace(1,0,n+1,dtype=np.float32))
        np.testing.assert_allclose(reference,actual,atol=2e-6,rtol=1e-6)
        assert len(calls)==n
    for h in (-.2,.1):
        t=.5
        def exact(z,v,t,tau):
            return (((t-tau)**2-t**2)[:,None,None,None,None])*jnp.ones_like(z)
        actual=gl_step(exact,x,jnp.ones_like(x)*t**2,t,h)
        np.testing.assert_allclose(actual,x-(t**3-(t-h)**3)/3,atol=2e-6)
    net=SideNet(16)
    c=(jnp.ones((1,4,24)),jnp.ones((1,4,24)),jnp.array([2]),jnp.array([1]))
    p=net.init(jax.random.key(0),x,x,.4,.1,c)["params"]
    initial=net.apply({"params":p},x,x,.4,.1,c)
    assert np.count_nonzero(np.asarray(initial))==0
    p["output"]["kernel"]=jnp.ones_like(p["output"]["kernel"])*.1
    apply=lambda h,cond:net.apply({"params":p},x,x,.4,h,cond)
    assert np.count_nonzero(np.asarray(apply(0.,c)))==0
    padded=(c[0].at[:,2:].set(999),c[1].at[:,1:].set(-999),*c[2:])
    np.testing.assert_array_equal(apply(.1,c),apply(.1,padded))
    assert not np.allclose(apply(.1,c),apply(.1,(c[0]*0,*c[1:])))
    print("PASS: official solver oracle for 1/5/8/12 NFE; signed GL3; masked Qwen conditioning; identity initialization")

if __name__=="__main__":main()

