"""Independent pinned official BA oracle and signed GL3 integration regression."""
from pathlib import Path
import numpy as np
import jax.numpy as jnp
from solver import sample,gl_step

def test():
    x=np.linspace(-1,1,32,dtype=np.float32).reshape(1,2,4,4)
    oracle=Path('$CODE_ROOT/moss-ba-jax/parity_oracle.npz')
    with np.load(oracle) as targets:
        for n in (1,5,8,12):
            calls=[]
            def teacher(z,t):calls.append(float(t));return .3*z+t**2
            def probe(z,v,t,h):return h[:,None,None,None,None]*(.2*z+.1*v+t)
            actual=sample(teacher,probe,jnp.asarray(x),np.linspace(1,0,n+1,dtype=np.float32))
            np.testing.assert_allclose(actual,targets[str(n)],atol=2e-6,rtol=1e-6)
            assert len(calls)==n
    for h in (-.2,.1):
        t=.5
        def exact(z,v,t,tau):return (((t-tau)**2-t**2)[:,None,None,None,None])*jnp.ones_like(z)
        actual=gl_step(exact,x,jnp.ones_like(x)*t**2,t,h)
        np.testing.assert_allclose(actual,x-(t**3-(t-h)**3)/3,atol=2e-6)
    print('PASS: pinned official PyTorch BA oracle at NFE 1/5/8/12; correct teacher call budget; GL3 signed intervals',flush=True)

if __name__=='__main__':test()
