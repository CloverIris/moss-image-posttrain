"""JAX BA integration. Positive h travels toward the clean endpoint."""
import math
import jax.numpy as jnp
import numpy as np

def gl_step(probe, x, v, t, h):
    r = math.sqrt(.6)
    nodes = jnp.float32(h)*jnp.array([(1-r)/2,.5,(1+r)/2],jnp.float32)
    dv = probe(x,v,t,nodes)
    return x - jnp.float32(h)*(v+(5*dv[0]+8*dv[1]+5*dv[2])/18)

def sample(teacher, probe, x, grid):
    grid = np.asarray(grid,np.float32)
    if len(grid)<2 or grid[0]!=1 or grid[-1]!=0 or not np.all(np.diff(grid)<0):
        raise ValueError("BA expects a strictly descending sigma grid from one to zero")
    alpha = np.float32((1-1/math.sqrt(5))/2)
    outer,inner = np.float32(1/12),np.float32(5/12)
    v = teacher(x,grid[0])
    for i,(t,end) in enumerate(zip(grid,grid[1:])):
        h = t-end
        offsets = jnp.array([alpha*h,(1-alpha)*h,h],dtype=jnp.float32)
        va,vb,ve = v + probe(x,v,t,offsets)
        predicted = x - h*(outer*(v+ve)+inner*(va+vb))
        if i==len(grid)-2:
            return predicted
        anchor = teacher(predicted,end)
        backward = anchor + probe(predicted,anchor,end,jnp.array([-alpha*h],jnp.float32))[0]
        x = x - h*(outer*(v+anchor)+inner*(va+backward))
        v = anchor
    raise AssertionError("unreachable")

