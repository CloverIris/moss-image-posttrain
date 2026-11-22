"""Single-controller JAX data-parallel graph; weights/state are distributed arrays."""
from functools import lru_cache
import numpy as np
import jax
from jax.sharding import Mesh,NamedSharding,PartitionSpec as P

class DataParallel:
    def __init__(self,size,autotune_level):
        devices=jax.local_devices()
        if len(devices)!=size:raise ValueError("Data-parallel mesh needs exactly the selected visible devices")
        self.size=size
        self.mesh=Mesh(np.asarray(devices),("dp",))
        self.replicated=NamedSharding(self.mesh,P())
        self.hidden=NamedSharding(self.mesh,P("dp",None,None))
        self.options={"xla_gpu_autotune_level":autotune_level} if autotune_level else None
        from .model import _dit_block,_block_vjp_impl
        self.kernels=(jax.jit(_dit_block,static_argnums=(5,6,7),out_shardings=self.hidden,compiler_options=self.options),
                      jax.jit(_block_vjp_impl,static_argnums=(6,7,8),out_shardings=(self.replicated,self.hidden),compiler_options=self.options))

    def parameter_sharding(self,key,value):return self.replicated

    def batch(self,rank):
        return NamedSharding(self.mesh,P("dp",*(None for _ in range(rank-1)))) if rank else self.replicated

    def put(self,value):
        if np.ndim(value) and np.shape(value)[0]%self.size:
            raise ValueError("Global data-parallel batch must divide mesh size")
        return jax.device_put(value,self.batch(np.ndim(value)))


@lru_cache(maxsize=8)
def get_data_parallel(size,autotune_level):
    # Graph only: no model arrays or optimizer state are closed over here.
    # Re-loading weights after VAE/reward reuses these compiled callables.
    return DataParallel(size,autotune_level)
