"""Two-device hidden/head tensor parallelism within one training worker.

Global array shapes stay identical to single-device MOSS. XLA partitions the
main DiT blocks; residual/checkpoint tensors shard their hidden axis and cuDNN
attention shards complete heads (never head_dim). This is TP, not FSDP.
"""
import numpy as np
import jax
from jax.sharding import Mesh,NamedSharding,PartitionSpec as P

class TensorParallel:
    def __init__(self,size,config,autotune_level):
        if type(size) is not int or size!=2:
            raise ValueError("tensor_parallel_size supports 1 or 2")
        devices=jax.local_devices()
        if len(devices)!=size:
            raise ValueError(f"TP worker needs exactly {size} visible devices, got {len(devices)}")
        if any(config[key]%size for key in ("dim","ffn_dim","num_heads")):
            raise ValueError("Hidden size, FFN size and attention heads must divide TP size")
        self.mesh=Mesh(np.asarray(devices),('tp',))
        self.replicated=NamedSharding(self.mesh,P())
        self.hidden=NamedSharding(self.mesh,P(None,None,'tp'))
        from .model import _dit_block,_block_vjp_impl
        options={"xla_gpu_autotune_level":autotune_level} if autotune_level else None
        def forward(*args): return _dit_block(*args,hidden_sharding=self.hidden)
        def backward(*args): return _block_vjp_impl(*args,hidden_sharding=self.hidden)
        self.kernels=(jax.jit(forward,static_argnums=(5,6,7),out_shardings=self.hidden,compiler_options=options),
                      jax.jit(backward,static_argnums=(6,7,8),out_shardings=(self.replicated,self.hidden),compiler_options=options))

    def parameter_sharding(self,key,value):
        if key.startswith('layers.') and key.endswith('.weight') and value.ndim==2:
            if any(key.endswith('.'+name+'.weight') for name in ('proj.q','proj.k','proj.v')):
                return NamedSharding(self.mesh,P('tp',None))
            # Row-parallel output/FFN projections. Keeping the fused up/gate
            # input axis sharded avoids splitting hidden vs gate across GPUs;
            # each split activation is explicitly hidden-sharded in the block.
            return NamedSharding(self.mesh,P(None,'tp'))
        return self.replicated
