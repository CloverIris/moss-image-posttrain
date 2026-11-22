"""Alias-preserving replay transfer and exact per-worker row/slot slicing."""
import numpy as np


def host_snapshot(value):
    """Copy each distinct device array once; pickle preserves the resulting aliases."""
    import jax
    memo={}
    def visit(x):
        key=id(x)
        if key in memo:return memo[key]
        if isinstance(x,jax.Array):out=np.asarray(x).copy()
        elif isinstance(x,dict):out=type(x)((k,visit(v)) for k,v in x.items())
        elif isinstance(x,list):out=[visit(v) for v in x]
        elif isinstance(x,tuple):out=tuple(visit(v) for v in x)
        else:out=x
        memo[key]=out
        return out
    return visit(value)


def canonical_contexts(contexts,groups):
    shared={}
    result=[]
    for pair,group in zip(contexts,groups):
        group=int(group)
        if group not in shared:shared[group]=pair
        result.append(shared[group])
    return result


def shard_replay(cache,stats,windows):
    """Retain only rows and selected slots consumed across this worker's windows."""
    rows=np.asarray(sorted({int(r) for window in windows for r,s in window}),np.int64)
    slots=np.asarray(sorted({int(s) for window in windows for r,s in window}),np.int64)
    group_ids,local_groups=np.unique(cache["groups"][rows],return_inverse=True)
    local={key:cache[key] for key in ("sigmas","ode_velocity_coefficients")}
    for key in ("parameter_version", "rollout_precision", "precision_contract"):
        if key in cache: local[key]=cache[key]
    for key in ("states","old"):
        local[key]=cache[key][rows[:,None],slots[None,:]]
    local["endpoints"]=cache["endpoints"][rows]
    local["selected"]=np.asarray(cache["selected"])[slots]
    local["groups"]=local_groups.astype(np.int32)
    local["contexts"]=canonical_contexts([cache["contexts"][r] for r in rows],local_groups)
    local["condition_lengths"]=[cache["condition_lengths"][g] for g in group_ids]
    local_stats={key:(value[group_ids] if key.startswith("group_") else
                      value[rows] if key.startswith("row_") else value)
                 for key,value in stats.items()}
    row_map={int(v):i for i,v in enumerate(rows)}
    slot_map={int(v):i for i,v in enumerate(slots)}
    local_windows=[np.asarray([(row_map[int(r)],slot_map[int(s)]) for r,s in window],np.int64).reshape(-1,2)
                   for window in windows]
    return local,{"stats":local_stats,"windows":local_windows}


def replay_identity(caches):
    """Validate shared sampler source version across every trajectory shard."""
    keys=("parameter_version", "rollout_precision", "precision_contract")
    first={k:caches[0][k] for k in keys if k in caches[0]} if caches else {}
    for cache in caches:
        current={k:cache[k] for k in keys if k in cache}
        if current != first:raise ValueError("Mixed replay parameter/quantization versions")
    return first
