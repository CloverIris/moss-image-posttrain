"""Logical prompt/trajectory/point budgets, independent of device microbatches."""
import numpy as np

def enabled(c):return c["training"]["prompts_per_update"]>0

def resolve(c):
    t,s=c["training"],c["sampling"]
    if t["prompts_per_update"]<0:raise ValueError("prompts_per_update must be nonnegative")
    if not enabled(c):return
    if t["rollouts_per_prompt"]<4 or t["rollouts_per_prompt"]%4:
        raise ValueError("rollouts_per_prompt must be a positive multiple of4")
    if t["rollouts_per_prompt"]%len(c["runtime"]["devices"]):
        raise ValueError("rollouts_per_prompt must divide evenly across selected workers")
    from .sampling import schedule
    _,sigmas=schedule(s["steps"],scheduler=s["scheduler"],shift=s["shift"],training_shift=s["training_shift"],
                      beta_alpha=s["beta_alpha"],beta_beta=s["beta_beta"])
    actual=len(sigmas)-1;points=t["training_points_per_trajectory"]
    if not 1<=points<=actual:
        raise ValueError("training_points_per_trajectory must lie between1 and actual denoising steps")
    s["group_size"]=t["rollouts_per_prompt"]
    s["prompts_per_rollout"]=1
    s["split_group_across_workers"]=True
    s["train_timesteps"]=np.linspace(0,actual-1,points,dtype=np.int64).tolist()
    t["rollouts_per_update"]=t["prompts_per_update"]

def unique_pool(records):
    seen=set();result=[]
    for row in records:
        key=(row["prompt"],row.get("negative_prompt",""))
        if key not in seen:seen.add(key);result.append(row)
    return result

def select(c,records,update,cursor,mix_plan=None,mix_cursors=None):
    count=c["training"]["prompts_per_update"]
    if mix_plan is None:
        pool=records;offset=cursor;seed=c["training"]["seed"]
    else:
        index=mix_plan.index(update);pool=mix_plan.buckets[index]
        offset=(mix_cursors or {}).get(mix_plan.labels[index],0)
        seed=c["training"]["seed"]+index*100003
    pool=unique_pool(pool)
    if count>len(pool):raise ValueError(f"Requested {count} distinct prompts but only {len(pool)} are available in this resolution bucket")
    from .rollout_mixing import enabled as mixed_sampling, select_balanced
    if mixed_sampling(c):
        return select_balanced(c,pool,count,seed,offset)
    order=np.random.default_rng((seed+offset)%(2**32)).choice(len(pool),size=count,replace=False)
    return [pool[int(index)] for index in order]

def following(c,records,update,cursor,prompts,round_index,mix_plan,mix_cursors):
    if round_index+1<len(prompts):return [prompts[round_index+1]]
    next_cursors=dict(mix_cursors)
    if mix_plan is not None:
        label=mix_plan.labels[mix_plan.index(update)]
        next_cursors[label]=next_cursors.get(label,0)+len(prompts)
    return select(c,records,update+c["training"]["updates_per_rollout"],cursor+len(prompts),mix_plan,next_cursors)[:1]
