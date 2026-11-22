"""Deterministic per-prompt resolution buckets and resumable bucket cursors."""
import copy
import hashlib
import json
import math
import numpy as np

PROFILE_KEYS = {"height", "width", "weight", "rollout_batch_size", "microbatch_size", "group_size", "rollouts_per_update"}

def validate_mixing(c):
    mix=c["resolution_mixing"]
    if not mix["enabled"]:return
    if not mix["profiles"]:raise ValueError("Resolution mixing requires profiles")
    if not c["sampling"]["split_group_across_workers"] or c["sampling"]["prompts_per_rollout"]!=1:
        raise ValueError("Resolution mixing currently requires split prompt groups")
    labels=set()
    for profile in mix["profiles"]:
        if not isinstance(profile,dict) or set(profile)-PROFILE_KEYS or not {"height","width","weight"}<=set(profile):
            raise ValueError("Invalid resolution profile fields")
        for key in ("height","width"):
            if type(profile[key]) is not int or profile[key]<=0 or profile[key]%16:
                raise ValueError("Profile dimensions must be positive multiples of16")
        weight=profile["weight"]
        if type(weight) not in (int,float) or not math.isfinite(weight) or weight<=0:
            raise ValueError("Profile weight must be positive and finite")
        label=f'{profile["height"]}x{profile["width"]}'
        if label in labels:raise ValueError("Duplicate resolution profile")
        labels.add(label)
        for key in PROFILE_KEYS-{"height","width","weight"}:
            if key in profile and (type(profile[key]) is not int or profile[key]<1):
                raise ValueError("Profile batch/group/round counts must be positive integers")
        if c["training"]["prompts_per_update"]>0:
            group=c["training"]["rollouts_per_prompt"];rounds=c["training"]["prompts_per_update"]
            if profile.get("group_size",group)!=group or profile.get("rollouts_per_update",rounds)!=rounds:
                raise ValueError("Logical batch profiles must preserve rollouts_per_prompt and prompts_per_update")
            profile["group_size"]=group;profile["rollouts_per_update"]=rounds
        for key in ("group_size", "rollout_batch_size"):
            profile.setdefault(key, c["sampling"][key])
        for key in ("microbatch_size", "rollouts_per_update"):
            profile.setdefault(key, c["training"][key])
        group=profile.get("group_size",c["sampling"]["group_size"])
        if group%2 or group%len(c["runtime"]["devices"]):raise ValueError("Profile group size must divide workers and two windows")
        if c["art"]["feedback"]=="pairwise" and group not in (2,4,8):raise ValueError("Invalid pairwise profile group size")


def _choice(key,weights):
    value=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],"big")/2**64
    target=value*sum(weights)
    for index,weight in enumerate(weights):
        target-=weight
        if target<0:return index
    return len(weights)-1

class ResolutionPlan:
    def __init__(self,c,records):
        self.seed=c["training"]["seed"]
        self.update_windows=c["training"]["updates_per_rollout"]
        self.profiles=copy.deepcopy(c["resolution_mixing"]["profiles"])
        self.weights=[p["weight"] for p in self.profiles]
        self.labels=[f'{p["height"]}x{p["width"]}' for p in self.profiles]
        self.buckets=[[] for _ in self.profiles]
        self._orders={}
        for record in records:self.buckets[self.assignment(record)].append(record)
        self.active=[i for i,b in enumerate(self.buckets) if b]
        if not self.active:raise ValueError("No records for resolution mixing")

    def assignment(self,record):
        # Text, not row ID/order, fixes the resolution even for duplicate records.
        body=json.dumps([record["prompt"],record.get("negative_prompt","")],ensure_ascii=False,separators=(",",":"))
        return _choice(f"prompt-resolution:{self.seed}:"+body,self.weights)

    def index(self,update):
        position=_choice(f"resolution-update:{self.seed}:{update}",[self.weights[i] for i in self.active])
        return self.active[position]

    def select(self,update,offset,count,cursors):
        index=self.index(update);bucket=self.buckets[index];label=self.labels[index]
        result=[]
        for pos in range(int(cursors.get(label,0))+offset,int(cursors.get(label,0))+offset+count):
            epoch,entry=divmod(pos,len(bucket));key=(index,epoch)
            if key not in self._orders:self._orders[key]=np.random.default_rng(self.seed+epoch+index*100003).permutation(len(bucket))
            result.append(bucket[int(self._orders[key][entry])])
        return result

    def following(self,update,round_index,count,rounds,cursors):
        if round_index+1<rounds:return self.select(update,(round_index+1)*count,count,cursors)
        next_cursors=dict(cursors);label=self.labels[self.index(update)]
        next_cursors[label]=next_cursors.get(label,0)+count*rounds
        return self.select(update+self.update_windows,0,count,next_cursors)


def apply_profile(c,pool,profile):
    sampling=copy.deepcopy(c["sampling"])
    training=copy.deepcopy(c["training"])
    for key in ("height","width","group_size","rollout_batch_size"):
        if key in profile:sampling[key]=profile[key]
    for key in ("microbatch_size","rollouts_per_update"):
        if key in profile:training[key]=profile[key]
    payload={"sampling":{k:sampling[k] for k in ("height","width","group_size","rollout_batch_size")},
             "microbatch_size":training["microbatch_size"]}
    # Mesh workers share pool.config. Let the worker compare its old shape
    # before committing the new supervisor configuration.
    pool.call("set_sampling_profile",[payload for _ in pool.workers])
    for section,value in (("sampling",sampling),("training",training)):
        c[section].update(value)
        pool.config[section].update(copy.deepcopy(value))
    return f'{sampling["height"]}x{sampling["width"]}'
