"""CPU-only diagnostics of ART's per-prompt centered/MAD reward weights.

Read immutable saved rewards; never change training inputs or model state.
The covariance direction is additionally RMS-normalized by ART afterwards.
"""
from __future__ import annotations
import argparse,hashlib,json,os,time
from collections import OrderedDict
from pathlib import Path
import numpy as np


def normalize_records(samples,rewards,group_size):
    raw=np.asarray(rewards,dtype=np.float32)
    if len(samples)!=len(raw) or raw.ndim!=1 or not len(raw) or not np.isfinite(raw).all():
        raise ValueError("Require one finite reward per saved image")
    labels=OrderedDict();ids=[]
    for index,row in enumerate(samples):
        if row.get("index")!=index or row.get("sample_id") is None:
            raise ValueError("Saved samples require ordered image indices and prompt IDs")
        label=row["sample_id"]
        if label not in labels:labels[label]=len(labels)
        ids.append(labels[label])
    ids=np.asarray(ids,np.int32);n=len(labels)
    counts=np.zeros(n,np.float32);sums=counts.copy();squares=counts.copy()
    np.add.at(counts,ids,1);np.add.at(sums,ids,raw);np.add.at(squares,ids,raw*raw)
    if not np.all(counts==group_size):raise ValueError("Incomplete prompt group; normalization must use complete K-image groups")
    means=sums/counts;stds=np.sqrt(np.maximum(squares/counts-means**2,0))
    maxima=np.full(n,-np.inf,np.float32);minima=np.full(n,np.inf,np.float32)
    np.maximum.at(maxima,ids,raw);np.minimum.at(minima,ids,raw)
    constant=maxima==minima;means=np.where(constant,maxima,means);stds=np.where(constant,0,stds)
    centered=np.where(constant[ids],np.float32(0),raw-means[ids])
    mad=np.zeros(n,np.float32);np.add.at(mad,ids,np.abs(centered));mad/=counts
    normalized=centered/np.where(mad[ids]>0,mad[ids],np.float32(1))
    groups=[dict(sample_id=label,count=int(counts[g]),reward_mean=float(means[g]),reward_std=float(stds[g]),
                 reward_mad=float(mad[g]),reward_min=float(minima[g]),reward_max=float(maxima[g]),constant=bool(constant[g])) for label,g in labels.items()]
    rows=[dict(index=i,sample_id=r["sample_id"],seed=r.get("seed"),raw_reward=float(raw[i]),
               group_reward_mean=float(means[ids[i]]),group_reward_std=float(stds[ids[i]]),group_reward_mad=float(mad[ids[i]]),
               centered_reward=float(centered[i]),mad_normalized_reward=float(normalized[i])) for i,r in enumerate(samples)]
    summary=dict(raw_mean=float(raw.mean()),raw_std=float(raw.std()),
        mad_normalized_mean=float(normalized.mean()),mad_normalized_std=float(normalized.std()),
        mad_normalized_min=float(normalized.min()),mad_normalized_max=float(normalized.max()),mad_normalized_abs_mean=float(np.abs(normalized).mean()),
        group_reward_std_mean=float(stds.mean()),group_reward_std_min=float(stds.min()),group_reward_std_max=float(stds.max()),
        group_reward_mad_mean=float(mad.mean()),group_reward_mad_min=float(mad.min()),group_reward_mad_max=float(mad.max()),
        constant_group_fraction=float(constant.mean()),groups=n,images=len(raw))
    return dict(groups=groups,rows=rows,summary=summary)


def record_training_rewards(c,output,logger,rollout,update,rewards,groups,outputs):
    """Inline CPU diagnostics; no auxiliary service needed for new training runs."""
    samples=[]
    for result in outputs:
        for prompt_id,seed in zip(result["sample_ids"],result["seeds"]):
            index=len(samples)
            samples.append(dict(index=index,sample_id=str(int(groups[index])),seed=seed,prompt_id=prompt_id))
    result=normalize_records(samples,rewards,c["sampling"]["group_size"])
    original={row["sample_id"]:row["prompt_id"] for row in samples}
    for row in result["groups"]+result["rows"]:
        row["group_id"]=int(row["sample_id"]);row["sample_id"]=original[row["sample_id"]]
    result.update(rollout=rollout,weights_update=update,height=c["sampling"]["height"],width=c["sampling"]["width"],
                  formula="(reward - per_prompt_mean) / per_prompt_MAD; zero for constant groups",
                  note="Diagnostics only; unchanged ART covariance subsequently RMS-normalizes its direction")
    destination=Path(output)/"reward_norm";destination.mkdir(parents=True,exist_ok=True)
    atomic_json(destination/f"rollout_{rollout:06d}.json",result)
    logger.write("reward_norm",rollout,result["summary"])
    logger.write(f'reward_norm/{result["height"]}x{result["width"]}',rollout,result["summary"])
    histogram(logger,rollout,[row["mad_normalized_reward"] for row in result["rows"]])
    with (destination/"summary.jsonl").open("a") as out:
        out.write(json.dumps(dict(rollout=rollout,weights_update=update,**result["summary"]),allow_nan=False)+"\n")


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value,indent=2,allow_nan=False));os.replace(temp,path)


def histogram(logger,step,values):
    if logger.writer is None:return
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.proto.summary_pb2 import Summary,HistogramProto
    values=np.asarray(values,np.float64);counts,edges=np.histogram(values,bins=24)
    hist=HistogramProto(min=float(values.min()),max=float(values.max()),num=len(values),sum=float(values.sum()),
        sum_squares=float(np.sum(values**2)),bucket_limit=edges[1:].tolist(),bucket=counts.astype(float).tolist())
    logger.writer.write(Event(wall_time=time.time(),step=step,summary=Summary(value=[Summary.Value(tag="reward_norm/mad_normalized",histo=hist)])).SerializeToString())
    logger.writer.flush()


def record_available(run,logger,recorded):
    run=Path(run);c=json.loads((run/"config.json").read_text());output=Path(c["runtime"]["output_dir"])
    if c.get("art",{}).get("feedback","scalar")!="scalar":raise ValueError("This recorder diagnoses scalar ART rewards")
    k=c["training"]["rollouts_per_prompt"];destination=output/"reward_norm";destination.mkdir(exist_ok=True)
    rollout_log=output/"rollouts.jsonl"
    resolution={r["rollout"]:r for r in (json.loads(line) for line in rollout_log.read_text().splitlines())} if rollout_log.exists() else {}
    count=0
    for source in sorted((output/"rewards").glob("rollout_*.json")):
        samples_path=output/"rollout_images"/source.stem/"samples.json"
        if not samples_path.exists():continue
        source_bytes=source.read_bytes();sample_bytes=samples_path.read_bytes()
        fingerprint=hashlib.sha256(source_bytes+b"\0"+sample_bytes).hexdigest()
        target=destination/source.name
        if source.name in recorded:
            if recorded[source.name]!=fingerprint:raise ValueError("A previously recorded reward file changed")
            if target.exists():continue
        payload=json.loads(source_bytes);rollout=payload["rollout"]
        if rollout not in resolution:continue  # trainer writes the resolution row just after reward JSON
        samples=json.loads(sample_bytes);result=normalize_records(samples,[r["reward"] for r in payload["details"]],k)
        info=resolution[rollout];label=f'{info["height"]}x{info["width"]}'
        result.update(rollout=rollout,weights_update=payload["update"],target_optimizer_update=payload["update"]+1,
                      height=info["height"],width=info["width"],source_sha256=fingerprint,
                      formula="(reward - per_prompt_mean) / per_prompt_mean_absolute_deviation; zero for constant reward groups",
                      note="Equivalent scalar weights for covariance/MAD; ART subsequently RMS-normalizes the covariance direction. Not a z-score or a new training reward.")
        atomic_json(target,result)
        logger.write("reward_norm",rollout,result["summary"]);logger.write("reward_norm/"+label,rollout,result["summary"])
        histogram(logger,rollout,[row["mad_normalized_reward"] for row in result["rows"]])
        summary=dict(rollout=rollout,weights_update=payload["update"],target_optimizer_update=payload["update"]+1,resolution=label,**result["summary"])
        with (destination/"summary.jsonl").open("a") as f:f.write(json.dumps(summary,allow_nan=False)+"\n")
        recorded[source.name]=fingerprint;atomic_json(destination/"recorded.json",recorded)
        print(json.dumps(summary),flush=True);count+=1
    return count


def alive(pid):
    status=Path(f"/proc/{pid}/stat")
    try:return status.read_text().split()[2]!="Z"
    except FileNotFoundError:return False


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--config",required=True);args=parser.parse_args(argv)
    options=json.loads(Path(args.config).read_text());run=Path(options["run_dir"]).resolve();c=json.loads((run/"config.json").read_text())
    destination=Path(c["runtime"]["output_dir"])/"reward_norm";destination.mkdir(parents=True,exist_ok=True)
    import fcntl
    from .tensorboard_log import ScalarLogger
    with (destination/".record.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        processed=destination/"recorded.json";recorded=json.loads(processed.read_text()) if processed.exists() else {}
        atomic_json(destination/"metadata.json",dict(formula="a_i=(r_i-mean(r_group))/mean(abs(r_group-mean(r_group)))",constant_groups="all zero",dtype="float32",group_size=c["training"]["rollouts_per_prompt"],x_axis="rollout index, same as rollout/reward_mean",training_math_unchanged=True,model_frameworks_imported=False))
        with ScalarLogger(c,c["runtime"]["output_dir"]) as logger:
            while True:
                record_available(run,logger,recorded)
                if not options.get("follow",False) or not alive(options["training_pid"]):break
                time.sleep(options.get("interval_seconds",10))

if __name__=="__main__":main()
