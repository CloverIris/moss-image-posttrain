import copy
import json
import numpy as np
from moss_art.config import DEFAULTS,validate
from moss_art.art import covariance
from moss_art.trainer import merge_rollout_rounds,paired_replay_payloads,aggregate_gradients
from test_paired_replay import worker,Pool
from test_batched_art import BatchedPolicy


def test_four_rounds_have_same_gradient_as_256_individual_points():
    c=copy.deepcopy(DEFAULTS);c["sampling"].update(height=32,width=32,steps=4,
        train_timesteps=[0,1,2,3],group_size=16,sampler="euler",scheduler="moss",guidance_scale=1.,skip_layers=[],rollout_batch_size=4)
    c["training"]["microbatch_size"]=4
    ws=[worker(c) for _ in range(4)]
    for w in ws:w.model=BatchedPolicy()
    record={"sample_id":"synthetic","prompt":"synthetic fixture","negative_prompt":""}
    all_outputs=[];all_caches=[];all_rewards=[];reference=[]
    for ri in range(4):
        outs=[w.rollout([record],list(range(ri*100+rank*4,ri*100+rank*4+4)),samples_per_prompt=4) for rank,w in enumerate(ws)]
        caches=[w.export_replay() for w in ws];all_outputs.append(outs);all_caches.append(caches)
        rewards=np.linspace(.1,.9,16,dtype=np.float32)**(1+ri*.1);all_rewards.append(rewards)
        ref=worker(c);ref.model=BatchedPolicy()
        cache={key:caches[0][key] for key in ("sigmas","selected","ode_velocity_coefficients","condition_lengths")}
        for key in ("states","old","endpoints"):cache[key]=np.concatenate([v[key] for v in caches])
        cache["groups"]=np.zeros(16,np.int32);cache["contexts"]=[p for v in caches for p in v["contexts"]]
        ref.install_replay(cache)
        stats={k:np.asarray(v) for k,v in covariance(cache["endpoints"],rewards,cache["groups"]).items()}
        ref.prepare_update(stats=stats,windows=[np.asarray([(row,slot) for row in range(16) for slot in range(4)])])
        reference.append(aggregate_gradients([ref.gradient(0)])[0]["theta"])
    outputs,caches,groups=merge_rollout_rounds(all_outputs,all_caches,1,True)
    assert np.bincount(groups).tolist()==[16]*4
    assert len(set(seed for out in outputs for seed in out["seeds"]))==64
    rewards=np.concatenate([np.concatenate([rr[rank*4:rank*4+4] for rr in all_rewards]) for rank in range(4)])
    endpoints=np.concatenate([o["endpoints"] for o in outputs])
    stats={k:np.asarray(v) for k,v in covariance(endpoints,rewards,groups).items()}
    windows=[np.asarray([(row,slot) for row in range(64) for slot in range(4)],np.int64)]
    pool=Pool(ws);ps=paired_replay_payloads(pool,groups,stats,windows,caches=caches);prep=pool.call("prepare_update",ps)
    assert [v["items_per_window"] for v in prep]==[[64]]*4
    grads,metrics=aggregate_gradients(pool.call("gradient",[{"window":0}]*4))
    assert metrics["items"]==256
    np.testing.assert_allclose(grads["theta"],np.mean(reference),rtol=5e-6,atol=5e-7)


def test_one_optimizer_step_after_four_collections_and_resume(tmp_path):
    from moss_art.trainer import train
    c=copy.deepcopy(DEFAULTS);c["model"].update(kind="tiny",dtype="float32",activation_storage="gpu",checkpoint_interval=2)
    c["runtime"].update(platform="cpu",devices=[0],cpu_threads=1,output_dir=str(tmp_path/"output"))
    c["sampling"].update(height=32,width=32,steps=4,train_timesteps=[0,1,2,3],group_size=4,
        prompts_per_rollout=1,sampler="euler",scheduler="moss",guidance_scale=1.,skip_layers=[],rollout_batch_size=4)
    c["training"].update(max_updates=1,rollouts_per_update=4,updates_per_rollout=1,microbatch_size=4,
        evaluate_every=0,checkpoint_every=1)
    c["reward"]["kind"]="toy_latent"
    data=tmp_path/"data.jsonl";data.write_text(json.dumps({"id":"synthetic","prompt":"synthetic fixture"})+"\n")
    c["data"]["prompts"]=str(data);validate(c)
    output=train(c)
    metrics=[json.loads(line) for line in (output/"metrics.jsonl").read_text().splitlines()]
    assert len(metrics)==1 and metrics[0]["update"]==1
    assert metrics[0]["items"]==64 and metrics[0]["rollout"]==4
    from moss_art.checkpoint import load_checkpoint
    ck=json.loads((output/"latest.json").read_text())["checkpoint"]
    _,_,metadata=load_checkpoint(ck)
    assert metadata["prompt_cursor"]==4 and metadata["update"]==1
    # Resume at a completed collection: zero further optimizer steps required.
    train(c,resume=ck)
