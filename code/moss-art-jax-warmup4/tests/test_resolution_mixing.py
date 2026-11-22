import copy,json
import numpy as np
import pytest
from moss_art.config import DEFAULTS,validate
from moss_art.resolution_mixing import ResolutionPlan

def config():
    c=copy.deepcopy(DEFAULTS)
    c["model"].update(kind="tiny",dtype="float32",activation_storage="gpu",checkpoint_interval=2)
    c["sampling"].update(height=32,width=32,steps=2,train_timesteps=[0,1],group_size=4,
        prompts_per_rollout=1,sampler="euler",scheduler="moss",guidance_scale=1.,skip_layers=[],
        rollout_batch_size=2,split_group_across_workers=True,pair_cached_points=True)
    c["runtime"].update(platform="cpu",devices=[0],cpu_threads=1)
    c["training"].update(max_updates=3,rollouts_per_update=1,updates_per_rollout=1,microbatch_size=2,
        evaluate_every=1,checkpoint_every=1)
    # Pair point distribution needs an even worker count; single-worker integration uses local replay.
    c["sampling"]["pair_cached_points"]=False
    c["reward"]["kind"]="toy_latent"
    c["resolution_mixing"]={"enabled":True,"profiles":[
        {"height":32,"width":32,"weight":2},
        {"height":48,"width":48,"weight":1},
        {"height":64,"width":64,"weight":1}]}
    return c

def records():
    return [dict(sample_id=str(i),prompt="synthetic fixture "+str(i),negative_prompt="") for i in range(60)]

def test_prompt_assignment_is_fixed_across_ids_order_updates_and_resume():
    c=config();validate(c);rows=records();plan=ResolutionPlan(c,rows)
    reordered=ResolutionPlan(c,list(reversed(rows)))
    for rec in rows:
        duplicate={**rec,"sample_id":"another_id"}
        assert plan.assignment(rec)==plan.assignment(duplicate)==reordered.assignment(rec)
    cursors={}
    for update in range(20):
        index=plan.index(update);label=plan.labels[index]
        selected=plan.select(update,0,4,cursors)
        assert all(plan.assignment(row)==index for row in selected)
        restored=ResolutionPlan(c,rows)
        assert [r["sample_id"] for r in selected]==[r["sample_id"] for r in restored.select(update,0,4,json.loads(json.dumps(cursors)))]
        future=plan.following(update,1,2,2,cursors)
        cursors[label]=cursors.get(label,0)+4
        assert [r["sample_id"] for r in future]==[r["sample_id"] for r in plan.select(update+1,0,2,cursors)]

@pytest.mark.parametrize("change",[lambda c:c["resolution_mixing"]["profiles"][0].update(height=33),
    lambda c:c["resolution_mixing"]["profiles"][0].update(weight=True),
    lambda c:c["resolution_mixing"]["profiles"][0].update(rollout_batch_size=0)])
def test_invalid_profiles_fail_before_gpu(change):
    c=config();change(c)
    with pytest.raises(ValueError):validate(c)

def test_mixed_training_keeps_prompt_resolution_and_evaluation_separate(tmp_path):
    from moss_art.trainer import train
    from moss_art.checkpoint import load_checkpoint
    c=config();rows=records()
    for seed in range(100):
        c["training"]["seed"]=seed;validate(c);plan=ResolutionPlan(c,rows)
        if set(plan.index(i) for i in range(3))=={0,1,2}:break
    else:raise AssertionError("Missing deterministic three-resolution fixture")
    file=tmp_path/"synthetic.jsonl";file.write_text("".join(json.dumps(row)+"\n" for row in rows))
    c["data"]["prompts"]=str(file);c["runtime"]["output_dir"]=str(tmp_path/"mixed")
    output=train(c)
    events=[json.loads(line) for line in (output/"rollouts.jsonl").read_text().splitlines()]
    assert {(e["height"],e["width"]) for e in events}=={(32,32),(48,48),(64,64)}
    for update in range(4):
        directory=output/"evaluation"/f"step_{update:08d}"
        assert {p.parent.name for p in directory.glob("*/metrics.json")}=={"32x32","48x48","64x64"}
    ck=json.loads((output/"latest.json").read_text())["checkpoint"]
    _,_,metadata=load_checkpoint(ck)
    assert sum(metadata["resolution_cursors"].values())==3
    assert metadata["update"]==3
    train(c,resume=ck)
