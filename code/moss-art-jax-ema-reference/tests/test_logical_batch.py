import copy,json
import numpy as np
import pytest
from moss_art.config import DEFAULTS,validate
from moss_art.batch_plan import select


def config():
    c=copy.deepcopy(DEFAULTS)
    c["model"].update(kind="tiny",dtype="float32",activation_storage="gpu",checkpoint_interval=2)
    c["runtime"].update(platform="cpu",devices=[0,1],cpu_threads=1)
    c["sampling"].update(height=32,width=32,steps=4,sampler="euler",scheduler="moss",guidance_scale=1.,skip_layers=[],pair_cached_points=True,rollout_batch_size=2)
    c["training"].update(prompts_per_update=3,rollouts_per_prompt=4,training_points_per_trajectory=3,
        max_updates=1,updates_per_rollout=1,microbatch_size=2,evaluate_every=0,checkpoint_every=1)
    c["reward"]["kind"]="toy_latent"
    return c

@pytest.mark.parametrize("k",[0,1,2,6,10])
def test_rollout_count_must_be_multiple_of_four(k):
    c=config();c["training"]["rollouts_per_prompt"]=k
    with pytest.raises(ValueError,match="multiple of4"):validate(c)

@pytest.mark.parametrize("points",[0,5])
def test_points_cannot_be_empty_or_exceed_denoising_steps(points):
    c=config();c["training"]["training_points_per_trajectory"]=points
    with pytest.raises(ValueError,match="training_points_per_trajectory"):validate(c)


def test_resolved_budget_and_unique_prompt_selection():
    c=config();validate(c)
    assert c["sampling"]["group_size"]==4
    assert c["sampling"]["train_timesteps"]==[0,1,3]
    assert c["training"]["rollouts_per_update"]==3
    assert c["training"]["updates_per_rollout"]==1
    rows=[dict(sample_id=str(i),prompt="synthetic "+str(i),negative_prompt="") for i in range(9)]
    chosen=select(c,rows,0,0)
    assert len({r["prompt"] for r in chosen})==3
    assert chosen==select(c,rows,0,0)
    with pytest.raises(ValueError,match="distinct prompts"):select(c,rows[:2],0,0)


def test_three_prompts_four_trajectories_three_points_one_update(tmp_path):
    from moss_art.trainer import train
    c=config();c["runtime"]["output_dir"]=str(tmp_path/"training")
    data=tmp_path/"data.jsonl";data.write_text("".join(json.dumps({"id":str(i),"prompt":"synthetic fixture "+str(i)})+"\n" for i in range(9)))
    c["data"]["prompts"]=str(data);validate(c)
    out=train(c)
    events=[json.loads(l) for l in (out/"metrics.jsonl").read_text().splitlines() if l.strip()]
    assert len(events)==1
    assert events[0]["update"]==1 and events[0]["items"]==3*4*3
    assert events[0]["trajectories"]==12
    assert events[0]["prompts_per_update"]==3
    assert events[0]["rollouts_per_prompt"]==4
    assert events[0]["training_points_per_trajectory"]==3
    from moss_art.checkpoint import load_checkpoint
    latest=json.loads((out/"latest.json").read_text())
    _,_,meta=load_checkpoint(latest["checkpoint"])
    assert meta["prompt_cursor"]==3
    train(c,resume=latest["checkpoint"])


def test_logical_budget_stays_fixed_with_resolution_profiles():
    c=config();c["resolution_mixing"]={"enabled":True,"profiles":[
        {"height":32,"width":32,"weight":1,"rollout_batch_size":2},
        {"height":48,"width":48,"weight":1,"rollout_batch_size":1}]}
    validate(c)
    for profile in c["resolution_mixing"]["profiles"]:
        assert profile["group_size"]==4 and profile["rollouts_per_update"]==3
    from moss_art.resolution_mixing import ResolutionPlan
    rows=[dict(sample_id=str(i),prompt="synthetic "+str(i),negative_prompt="") for i in range(60)]
    plan=ResolutionPlan(c,rows)
    for update in range(10):
        chosen=select(c,rows,update,update*3,plan,{})
        assert len({r["prompt"] for r in chosen})==3
        assert len({plan.assignment(r) for r in chosen})==1
