import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from moss_art.rollout_only import load_config,read_prompt_file,jobs_for


def test_txt_jsonl_and_seed_plan(tmp_path):
    txt=tmp_path/"p.txt";txt.write_text("\ufeffcat\n\n dog \n",encoding="utf-8")
    rows=read_prompt_file(txt,"bad")
    assert [(r.prompt,r.negative_prompt) for r in rows]==[("cat","bad"),("dog","bad")]
    js=tmp_path/"p.jsonl";js.write_text(json.dumps({"id":"../../cat","prompt":"cat","negative_prompt":"blurry"})+"\n")
    assert read_prompt_file(js)[0].negative_prompt=="blurry"
    records=[dict(sample_id=r.id,prompt=r.prompt) for r in rows]
    positions={r.id:i for i,r in enumerate(rows)}
    a=list(jobs_for(records,positions,3,2**32-2))
    b=sum([list(jobs_for([r],positions,3,2**32-2)) for r in records],[])
    assert a==b and len({j["seed"] for j in a})==6
    txt.write_text("\n \n")
    with pytest.raises(ValueError,match="empty"):read_prompt_file(txt)


@pytest.mark.parametrize("patch",[{"rollout":{"count":0}},{"rollout":{"count":True}},
    {"rollout":{"seed":-1}},{"rollout":{"prompts_per_batch":0}},
    {"rollout":{"decode":False,"save_latents":False}},
    {"sampling":{"sampler":"unknown"}},{"sampling":{"steps":0}},
    {"optimizer":{"name":"muon"}},{"run":{"resume":"x"}},
    {"preparation":{"device":0}}])
def test_reject_bad_inference_config(tmp_path,patch):
    f=tmp_path/"c.json";f.write_text(json.dumps(patch))
    with pytest.raises(ValueError):load_config(f)


def test_one_step_odd_counts_and_arbitrary_prompt_batch_are_valid(tmp_path):
    f=tmp_path/"c.json";f.write_text(json.dumps({"sampling":{"steps":1},
        "rollout":{"count":1,"prompts_per_batch":3},"runtime":{"devices":[0,1,2,3]}}))
    assert load_config(f)["rollout"]["count"]==1
    for f in Path("configs/rollout").glob("*.json"):load_config(f)


def test_tiny_cli_async_tail_matches_sync_single_worker(tmp_path):
    prompts=tmp_path/"p.jsonl"
    prompts.write_text("".join(json.dumps(dict(id=f"../id-{i}",prompt=f"cat {i}",negative_prompt="blurry"))+"\n" for i in range(3)))
    base={"data":{"prompts":str(prompts),"cache":str(tmp_path/"cache"),"short_context_length":4,"context_length":8},
          "model":{"kind":"tiny","dtype":"float32","attention_backend":"native","dit_autotune_level":0,"vae_conv_backend":"xla"},
          "sampling":{"height":32,"width":32,"steps":2,"skip_layers":[]},
          "rollout":{"count":3,"seed":999,"prompts_per_batch":1,"decode":False,"save_latents":True},
          "runtime":{"platform":"cpu","devices":[0,1],"cpu_threads":1}}
    env=dict(os.environ,JAX_PLATFORMS="cpu",CUDA_VISIBLE_DEVICES="",OPENBLAS_NUM_THREADS="1")
    manifests=[]
    for async_mode,devices in [(True,[0,1]),(False,[0])]:
        c=copy.deepcopy(base);out=tmp_path/str(async_mode)
        c["runtime"].update(devices=devices,output_dir=str(out))
        c["preparation"]={"async_prefetch":async_mode}
        f=tmp_path/f"{async_mode}.json";f.write_text(json.dumps(c))
        result=subprocess.run([sys.executable,"run.py","rollout-only","--config",str(f)],env=env,
            capture_output=True,text=True,timeout=180)
        assert result.returncode==0,result.stdout+result.stderr
        complete=json.loads((out/"complete.json").read_text())
        assert complete["images"]==9 and complete["prompts"]==3 and not complete["training"] and not complete["reward"]
        rows=[json.loads(x) for x in (out/"samples.jsonl").read_text().splitlines()]
        assert len(rows)==9 and len({r["seed"] for r in rows})==9
        assert all(r["actual_steps"]==2 and r["negative_prompt"]=="blurry" for r in rows)
        assert len(list(out.glob("*.npy")))==9 and not list(out.glob("*.png"))
        manifests.append([(r,np.load(out/r["latent"])) for r in rows])
        if async_mode:
            events=[json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
            ready=[e for e in events if e.get("event")=="qwen_buffer_ready"]
            assert len(ready)==3 and sum(e["encoded"] for e in ready)==4
    for (left,a),(right,b) in zip(*manifests):
        assert left==right
        np.testing.assert_array_equal(a,b)
    # Existing images/latents must never be silently replaced.
    result=subprocess.run([sys.executable,"run.py","rollout","--config",str(f)],env=env,capture_output=True,text=True,timeout=30)
    assert result.returncode!=0 and "not empty" in result.stderr
