import copy
import json
from pathlib import Path
import numpy as np
import pytest
from moss_art.sft_data import SFTCache, load_manifest, write_npz, digest
from moss_art.sft_worker import training_example
from moss_art.data import atomic_json

def fixture_cache(directory):
    directory.mkdir()
    rng=np.random.default_rng(17)
    empty=write_npz(directory,"empty.npz",context=np.zeros((1,2,24),np.float32))
    rows=[]
    for i in range(2):
        ids=np.asarray([[1,100,100,2,3] if i else [1,2,3]],np.int64)
        context=write_npz(directory,f"context-{i}.npz",context=rng.normal(size=(1,ids.size,24)).astype(np.float32),
                          input_ids=ids,attention_mask=np.ones_like(ids,bool))
        latent=write_npz(directory,f"latent-{i}.npz",latent=rng.normal(size=(1,8,2,2)).astype(np.float32))
        rows.append(dict(id=str(i),context=context,latent=latent,width=32,height=32,
            conditioning_images=["reference.png"] if i else [],visual_tokens=2 if i else 0,
            sequence_length=ids.size,prompt="test"))
    identity={"fixture":True}
    atomic_json(directory/"manifest.json",dict(version=1,complete=True,
        conditioning="all_multimodal_hidden_states",identity=identity,identity_hash=digest(identity),
        entries=rows,empty_context=empty,image_token_id=100,latent_channels=8))
    return SFTCache(directory)

def test_manifest_paths_and_both_modes(tmp_path):
    (tmp_path/"target.png").write_bytes(b"x")
    (tmp_path/"ref.png").write_bytes(b"x")
    path=tmp_path/"data.jsonl"
    rows=[dict(prompt="draw",target_image="target.png"),
          dict(prompt="edit",target_image="target.png",conditioning_image="ref.png",width=512,height=768)]
    path.write_text("\n".join(map(json.dumps,rows)))
    result=load_manifest(path)
    assert result[0]["conditioning_images"]==[]
    assert result[1]["conditioning_images"]==[str(tmp_path/"ref.png")]
    assert result[1]["target_image"]!=result[1]["conditioning_images"][0]
    assert result[1]["width"]==512

@pytest.mark.parametrize("row",[
    {"prompt":"x"},{"prompt":"","target_image":"image"},
    {"prompt":"x","target_image":"image","width":15},
    {"prompt":"x","target_image":"image","conditioning_images":"image"},
    {"prompt":"x","target_image":"image","unknown":1},
    {"prompt":"x","target_image":"image","conditioning_image":"image","conditioning_images":[]},
])
def test_invalid_manifest(tmp_path,row):
    (tmp_path/"image").touch()
    path=tmp_path/"data.jsonl"
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError):
        load_manifest(path)

def test_conditioning_only(tmp_path):
    path=tmp_path/"inference.jsonl"
    path.write_text(json.dumps({"prompt":"new image"}))
    assert load_manifest(path,conditioning_only=True)[0]["target_image"] is None

def test_full_token_cache_and_corruption(tmp_path):
    cache=fixture_cache(tmp_path/"cache")
    assert cache.load(cache.rows[1])[1].shape==(1,5,24)
    # A missing visual token must never be silently treated as text-only.
    manifest=cache.manifest
    manifest["entries"][1]["visual_tokens"]=1
    atomic_json(cache.directory/"manifest.json",manifest)
    with pytest.raises(ValueError,match="Visual-token"):
        SFTCache(cache.directory)
    manifest["entries"][1]["visual_tokens"]=2
    atomic_json(cache.directory/"manifest.json",manifest)
    (cache.directory/"context-1.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError,match="Corrupted"):
        SFTCache(cache.directory)

@pytest.mark.parametrize("distribution",["uniform","logit_normal"])
def test_flow_target_and_seed(distribution):
    from moss_art.sft import DEFAULTS
    options=copy.deepcopy(DEFAULTS["sft"])
    options.update(sigma_distribution=distribution,conditioning_dropout=1.)
    latent=np.arange(32,dtype=np.float32).reshape(1,8,2,2)
    state,target,sigma,dropped=training_example(latent,91,options)
    assert 0<sigma<1 and dropped
    np.testing.assert_allclose(state,latent+sigma*target)
    # The same seed exactly reproduces noise, timestep and dropout on resume.
    again=training_example(latent,91,options)
    for a,b in zip((state,target,sigma,dropped),again):
        np.testing.assert_array_equal(a,b)

def test_sft_gradient_depends_on_full_condition():
    import jax.numpy as jnp
    from moss_art.model import MossModel
    from moss_art.sft import DEFAULTS
    model=MossModel.tiny(seed=9,rank=2,targets="all")
    adapters=model.init_lora(3)
    rng=np.random.default_rng(18)
    latent=rng.normal(size=(1,8,2,2)).astype(np.float32)
    state,target,sigma,_=training_example(latent,91,DEFAULTS["sft"])
    context=rng.normal(size=(1,5,24)).astype(np.float32)
    results=[]
    for condition in (context,context[:,2:]):
        loss,grad=model.loss_and_grad(adapters,state,sigma,condition,
                                      lambda v:jnp.mean((v-jnp.asarray(target))**2))
        assert np.isfinite(float(loss))
        results.append(np.concatenate([np.asarray(v).ravel() for v in grad.values()]))
    assert np.linalg.norm(results[0])>0
    assert not np.allclose(results[0],results[1],rtol=1e-5,atol=1e-7)

def test_training_resume_and_two_worker_reduction(tmp_path):
    from moss_art.sft import DEFAULTS,train
    from moss_art.checkpoint import load_checkpoint
    fixture_cache(tmp_path/"cache")
    c=copy.deepcopy(DEFAULTS)
    c["model"].update(kind="tiny",dtype="float32",attention_backend="native",dit_autotune_level=0)
    c["data"]["cache"]=str(tmp_path/"cache")
    c["data"]["context_length"]=16
    c["sampling"]["skip_layers"]=[]
    c["runtime"].update(platform="cpu",devices=[0,1],cpu_threads=1,output_dir=str(tmp_path/"split"))
    c["training"].update(max_updates=1,batch_size=2,checkpoint_every=1)
    c["lora"].update(rank=2,alpha=2.)
    train(c)
    first=tmp_path/"split/checkpoints/step_00000001"
    c["training"]["max_updates"]=2
    train(c,resume=str(first))
    resumed=load_checkpoint(tmp_path/"split/checkpoints/step_00000002")
    c["runtime"].update(devices=[0],output_dir=str(tmp_path/"continuous"))
    train(c)
    continuous=load_checkpoint(tmp_path/"continuous/checkpoints/step_00000002")
    for key in resumed[0]:
        np.testing.assert_allclose(resumed[0][key],continuous[0][key],atol=1e-7,rtol=1e-6)
    assert resumed[2]["sample_cursor"]==continuous[2]["sample_cursor"]==4
    assert (first/"adapter.safetensors").is_file()


def test_shipped_sft_configs_and_invalid_settings():
    from moss_art.sft import load_config, DEFAULTS, validate
    for path in Path("configs/sft").glob("*.yaml"):
        load_config(path)
    for section,key,value in [("training","max_updates",0),("runtime","devices",[]),
                              ("sft","sigma_shift",0),("optimizer","betas",[0.9,float("nan")]),
                              ("sampling","sampler","bad")]:
        c=copy.deepcopy(DEFAULTS)
        c[section][key]=value
        with pytest.raises(ValueError):
            validate(c)


def test_conditioning_only_generation(tmp_path):
    from moss_art.sft import DEFAULTS,sample
    cache=fixture_cache(tmp_path/"cache")
    for row in cache.manifest["entries"]:
        row["latent"]=None
    atomic_json(cache.directory/"manifest.json",cache.manifest)
    with pytest.raises(ValueError,match="cannot train"):
        SFTCache(cache.directory)
    c=copy.deepcopy(DEFAULTS)
    c["model"].update(kind="tiny",dtype="float32",attention_backend="native",dit_autotune_level=0)
    c["data"]["cache"]=str(cache.directory)
    c["data"]["context_length"]=16
    c["runtime"].update(platform="cpu",devices=[0],cpu_threads=1,output_dir=str(tmp_path/"samples"))
    c["sampling"].update(steps=2,skip_layers=[],guidance_scale=1.)
    sample(c,1,42)
    latent=np.load(tmp_path/"samples/sample-0001-seed-42.npy")
    assert latent.shape==(1,8,2,2) and np.isfinite(latent).all()
    meta=json.loads((tmp_path/"samples/sample-0001-seed-42.json").read_text())
    assert meta["visual_tokens"]==2 and meta["steps"]==2


def test_shape_compilation_cache_bounded_without_padding(monkeypatch):
    import jax
    from moss_art.sft_worker import SFTWorker
    worker=object.__new__(SFTWorker)
    worker._compiled_shapes=set()
    cleared=[]
    monkeypatch.setattr(jax,"clear_caches",lambda:cleared.append(True))
    for length in (10,20,10,30):
        worker._manage_compilation_cache((1,8,2,2),(1,length,24),limit=2)
    assert cleared==[True]
    assert worker._compiled_shapes=={((1,8,2,2),(1,30,24))}


@pytest.mark.parametrize("mixed_resolution",[False,True])
@pytest.mark.parametrize("dropout",[0.,.5,1.])
def test_true_microbatch_matches_sample_sum_with_tail(tmp_path,dropout,mixed_resolution):
    from moss_art.sft import DEFAULTS
    from moss_art.sft_worker import SFTWorker
    cache=fixture_cache(tmp_path/"cache")
    if mixed_resolution:
        row=cache.manifest["entries"][1]
        row["width"]=48
        row["latent"]=write_npz(cache.directory,"mixed-latent.npz",
            latent=np.random.default_rng(51).normal(size=(1,8,2,3)).astype(np.float32))
        atomic_json(cache.directory/"manifest.json",cache.manifest)
    c=copy.deepcopy(DEFAULTS)
    c["model"].update(kind="tiny",dtype="float32",attention_backend="native",dit_autotune_level=0)
    c["runtime"].update(platform="cpu",devices=[0])
    c["data"].update(cache=str(tmp_path/"cache"),context_length=8)
    c["sft"]["conditioning_dropout"]=dropout
    c["lora"].update(rank=2,alpha=2.)
    worker=SFTWorker(c,0)
    rng=np.random.default_rng(871)
    worker.set_lora({k:rng.normal(0,.01,v.shape).astype(np.float32) for k,v in worker.lora.items()})
    jobs=[{"index":0,"seed":91},{"index":1,"seed":27},{"index":0,"seed":63}]
    one=worker.gradient(jobs)
    c["training"]["microbatch_size"]=2
    two=worker.gradient(jobs)
    assert two["microbatch_sizes"]==([1,1,1] if mixed_resolution else [2,1])
    assert two["denominator"]==one["denominator"]==3
    assert two["sigmas"]==one["sigmas"]
    assert two["dropped"]==one["dropped"]
    np.testing.assert_allclose(two["loss_sum"],one["loss_sum"],rtol=2e-6)
    for key in one["gradients"]:
        np.testing.assert_allclose(two["gradients"][key],one["gradients"][key],rtol=2e-4,atol=2e-7,err_msg=key)


def test_microbatch_size_validation():
    from moss_art.sft import DEFAULTS,validate
    for value in (0,-1,True,1.5):
        c=copy.deepcopy(DEFAULTS)
        c["training"]["microbatch_size"]=value
        with pytest.raises(ValueError):
            validate(c)
