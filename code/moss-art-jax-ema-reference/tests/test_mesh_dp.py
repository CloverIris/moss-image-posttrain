import copy,json
import jax,jax.numpy as jnp,numpy as np
import pytest
from moss_art.config import DEFAULTS,validate
from moss_art.model import MossModel
from moss_art.mesh_runtime import MeshPool
from moss_art.worker import NativeWorker
from moss_art.trainer import make_payloads,paired_replay_payloads,aggregate_gradients
from moss_art.art import covariance,prompt_windows

pytestmark=pytest.mark.skipif(len(jax.local_devices())!=4,reason="requires four logical devices")

def configuration():
 c=copy.deepcopy(DEFAULTS)
 c["model"].update(kind="tiny",dtype="float32",activation_storage="gpu",checkpoint_interval=2)
 c["runtime"].update(platform="cpu",devices=[0,1,2,3],execution_backend="mesh",cpu_threads=2)
 c["data"].update(context_length=16,short_context_length=8)
 c["sampling"].update(height=32,width=32,steps=4,train_timesteps=[0,1,2],group_size=4,prompts_per_rollout=1,
  split_group_across_workers=True,pair_cached_points=True,rollout_batch_size=1,guidance_scale=1.5,skip_layers=[1])
 c["training"].update(microbatch_size=2,max_updates=1,updates_per_rollout=1,evaluate_every=0)
 c["reward"]["kind"]="toy_latent"
 return c

def test_mesh_predict_and_pullback_match_reference():
 m=MossModel.tiny(seed=3,offload_activations=False,data_parallel_size=4)
 ref=MossModel.tiny(seed=3,offload_activations=False)
 rng=np.random.default_rng(9);z=rng.normal(size=(4,8,2,2)).astype(np.float32);ctx=rng.normal(size=(4,6,24)).astype(np.float32)
 a=m.init_lora(4);ar=ref.init_lora(4)
 y,pb=m.predict_with_pullback(a,z,.5,ctx,context_lengths=np.array([3,4,5,6],np.int32))
 yr,pbr=ref.predict_with_pullback(ar,z,.5,ctx,context_lengths=np.array([3,4,5,6],np.int32))
 np.testing.assert_allclose(y,yr,rtol=3e-5,atol=3e-6)
 assert len(y.addressable_shards)==4 and not y.is_fully_replicated
 g,gr=pb(jnp.ones_like(y)),pbr(jnp.ones_like(yr))
 for k in g:
  np.testing.assert_allclose(g[k],gr[k],rtol=4e-5,atol=4e-6)
  assert g[k].is_fully_replicated and len(g[k].addressable_shards)==4

@pytest.mark.parametrize("feedback",["scalar","pairwise"])
def test_mesh_rollout_and_art_gradient_match_serial(feedback):
 c=configuration();c["art"]["feedback"]=feedback
 record=dict(sample_id="fixture",prompt="synthetic fixture",negative_prompt="")
 ref=NativeWorker(c,0)
 with MeshPool(c) as pool:
  outputs=pool.call("rollout",make_payloads(c,[record],0))
  expected=ref.rollout([record],list(range(c["training"]["seed"],c["training"]["seed"]+4)))
  endpoints=np.concatenate([o["endpoints"] for o in outputs]);np.testing.assert_allclose(endpoints,expected["endpoints"],rtol=5e-5,atol=5e-6)
  groups=np.zeros(4,np.int32);rewards=np.array([.1,.4,.7,.95],np.float32)
  stats={k:np.asarray(v) for k,v in covariance(endpoints,rewards,groups,mode=feedback,group_size=4).items()}
  windows=[np.concatenate(prompt_windows(groups,3,seed=7))]
  ps=paired_replay_payloads(pool,groups,stats,windows);pool.call("prepare_update",ps)
  result=pool.call("gradient",[{"window":0}]*4)
  actual,metrics=aggregate_gradients(result)
  ref.prepare_update(stats=stats,windows=windows)
  expected,reference=aggregate_gradients([ref.gradient(0)])
  for k in actual:
   np.testing.assert_allclose(actual[k],expected[k],rtol=2e-4,atol=2e-6)
   assert isinstance(actual[k],jax.Array) and actual[k].is_fully_replicated
  assert metrics["items"]==reference["items"]==12
  pool.call("clear_replay")
  before={k:np.asarray(v).copy() for k,v in pool.worker.lora.items()}
  graph=pool.parallel
  pool.pause();assert pool.worker is None
  pool.resume()
  assert pool.parallel is graph
  for k,v in before.items():np.testing.assert_array_equal(pool.worker.lora[k],v)


def test_mesh_full_training_checkpoint_resume_and_eval(tmp_path):
 from moss_art.trainer import train
 c=configuration();c["training"].update(prompts_per_update=2,rollouts_per_prompt=4,training_points_per_trajectory=3,evaluate_every=1)
 c["runtime"]["output_dir"]=str(tmp_path/"training")
 data=tmp_path/"data.jsonl";data.write_text(''.join(json.dumps(dict(id=str(i),prompt="synthetic fixture "+str(i)))+'\n' for i in range(5)))
 c["data"]["prompts"]=str(data);validate(c)
 output=train(c);rows=[json.loads(l) for l in (output/"metrics.jsonl").read_text().splitlines()]
 assert len(rows)==1 and rows[0]["items"]==24
 latest=json.loads((output/"latest.json").read_text())
 train(c,resume=latest["checkpoint"])
 assert (output/"evaluation/step_00000001/metrics.json").exists()
