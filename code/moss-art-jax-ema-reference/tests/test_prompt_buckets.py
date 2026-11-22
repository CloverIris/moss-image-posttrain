import copy
import json
import sys
import numpy as np
import pytest
from moss_art.prompt_buckets import PromptBuckets,configure_sft_cache
from moss_art.prompt_lengths import reference_grid
from PIL import Image


def test_nearest_rank_quantile_and_boundary():
    b=PromptBuckets.resolve([10]*7+[200,300,500,9000],0,1024,emit=False)
    assert b.short==64 and b.full==1024
    assert b.capacity(64)==64 and b.capacity(65)==1024 and b.capacity(1024)==1024
    with pytest.raises(ValueError):b.capacity(1025)
    with pytest.raises(ValueError):PromptBuckets.resolve([1025],0,1024)


def test_short_auto_empty_condition_minimum():
    assert PromptBuckets.resolve([5],0,128,minimum=70,emit=False).short==128
    with pytest.raises(ValueError):PromptBuckets.resolve([5],32,128,minimum=70,emit=False)


def test_image_header_grids(tmp_path):
    for (w,h),grid in [((1024,1024),[1,32,32]),((1024,768),[1,26,36]),((2048,1024),[1,22,44]),((256,256),[1,16,16])]:
        p=tmp_path/f"{w}-{h}.png";Image.new("RGB",(w,h)).save(p)
        assert reference_grid(p)==grid


def test_filter_cached_sft_before_sampling_preserves_indices(tmp_path,capsys):
    from test_sft import fixture_cache
    cache=fixture_cache(tmp_path/"cache")
    b=configure_sft_cache(cache,{"context_length":4,"short_context_length":3})
    assert [r["id"] for r in cache.rows]==["0"]
    assert b.capacity(3)==3
    assert 'prompt_dropped' in capsys.readouterr().out


def test_art_bucket_lengths_used_in_rollout_and_both_replays(monkeypatch):
    from test_sampling_worker import bare_worker,RECORDS
    import jax.numpy as jnp
    worker=bare_worker(steps=2,selected=(0,1))
    worker.c["data"].update(short_context_length=4,context_length=8)
    worker._conditioning=lambda record:(jnp.zeros((1,3 if record["sample_id"]=="first" else 6,1)),jnp.zeros((1,2,1)))
    calls=[]
    original=worker.model.predict
    def predict(*args,**kw):
        calls.append((args[3].shape[1],kw["negative_context"].shape[1],int(kw["context_lengths"][0]),int(kw["negative_context_lengths"][0])))
        return original(*args,**kw)
    worker.model.predict=predict
    worker.rollout(RECORDS,[1,2,3,4])
    assert set(calls)=={(4,4,3,2),(8,8,6,2)}
    assert [tuple(int(a[0]) for a in lens) for lens in worker.cache["condition_lengths"]]==[(3,2),(6,2)]


def tiny_text_model():
    from vendor.moss_jax import text_encoder as te
    rng=np.random.default_rng(4)
    cfg=dict(hidden_size=8,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=4,
        intermediate_size=16,linear_num_key_heads=1,linear_num_value_heads=2,linear_key_head_dim=2,
        linear_value_head_dim=2,linear_conv_kernel_dim=4,rms_norm_eps=1e-6,
        layer_types=["linear_attention","full_attention"],rope_parameters=dict(partial_rotary_factor=1.,rope_theta=10000.,mrope_section=[0,1,1]))
    p={"language_model.embed_tokens.weight":rng.normal(size=(32,8)).astype(np.float32),"language_model.norm.weight":np.zeros(8,np.float32)}
    for i,kind in enumerate(cfg["layer_types"]):
        shapes={"input_layernorm.weight":(8,),"post_attention_layernorm.weight":(8,),
            "mlp.gate_proj.weight":(16,8),"mlp.up_proj.weight":(16,8),"mlp.down_proj.weight":(8,16)}
        shapes.update({"linear_attn.in_proj_qkv.weight":(8,8),"linear_attn.in_proj_z.weight":(4,8),
            "linear_attn.in_proj_b.weight":(2,8),"linear_attn.in_proj_a.weight":(2,8),"linear_attn.conv1d.weight":(8,1,4),
            "linear_attn.A_log":(2,),"linear_attn.dt_bias":(2,),"linear_attn.norm.weight":(2,),
            "linear_attn.out_proj.weight":(8,4)} if kind=="linear_attention" else {
            "self_attn.q_proj.weight":(16,8),"self_attn.k_proj.weight":(4,8),"self_attn.v_proj.weight":(4,8),
            "self_attn.o_proj.weight":(8,8),"self_attn.q_norm.weight":(4,),"self_attn.k_norm.weight":(4,)})
        p.update({f"language_model.layers.{i}."+k:rng.normal(0,.1,size=shape).astype(np.float32) for k,shape in shapes.items()})
    return p,cfg


def test_jax_prefill_right_padding_preserves_valid_hidden_states():
    from vendor.moss_jax.text_encoder import text_encode,decoder_layer
    p,c=tiny_text_model();ids=np.asarray([[1,2,3,4,5]],np.int32)
    exact=np.asarray(text_encode(p,ids,np.ones_like(ids),c))
    before=decoder_layer._cache_size()
    for length in (5,4,3):
        padded=np.pad(ids[:,:length],((0,0),(0,8-length)))
        out=np.asarray(text_encode(p,padded,np.arange(8)[None]<length,c))
        np.testing.assert_allclose(out[:,:length],exact[:,:length],rtol=2e-5,atol=2e-5)
    assert decoder_layer._cache_size()==before+2 # one per block type, not per real length


def test_multimodal_positions_two_images():
    from moss_art.qwen_prefill import multimodal_positions
    ids=np.asarray([[1,60,60,60,60,2,60,60,3]])
    pos=multimodal_positions(ids,[[1,4,4],[1,2,4]],60)
    np.testing.assert_array_equal(pos[:,0,1:5],[[1,1,1,1],[1,1,2,2],[1,2,1,2]])
    np.testing.assert_array_equal(pos[:,0,6:8],[[4,4],[4,4],[4,5]])
    assert list(pos[:,0,-1])==[6,6,6]


def test_scanner_import_is_accelerator_free():
    import subprocess
    subprocess.run([sys.executable,"-c","import sys; import moss_art.prompt_lengths; assert not any(m in sys.modules for m in ('jax','torch','tensorflow','transformers'))"],check=True)


def test_art_drop_identity_is_bound_to_text_not_only_id(monkeypatch,tmp_path):
    from moss_art.trainer import read_records
    from moss_art.data import text_key
    from moss_art.config import DEFAULTS
    c=copy.deepcopy(DEFAULTS);path=tmp_path/"prompts.jsonl"
    c["data"]["prompts"]=str(path)
    path.write_text(json.dumps({"id":"same","prompt":"new prompt"}))
    class FakeCache:
        def __init__(self,*a,**kw):
            self.manifest={"entries":{},"dropped_records":[dict(id="same",tokens=9999,text_keys=[text_key("old prompt"),text_key("")])]}
    monkeypatch.setattr("moss_art.data.PromptCache",FakeCache)
    with pytest.raises(ValueError,match="Missing cached condition"):
        read_records(c)
