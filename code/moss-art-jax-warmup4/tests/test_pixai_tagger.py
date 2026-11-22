"""Native PixAI inference against an independent Torch synthetic oracle."""
from pathlib import Path
import json
import math
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from PIL import Image
from moss_art.pixai_tagger import (METADATA,PixAITagger,expected_shapes,forward,load_parameters,preprocess,validate_config)

FIXTURE=Path(__file__).parent/'fixtures/pixai_tiny'

@pytest.fixture(scope='module')
def config():return json.loads((FIXTURE/'tiny_config.json').read_text())

@pytest.fixture(scope='module')
def oracle():
    with np.load(FIXTURE/'torch_reference.npz') as f:yield {k:f[k] for k in f.files}

@pytest.fixture(scope='module')
def model(config):
    p=load_parameters(FIXTURE/'tiny.safetensors',config)
    model=PixAITagger(p,config)
    yield model
    model.close()


def test_released_checkpoint_schema():
    c=json.loads((METADATA/'config.json').read_text());m=json.loads((METADATA/'weights_manifest.json').read_text());s=expected_shapes(c)
    assert set(s)==set(m['tensors'])
    assert all(shape==tuple(m['tensors'][k]['shape']) for k,shape in s.items())
    assert sum(math.prod(shape) for shape in s.values())==486346909
    assert m['weight_data_transferred'] is False


@pytest.mark.parametrize('dtype,prefix,limit',[('float32','',2e-6),('bfloat16','bf16_',.025)])
def test_torch_layerwise_parity(config,oracle,dtype,prefix,limit):
    p=load_parameters(FIXTURE/'tiny.safetensors',config,dtype=getattr(jnp,dtype))
    try:
        logits,trace=forward(p,jnp.asarray(oracle[prefix+'input']),config,capture=True)
        for name,value in trace.items():
            np.testing.assert_allclose(np.asarray(value,np.float32),oracle[prefix+name],atol=limit,rtol=0)
        error=np.linalg.norm(np.asarray(logits)-oracle[prefix+'logits'])/np.linalg.norm(oracle[prefix+'logits'])
        assert error<(2e-5 if dtype=='float32' else .01)
    finally:
        for v in p.values():v.delete()


@pytest.mark.parametrize('i',range(7))
def test_original_preprocessing(oracle,i):
    # Upscale/downscale, square/wide/tall, single-pixel axes.
    np.testing.assert_allclose(preprocess(oracle['raw_'+str(i)],24),oracle['preprocessed_'+str(i)],atol=1e-6,rtol=0)


def test_alpha_and_padding(oracle):
    np.testing.assert_allclose(preprocess(Image.fromarray(oracle['raw_rgba']),24),oracle['preprocessed_rgba'],atol=1e-6,rtol=0)
    assert np.all(preprocess(np.zeros((3,12,3),np.uint8),24)[:,:,:9]==-1)


def test_rewards_are_dense_deterministic_and_deduplicate(model,oracle):
    images=[oracle['raw_0'],oracle['raw_1'],oracle['raw_2']]
    dense=model.probabilities(images,batch_size=2)
    repeated=model.probabilities(images,batch_size=2)
    np.testing.assert_array_equal(dense,repeated)
    records=model.rewards(images,[['blue hair','hat','hat'],['1girl'],['coat','flower']],batch_size=2)
    np.testing.assert_allclose([x['reward'] for x in records],[dense[0,[1,2]].mean(),dense[1,0],dense[2,[3,5]].mean()],atol=1e-7)
    assert records[0]['used_tags']==['blue_hair','hat']
    assert len(records)==3 and dense.shape==(3,7)
    # Fixed-size padded final batch must not change which score belongs to an image.
    np.testing.assert_allclose(model.probabilities(images,batch_size=1),dense,atol=1e-6,rtol=0)


def test_unknown_tags_and_category(model):
    with pytest.raises(ValueError,match='Unknown or out-of-category'):model.target_indices(['not_in_vocabulary'])
    with pytest.raises(ValueError,match='Unknown or out-of-category'):model.target_indices(['sample_character'])
    assert model.target_indices(['sample_character'],category='character')[1]==['sample_character']
    with pytest.warns(RuntimeWarning):
        ids,tags,unknown=model.target_indices(['hat','not_in_vocabulary'],strict=False)
    assert tags==['hat'] and unknown==['not_in_vocabulary']
    with pytest.raises(ValueError,match='No supported'):model.target_indices([])


def test_no_pretrained_weight_fallback(tmp_path):
    with pytest.raises(FileNotFoundError,match='weights are required'):PixAITagger.from_pretrained(tmp_path/'missing.safetensors')


def test_unsupported_architecture_rejected(config):
    c=dict(config,retain_cls_token=True)
    with pytest.raises(ValueError,match='CLS'):validate_config(c)


def test_bad_images():
    with pytest.raises(ValueError,match='uint8'):preprocess(np.zeros((8,8,3),np.float32),24)
    with pytest.raises(ValueError,match='empty resized'):preprocess(np.zeros((1,48,3),np.uint8),24)


def test_full_release_abstract_forward():
    c=json.loads((METADATA/'config.json').read_text())
    weights={k:jax.ShapeDtypeStruct(shape,jnp.bfloat16) for k,shape in expected_shapes(c).items()}
    pixels=jax.ShapeDtypeStruct((1,3,1008,1008),jnp.float32)
    out=jax.eval_shape(lambda p,x:forward(p,x,c),weights,pixels)
    assert out.shape==(1,30877) and out.dtype==jnp.float32


def test_cudnn_requires_gpu_and_low_precision(model):
    with pytest.raises(ValueError,match='cuDNN'):PixAITagger(model.params,model.config,attention_backend='cudnn')


def test_malformed_checkpoint_rejected(tmp_path,config):
    from safetensors.numpy import save_file
    file=tmp_path/'bad.safetensors';save_file({'unexpected.weight':np.zeros((2,2),np.float32)},str(file))
    with pytest.raises(ValueError,match='Weight keys differ'):load_parameters(file,config)


def test_wrong_input_shape_and_target_count(model):
    with pytest.raises(ValueError,match='normalized NCHW'):model.logits(np.zeros((1,3,8,8),np.float32))
    with pytest.raises(ValueError,match='Nonfinite'):model.logits(np.full((1,3,24,24),np.nan,np.float32))
    with pytest.raises(ValueError,match='One target-tag list'):model.rewards([np.zeros((24,24,3),np.uint8)],[['hat'],['coat']])


@pytest.mark.skipif(len(jax.local_devices())!=4,reason="requires four logical devices")
def test_mesh_tagger_matches_serial(config,oracle):
    from moss_art.data_parallel import get_data_parallel
    dp=get_data_parallel(4,0)
    params=load_parameters(FIXTURE/'tiny.safetensors',config,device=dp.replicated)
    model=PixAITagger(params,config,parallel=dp)
    pixels=np.repeat(oracle['input'][:1],4,axis=0)
    try:
        logits=model.logits(pixels)
        assert logits.sharding.is_equivalent_to(dp.batch(2),2)
        np.testing.assert_allclose(np.asarray(logits),np.repeat(oracle['logits'][:1],4,axis=0),atol=2e-6)
    finally:model.close()
