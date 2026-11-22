"""Online/offline FP8 bit parity, additive LoRA and original BF16 recovery."""
import copy
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from moss_art.config import DEFAULTS, validate, fingerprint
from moss_art.model import MossModel, tiny_base
from moss_art.mxfp8_cache import write_cache, load_cache

CFG = dict(dim=128, ffn_dim=256, num_heads=2, num_layers=1, text_dim=128,
           freq_dim=16, latent_channels=8, rope_base=10000.)


@pytest.mark.parametrize("transpose", [False, True])
def test_cached_weight_bits_predictions_and_lora_gradients_match_online(tmp_path, transpose, monkeypatch):
    base = tiny_base(CFG, seed=12, dtype=jnp.bfloat16)
    source = {"synthetic": "BF16 fixture"}
    directory = tmp_path/"weights"
    write_cache(directory, base, source_identity=source, include_transpose=transpose)
    host = load_cache(directory, source_identity=source, require_transpose=transpose)
    dgrad = "mxfp8" if transpose else "bf16"
    online = MossModel(base, CFG, rank=2, offload_activations=False, base_precision="mxfp8", mxfp8_dgrad=dgrad)
    from moss_art import mxfp8
    monkeypatch.setattr(mxfp8, "quantize", lambda *a: pytest.fail("Offline weight loader requantized a weight"))
    offline = MossModel(host, CFG, rank=2, offload_activations=False, base_precision="mxfp8",
                        mxfp8_dgrad=dgrad, prequantized=True)
    for key, value in online.base.items():
        assert str(value.dtype) == str(offline.base[key].dtype)
        np.testing.assert_array_equal(np.asarray(value).view(np.uint8), np.asarray(offline.base[key]).view(np.uint8))
    assert online.base_storage_bytes == offline.base_storage_bytes
    # Restore dynamic activation quantization; only weight loading was forbidden.
    monkeypatch.undo()
    rng = np.random.default_rng(81)
    adapters = {k: v+jnp.asarray(rng.normal(0, .02, v.shape), jnp.float32) for k,v in online.init_lora().items()}
    state = jnp.asarray(rng.normal(size=(1,8,2,2)), jnp.float32)
    context = jnp.asarray(rng.normal(size=(1,5,128)), jnp.bfloat16)
    actual, pb = offline.predict_with_pullback(adapters, state, .4, context)
    expected, reference_pb = online.predict_with_pullback(adapters, state, .4, context)
    np.testing.assert_array_equal(actual, expected)
    for a, b in zip(jax.tree.leaves(pb(jnp.ones_like(actual))), jax.tree.leaves(reference_pb(jnp.ones_like(expected)))):
        np.testing.assert_array_equal(a, b)
    changed = {k: v*1.3 for k,v in adapters.items()}
    np.testing.assert_array_equal(offline.predict(changed,state,.4,context), online.predict(changed,state,.4,context))
    # A forward-only caller can load a transpose-rich cache without its extra copies.
    if transpose:
        forward = load_cache(directory, source_identity=source)
        assert not any(".mx_transpose" in key for key in forward)


def test_cache_rejects_source_changes_and_corruption(tmp_path):
    base = tiny_base(CFG, seed=12, dtype=jnp.bfloat16)
    cache = tmp_path/"weights"
    write_cache(cache, base, source_identity={"source": 1})
    with pytest.raises(ValueError, match="source model"):
        load_cache(cache, source_identity={"source": 2})
    with pytest.raises(ValueError, match="include_transpose"):
        load_cache(cache, source_identity={"source": 1}, require_transpose=True)
    with pytest.raises(FileExistsError):
        write_cache(cache, base, source_identity={"source": 1})
    manifest = json.loads((cache/"manifest.json").read_text())
    file = cache/next(iter(manifest["entries"].values()))["file"]
    with file.open("r+b") as stream:
        stream.seek(-1,2); stream.write(b"x")
    with pytest.raises(ValueError, match="checksum"):
        load_cache(cache, source_identity={"source": 1})


def test_io_mode_does_not_change_algorithm_fingerprint_and_requires_cache_path():
    c = copy.deepcopy(DEFAULTS)
    original = fingerprint(c)
    c["model"].update(mxfp8_weight_mode="offline", mxfp8_cache_dir="/prepared/fp8", mxfp8_cache_verify=False)
    validate(c)
    assert fingerprint(c) == original
    c["model"]["mxfp8_cache_dir"] = ""
    with pytest.raises(ValueError, match="cache_dir"):
        validate(c)


def test_pretrained_bf16_path_ignores_fp8_cache_and_loads_original_weights(monkeypatch):
    import moss_art.model as module
    base = tiny_base(CFG, seed=12, dtype=jnp.bfloat16)
    monkeypatch.setattr(module,"load_config",lambda *a: CFG)
    monkeypatch.setattr(module,"load_host_params",lambda *a: base)
    from moss_art import mxfp8_cache
    monkeypatch.setattr(mxfp8_cache,"load_pretrained_cache",lambda *a,**kw:pytest.fail("BF16 learner read FP8 weights"))
    model = MossModel.from_pretrained("/synthetic", rank=2, offload_activations=False,
        base_precision="bf16", mxfp8_weight_mode="offline", mxfp8_cache_dir="/does-not-exist")
    for key, value in base.items():
        np.testing.assert_array_equal(model.base[key], value)


def test_sft_worker_forwards_offline_options_to_shared_pretrained_loader(monkeypatch):
    from moss_art.sft import DEFAULTS as SFT_DEFAULTS
    from moss_art.sft_worker import SFTWorker
    from moss_art.prompt_buckets import PromptBuckets
    from types import SimpleNamespace
    c=copy.deepcopy(SFT_DEFAULTS)
    c["model"].update(kind="moss",base_precision="mxfp8",attention_backend="native",dit_autotune_level=0,
                      mxfp8_weight_mode="offline",mxfp8_cache_dir="/prepared/weights")
    captured={}
    def loader(path,**options):
        captured.update(options)
        for key in ("mxfp8_weight_mode","mxfp8_cache_dir","mxfp8_cache_verify"):options.pop(key)
        return MossModel.tiny(config=CFG,dtype=jnp.bfloat16,**options)
    monkeypatch.setattr(MossModel,"from_pretrained",loader)
    monkeypatch.setattr("moss_art.sft_data.SFTCache",lambda *a,**kw:SimpleNamespace(empty_context=lambda:np.zeros((1,2,128),np.float32)))
    monkeypatch.setattr("moss_art.prompt_buckets.configure_sft_cache",lambda *a,**kw:PromptBuckets(8,16))
    SFTWorker(c,0)
    assert captured["mxfp8_weight_mode"]=="offline"
    assert captured["mxfp8_cache_dir"]=="/prepared/weights" and captured["mxfp8_cache_verify"] is True
