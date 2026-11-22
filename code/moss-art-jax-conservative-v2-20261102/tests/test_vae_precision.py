"""Tiny CPU checks for experimental VAE stage precision and native semantics."""
from unittest.mock import Mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from moss_art import vae_precision as mixed
from vendor.moss_jax import vae


@pytest.fixture(scope="module")
def fixture():
    rng = np.random.default_rng(191)
    config = dict(
        latent_channels=2, patch_size=(2, 2), batch_norm_eps=1e-4,
        block_out_channels=(4,) * 4, decoder_block_out_channels=(4,) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        layers_per_block=0, norm_num_groups=2, mid_block_add_attention=True,
        use_post_quant_conv=True, sample_size=32, act_fn="silu",
    )
    params = {
        "bn.running_mean": rng.normal(0, .1, 8).astype(np.float32),
        "bn.running_var": rng.uniform(.5, 1.5, 8).astype(np.float32),
    }

    def norm(prefix, channels=4):
        params[prefix + ".weight"] = np.ones(channels, np.float32)
        params[prefix + ".bias"] = rng.normal(0, .02, channels).astype(np.float32)

    def conv(prefix, out_channels=4, in_channels=4, size=3):
        params[prefix + ".weight"] = rng.normal(
            0, .15 / np.sqrt(in_channels * size**2),
            (out_channels, in_channels, size, size),
        ).astype(np.float32)
        params[prefix + ".bias"] = rng.normal(0, .03, out_channels).astype(np.float32)

    def resnet(prefix, shortcut=False):
        norm(prefix + ".norm1")
        norm(prefix + ".norm2")
        conv(prefix + ".conv1")
        conv(prefix + ".conv2")
        if shortcut:
            conv(prefix + ".conv_shortcut", size=1)

    conv("post_quant_conv", 2, 2, 1)
    conv("decoder.conv_in", 4, 2)
    for i in range(2):
        resnet(f"decoder.mid_block.resnets.{i}")
    prefix = "decoder.mid_block.attentions.0"
    norm(prefix + ".group_norm")
    for suffix in ("to_q", "to_k", "to_v", "to_out.0"):
        params[f"{prefix}.{suffix}.weight"] = rng.normal(0, .08, (4, 4)).astype(np.float32)
        params[f"{prefix}.{suffix}.bias"] = rng.normal(0, .01, 4).astype(np.float32)
    for i in range(4):
        prefix = f"decoder.up_blocks.{i}"
        resnet(prefix + ".resnets.0", shortcut=(i == 0))
        if i != 3:
            conv(prefix + ".upsamplers.0.conv")
    norm("decoder.conv_norm_out")
    conv("decoder.conv_out", 3, 4)
    params["decoder.conv_out.bias"] += 1.05  # Exercise clamp as well as raw RGB.
    latents = rng.normal(0, .3, (2, 8, 2, 3)).astype(np.float32)
    return {key: jnp.asarray(value) for key, value in params.items()}, jnp.asarray(latents), config


@pytest.mark.parametrize("precision", ["fp32", "bfloat16", "float16"])
def test_reference_paths_delegate_all_options(monkeypatch, precision):
    sentinel = object()
    delegate = Mock(return_value=sentinel)
    monkeypatch.setattr(vae, "decode_latents", delegate)
    params, latents, config = {}, object(), {}
    # FP32 must stay native even if a valid stage was selected.
    stages = ("mid",) if precision == "fp32" else ()
    actual = mixed.decode_latents(
        params, latents, config, precision=precision, low_precision_stages=stages,
        tiling=True, slicing=True, attention_chunk_size=7, tile_overlap_factor=.5, clamp=True,
    )
    assert actual is sentinel
    delegate.assert_called_once_with(
        params, latents, config, tiling=True, slicing=True, attention_chunk_size=7,
        tile_overlap_factor=.5, clamp=True,
    )


@pytest.mark.parametrize("kwargs", [
    {"precision": "bf16"}, {"precision": None},
    {"low_precision_stages": ("up4",)}, {"low_precision_stages": ("Mid",)},
    {"low_precision_stages": "mid"}, {"low_precision_stages": None},
])
def test_invalid_precision_or_stage_fails_before_decode(monkeypatch, kwargs):
    delegate = Mock(side_effect=AssertionError("invalid request reached decoder"))
    monkeypatch.setattr(vae, "decode_latents", delegate)
    with pytest.raises(ValueError):
        mixed.decode_latents({}, None, **kwargs)
    delegate.assert_not_called()


def test_fp32_jitted_fixture_exactly_matches_vendor(fixture):
    params, latents, config = fixture
    for tiling in (False, True):
        kwargs = dict(tiling=tiling, slicing=True, attention_chunk_size=3, clamp=True)
        reference = jax.jit(lambda p, z: vae.decode_latents(p, z, config, **kwargs))(params, latents)
        actual = jax.jit(lambda p, z: mixed.decode_latents(p, z, config, **kwargs))(params, latents)
        np.testing.assert_array_equal(actual, reference)
        assert actual.dtype == jnp.float32


def prefix_stage(prefix):
    if prefix in ("post_quant_conv", "decoder.conv_in"):
        return "input"
    if prefix.startswith("decoder.mid_block."):
        return "mid"
    if prefix.startswith("decoder.up_blocks."):
        return "up" + prefix.split(".")[2]
    assert prefix == "decoder.conv_out"
    return "output"


@pytest.mark.parametrize("stage", mixed.STAGES)
def test_only_selected_stage_uses_low_precision_and_results_are_fp32(monkeypatch, fixture, stage):
    params, latents, config = fixture
    calls = []
    conv, linear = mixed._conv2d, mixed._linear
    norm, silu = vae._group_norm, vae._silu

    def record(function):
        def wrapped(params, x, prefix, dtype, *conv_options):
            assert x.dtype == jnp.float32
            calls.append((prefix, dtype))
            result = function(params, x, prefix, dtype, *conv_options)
            assert result.dtype == jnp.float32
            return result
        return wrapped

    def fp32_norm(params, x, *args):
        assert x.dtype == jnp.float32
        result = norm(params, x, *args)
        assert result.dtype == jnp.float32
        return result

    def fp32_silu(x):
        assert x.dtype == jnp.float32
        result = silu(x)
        assert result.dtype == jnp.float32
        return result

    monkeypatch.setattr(mixed, "_conv2d", record(conv))
    monkeypatch.setattr(mixed, "_linear", record(linear))
    monkeypatch.setattr(vae, "_group_norm", fp32_norm)
    monkeypatch.setattr(vae, "_silu", fp32_silu)
    result = mixed.decode_latents(
        params, latents[:1, :, :1, :1], config,
        precision="bfloat16", low_precision_stages=(stage,), attention_chunk_size=3,
    )
    assert result.dtype == jnp.float32 and np.isfinite(result).all()
    assert {prefix_stage(prefix) for prefix, _ in calls} == set(mixed.STAGES)
    for prefix, dtype in calls:
        assert dtype == (jnp.bfloat16 if prefix_stage(prefix) == stage else jnp.float32)
    assert all(value.dtype == jnp.float32 for value in params.values())


@pytest.mark.parametrize("precision", ["bfloat16", "float16"])
def test_mixed_decode_supports_jit_tiling_slicing_clamp(fixture, precision):
    params, latents, config = fixture
    kwargs = dict(precision=precision, low_precision_stages=mixed.STAGES,
                  tiling=True, slicing=True, attention_chunk_size=3)
    decode = jax.jit(lambda p, z: mixed.decode_latents(p, z, config, **kwargs))
    actual = decode(params, latents)
    separately = jnp.concatenate([decode(params, latents[i:i+1]) for i in range(2)], axis=0)
    np.testing.assert_allclose(actual, separately, rtol=2e-7, atol=2e-7)
    reference = vae.decode_latents(params, latents, config, tiling=True, slicing=True, attention_chunk_size=3)
    assert actual.dtype == jnp.float32
    assert actual.shape == reference.shape == (2, 3, 32, 48)
    assert np.isfinite(actual).all()
    assert np.max(np.abs(np.asarray(actual) - np.asarray(reference))) < .04
    # Compare clamp to raw under the same eager execution; mixing eager/JIT
    # changes low-precision rounding through fusion and is a different test.
    eager_raw = mixed.decode_latents(params, latents, config, **kwargs)
    clamped = mixed.decode_latents(params, latents, config, clamp=True, **kwargs)
    np.testing.assert_array_equal(clamped, jnp.clip(eager_raw, -1, 1))
    assert float(jnp.max(actual)) > 1


def test_low_conv_output_is_rounded_before_fp32_bias(fixture):
    params, _, _ = fixture
    x = jnp.full((1, 2, 3, 3), .12345, jnp.float32)
    actual = mixed._conv2d(params, x, "post_quant_conv", jnp.bfloat16)
    weight = params["post_quant_conv.weight"]
    expected = jax.lax.conv_general_dilated(
        x.astype(jnp.bfloat16), weight.astype(jnp.bfloat16),
        window_strides=(1, 1), padding=((0, 0), (0, 0)),
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
        precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.bfloat16,
    ).astype(jnp.float32) + params["post_quant_conv.bias"][None, :, None, None]
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == jnp.float32


def test_attention_scores_and_value_reduction_remain_float32(monkeypatch, fixture):
    params, _, _ = fixture
    matmul = jnp.matmul
    reductions = []

    def checked_matmul(left, right, **kwargs):
        if left.ndim == right.ndim == 3:
            assert left.dtype == right.dtype == jnp.float32
            reductions.append((left.shape, right.shape))
        result = matmul(left, right, **kwargs)
        assert result.dtype == jnp.float32
        return result

    monkeypatch.setattr(jnp, "matmul", checked_matmul)
    result = mixed._attention(
        params, jnp.ones((1, 4, 2, 3), jnp.float32),
        "decoder.mid_block.attentions.0", 2, 3, jnp.bfloat16,
    )
    assert result.dtype == jnp.float32 and np.isfinite(result).all()
    assert len(reductions) == 2


def test_tiling_threshold_and_blending_match_vendor_without_quantization(monkeypatch, fixture):
    params, latents, config = fixture
    # Exercise the copied tile dispatcher with the unchanged native direct
    # decoder so semantic differences cannot be hidden by reduced precision.
    monkeypatch.setattr(
        mixed, "_decode_direct",
        lambda p, z, c, chunk, dtype, stages, *conv_options: vae._decode_direct(p, z, c, chunk),
    )
    for packed in (latents[:1, :, :2, :2], latents):
        actual = mixed.decode_latents(
            params, packed, config, precision="bfloat16",
            low_precision_stages=("input",), tiling=True, slicing=True, attention_chunk_size=3,
        )
        reference = vae.decode_latents(
            params, packed, config, tiling=True, slicing=True, attention_chunk_size=3,
        )
        np.testing.assert_array_equal(actual, reference)
