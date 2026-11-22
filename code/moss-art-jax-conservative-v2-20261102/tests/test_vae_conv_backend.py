"""CPU integration tests for VAE convolution backend routing and precision."""
from pathlib import Path
import sys
import types
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from moss_art import vae_precision as mixed
from moss_art.config import load_config, validate
from vendor.moss_jax import vae
from test_vae_precision import fixture, prefix_stage

BACKENDS = ("cudnn_bounded", "tiled_gemm")


def _proxy(monkeypatch, backend):
    calls = []

    def conv2d_nchw(x, weight, **options):
        calls.append((x.shape, weight.shape, x.dtype, weight.dtype, options))
        assert x.dtype == weight.dtype
        kh, kw = weight.shape[-2:]
        return jax.lax.conv_general_dilated(
            x, weight, (1, 1), ((kh // 2, kh // 2), (kw // 2, kw // 2)),
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
            precision=jax.lax.Precision.HIGHEST, preferred_element_type=x.dtype,
        )

    name = "moss_art.cudnn_conv" if backend == "cudnn_bounded" else "moss_art.tiled_conv"
    module = types.ModuleType(name)
    module.conv2d_nchw = conv2d_nchw
    monkeypatch.setitem(sys.modules, name, module)
    return calls


@pytest.mark.parametrize("backend", BACKENDS)
def test_fp32_backend_reaches_every_conv_including_shortcuts(monkeypatch, fixture, backend):
    params, latents, config = fixture
    latents = latents[:1, :, :1, :1]
    reference = vae.decode_latents(params, latents, config, attention_chunk_size=3)
    calls = _proxy(monkeypatch, backend)
    prefixes = []
    original = mixed._conv2d

    def record(p, x, prefix, dtype, selected_backend, cap):
        prefixes.append(prefix)
        assert dtype == jnp.float32
        assert selected_backend == backend and cap == 17
        return original(p, x, prefix, dtype, selected_backend, cap)

    monkeypatch.setattr(mixed, "_conv2d", record)
    monkeypatch.setattr(vae, "_conv2d",
                        lambda *_a, **_k: pytest.fail("explicit backend used native convolution"))
    actual = mixed.decode_latents(
        params, latents, config, precision="fp32", conv_backend=backend,
        conv_workspace_mib=17, attention_chunk_size=3,
    )
    expected = {key[:-7] for key, value in params.items()
                if key.endswith(".weight") and value.ndim == 4}
    assert set(prefixes) == expected
    assert len(prefixes) == len(expected) == len(calls) == 19
    assert all(call[2:4] == (jnp.float32, jnp.float32) for call in calls)
    if backend == "cudnn_bounded":
        assert all(call[4]["workspace_mib"] == 17 for call in calls)
    np.testing.assert_array_equal(actual, reference)


@pytest.mark.parametrize("backend", BACKENDS)
def test_selected_stage_precision_and_sensitive_math_remain_correct(monkeypatch, fixture, backend):
    params, latents, config = fixture
    calls = _proxy(monkeypatch, backend)
    selected = {"input", "up1", "output"}
    prefixes = []
    original, norm, silu = mixed._conv2d, vae._group_norm, vae._silu

    def record(p, x, prefix, dtype, selected_backend, cap):
        assert x.dtype == jnp.float32
        expected = jnp.float16 if prefix_stage(prefix) in selected else jnp.float32
        assert dtype == expected
        prefixes.append(prefix)
        value = original(p, x, prefix, dtype, selected_backend, cap)
        assert value.dtype == jnp.float32
        return value

    def checked_norm(p, x, *args):
        assert x.dtype == jnp.float32
        result = norm(p, x, *args)
        assert result.dtype == jnp.float32
        return result

    def checked_silu(x):
        assert x.dtype == jnp.float32
        result = silu(x)
        assert result.dtype == jnp.float32
        return result

    monkeypatch.setattr(mixed, "_conv2d", record)
    monkeypatch.setattr(vae, "_group_norm", checked_norm)
    monkeypatch.setattr(vae, "_silu", checked_silu)
    result = mixed.decode_latents(
        params, latents[:1, :, :1, :1], config, precision="float16",
        low_precision_stages=selected, conv_backend=backend,
        conv_workspace_mib=23, attention_chunk_size=3,
    )
    assert len(calls) == len(prefixes) == 19
    assert {call[2] for call in calls} == {np.dtype("float32"), np.dtype("float16")}
    assert result.dtype == jnp.float32 and np.isfinite(result).all()
    assert all(value.dtype == jnp.float32 for value in params.values())


@pytest.mark.parametrize("backend", BACKENDS)
def test_low_conv_output_rounds_before_fp32_bias(monkeypatch, fixture, backend):
    params, _, _ = fixture
    _proxy(monkeypatch, backend)
    x = jnp.full((1, 2, 3, 5), .12345, jnp.float32)
    weight = params["post_quant_conv.weight"]
    expected = jax.lax.conv_general_dilated(
        x.astype(jnp.float16), weight.astype(jnp.float16), (1, 1), ((0, 0), (0, 0)),
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
        precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float16,
    ).astype(jnp.float32) + params["post_quant_conv.bias"][None, :, None, None]
    actual = mixed._conv2d(params, x, "post_quant_conv", jnp.float16, backend, 7)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == jnp.float32


@pytest.mark.parametrize("backend", BACKENDS)
def test_backend_survives_slicing_tiling_and_clamp(monkeypatch, fixture, backend):
    params, latents, config = fixture
    calls = _proxy(monkeypatch, backend)
    original = mixed._decode_direct
    tiles = []

    def record(p, z, c, chunk, dtype, stages, selected_backend, cap):
        tiles.append(z.shape)
        assert selected_backend == backend and cap == 31 and chunk == 3
        assert z.shape[0] == 1
        return original(p, z, c, chunk, dtype, stages, selected_backend, cap)

    monkeypatch.setattr(mixed, "_decode_direct", record)
    kwargs = dict(tiling=True, slicing=True, attention_chunk_size=3,
                  tile_overlap_factor=.5, clamp=True)
    actual = mixed.decode_latents(
        params, latents, config, conv_backend=backend, conv_workspace_mib=31, **kwargs,
    )
    reference = vae.decode_latents(params, latents, config, **kwargs)
    np.testing.assert_array_equal(actual, reference)
    assert actual.shape == (2, 3, 32, 48)
    assert len(tiles) == 12
    assert {s[-2:] for s in tiles} == {(4, 4), (4, 2), (2, 4), (2, 2)}
    assert len(calls) == len(tiles) * 19


def test_native_default_and_empty_stages_stay_bitwise_native(monkeypatch, fixture):
    params, latents, config = fixture
    bounded_calls = _proxy(monkeypatch, "cudnn_bounded")
    tiled_calls = _proxy(monkeypatch, "tiled_gemm")
    latents = latents[:1, :, :1, :1]
    options = dict(tiling=True, slicing=True, attention_chunk_size=3, clamp=True)
    native = jax.jit(lambda p, z: vae.decode_latents(p, z, config, **options))(params, latents)
    for precision in ("fp32", "float16", "bfloat16"):
        result = jax.jit(lambda p, z: mixed.decode_latents(
            p, z, config, precision=precision, low_precision_stages=(), **options,
        ))(params, latents)
        np.testing.assert_array_equal(result, native)
    assert not bounded_calls and not tiled_calls


def _config(backend="cudnn_bounded"):
    config = load_config(Path(__file__).resolve().parents[1] / "configs/smoke_cpu.yaml")
    config["model"]["vae_conv_backend"] = backend
    config["runtime"]["platform"] = "cuda"
    return config


@pytest.mark.parametrize("workspace", [-1, 513, True, 1.5, "256", None])
def test_bad_workspace_rejected_by_config_and_decoder(workspace):
    config = _config()
    config["model"]["vae_conv_workspace_mib"] = workspace
    with pytest.raises(ValueError):
        validate(config)
    with pytest.raises(ValueError):
        mixed.decode_latents({}, None, conv_backend="cudnn_bounded", conv_workspace_mib=workspace)


@pytest.mark.parametrize("workspace", [0, 256, 512])
def test_bounded_workspace_boundaries_are_valid(workspace):
    config = _config()
    config["model"]["vae_conv_workspace_mib"] = workspace
    validate(config)


@pytest.mark.parametrize("change", ["cpu_platform", "cpu_decode", "bfloat16"])
def test_bounded_rejects_unsupported_platform_or_precision(change):
    config = _config()
    if change == "cpu_platform":
        config["runtime"]["platform"] = "cpu"
    elif change == "cpu_decode":
        config["reward"]["cpu_decode"] = True
    else:
        config["model"]["vae_precision"] = "bfloat16"
        config["model"]["vae_low_precision_stages"] = ["up1"]
    with pytest.raises(ValueError):
        validate(config)


def test_tiled_gemm_allows_cpu_decode():
    config = _config("tiled_gemm")
    config["runtime"]["platform"] = "cpu"
    config["reward"]["cpu_decode"] = True
    validate(config)


def test_unknown_backend_fails_before_decode():
    with pytest.raises(ValueError):
        validate(_config("missing"))
    with pytest.raises(ValueError):
        mixed.decode_latents({}, None, conv_backend="missing")


def test_real_tiled_decoder_jaxpr_has_no_convolutions_and_matches_native(fixture):
    params, latents, config = fixture
    latents = latents[:1, :, :1, :1]
    def decode(p, z):
        return mixed.decode_latents(
            p, z, config, conv_backend="tiled_gemm", attention_chunk_size=3,
        )
    program = str(jax.make_jaxpr(decode)(params, latents))
    assert "conv_general_dilated" not in program
    assert "dot_general" in program
    actual = jax.jit(decode)(params, latents)
    native = jax.jit(lambda p, z: vae.decode_latents(
        p, z, config, attention_chunk_size=3,
    ))(params, latents)
    np.testing.assert_allclose(actual, native, rtol=2e-6, atol=2e-6)
    assert actual.dtype == jnp.float32 and np.isfinite(actual).all()
