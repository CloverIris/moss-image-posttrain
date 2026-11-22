import jax
import jax.numpy as jnp
import numpy as np
import pytest
from safetensors.numpy import load_file

from moss_art import lora
from moss_art.model import MossModel


def test_initialization_and_target_selection():
    model = MossModel.tiny()
    a = model.init_lora(99)
    b = model.init_lora(99)
    assert len(a) == 2 * 6 * 2
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
        if key.endswith(lora.UP):
            assert np.all(a[key] == 0)
        else:
            assert np.any(a[key] != 0)
    attention = lora.init_lora(model.base, rank=4, targets="attention")
    assert len(attention) == 2 * 4 * 2
    assert lora.parameter_count(attention) > 0
    with pytest.raises(ValueError):
        lora.init_lora(model.base, rank=0)


@pytest.mark.parametrize("style", ["native", "diffusion_model", "lora_unet"])
def test_kohaku_export_roundtrip_and_alpha_scaling(tmp_path, style):
    model = MossModel.tiny(rank=2, alpha=1)
    a = model.init_lora()
    a = {k: v + jnp.float32(0.01) for k,v in a.items()}
    filename = tmp_path / "adapter.safetensors"
    lora.export_safetensors(filename, a, alpha=1.0, style=style)
    b = lora.import_safetensors(filename, model.init_lora(), alpha=2.0)
    for key in a:
        expected = a[key] * (0.5 if key.endswith(lora.UP) else 1)
        np.testing.assert_array_equal(b[key], expected)
    for key, value in load_file(str(filename)).items():
        assert value.dtype == np.float32
    # Different loader alpha still represents exactly the same delta W.
    first = next(k[:-len(lora.DOWN)] for k in a if k.endswith(lora.DOWN))
    np.testing.assert_allclose((a[first+lora.UP] @ a[first+lora.DOWN]) * 0.5,
                               b[first+lora.UP] @ b[first+lora.DOWN], atol=1e-6)


def test_import_rejects_rank_and_target_mismatch(tmp_path):
    model = MossModel.tiny()
    filename = tmp_path / "adapter.safetensors"
    lora.export_safetensors(filename, model.init_lora(), alpha=2)
    wrong = lora.init_lora(model.base, rank=3)
    with pytest.raises(ValueError, match="Shape mismatch"):
        lora.import_safetensors(filename, wrong, alpha=2)
    wrong_targets = lora.init_lora(model.base, rank=2, targets="attention")
    with pytest.raises(ValueError, match="Unsupported or untargeted"):
        lora.import_safetensors(filename, wrong_targets, alpha=2)
