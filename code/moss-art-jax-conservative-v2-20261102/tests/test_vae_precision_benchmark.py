"""CPU-only metrics, reference mapping and input validation for the VAE probe."""
import json
from pathlib import Path
import numpy as np
import pytest

from scripts.benchmark_vae_precision import (
    parse_indices, parse_stages, load_latents, load_reference, comparison_metrics,
    rgb_pixels, rgb_ssim, sha256, comparison_panel,
)


def test_indices_and_stages_are_explicit():
    assert parse_indices("3,1", 4) == [3, 1]
    assert parse_indices("all", 4) == [0, 1, 2, 3]
    assert parse_stages("input,up2,output") == ("input", "up2", "output")
    for value in ("", "-1", "0,0", "4", "x"):
        with pytest.raises(ValueError):
            parse_indices(value, 4)
    for value in ("foo", "up1,up1"):
        with pytest.raises(ValueError):
            parse_stages(value)


def test_identity_comparison_and_nonfinite_detection():
    raw = np.linspace(-1.5, 1.5, 3 * 16 * 16, dtype=np.float32).reshape(1, 3, 16, 16)
    result = comparison_metrics(raw, raw)
    assert result["raw_finite"] and result["raw_bitwise_equal"]
    assert result["raw_relative_l2"] == 0 and result["raw_max_abs"] == 0
    assert result["pixels_equal"] and result["pixel_mae_0_255"] == 0
    assert result["pixel_psnr_db"] is None and result["pixel_psnr_infinite"]
    assert result["ssim"]["rgb_mean"] == 1
    bad = raw.copy()
    bad[0, 0, 0, 0] = np.nan
    assert comparison_metrics(bad, raw) == {"raw_finite": False}
    with pytest.raises(ValueError, match="Reference shape differs"):
        comparison_metrics(raw, raw[..., :-1])


def test_clipped_pixel_metrics_and_raw_difference_are_separate():
    reference = np.full((1, 3, 16, 16), 2, np.float32)
    raw = np.full_like(reference, 3)
    result = comparison_metrics(raw, reference)
    assert result["raw_relative_l2"] == .5
    assert result["raw_max_abs"] == 1
    assert result["pixels_equal"] and result["pixel_mae_0_255"] == 0
    assert rgb_pixels(raw).dtype == np.uint8
    # Every candidate pixel changes by 127 relative to the reference's 128.
    reference = np.zeros_like(reference)
    raw = np.ones_like(reference)
    result = comparison_metrics(raw, reference)
    assert result["raw_relative_l2"] is None
    assert result["pixel_mae_0_255"] == 127
    assert result["pixel_mse"] == 127 ** 2
    assert result["pixel_psnr_db"] == pytest.approx(10 * np.log10(255 ** 2 / 127 ** 2))


def test_ssim_matches_direct_gaussian_valid_windows():
    rng = np.random.default_rng(93)
    a = rng.integers(0, 256, (13, 13, 3), dtype=np.uint8)
    b = rng.integers(0, 256, (13, 13, 3), dtype=np.uint8)
    axis = np.arange(-5, 6, dtype=np.float64)
    kernel = np.exp(-(axis ** 2) / (2 * 1.5 ** 2))
    kernel /= kernel.sum()
    weights = kernel[:, None] * kernel[None, :]
    expected = []
    c1, c2 = (.01 * 255) ** 2, (.03 * 255) ** 2
    for channel in range(3):
        scores = []
        for y in range(3):
            for x in range(3):
                pa = a[y:y+11, x:x+11, channel].astype(np.float64)
                pb = b[y:y+11, x:x+11, channel].astype(np.float64)
                ma, mb = np.sum(weights * pa), np.sum(weights * pb)
                va, vb = np.sum(weights * pa * pa) - ma ** 2, np.sum(weights * pb * pb) - mb ** 2
                covariance = np.sum(weights * pa * pb) - ma * mb
                scores.append(((2 * ma * mb + c1) * (2 * covariance + c2)) /
                              ((ma * ma + mb * mb + c1) * (va + vb + c2)))
        expected.append(np.mean(scores))
    actual = rgb_ssim(a, b)
    np.testing.assert_allclose(actual["per_channel"], expected, atol=1e-12)
    assert actual["rgb_mean"] == pytest.approx(np.mean(expected), abs=1e-12)


def test_reference_directory_maps_original_indices_and_checks_identity(tmp_path):
    reference = tmp_path / "reference"
    reference.mkdir()
    rows = []
    for index in range(4):
        raw_path = f"sample-{index}.raw.npy"
        np.save(reference / raw_path, np.full((1, 3, 16, 16), index, np.float32))
        rows.append({"sample_index": index, "raw_path": raw_path})
    report = {"status": "passed", "input_sha256": "abc", "precision": "fp32",
              "source_hashes": {"source.py": "hash"}, "repeats": [{"samples": rows}]}
    (reference / "summary.json").write_text(json.dumps(report))
    arrays, metadata = load_reference(reference, "abc", [3, 1], 4)
    np.testing.assert_array_equal(arrays[3], 3)
    np.testing.assert_array_equal(arrays[1], 1)
    assert metadata["summary_sha256"] == sha256(reference / "summary.json")
    with pytest.raises(ValueError, match="SHA256"):
        load_reference(reference, "wrong", [3], 4)
    with pytest.raises(ValueError, match="lacks"):
        load_reference(reference, "abc", [5], 6)
    report["status"] = "running"
    (reference / "summary.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="completed"):
        load_reference(reference, "abc", [3], 4)


def test_input_npy_npz_and_reference_raw_validation(tmp_path):
    values = np.zeros((4, 128, 1, 1), np.float32)
    path = tmp_path / "latents.npz"
    np.savez(path, endpoints=values)
    np.testing.assert_array_equal(load_latents(path), values)
    raw = np.arange(4 * 3 * 16 * 16, dtype=np.float32).reshape(4, 3, 16, 16)
    path = tmp_path / "raw.npy"
    np.save(path, raw)
    selected, info = load_reference(path, "unchecked", [3], 4)
    np.testing.assert_array_equal(selected[3], raw[3:4])
    assert info["source_identity_verified"] is False
    np.save(path, values[:, :3])
    with pytest.raises(ValueError, match="packed"):
        load_latents(path)
    np.save(path, values * np.nan)
    with pytest.raises(ValueError, match="nonfinite"):
        load_latents(path)


def test_comparison_panel_preserves_full_resolution(tmp_path):
    from PIL import Image
    raw = np.zeros((1, 3, 16, 16), np.float32)
    path = tmp_path / "panel.png"
    comparison_panel(raw, raw + .01, path, "bf16 up3")
    with Image.open(path) as panel:
        assert panel.size == (48, 48)
