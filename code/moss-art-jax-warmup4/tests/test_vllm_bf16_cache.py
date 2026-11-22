"""Portable BF16 bits, final-normalized contract, and training reader checks."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from moss_art.data import CACHE_VERSION, PromptCache, atomic_json, sha256_file, text_key
from moss_art.vllm_contract import storage_contract, encode_embedding_arrays, decode_embedding_arrays


def test_bf16_round_to_even_and_signed_zero():
    values = np.asarray([1.00390625, 1.01171875, -1.00390625, 0., -0., 2**-133], np.float32)
    bits = encode_embedding_arrays(values, "bfloat16")
    np.testing.assert_array_equal(bits, np.asarray([0x3f80, 0x3f82, 0xbf80, 0, 0x8000, 1], np.uint16))
    decoded = decode_embedding_arrays(bits, storage_contract("bfloat16"))
    np.testing.assert_array_equal(decoded, np.asarray([1., 1.015625, -1., 0., -0., 2**-133], np.float32))
    assert np.signbit(decoded[4])


def test_bf16_numpy_archive_preserves_bits(tmp_path):
    bits = np.arange(65536, dtype=np.uint16)
    path = tmp_path / "all-bit-patterns.npz"
    np.savez_compressed(path, embedding=bits)
    with np.load(path, allow_pickle=False) as archive:
        assert archive["embedding"].dtype == np.uint16
        np.testing.assert_array_equal(archive["embedding"], bits)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, np.finfo(np.float32).max])
def test_invalid_or_overflow_features_rejected(value):
    with pytest.raises(ValueError):
        encode_embedding_arrays(np.asarray([value], np.float32), "bfloat16")


def test_cache_reader_decodes_bf16_to_existing_training_dtype(tmp_path):
    contract = storage_contract("bfloat16")
    value = np.full((1, 3, 4096), 1.015625, np.float32)
    ids = np.asarray([[11, 12, 13]], np.int32)
    file = tmp_path / "features.npz"
    np.savez_compressed(file, embedding=encode_embedding_arrays(value, "bfloat16"),
                        attention_mask=np.ones((1, 3), bool), input_ids=ids)
    key = text_key("fixture")
    manifest = {"schema_version": CACHE_VERSION, "encoder": "moss-vllm-qwen3.5",
                "model_identity": {"fingerprint": "fixture"}, "conditioning_contract": contract,
                "entries": {key: {"file": file.name, "sha256": sha256_file(file),
                                  "shape": list(value.shape), "token_count": 3,
                                  "input_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
                                  "embedding_storage_dtype": "bfloat16"}}}
    atomic_json(tmp_path / "manifest.json", manifest)
    loaded, mask = PromptCache(tmp_path).load_text("fixture")
    assert loaded.dtype == np.float32 and mask.all()
    np.testing.assert_array_equal(loaded, value)
    with np.load(file, allow_pickle=False) as archive:
        assert archive["embedding"].nbytes == value.nbytes // 2


def test_storage_contracts_reject_dtype_confusion():
    with pytest.raises(ValueError):
        decode_embedding_arrays(np.ones((1, 3, 4096), np.float32), storage_contract("bfloat16"))
    with pytest.raises(ValueError):
        decode_embedding_arrays(np.ones((1, 3, 4096), np.uint16), storage_contract("float32"))
