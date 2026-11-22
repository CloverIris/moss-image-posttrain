"""Host-only backend dispatch and immutable MOSS conditioning-contract checks."""
import copy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from moss_art.config import DEFAULTS, load_config
from moss_art.data import CACHE_VERSION, PromptCache, atomic_json, sha256_file, text_key
from moss_art.vllm_contract import CONDITIONING_CONTRACT


def config_file(tmp_path, **preparation):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"preparation": {"backend": "vllm", **preparation}}))
    return path


def test_default_backend_preserves_jax_and_vllm_is_offline(tmp_path):
    path = tmp_path / "default.json"
    path.write_text("{}")
    assert load_config(path)["preparation"]["backend"] == "jax"
    c = load_config(config_file(tmp_path))
    assert c["preparation"]["vllm"]["python"].endswith("/vllm_env/bin/python")
    with pytest.raises(ValueError, match="offline"):
        load_config(config_file(tmp_path, async_prefetch=True))
    with pytest.raises(ValueError, match="backend"):
        load_config(config_file(tmp_path, backend="unknown"))


@pytest.mark.parametrize("settings", [
    {"devices": []}, {"devices": [0, 0]}, {"devices": [True, 1]},
    {"tensor_parallel_size": 1}, {"tensor_parallel_size": 0},
    {"batch_size": 0}, {"max_num_seqs": 0}, {"max_num_batched_tokens": 0},
    {"gpu_memory_utilization": 0.0}, {"gpu_memory_utilization": 1.01},
    {"max_reference_pixels": 1}, {"python": " "}, {"processor_dir": ""},
])
def test_invalid_vllm_configuration_fails_before_accelerator_import(tmp_path, settings):
    with pytest.raises(ValueError):
        load_config(config_file(tmp_path, vllm=settings))


def test_vllm_requires_cuda_moss(tmp_path):
    path = config_file(tmp_path)
    value = json.loads(path.read_text())
    value["runtime"] = {"platform": "cpu"}
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="MOSS model and CUDA"):
        load_config(path)


def test_vllm_token_budget_includes_one_discarded_output_token(tmp_path):
    path = config_file(tmp_path, vllm={"max_num_batched_tokens": 4096})
    with pytest.raises(ValueError, match=r"context_length \+ 1"):
        load_config(path)
    path = config_file(tmp_path, vllm={"max_num_batched_tokens": 4097})
    assert load_config(path)["preparation"]["vllm"]["max_num_batched_tokens"] == 4097
    # A disabled optional backend must not restrict the existing JAX encoder.
    path = config_file(tmp_path, backend="jax", vllm={"max_num_batched_tokens": 1})
    assert load_config(path)["preparation"]["backend"] == "jax"


def test_cli_dispatches_vllm_before_jax_runtime_setup(tmp_path):
    from moss_art.cli import main
    path = config_file(tmp_path)
    with patch("moss_art.prepare.launch_vllm_prepare", return_value="finished") as call, \
            patch("moss_art.runtime.configure_process") as configure:
        assert main(["prepare-text", "--config", str(path)]) == "finished"
    configure.assert_not_called()
    assert call.call_args.args[0]["preparation"]["backend"] == "vllm"


def test_launcher_preserves_config_and_uses_independent_interpreter(tmp_path):
    from moss_art.prepare import launch_vllm_prepare
    c = load_config(config_file(tmp_path))
    before = copy.deepcopy(c)
    observed = {}

    def run(command, **kwargs):
        observed.update(command=command, kwargs=kwargs,
                        config=json.loads(Path(command[-1]).read_text()))
        return "child exited"

    with patch("moss_art.runtime.preflight") as preflight, patch("subprocess.run", side_effect=run):
        assert launch_vllm_prepare(c) == "child exited"
    assert c == before
    assert observed["command"][:4] == [c["preparation"]["vllm"]["python"], "-m", "moss_art.vllm_prepare", "--config"]
    assert observed["config"] == before
    assert observed["kwargs"]["check"] is True
    assert preflight.call_args.args[0]["runtime"]["devices"] == [1, 2]
    assert not Path(observed["command"][-1]).exists()


def test_training_ensure_cache_consumes_offline_cache_without_starting_qwen(tmp_path):
    from moss_art.prepare import ensure_cache
    c = load_config(config_file(tmp_path))
    c["data"]["cache"] = str(tmp_path / "offline")
    Path(c["data"]["cache"]).mkdir()
    (Path(c["data"]["cache"]) / "manifest.json").write_text("{}")
    with patch("moss_art.prepare.launch_vllm_prepare") as call, \
            patch("moss_art.data.PromptCache") as cache:
        cache.return_value.manifest = {"complete": True}
        ensure_cache(c)
    call.assert_not_called()
    cache.assert_called_once_with(c["data"]["cache"], model_dir=c["model"]["path"])
    c["preparation"]["auto_prepare"] = False
    with patch("moss_art.prepare.launch_vllm_prepare") as call, patch("moss_art.data.PromptCache"):
        ensure_cache(c)
    call.assert_not_called()


def test_training_vllm_requires_complete_standalone_cache(tmp_path):
    from moss_art.prepare import ensure_cache
    c = load_config(config_file(tmp_path))
    c["data"]["cache"] = str(tmp_path / "offline")
    with pytest.raises(ValueError, match="prefill_qwen.py"):
        ensure_cache(c)
    Path(c["data"]["cache"]).mkdir()
    (Path(c["data"]["cache"]) / "manifest.json").write_text("{}")
    with patch("moss_art.data.PromptCache") as cache:
        cache.return_value.manifest = {"complete": False}
        with pytest.raises(ValueError, match="incomplete"):
            ensure_cache(c)


def synthetic_cache(tmp_path, *, encoder="moss-vllm-qwen3.5", hidden=4096,
                    contract=None, dtype=np.float32, ids=True, token_count=3, all_zero=False):
    folder = tmp_path / "cache"
    folder.mkdir()
    file = folder / "features.npz"
    arrays = {"embedding": np.full((1, 3, hidden), 0.0 if all_zero else 0.25, dtype=dtype),
              "attention_mask": np.ones((1, 3), bool)}
    if ids:
        arrays["input_ids"] = np.asarray([[1, 2, 3]], dtype=np.int32)
    np.savez(file, **arrays)
    manifest = {
        "schema_version": CACHE_VERSION, "encoder": encoder,
        "model_identity": {"fingerprint": "synthetic-only"},
        "conditioning_contract": copy.deepcopy(CONDITIONING_CONTRACT if contract is None else contract),
        "entries": {text_key("fixture"): {"file": file.name, "sha256": sha256_file(file),
                                           "shape": [1, 3, hidden], "token_count": token_count}},
    }
    atomic_json(folder / "manifest.json", manifest)
    return folder


def test_vllm_cache_full_sequence_float32_is_accepted(tmp_path):
    folder = synthetic_cache(tmp_path)
    embedding, mask = PromptCache(folder).load_text("fixture")
    assert embedding.shape == (1, 3, 4096) and embedding.dtype == np.float32
    assert mask.all()


def test_vllm_cache_all_zero_features_cannot_train(tmp_path):
    folder = synthetic_cache(tmp_path, all_zero=True)
    with pytest.raises(ValueError, match="all-zero features cannot be used for training"):
        PromptCache(folder).load_text("fixture")


@pytest.mark.parametrize("field,value", [
    ("output_norm", False), ("vllm_layer_ids", [31]),
    ("preserve_all_prompt_tokens", False), ("hidden_size", 8192),
    ("version", True), ("storage_dtype", "bfloat16"),
])
def test_vllm_cache_contract_mismatch_is_rejected(tmp_path, field, value):
    contract = copy.deepcopy(CONDITIONING_CONTRACT)
    contract[field] = value
    folder = synthetic_cache(tmp_path, contract=contract)
    with pytest.raises(ValueError, match="conditioning contract"):
        PromptCache(folder)


@pytest.mark.parametrize("settings", [
    {"hidden": 4}, {"dtype": np.float16}, {"ids": False}, {"token_count": 2},
])
def test_vllm_cache_invalid_hidden_or_storage_is_rejected(tmp_path, settings):
    folder = synthetic_cache(tmp_path, **settings)
    with pytest.raises(ValueError, match="vLLM"):
        PromptCache(folder).load_text("fixture")


def test_legacy_jax_cache_has_no_new_contract_requirement(tmp_path):
    folder = synthetic_cache(tmp_path, encoder="moss-native-jax-qwen3.5", hidden=4, ids=False, all_zero=True)
    manifest = json.loads((folder / "manifest.json").read_text())
    manifest.pop("conditioning_contract")
    atomic_json(folder / "manifest.json", manifest)
    assert PromptCache(folder).load_text("fixture")[0].shape == (1, 3, 4)
