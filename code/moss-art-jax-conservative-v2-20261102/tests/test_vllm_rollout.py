"""Host-only rollout JSON configuration to offline-Qwen engine contract."""
import json
from unittest.mock import patch

import pytest

from moss_art.config import DEFAULTS as ART_DEFAULTS
from moss_art.rollout_only import DEFAULTS, engine_config, load_config


def test_rollout_keeps_original_jax_async_defaults(tmp_path):
    path = tmp_path / "rollout.json"
    path.write_text("{}")
    c = load_config(path)
    assert c["preparation"]["backend"] == "jax"
    assert c["preparation"]["async_prefetch"] is True
    assert ART_DEFAULTS["preparation"]["async_prefetch"] is False
    assert DEFAULTS["preparation"] is not ART_DEFAULTS["preparation"]
    assert DEFAULTS["preparation"]["vllm"] is not ART_DEFAULTS["preparation"]["vllm"]


def test_rollout_json_preserves_offline_vllm_engine_settings(tmp_path):
    path = tmp_path / "rollout.json"
    path.write_text(json.dumps({
        "model": {"path": "$MODEL_ROOT/MOSS-Image-Base"},
        "preparation": {"backend": "vllm", "async_prefetch": False,
                        "vllm": {"devices": [1, 2], "tensor_parallel_size": 2, "batch_size": 3}},
        "data": {"context_length": 512, "short_context_length": 128, "cache": str(tmp_path / "cache")},
        "rollout": {"count": 4},
    }))
    c = load_config(path)
    engine = engine_config(c)
    assert engine["preparation"] == c["preparation"]
    assert engine["preparation"]["vllm"]["batch_size"] == 3
    assert engine["preparation"]["vllm"]["devices"] == [1, 2]
    assert engine["preparation"]["backend"] == "vllm"
    assert engine["preparation"]["async_prefetch"] is False
    assert engine["model"]["base_precision"] == "bf16"
    assert engine["data"]["context_length"] == 512
    assert engine["preparation"] is not c["preparation"]
    # run() already removes the rollout-only precision annotation before this call.
    engine["model"].pop("base_precision")
    from moss_art.prepare import ensure_cache
    from pathlib import Path
    folder = Path(engine["data"]["cache"])
    folder.mkdir()
    (folder / "manifest.json").write_text("{}")
    with patch("moss_art.prepare.launch_vllm_prepare") as launch, patch("moss_art.data.PromptCache") as cache:
        cache.return_value.manifest = {"complete": True}
        ensure_cache(engine)
    launch.assert_not_called()


@pytest.mark.parametrize("settings,match", [
    ({"backend": "vllm"}, "offline"),
    ({"backend": "unknown"}, "backend"),
    ({"backend": "vllm", "async_prefetch": False,
      "vllm": {"max_num_batched_tokens": 4096}}, r"context_length \+ 1"),
])
def test_rollout_rejects_vllm_conflicts_before_accelerator_start(tmp_path, settings, match):
    path = tmp_path / "rollout.json"
    path.write_text(json.dumps({"preparation": settings}))
    with pytest.raises(ValueError, match=match):
        load_config(path)
