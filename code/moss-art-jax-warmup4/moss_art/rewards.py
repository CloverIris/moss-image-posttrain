"""Detached image rewards. Neural rewards execute in a separate process.

`toy` is mean image brightness ONLY, useful for pipeline checks and never a
quality metric. `siglip` uses a local pretrained SigLIP image/text model.
`http` POSTs PNG pixels and prompts to a configured remote reward URL.
`command` exchanges JSON files with any explicitly configured executable.
No Torch object, computation graph, or gradient enters the JAX trainer.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

import numpy as np

DEFAULT_SIGLIP_MODEL = "$HOME/.cache/huggingface/hub/models--google--siglip2-so400m-patch14-384/snapshots/e8e487298228002f3d8a82e0cd5c8ea9c567f57f"
DEFAULT_REWARD_PYTHON = "$CONDA_PREFIX/envs/torch_env/bin/python"


def as_image_batch(images):
    images = np.asarray(images)
    if images.ndim == 3:
        images = images[None]
    if images.ndim != 4 or images.shape[-1] != 3 or images.shape[0] < 1:
        raise ValueError("Expected nonempty RGB images with shape [B,H,W,3]")
    if images.dtype == np.uint8:
        return images
    if not np.issubdtype(images.dtype, np.floating) or not np.isfinite(images).all():
        raise ValueError("Images must be uint8 or finite floats in [0,1]")
    if images.min() < 0 or images.max() > 1:
        raise ValueError("Floating image pixels must lie in [0,1]")
    return np.rint(images * 255).astype(np.uint8)


def _validate_scores(values, count):
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (count,) or not np.isfinite(result).all():
        raise ValueError(f"Reward must return {count} finite scalar scores, got shape {result.shape}")
    return result


def _command_argv(command, request, output):
    if isinstance(command, str):
        command = shlex.split(command)
    if not isinstance(command, (list, tuple)) or not command or not all(isinstance(v, str) for v in command):
        raise ValueError("reward command must be a nonempty argv list or shell-quoted command string")
    placeholders = any("{request}" in part or "{output}" in part for part in command)
    result = [part.replace("{request}", str(request)).replace("{output}", str(output)) for part in command]
    return result if placeholders else result + ["--request", str(request), "--output", str(output)]


def score_images(images, prompts, config=None):
    """Return finite float32 scores [B]. Higher is better for every backend.

    Command request: {version:1,items:[{index,prompt,image}],config:{...}}.
    The executable receives --request PATH --output PATH (or configured
    {request}/{output} argv substitutions), and writes {scores:[number,...]}.
    Paths are absolute PNGs. Ordering must be preserved. Failures raise,
    rather than silently substituting zero rewards. Shell execution is off.
    """
    config = dict(config or {"kind": "toy"})
    images = as_image_batch(images)
    prompts = list(prompts)
    if len(prompts) != len(images) or not all(isinstance(p, str) for p in prompts):
        raise ValueError("One string prompt is required per image")
    kind = config.get("kind", "toy")
    if kind == "toy":
        # Deliberately simple and documented: optimizing this makes images white.
        return images.astype(np.float32).mean(axis=(1, 2, 3)) / 255.0
    if kind == "http":
        from .http_reward import score_http
        return score_http(images, prompts, config)
    if kind not in ("siglip", "command"):
        raise ValueError(f"Unknown reward kind: {kind}")
    from PIL import Image
    with tempfile.TemporaryDirectory(prefix="moss-art-reward-") as temporary:
        directory = Path(temporary)
        items = []
        for index, (pixels, prompt) in enumerate(zip(images, prompts)):
            file = directory / f"image-{index:06d}.png"
            Image.fromarray(pixels).save(file)
            items.append({"index": index, "prompt": prompt, "image": str(file)})
        request, output = directory / "request.json", directory / "result.json"
        # Do not pass arbitrary trainer state or credentials into a reward request.
        worker_config = {key: config[key] for key in ("model_path", "device", "batch_size", "metric", "threads")
                         if key in config}
        request.write_text(json.dumps({"version": 1, "items": items, "config": worker_config}), encoding="utf-8")
        environment = os.environ.copy()
        if kind == "siglip":
            worker = Path(__file__).with_name("reward_worker.py").resolve()
            python = config.get("python", DEFAULT_REWARD_PYTHON if Path(DEFAULT_REWARD_PYTHON).exists() else sys.executable)
            argv = [str(python), str(worker), "--request", str(request), "--output", str(output)]
            environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
            if config.get("device", "cpu") == "cpu":
                environment["CUDA_VISIBLE_DEVICES"] = ""
        else:
            argv = _command_argv(config.get("command"), request, output)
        result = subprocess.run(argv, env=environment, text=True, capture_output=True,
                                timeout=float(config.get("timeout", 900)), check=False)
        if result.returncode:
            raise RuntimeError(f"Reward subprocess failed ({result.returncode}): {result.stderr[-8000:]}")
        if not output.is_file():
            raise ValueError(f"Reward subprocess did not produce {output.name}; stdout: {result.stdout[-2000:]}")
        response = json.loads(output.read_text(encoding="utf-8"))
        if not isinstance(response, dict) or "scores" not in response:
            raise ValueError("Reward response must contain a scores array")
        return _validate_scores(response["scores"], len(images))
