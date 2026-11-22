"""Explicitly enabled eager-only layer dumps for synthetic alignment tests."""

from functools import wraps
import json
import os
from pathlib import Path

_installed = False


def _eligible(tensor):
    import torch
    from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

    return (
        not torch.compiler.is_compiling()
        and isinstance(tensor, torch.Tensor)
        and tensor.ndim == 2
        and tensor.shape[0] == int(os.environ.get("MOSS_VLLM_LAYER_DUMP_TOKENS", "478"))
        and not torch.cuda.is_current_stream_capturing()
        and get_tensor_model_parallel_rank() == 0
    )


def _save(name, **arrays):
    import numpy as np

    destination = Path(os.environ["MOSS_VLLM_LAYER_DUMP_DIR"])
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / f"{name}.npz"
    temporary = destination / f"{name}.tmp.npz"
    np.savez(temporary, **arrays)
    os.replace(temporary, output)
    print(json.dumps({"event": "moss_vllm_layer_dump", "name": name, "path": str(output)}), flush=True)


def install():
    global _installed
    if _installed or not os.environ.get("MOSS_VLLM_LAYER_DUMP_DIR"):
        return
    from vllm.model_executor.models.qwen3_5 import Qwen3_5DecoderLayer
    from .model import MossQwen3_5ForConditionalGeneration

    original_layer_forward = Qwen3_5DecoderLayer.forward

    @wraps(original_layer_forward)
    def layer_forward(self, *args, **kwargs):
        result = original_layer_forward(self, *args, **kwargs)
        hidden, residual = result
        if _eligible(hidden):
            summed_fp32 = hidden.float() + residual.float()
            _save(
                f"layer_{self.layer_idx:02d}",
                sum_fp32=summed_fp32.detach().cpu().numpy(),
                sum_bf16=summed_fp32.to(hidden.dtype).float().detach().cpu().numpy(),
            )
        return result

    Qwen3_5DecoderLayer.forward = layer_forward
    original_forward = MossQwen3_5ForConditionalGeneration.forward

    @wraps(original_forward)
    def forward(self, *args, **kwargs):
        result = original_forward(self, *args, **kwargs)
        hidden = result[0] if isinstance(result, tuple) else result
        if _eligible(hidden):
            _save("final_normalized", hidden=hidden.float().detach().cpu().numpy())
        return result

    MossQwen3_5ForConditionalGeneration.forward = forward
    _installed = True
