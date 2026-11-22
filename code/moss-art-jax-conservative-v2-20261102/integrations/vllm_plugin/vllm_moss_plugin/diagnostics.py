"""Opt-in numeric probes; never print tokens, prompts, or feature contents.

Enable only for a tiny eager diagnostic run with MOSS_VLLM_DEBUG=1. Patches
are process-local and do not modify any vLLM files. GPU probes synchronize;
production preprocessing must leave this option unset.
"""

from __future__ import annotations

from functools import wraps
import hashlib
import json
import os

_installed = False


def _eligible(tensor):
    import torch

    if not isinstance(tensor, torch.Tensor) or tensor.ndim == 0:
        return False
    if not 0 < tensor.shape[0] <= 512:
        return False
    if torch.compiler.is_compiling():
        return False
    if tensor.is_cuda and torch.cuda.is_current_stream_capturing():
        return False
    return True


def _emit(stage, tensor, **extra):
    if not _eligible(tensor):
        return
    import torch

    value = tensor.detach().float()
    # Ignore the extractor's singleton auxiliary-layer axis in the digest.
    # Hash raw BF16 bytes, preserving token order and every feature bit.
    contiguous = tensor.detach().contiguous()
    content_sha256 = hashlib.sha256(
        contiguous.view(torch.uint8).cpu().numpy().tobytes()
    ).hexdigest()
    finite = bool(torch.isfinite(value).all().item())
    record = {
        "event": "moss_vllm_debug",
        "stage": stage,
        "pid": os.getpid(),
        "rank": os.environ.get("RANK", os.environ.get("LOCAL_RANK", "unknown")),
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "finite": finite,
        "max_abs": float(value.abs().max().item()),
        "rms": float(value.square().mean().sqrt().item()),
        "nonzero": int(value.count_nonzero().item()),
        "content_sha256": content_sha256,
        **extra,
    }
    print(json.dumps(record, allow_nan=True), flush=True)


def install():
    """Install once in this process; no effect unless explicitly enabled."""
    global _installed
    if _installed or os.environ.get("MOSS_VLLM_DEBUG") != "1":
        return

    from .model import MossQwen3_5ForConditionalGeneration
    from vllm.v1.spec_decode.extract_hidden_states import ExtractHiddenStatesProposer
    from vllm.model_executor.models.extract_hidden_states import CacheOnlyAttentionImpl
    from vllm.distributed.kv_transfer.kv_connector.v1 import (
        example_hidden_states_connector as connector,
    )

    original_forward = MossQwen3_5ForConditionalGeneration.forward

    @wraps(original_forward)
    def forward(self, *args, **kwargs):
        result = original_forward(self, *args, **kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            _emit("model_normalized", result[0])
            if isinstance(result[1], list):
                for index, tensor in enumerate(result[1]):
                    _emit("model_auxiliary", tensor, auxiliary_index=index)
        return result

    MossQwen3_5ForConditionalGeneration.forward = forward
    original_propose = ExtractHiddenStatesProposer.propose

    @wraps(original_propose)
    def propose(self, *args, **kwargs):
        targets = kwargs.get("target_hidden_states")
        if targets is None and len(args) >= 3:
            targets = args[2]
        if isinstance(targets, list):
            for index, tensor in enumerate(targets):
                _emit("proposer_target_before", tensor, auxiliary_index=index)
        result = original_propose(self, *args, **kwargs)
        if isinstance(targets, list) and targets:
            _emit("proposer_buffer_after", self.hidden_states[:targets[0].shape[0]])
        return result

    ExtractHiddenStatesProposer.propose = propose
    original_update = CacheOnlyAttentionImpl.do_kv_cache_update

    @wraps(original_update)
    def update(self, layer, to_cache, kv_cache, slot_mapping):
        extra = {}
        if _eligible(to_cache):
            slots = slot_mapping.detach()
            extra = {
                "slot_min": int(slots.min().item()),
                "slot_max": int(slots.max().item()),
                "valid_slots": int((slots >= 0).count_nonzero().item()),
                "cache_shape": list(kv_cache.shape),
                "cache_stride": list(kv_cache.stride()),
            }
        _emit("cache_write_input", to_cache, **extra)
        result = original_update(self, layer, to_cache, kv_cache, slot_mapping)
        if _eligible(to_cache):
            slots = slot_mapping[slot_mapping >= 0]
            if slots.numel():
                selected = kv_cache[
                    slots // kv_cache.shape[1], slots % kv_cache.shape[1]
                ]
                _emit("cache_write_selected_after", selected, **extra)
        return result

    CacheOnlyAttentionImpl.do_kv_cache_update = update
    original_extract = connector.extract_from_kv_cache

    @wraps(original_extract)
    def extract(kv_cache, slot_mapping, num_tokens):
        result = original_extract(kv_cache, slot_mapping, num_tokens)
        extra = {"cache_shape": list(kv_cache.shape), "num_tokens": int(num_tokens)}
        if _eligible(result):
            slots = slot_mapping[:num_tokens]
            extra.update(slot_min=int(slots.min().item()), slot_max=int(slots.max().item()))
        _emit("connector_extracted", result, **extra)
        return result

    connector.extract_from_kv_cache = extract
    _installed = True
    print(json.dumps({"event": "moss_vllm_debug_installed", "pid": os.getpid()}), flush=True)
