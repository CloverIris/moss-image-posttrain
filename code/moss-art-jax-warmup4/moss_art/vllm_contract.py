"""Host-only feature contract shared by preprocessing and JAX cache readers."""

import copy

import numpy as np

ENCODER = "moss-vllm-qwen3.5"
CONDITIONING_CONTRACT = {
    "version": 1,
    "hf_output": "last_hidden_state",
    "vllm_layer_ids": [32],
    "output_norm": "qwen3_5_final_rmsnorm",
    "hidden_size": 4096,
    "preserve_all_prompt_tokens": True,
    "prompt_template": "moss_qwen_thinking_v1",
    "source_dtype": "bfloat16",
    "storage_dtype": "float32",
}


def storage_contract(storage_dtype="float32"):
    contract = copy.deepcopy(CONDITIONING_CONTRACT)
    if storage_dtype == "bfloat16":
        # NumPy's .npy format does not preserve ml_dtypes' BF16 descriptor.
        # Explicit uint16 bits are portable and require no accelerator import.
        contract.update(version=2, storage_dtype="bfloat16_uint16")
    elif storage_dtype != "float32":
        raise ValueError("hidden-state storage_dtype must be float32 or bfloat16")
    return contract


def validate_conditioning_contract(contract):
    import json
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"))
    for dtype in ("float32", "bfloat16"):
        if canonical(contract) == canonical(storage_contract(dtype)):
            return dtype
    raise ValueError("vLLM prompt cache conditioning contract mismatch")


def encode_embedding_arrays(embedding, storage_dtype):
    value = np.asarray(embedding, dtype=np.float32)
    if not np.isfinite(value).all():
        raise ValueError("Nonfinite hidden-state cache value")
    if storage_dtype == "float32":
        return value
    if storage_dtype != "bfloat16":
        raise ValueError("hidden-state storage_dtype must be float32 or bfloat16")
    bits = value.view(np.uint32)
    # Round-to-nearest, ties-to-even, including values supplied by host tools.
    rounded = bits + np.uint32(0x7fff) + ((bits >> np.uint32(16)) & np.uint32(1))
    result = (rounded >> np.uint32(16)).astype(np.uint16)
    if not np.isfinite((result.astype(np.uint32) << 16).view(np.float32)).all():
        raise ValueError("Hidden-state value overflows BF16 storage")
    return result


def decode_embedding_arrays(array, contract):
    dtype = validate_conditioning_contract(contract)
    array = np.asarray(array)
    if dtype == "bfloat16":
        if array.dtype != np.uint16:
            raise ValueError("BF16 hidden-state cache must contain uint16 bits")
        return (array.astype(np.uint32) << np.uint32(16)).view(np.float32)
    if array.dtype != np.float32:
        raise ValueError("FP32 hidden-state cache must contain float32 values")
    return array


def format_prompt(prompt):
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    return "<|im_start|>user\n" + prompt.strip() + "<|im_end|>\n<|im_start|>assistant\n<think>\n"
