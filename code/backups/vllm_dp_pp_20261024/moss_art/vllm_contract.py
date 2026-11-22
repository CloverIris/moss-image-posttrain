"""Host-only feature contract shared by preprocessing and JAX cache readers."""

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


def format_prompt(prompt):
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    return "<|im_start|>user\n" + prompt.strip() + "<|im_end|>\n<|im_start|>assistant\n<think>\n"
