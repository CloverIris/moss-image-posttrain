"""Reuse vLLM's Qwen3.5 implementation with the MOSS checkpoint contract.

MOSS stores the backbone as ``language_model.*`` and omits an LM head. The
model view enables tied embeddings for the one discarded output token; no
language or visual checkpoint tensors are rewritten or copied here.

The official extractor normally saves decoder outputs before the final norm.
MOSS instead consumes the native final normalized hidden state. Replacing the
single auxiliary output with the already normalized main output avoids doing
another norm on a separately rounded BF16 residual sum.
"""

from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
)
from vllm.model_executor.models.utils import WeightsMapper
from vllm_moss_plugin.pp_compat import (
    install as install_pp_compat,
    is_extract_hidden_states_pp,
)


class MossQwen3_5ForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    """MOSS's 32-layer Qwen3.5 backbone, preserving multimodal processing."""

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "language_model.": "language_model.model.",
        }
    )

    def __init__(self, *args, **kwargs):
        # Worker-side construction runs after the runner and PP group exist;
        # keep registration/model-class inspection free of eager runner imports.
        vllm_config = kwargs.get("vllm_config")
        if is_extract_hidden_states_pp(vllm_config):
            install_pp_compat(vllm_config)
        super().__init__(*args, **kwargs)

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        layers = tuple(layers)
        if layers != (32,):
            raise ValueError(
                "MOSS prefill requires exactly auxiliary layer (32,); "
                f"received {layers!r}. Its exported value is final normalized."
            )
        num_layers = self.config.text_config.num_hidden_layers
        if num_layers != 32:
            raise ValueError(
                "The MOSS prefill adapter requires a 32-layer backbone; "
                f"the model config contains {num_layers}."
            )
        super().set_aux_hidden_state_layers(layers)

    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
        **kwargs,
    ):
        result = super().forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        if (
            isinstance(result, tuple)
            and len(result) == 2
            and isinstance(result[1], list)
            and result[1]
        ):
            normalized, auxiliary = result
            if len(auxiliary) != 1:
                raise ValueError(
                    "MOSS prefill expected one auxiliary hidden state; "
                    f"received {len(auxiliary)}."
                )
            return normalized, [normalized]
        # Profiling and non-final pipeline stages may have no auxiliary output.
        return result
