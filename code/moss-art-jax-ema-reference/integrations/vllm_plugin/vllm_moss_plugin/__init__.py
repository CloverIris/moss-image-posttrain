"""Register MOSS without importing its model or initializing CUDA."""

MODEL_ARCHITECTURE = "MossQwen3_5ForConditionalGeneration"
MODEL_IMPORT = "vllm_moss_plugin.model:MossQwen3_5ForConditionalGeneration"


def register() -> None:
    """vLLM runs this entry point in the parent and each worker process."""
    from vllm import ModelRegistry

    ModelRegistry.register_model(MODEL_ARCHITECTURE, MODEL_IMPORT)
    import os

    if os.environ.get("MOSS_VLLM_DEBUG") == "1":
        from .diagnostics import install

        install()
    if os.environ.get("MOSS_VLLM_LAYER_DUMP_DIR"):
        from .layer_diagnostics import install as install_layer_diagnostics

        install_layer_diagnostics()
