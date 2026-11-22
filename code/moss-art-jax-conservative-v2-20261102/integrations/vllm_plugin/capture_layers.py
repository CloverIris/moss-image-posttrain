"""Capture a synthetic token sequence's layer outputs with an eager engine."""

import argparse
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="NPZ with synthetic input_ids")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import numpy as np

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    ids = np.load(args.input)["input_ids"]
    os.environ["MOSS_VLLM_LAYER_DUMP_DIR"] = str(output / "layers")
    os.environ["MOSS_VLLM_LAYER_DUMP_TOKENS"] = str(len(ids))
    sys.path.insert(0, "$CODE_ROOT/moss-art-jax")
    from moss_art.vllm_qwen import VllmPrefill

    engine = VllmPrefill(
        "$MODEL_ROOT/MOSS-Image-Base",
        {
            "tensor_parallel_size": 2,
            "max_num_batched_tokens": 1024,
            "max_num_seqs": 2,
            "gpu_memory_utilization": 0.8,
            "enforce_eager": True,
            "processor_dir": "$CODE_ROOT/moss-image/reference-lab/processor",
            "max_reference_pixels": 262144,
        },
        output / "engine",
        max_sequence_length=512,
    )
    features = engine.encode_token_lists([ids])[0]
    np.savez(output / "exported.npz", embedding=features)
    print(json.dumps({"completed": True, "tokens": len(ids), "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
