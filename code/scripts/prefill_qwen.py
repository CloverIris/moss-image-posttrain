#!/usr/bin/env python3
"""Standalone BF16 Qwen hidden-state producer; never starts ART training."""
from pathlib import Path
import argparse
import os
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    from moss_art.config import load_config
    config = load_config(args.config)
    python = config["preparation"]["vllm"]["python"]
    if os.path.abspath(sys.executable) != os.path.abspath(python):
        if not Path(python).is_file():
            raise FileNotFoundError(f"Missing vLLM Python: {python}")
        os.execv(python, [python, str(Path(__file__).resolve()), *sys.argv[1:]])
    from moss_art.vllm_prepare import main as prefill_main
    return prefill_main(sys.argv[1:])


if __name__ == "__main__":
    main()
