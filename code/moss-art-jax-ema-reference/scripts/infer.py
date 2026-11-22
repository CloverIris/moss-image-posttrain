"""Run one collection and save selected latent points plus rendered images."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moss_art.stage_pipeline import stage_main

if __name__ == "__main__":
    stage_main("infer")
