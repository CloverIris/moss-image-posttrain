"""Synthetic, nonrepeating MOSS conditions; never reads the user's dataset."""
import json
from pathlib import Path
import time

import numpy as np


def main():
    from moss_art.config import load_config
    from moss_art.vllm_prepare import select_devices
    c = load_config("configs/preparation/vllm_qwen_prefill.json")
    select_devices(c["preparation"]["vllm"]["devices"])
    from moss_art.vllm_qwen import VllmPrefill
    from PIL import Image, ImageDraw
    base = Path("$MOSS_ROOT/audits/vllm_prefill_20261024/representative")
    root = base / "numeric_validation"
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(411)
    yy, xx = np.mgrid[:256, :256]
    pixels = np.stack([xx, yy, 100 + 60*np.sin(xx/23) + 40*np.cos(yy/19)], axis=-1)
    pixels = np.uint8(np.clip(pixels + rng.normal(0, 9, pixels.shape), 0, 255))
    image = Image.fromarray(pixels)
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 40, 95, 160), fill=(170, 75, 45), outline=(245, 220, 120), width=4)
    draw.ellipse((120, 100, 230, 210), fill=(40, 120, 180), outline=(240, 250, 240), width=3)
    draw.line((0, 250, 210, 20), fill=(240, 90, 150), width=7)
    image.save(root / "texture.png")
    natural = (
        "prompt:\nCreate a detailed anime illustration of an adult botanist visiting a hillside railway station in early autumn. "
        "Her copper hair is tied with a narrow teal ribbon, and she wears a cream jacket over a dark green dress with practical leather boots. "
        "She carries a worn notebook, a brass magnifying glass, and a small basket containing fern leaves, yellow asters, and a folded map. "
        "A sleepy orange cat sits on a wooden bench beside her. The station has pale stone walls, a red tiled roof, hanging flower boxes, "
        "an old circular clock, and tall windows reflecting the valley. Beyond the platform, silver rails curve through terraced fields "
        "toward a distant village and a mountain ridge partly hidden by mist. Place the character slightly left of center in a three-quarter "
        "standing pose, looking down at a pressed flower between the notebook pages. A gentle breeze lifts loose hair strands and the edge "
        "of her scarf. Late afternoon sunlight crosses the scene from the right, leaving cool violet shadows beneath the roof and warm "
        "highlights along the basket, cheek, and jacket seams. Use precise facial drawing, expressive amber eyes, delicate colored line art, "
        "subtle cel shading, soft watercolor textures in the scenery, restrained film grain, and a balanced palette of ochre, sage, blue, "
        "and terracotta. Show scattered fallen leaves, small reflections in a shallow puddle, weathered signposts, and distant birds, "
        "while keeping a clear silhouette and comfortable breathing space around the face and hands. The mood is quiet curiosity and "
        "the anticipation of a short journey. Avoid lettering over the character.\nnegative prompt:\nwatermark, jpeg artifact, bad anatomy"
    )
    tags = (
        "prompt:\nbest quality, highres, anime style, 1woman, solo, adult, botanist, copper hair, long hair, "
        "teal ribbon, amber eyes, looking down, gentle smile, cream jacket, green dress, leather boots, scarf, "
        "standing, three-quarter view, holding notebook, pressed flower, woven basket, fern, yellow aster, "
        "brass magnifying glass, folded map, railway station, hillside, autumn, orange cat, wooden bench, "
        "pale stone wall, red roof, hanging flower box, circular clock, tall window, reflection, platform, "
        "curving rails, terraced field, distant village, mountain ridge, mist, late afternoon, warm sunlight, "
        "cool shadow, rim light, breeze, windblown hair, fallen leaves, puddle, weathered signpost, flying birds, "
        "detailed face, expressive eyes, delicate lineart, colored outlines, cel shading, watercolor background, "
        "film grain, ochre palette, sage green, muted blue, terracotta, clear silhouette, balanced composition, "
        "environmental storytelling, quiet atmosphere, natural hands, depth of field, detailed fabric, "
        "stitched seams, basket handle, loose strands, stone paving, glass highlights, distant perspective\n"
        "negative prompt:\nworst quality, watermark, jpeg artifact, bad anatomy"
    )
    examples = [
        {"id": "natural", "prompt": natural, "images": []},
        {"id": "tags", "prompt": tags, "images": []},
        {"id": "texture_image", "prompt": "prompt:\nAn adult artist studying this colorful textured geometric painting, seated in a bright studio, delicate anime line art, warm afternoon light.\nnegative prompt:\nwatermark", "images": [str(root / "texture.png")]},
    ]
    (root / "examples.json").write_text(json.dumps(examples, indent=2))
    c["data"]["cache"] = str(base)
    (base / "config.json").write_text(json.dumps(c, indent=2))
    start = time.perf_counter()
    engine = VllmPrefill(c["model"]["path"], c["preparation"]["vllm"], base / "engine", max_sequence_length=512)
    report = {"initialization_seconds": time.perf_counter()-start, "backend": engine.metadata, "examples": []}
    for group in [examples[:2], examples[2:]]:
        start = time.perf_counter()
        features, ids = engine.encode_requests(group)
        for ex, feature, tokens in zip(group, features, ids):
            np.savez(root / f"{ex['id']}_vllm.npz", embedding=feature, input_ids=tokens)
        report["examples"].append({"ids": [ex["id"] for ex in group], "token_counts": [len(t) for t in ids], "seconds": time.perf_counter()-start})
    (root / "vllm.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
