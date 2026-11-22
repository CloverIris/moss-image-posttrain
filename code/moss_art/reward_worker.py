"""Independent, inference-only SigLIP image-text reward process (no JAX).

The trainer launches this interpreter separately. PyTorch is used solely as
an external black-box reward implementation under torch.inference_mode().
All model loading is local_files_only and trust_remote_code=False.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

DEFAULT_MODEL = "$HOME/.cache/huggingface/hub/models--google--siglip2-so400m-patch14-384/snapshots/e8e487298228002f3d8a82e0cd5c8ea9c567f57f"


def run(request_path, output_path):
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    if request.get("version") != 1 or not isinstance(request.get("items"), list) or not request["items"]:
        raise ValueError("Expected reward protocol version 1 and nonempty items")
    config = request.get("config", {})
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor
    torch.set_num_threads(int(config.get("threads", 8)))
    device = str(config.get("device", "cpu"))
    if device != "cpu" and not device.startswith("cuda"):
        raise ValueError("SigLIP reward device must be cpu or explicit cuda device")
    model_path = config.get("model_path", DEFAULT_MODEL)
    model = AutoModel.from_pretrained(model_path, local_files_only=True,
                                      trust_remote_code=False, dtype=torch.float32).eval().to(device)
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    metric = config.get("metric", "cosine")
    if metric not in {"cosine", "logit"}:
        raise ValueError("SigLIP metric must be cosine or logit")
    batch_size = int(config.get("batch_size", 1))
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    scores = []
    with torch.inference_mode():
        for start in range(0, len(request["items"]), batch_size):
            items = request["items"][start:start + batch_size]
            images = []
            for item in items:
                with Image.open(item["image"]) as image:
                    images.append(image.convert("RGB"))
            inputs = processor(text=[item["prompt"] for item in items], images=images,
                               padding="max_length", truncation=True, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            result = model(**inputs)
            if metric == "logit":
                values = result.logits_per_image.diag()
            else:
                image_features = torch.nn.functional.normalize(result.image_embeds.float(), dim=-1)
                text_features = torch.nn.functional.normalize(result.text_embeds.float(), dim=-1)
                values = (image_features * text_features).sum(dim=-1)
            if not torch.isfinite(values).all():
                raise ValueError("SigLIP returned nonfinite rewards")
            scores.extend(values.cpu().tolist())
    Path(output_path).write_text(json.dumps({"scores": scores, "backend": "siglip", "metric": metric,
                                           "model_path": str(model_path), "device": device}, allow_nan=False),
                                encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.request, args.output)
