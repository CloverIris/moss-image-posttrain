#!/usr/bin/env python3
"""Standalone Kawai worker; importing validation helpers never imports Torch.

Input: {items:[{id,image,prompt_id}]}.
Output: full logits/probabilities and the historical expected-class score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time


MODEL_ID = 'kawaimasa/kawai-aesthetic-scorer-convnextv2'
CLASS_LABELS = ['SS_tier', 'S_tier', 'A_tier', 'B_tier', 'C_tier']
CLASS_WEIGHTS = [1., .9, .5, -.5, -1.]
PREPROCESS = {
    'size': 768, 'resize': 'aspect_preserving_LANCZOS',
    'scaled_dimensions': 'int(original_dimension * 768 / max(width,height))',
    'padding': 'centered_black_letterbox', 'padding_rgb': [0, 0, 0],
    'channels': 'RGB', 'pixel_scale': 'uint8 / 255',
    'mean': [.485, .456, .406], 'std': [.229, .224, .225],
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                      allow_nan=False).encode()).hexdigest()


def validate_class_contract(id2label, expected_labels, weights):
    if expected_labels != CLASS_LABELS:
        raise ValueError('expected_id2label must preserve the pinned Kawai class order')
    if weights != CLASS_WEIGHTS or any(type(x) not in (int, float) or not math.isfinite(x) for x in weights):
        raise ValueError('class_weights must equal [1,0.9,0.5,-0.5,-1] in pinned class order')
    if not isinstance(id2label, dict):
        raise ValueError('Model config must contain id2label')
    normalized = {str(k): v for k, v in id2label.items()}
    required = {str(i): label for i, label in enumerate(expected_labels)}
    if normalized != required:
        raise ValueError(f'Kawai id2label mismatch: expected {required}, received {normalized}')
    return required


def validate_settings(cfg, config_dir):
    if type(cfg.get('device')) is not int or cfg['device'] not in (0, 1, 2, 3):
        raise ValueError('device must be physical GPU index 0..3; GPU4 is reserved')
    batch_size = cfg.get('batch_size', 2)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    dtype = cfg.get('dtype', 'float32')
    if dtype not in ('float32', 'bfloat16'):
        raise ValueError('dtype must be float32 or bfloat16; no silent fallback')
    threads = cfg.get('threads', 4)
    if type(threads) is not int or threads < 1:
        raise ValueError('threads must be a positive integer')
    def resolve(value):
        path = Path(value).expanduser()
        return (path if path.is_absolute() else config_dir / path).resolve()
    for field in ('input', 'output', 'model_path'):
        if not isinstance(cfg.get(field), str) or not cfg[field]:
            raise ValueError(f'{field} must be a nonempty path string')
    paths = {field: resolve(cfg[field]) for field in ('input', 'output', 'model_path')}
    if paths['input'] == paths['output']:
        raise ValueError('Input and output paths must differ')
    model_config = paths['model_path'] / 'config.json'
    model_weights = paths['model_path'] / 'model.safetensors'
    if not model_config.is_file() or not model_weights.is_file():
        raise FileNotFoundError('Expected local Kawai config.json and model.safetensors')
    labels = cfg.get('expected_id2label', CLASS_LABELS)
    weights = cfg.get('class_weights', CLASS_WEIGHTS)
    normalized_labels = validate_class_contract(json.loads(model_config.read_text())['id2label'], labels, weights)
    return dict(cfg, **{k: str(v) for k, v in paths.items()}, batch_size=batch_size,
                dtype=dtype, threads=threads, expected_id2label=labels,
                id2label=normalized_labels, class_weights=weights)


def load_items(path):
    path = Path(path)
    value = json.loads(path.read_text(encoding='utf-8-sig'))
    rows = value.get('items') if isinstance(value, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError('Input must contain a nonempty items list')
    seen, result = set(), []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('Every item must be an object')
        identity = row.get('id')
        if not isinstance(identity, str) or not identity or identity in seen:
            raise ValueError('Every item needs a unique nonempty string id')
        seen.add(identity)
        prompt_id = row.get('prompt_id')
        if type(prompt_id) not in (str, int) or prompt_id == '':
            raise ValueError('Every item needs a string or integer prompt_id')
        if not isinstance(row.get('image'), str) or not row['image']:
            raise ValueError('Every item needs a nonempty image path')
        image = Path(row['image']).expanduser()
        image = (image if image.is_absolute() else path.parent / image).resolve()
        if not image.is_file():
            raise FileNotFoundError(f'Image missing for id={identity}: {image}')
        result.append(dict(row, image=str(image)))
    return result


def process_image(image, normalize):
    # Importing this worker/validating input does not load imaging/ML libraries.
    from PIL import Image
    image = image.convert('RGB')
    width, height = image.size
    if width < 1 or height < 1:
        raise ValueError('Image dimensions must be positive')
    scale = 768 / max(width, height)
    nw, nh = int(width * scale), int(height * scale)
    if min(nw, nh) < 1:
        raise ValueError('Aspect ratio is too extreme for the pinned 768px preprocessing')
    image = image.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new('RGB', (768, 768), (0, 0, 0))
    canvas.paste(image, ((768 - nw) // 2, (768 - nh) // 2))
    return normalize(canvas)


def score_items(items, cfg, trace_path):
    # GPU visibility is selected before Torch or Transformers is imported.
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg['device'])
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    import torchvision.transforms as transforms
    from PIL import Image
    from transformers import AutoModelForImageClassification
    torch.set_num_threads(cfg['threads'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA Kawai inference, but CUDA is unavailable')
    device, desired_dtype = 'cuda:0', {'float32':torch.float32, 'bfloat16':torch.bfloat16}[cfg['dtype']]
    normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=PREPROCESS['mean'], std=PREPROCESS['std'])])
    loading_started = time.perf_counter()
    model = AutoModelForImageClassification.from_pretrained(cfg['model_path'], local_files_only=True,
                                                            use_safetensors=True, torch_dtype=desired_dtype).to(device).eval()
    id2label = validate_class_contract(model.config.id2label, cfg['expected_id2label'], cfg['class_weights'])
    actual_dtype = str(next(model.parameters()).dtype)
    if next(model.parameters()).dtype != desired_dtype:
        raise RuntimeError(f'Model dtype changed unexpectedly: {actual_dtype}')
    torch.cuda.synchronize()
    loading_seconds = time.perf_counter() - loading_started
    weights = torch.tensor(cfg['class_weights'], device=device, dtype=torch.float32)
    effective_weights = weights.cpu().tolist()
    results = []
    inference_started = time.perf_counter()
    with trace_path.open('x', encoding='utf-8') as trace, torch.inference_mode():
        for start in range(0, len(items), cfg['batch_size']):
            batch = items[start:start+cfg['batch_size']]
            pixels = []
            for row in batch:
                with Image.open(row['image']) as image:
                    pixels.append(process_image(image, normalize))
            logits = model(pixel_values=torch.stack(pixels).to(device=device, dtype=desired_dtype)).logits.float()
            if tuple(logits.shape) != (len(batch), 5) or not bool(torch.isfinite(logits).all()):
                raise ValueError('Kawai returned invalid logits shape or nonfinite values')
            probabilities = torch.softmax(logits, dim=-1)
            values = (probabilities * weights).sum(dim=-1)
            logits_rows = logits.cpu().tolist()
            probability_rows = probabilities.cpu().tolist()
            scores = values.cpu().tolist()
            for item, value, raw_logits, probs in zip(batch, scores, logits_rows, probability_rows):
                if not math.isfinite(value):
                    raise ValueError('Nonfinite Kawai score')
                row = dict(id=item['id'], prompt_id=item['prompt_id'], image=item['image'],
                           image_sha256=sha256(item['image']), kawai=value, logits=raw_logits,
                           probabilities=probs, class_weights=effective_weights, id2label=id2label)
                trace.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n'); trace.flush()
                results.append(row)
            print(json.dumps(dict(event='kawai_progress', images=len(results), total=len(items))), flush=True)
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter()-inference_started
    return results, dict(dtype=actual_dtype, load_seconds=loading_seconds,
                         inference_seconds=inference_seconds, batch_size=cfg['batch_size'],
                         timing_scope='preprocessing, model forward, score extraction, image hashing and row trace',
                         class_weights=effective_weights)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    settings = validate_settings(json.loads(args.config.read_text(encoding='utf-8-sig')), args.config.resolve().parent)
    items = load_items(settings['input'])
    output = Path(settings['output'])
    trace = output.with_name(output.stem+'.rows.jsonl')
    if output.exists() or trace.exists():
        raise FileExistsError('Choose new output/trace paths; existing scores are preserved')
    occupancy = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], text=True)
    used = {int(line.split(',')[0]):int(line.split(',')[1]) for line in occupancy.splitlines()}
    if used[settings['device']] > settings.get('busy_memory_mib', 1000):
        raise RuntimeError('Kawai GPU is occupied; no unrelated process will be stopped')
    output.parent.mkdir(parents=True, exist_ok=True)
    provenance_started = time.perf_counter()
    metadata = dict(model=MODEL_ID, model_path=settings['model_path'], preprocessing=PREPROCESS,
                    preprocessing_sha256=canonical_sha256(PREPROCESS),
                    model_sha256=sha256(Path(settings['model_path'])/'model.safetensors'),
                    model_config_sha256=sha256(Path(settings['model_path'])/'config.json'),
                    worker_sha256=sha256(__file__), input_sha256=sha256(settings['input']),
                    settings=settings, class_weights=settings['class_weights'], id2label=settings['id2label'],
                    raw_rows=str(trace))
    metadata['provenance_seconds'] = time.perf_counter()-provenance_started
    rows, runtime = score_items(items, settings, trace)
    metadata.update(runtime, images=len(rows))
    with output.open('x', encoding='utf-8') as stream:
        json.dump(dict(items=rows, metadata=metadata), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(event='kawai_done', images=len(rows), output=str(output), **runtime)), flush=True)


if __name__ == '__main__':
    main()
