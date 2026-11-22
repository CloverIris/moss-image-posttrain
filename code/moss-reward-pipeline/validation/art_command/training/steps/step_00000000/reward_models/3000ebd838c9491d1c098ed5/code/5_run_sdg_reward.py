#!/usr/bin/env python3
"""SDG vLLM worker for ART's --request/--output command reward protocol."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time
import uuid

from sdg_artifact_reward import ArtifactScorer


def validate_request(request):
    if request.get('version') != 1:
        raise ValueError('Expected reward request version=1')
    rows = request.get('items')
    if not isinstance(rows, list) or not rows:
        raise ValueError('Expected a nonempty items list')
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get('prompt'), str):
            raise ValueError('Each item needs an unchanged string prompt')
        if not isinstance(row.get('image'), str) or not Path(row['image']).is_file():
            raise ValueError('Each item needs an existing saved image path')
    return rows


def worker_settings(cfg, request_cfg):
    if cfg.get('mode') != 'artifact_only' or cfg.get('alpha_misalignment', 0.) != 0.:
        raise ValueError('Only artifact_only with zero misalignment weight is supported')
    settings = dict(cfg)
    for key in ('model_path', 'device', 'batch_size', 'metric', 'threads'):
        if key in request_cfg:
            settings[key] = request_cfg[key]
    if settings.get('metric', 'artifact_penalty') not in ('artifact_penalty', 'artifact_only'):
        raise ValueError('Set reward.metric=artifact_penalty for the SDG worker')
    device = str(settings.get('device', 'cuda:0'))
    if device.startswith('cuda:'):
        device = device[5:]
    if not device.isdigit() or int(device) not in (0, 1, 2, 3):
        raise ValueError('Choose a physical GB203(SM120) GPU 0..3; GPU4 is reserved')
    settings['device'] = int(device)
    for key in ('batch_size', 'threads', 'resolution', 'max_tokens'):
        if type(settings.get(key)) is not int or settings[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if settings.get('resize_policy') not in ('aspect', 'square'):
        raise ValueError('resize_policy must be aspect or square')
    return settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding='utf-8-sig'))
    request = json.loads(args.request.read_text(encoding='utf-8-sig'))
    rows = validate_request(request)
    settings = worker_settings(cfg, request.get('config', {}))
    if args.output.exists():
        raise FileExistsError('Reward output already exists')
    occupancy = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used',
                                         '--format=csv,noheader,nounits'], text=True)
    memory = {int(s.split(',')[0]):int(s.split(',')[1]) for s in occupancy.splitlines()}
    if memory[settings['device']] > settings.get('occupied_gpu_limit_mib', 1000):
        raise RuntimeError('SDG GPU is occupied; no other process will be stopped')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(settings['device'])
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    # vLLM uses ninja from the selected environment; no system installation.
    import sys
    os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', '')
    from PIL import Image
    import torch
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from run_sdg_pilot import upstream_template
    torch.set_num_threads(settings['threads'])
    work = args.output.parent / (args.output.stem + '.sdg') / uuid.uuid4().hex
    work.mkdir(parents=True, exist_ok=False)
    (work / 'resolved_config.json').write_text(json.dumps(settings, indent=2) + '\n')
    template = upstream_template(settings['repo'])
    scorer = ArtifactScorer(settings['repo'], alpha_artifact=settings.get('alpha_artifact', .5),
                            latent_size=settings.get('latent_size', 64),
                            penalty_weight=settings.get('penalty_weight', 1.))
    t = time.perf_counter()
    model = LLM(model=settings['model_path'], tensor_parallel_size=1, dtype='bfloat16',
                max_model_len=settings.get('max_model_len', 8192),
                max_num_seqs=settings['batch_size'], max_num_batched_tokens=2048,
                gpu_memory_utilization=settings.get('gpu_memory_utilization', .8),
                enable_prefix_caching=True, enable_chunked_prefill=True,
                limit_mm_per_prompt={'image':1, 'video':0}, mm_processor_cache_gb=0,
                mm_processor_kwargs={'size':{'longest_edge':settings['resolution']**2,'shortest_edge':65536}},
                generation_config='vllm')
    load_seconds = time.perf_counter()-t
    processor = AutoProcessor.from_pretrained(settings['model_path'], local_files_only=True)
    sampling = SamplingParams(temperature=0, max_tokens=settings['max_tokens'], seed=42)
    scores, details = [], []
    raw_path = work / 'predictions.jsonl'
    inference_seconds = 0.
    with raw_path.open('x', encoding='utf-8') as stream:
        for start in range(0, len(rows), settings['batch_size']):
            batch = rows[start:start+settings['batch_size']]
            inputs, dimensions = [], []
            for row in batch:
                with Image.open(row['image']) as image:
                    im = image.convert('RGB')
                source_size = list(im.size)
                edge = settings['resolution']
                if settings['resize_policy'] == 'square':
                    target = (edge, edge)
                else:
                    scale = edge / max(im.size)
                    target = tuple(max(32, round(v*scale/32)*32) for v in im.size)
                im = im.resize(target, Image.Resampling.BICUBIC)
                text = template.format(caption=row['prompt'])
                messages = [{'role':'user','content':[{'type':'image'}, {'type':'text','text':text}]}]
                prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs.append({'prompt':prompt,'multi_modal_data':{'image':im}})
                dimensions.append((source_size, list(im.size)))
            t = time.perf_counter()
            generated = model.generate(inputs, sampling, use_tqdm=False)
            inference_seconds += time.perf_counter()-t
            if len(generated) != len(batch):
                raise RuntimeError('SDG returned the wrong number of predictions')
            for offset, (row, sample, sizes) in enumerate(zip(batch, generated, dimensions)):
                answer = sample.outputs[0]
                prediction = dict(index=start+offset, status='ok', resolution=edge,
                                  source_size=sizes[0], processed_size=sizes[1],
                                  raw_response=answer.text, finish_reason=answer.finish_reason,
                                  output_tokens=len(answer.token_ids), prompt_tokens=len(sample.prompt_token_ids))
                # Write even invalid/truncated responses before failing the stage.
                try:
                    result = scorer.score_prediction(prediction)
                except (ValueError, TypeError) as exc:
                    prediction.update(status='invalid_output', error=str(exc))
                    stream.write(json.dumps(prediction, ensure_ascii=False)+'\n'); stream.flush()
                    raise RuntimeError(f'SDG prediction {start+offset} invalid; see {raw_path}') from exc
                prediction['boxes'] = result['sdg_boxes']
                stream.write(json.dumps(prediction, ensure_ascii=False)+'\n'); stream.flush()
                detail = dict(**result, raw_predictions=str(raw_path), source_index=start+offset,
                              processed_size=sizes[1], output_tokens=len(answer.token_ids))
                for key in ('id', 'index', 'sample_id', 'prompt_index', 'rollout_id'):
                    if key in row:
                        detail[key] = row[key]
                scores.append(result['reward']); details.append(detail)
            print(json.dumps({'event':'sdg_batch_done','completed':len(scores),'total':len(rows)}), flush=True)
    if len(scores) != len(rows) or any(not math.isfinite(v) for v in scores):
        raise RuntimeError('Expected exactly one finite artifact reward per input image')
    response = dict(scores=scores, details=details, mode='artifact_only',
                    load_seconds=load_seconds, inference_seconds=inference_seconds,
                    raw_predictions=str(raw_path), model=settings['model_path'])
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(response, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'event':'sdg_reward_done','images':len(scores),'load_seconds':load_seconds,
                      'inference_seconds':inference_seconds,'output':str(args.output)}), flush=True)


if __name__ == '__main__':
    main()
