#!/usr/bin/env python3
"""Environment-specific workers; GPU selection happens in run.py before imports."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from dataclasses import asdict

from run import save
from defect_prompt import PROMPT_VERSION, build_messages
from defect_response import response_fields


def defect_output_schema(categories):
    return {
        'type':'object','additionalProperties':False,
        'required':['has_defect','defects','uncertain','quality_penalty','notes'],
        'properties':{
            'has_defect':{'type':'boolean'},
            'defects':{'type':'array','maxItems':8,'items':{
                'type':'object','additionalProperties':False,
                'required':['category','severity','region','description','confidence'],
                'properties':{
                    'category':{'type':'string','enum':categories},
                    'severity':{'type':'integer','minimum':1,'maximum':5},
                    'region':{'type':'string','maxLength':100},
                    'description':{'type':'string','maxLength':200},
                    'confidence':{'type':'string','enum':['low','medium','high']}}}},
            'uncertain':{'type':'array','maxItems':8,'items':{
                'type':'object','additionalProperties':False,'required':['category','reason'],
                'properties':{'category':{'type':'string','enum':categories},
                              'reason':{'type':'string','maxLength':160}}}},
            'quality_penalty':{'type':'number','minimum':0,'maximum':1},
            'notes':{'type':'string','maxLength':200}}}


# Keep existing Gemma comparison scripts compatible.
gemma_output_schema = defect_output_schema


def completed(path):
    if not path.exists(): return {}
    data = {}
    with path.open('rb') as stream:
        offset = 0
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # An interrupted final write can be removed, without discarding completed rows.
                if stream.read(): raise
                with path.open('r+b') as repair: repair.truncate(offset)
                break
            if row['id'] in data: raise ValueError('Duplicate completed item')
            data[row['id']] = row['result']
            offset = stream.tell()
    return data


def append(path, identity, value):
    with path.open('a') as stream:
        stream.write(json.dumps(dict(id=identity,result=value),ensure_ascii=False,allow_nan=False)+'\n')
        stream.flush()


def progress(stage, count, total, start):
    print(json.dumps(dict(event='progress',stage=stage,images=count,total=total,
                          elapsed_seconds=time.monotonic()-start)),flush=True)


def batch_barrier(config):
    """Synchronize independent DP replicas after loading/warmup using small files."""
    batch=config.get('_batch')
    if not batch:return
    directory=Path(batch['barrier_dir'])
    directory.mkdir(parents=True,exist_ok=True)
    (directory/('dp%02d.ready'%batch['dp_rank'])).write_text(str(os.getpid())+'\n')
    deadline=time.monotonic()+batch.get('barrier_timeout_seconds',900)
    while not all((directory/('dp%02d.ready'%rank)).exists() for rank in batch['participants']):
        if time.monotonic()>deadline:raise TimeoutError('DP replicas did not all finish warmup')
        time.sleep(.05)


def speculative_snapshot(llm):
    from vllm.v1.metrics.reader import Counter, Vector
    return [asdict(metric) for metric in llm.get_metrics()
            if 'spec_decode' in metric.name and isinstance(metric, (Counter, Vector))]


def context_capacity_probe(llm, tokenizer, settings, result_path):
    """Optional synthetic text pressure test, outside the scored image timing."""
    probe = settings.get('context_probe')
    if not probe:
        return None
    from vllm import SamplingParams
    maximum = llm.llm_engine.vllm_config.model_config.max_model_len
    output_count = int(probe.get('output_tokens', 32))
    reserve = int(probe.get('reserve_tokens', 64))
    if reserve < output_count or maximum <= reserve:
        raise ValueError('Invalid context probe token budget')
    target = maximum - reserve
    pattern = tokenizer.encode(
        'This is a synthetic context memory test. Read this neutral text and continue. ',
        add_special_tokens=False)
    token_ids = (pattern * ((target + len(pattern) - 1) // len(pattern)))[:target]
    params = SamplingParams(temperature=0, max_tokens=output_count,
                            ignore_eos=True, logprobs=1)
    print(json.dumps(dict(event='context_probe_started', prompt_tokens=target,
                          output_tokens=output_count, resolved_max_model_len=maximum)), flush=True)
    started = time.monotonic()
    output = llm.generate([dict(prompt_token_ids=token_ids)], params, use_tqdm=False)[0]
    elapsed = time.monotonic() - started
    sample = output.outputs[0]
    finite = (sample.logprobs is not None
              and len(sample.logprobs) == len(sample.token_ids)
              and all(math.isfinite(probs[token].logprob)
                      for token, probs in zip(sample.token_ids, sample.logprobs)))
    if len(output.prompt_token_ids) != target or len(sample.token_ids) != output_count or not finite:
        raise ValueError('Context capacity probe returned invalid lengths or non-finite logits')
    result = dict(kind='synthetic_text_only', resolved_max_model_len=maximum,
                  prompt_tokens=target, generated_tokens=len(sample.token_ids),
                  total_tokens=target + len(sample.token_ids), finite_logprobs=finite,
                  finish_reason=sample.finish_reason, seconds=elapsed)
    save(result_path.with_suffix('.context_probe.json'), result)
    print(json.dumps(dict(event='context_probe_complete', **result)), flush=True)
    return result


def visual_defects(config, items, result_path, stage):
    settings=config[stage]
    template_kwargs={'enable_thinking':False, **settings.get('chat_template_kwargs',{})}
    thinking_enabled=bool(template_kwargs.get('enable_thinking'))
    parser_name=settings['llm'].get('reasoning_parser') or (settings['llm'].get('structured_outputs_config') or {}).get('reasoning_parser')
    if thinking_enabled and not parser_name:
        raise ValueError('Thinking requires a vLLM reasoning_parser for final JSON extraction')
    if settings.get('prompt_version')!=PROMPT_VERSION:
        raise ValueError(stage+' prompt version changed; use a new output path')
    previous=completed(result_path)
    if any(value.get('prompt_version')!=PROMPT_VERSION for value in previous.values()):
        raise ValueError('Old defect results cannot be resumed with the current prompt version')
    if any(value.get('model', settings['model']) != settings['model'] for value in previous.values()):
        raise ValueError('Defect evaluator model changed; use a new output path')
    if any(value.get('thinking_enabled',False) != thinking_enabled for value in previous.values()):
        raise ValueError('Thinking mode changed; use a new output path')
    pending=[item for item in items if item['id'] not in previous]
    started=time.monotonic()
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    from PIL import Image
    from transformers import AutoProcessor
    reasoning_parser_class=None
    if thinking_enabled:
        from vllm.reasoning import ReasoningParserManager
        reasoning_parser_class=ReasoningParserManager.get_reasoning_parser(parser_name)
    sys.path.insert(0,config['ranker_root'])
    from pipeline.defect import DEFECT_CATEGORIES, parse_defect_text
    schema=defect_output_schema(DEFECT_CATEGORIES)
    params=SamplingParams(temperature=0,max_tokens=settings['max_tokens'],
        structured_outputs=StructuredOutputsParams(json=schema) if settings.get('structured_outputs') else None)
    engine_start=time.monotonic()
    llm=LLM(model=settings['model'],**settings['llm'])
    load_seconds=time.monotonic()-engine_start
    processor=AutoProcessor.from_pretrained(settings['model'],local_files_only=True)
    image_token_id=getattr(processor,'image_token_id',None)
    # Standard HF chat rendering + vLLM multimodal pixel inputs, no HTTP/base64 layer.
    def inputs(batch):
        requests=[]
        for item in batch:
            with Image.open(item['image']) as image: pixels=image.convert('RGB').copy()
            messages=build_messages(item)
            prompt=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,**template_kwargs)
            requests.append(dict(prompt=prompt,multi_modal_data={'image':pixels}))
        return requests
    warmup_start=time.monotonic()
    warmup_reasoning=[]
    if pending and settings.get('warmup_images',0):
        warm_outputs=llm.generate(inputs(pending[:settings['warmup_images']]),params,use_tqdm=False)
        if thinking_enabled:
            diagnostics=[]
            for output in warm_outputs:
                sample=output.outputs[0]
                fields=response_fields(sample,processor.tokenizer,reasoning_parser_class)
                diagnostics.append(dict(**fields,finish_reason=sample.finish_reason,
                                        generated_tokens=len(sample.token_ids),
                                        generated_token_ids=list(sample.token_ids)))
                save(result_path.with_suffix('.warmup.json'),dict(outputs=diagnostics))
                if sample.finish_reason=='length':
                    raise ValueError('Thinking warmup reached the output token limit; see the warmup diagnostic')
                if not fields['reasoning_closed']:
                    raise ValueError('Thinking warmup has no explicit reasoning boundary; see the warmup diagnostic')
                json.loads(fields['raw'])
                warmup_reasoning.append(dict(reasoning_tokens=fields['reasoning_tokens'],final_tokens=fields['final_tokens']))
    warmup_seconds=time.monotonic()-warmup_start
    context_probe = context_capacity_probe(llm, processor.tokenizer, settings, result_path)
    speculative_config=settings['llm'].get('speculative_config')
    spec_before=speculative_snapshot(llm) if speculative_config else []
    prompt_tokens=output_tokens=reasoning_tokens=final_tokens=reasoning_images=errors=0
    batch_barrier(config)
    inference_started_at=time.time()
    inference_start=time.monotonic()
    for start in range(0,len(pending),settings['submit_batch_size']):
        batch=pending[start:start+settings['submit_batch_size']]
        requests=inputs(batch)
        outputs=llm.generate(requests,params,use_tqdm=False)
        if len(outputs)!=len(batch):raise ValueError('vLLM returned incorrect output count')
        for item,request,output in zip(batch,requests,outputs):
            sample=output.outputs[0]
            prompt_count=len(output.prompt_token_ids)
            generated_count=len(sample.token_ids)
            fields=response_fields(sample,processor.tokenizer,reasoning_parser_class)
            prompt_tokens+=prompt_count
            output_tokens+=generated_count
            reasoning_tokens+=fields['reasoning_tokens']
            final_tokens+=fields['final_tokens']
            reasoning_images+=int(fields.get('reasoning_present',False))
            value=dict(status='ok',image_id=item['id'],prompt_version=PROMPT_VERSION,
                       model=settings['model'],evaluator=stage,
                       rendered_prompt_sha256=hashlib.sha256(request['prompt'].encode()).hexdigest(),
                       **fields,finish_reason=sample.finish_reason,
                       prompt_tokens=prompt_count,generated_tokens=generated_count)
            if image_token_id is not None:
                value['image_tokens']=output.prompt_token_ids.count(image_token_id)
            try:
                # Strict JSON validity, then the existing project's normalizer.
                if thinking_enabled and not fields['reasoning_closed']:
                    raise ValueError('Thinking output has no completed reasoning boundary')
                json.loads(fields['raw'])
                value['parsed']=parse_defect_text(fields['raw'])
                if sample.finish_reason=='length':raise ValueError('Generation reached token limit')
            except Exception as error:
                errors+=1
                value.update(status='error',error=str(error),parsed=None)
            append(result_path,item['id'],value)
        progress(stage,len(previous)+min(start+len(batch),len(pending)),len(items),inference_start)
    inference_seconds=time.monotonic()-inference_start
    inference_finished_at=time.time()
    spec_after=speculative_snapshot(llm) if speculative_config else []
    return dict(model=settings['model'],backend='vllm.LLM.generate',vllm=__import__('vllm').__version__,
                prompt_version=PROMPT_VERSION,
                evaluator=stage,tp_size=settings['llm'].get('tensor_parallel_size',1),
                pp_size=settings['llm'].get('pipeline_parallel_size',1),
                gpus=settings['devices'],load_seconds=load_seconds,warmup_seconds=warmup_seconds,
                inference_seconds=inference_seconds,images=len(pending),total_images=len(items),
                inference_started_at=inference_started_at,inference_finished_at=inference_finished_at,
                images_per_second=len(pending)/inference_seconds if pending else None,
                prompt_tokens=prompt_tokens,generated_tokens=output_tokens,
                generated_tokens_per_second=output_tokens/inference_seconds,errors=errors,
                thinking_enabled=thinking_enabled,reasoning_parser=parser_name,
                reasoning_tokens=reasoning_tokens,final_tokens=final_tokens,reasoning_images=reasoning_images,
                warmup_reasoning=warmup_reasoning,
                resolved_max_model_len=llm.llm_engine.vllm_config.model_config.max_model_len,
                max_num_seqs=settings['llm'].get('max_num_seqs'),
                context_probe=context_probe,
                speculative_config=speculative_config,
                speculative_metrics_before=spec_before,speculative_metrics_after=spec_after,
                max_tokens=settings['max_tokens'],
                max_soft_tokens=settings['llm'].get('mm_processor_kwargs',{}).get('max_soft_tokens'),
                mm_processor_kwargs=settings['llm'].get('mm_processor_kwargs',{}),
                chat_template_kwargs=template_kwargs,
                task='Existing single-image defect inspection; original prompt retained',
                process_seconds=time.monotonic()-started)


def gemma(config, items, result_path):
    return visual_defects(config, items, result_path, 'gemma')


def qwen(config, items, result_path):
    return visual_defects(config, items, result_path, 'qwen')


def hps(config, items, result_path):
    settings=config['hps']
    previous=completed(result_path)
    pending=[item for item in items if item['id'] not in previous]
    started=time.monotonic()
    sys.path.insert(0,settings['project'])
    from hpsv3pp_torch.engine import NativeHPS
    import torch
    native=json.loads(Path(settings['native_config']).read_text())
    base=Path(native['model_config'])
    if not base.is_absolute():base=Path(settings['project'])/base
    model=json.loads(base.read_text())
    model['torch_native']=dict(native['torch_native'],cpu_threads=settings.get('cpu_threads',4))
    model['runtime'].update(batch_size=settings['batch_size'],devices=settings['devices'])
    engine=NativeHPS(model)
    try:
        load_seconds=engine.maximum(time.monotonic()-started)
        engine.barrier()
        warmup_start=time.monotonic()
        if pending and settings.get('warmup_images',0):
            warm=pending[:settings['warmup_images']]
            engine.score([x['image'] for x in warm],[x['prompt'] for x in warm])
        engine.barrier()
        warmup_seconds=engine.maximum(time.monotonic()-warmup_start)
        if engine.rank==0:batch_barrier(config)
        engine.barrier()
        inference_started_at=time.time()
        inference_start=time.monotonic()
        for start in range(0,len(pending),settings['batch_size']):
            batch=pending[start:start+settings['batch_size']]
            output=engine.score([x['image'] for x in batch],[x['prompt'] for x in batch])
            if engine.rank==0:
                for item,value in zip(batch,output):
                    append(result_path,item['id'],dict(status='ok',**value))
                if (start+len(batch))%8==0 or start+len(batch)==len(pending):
                    progress('hps',len(previous)+start+len(batch),len(items),inference_start)
        engine.barrier()
        inference_seconds=engine.maximum(time.monotonic()-inference_start)
        inference_finished_at=time.time()
        peak=engine.maximum(torch.cuda.max_memory_allocated(engine.device))
        value=dict(model='HPSv3++',backend='Native PyTorch DTensor TP2',gpus=settings['devices'],
                   attention=model['torch_native']['attention'],batch_size=settings['batch_size'],
                   load_seconds=load_seconds,warmup_seconds=warmup_seconds,inference_seconds=inference_seconds,
                   inference_started_at=inference_started_at,inference_finished_at=inference_finished_at,
                   images=len(pending),total_images=len(items),images_per_second=len(pending)/inference_seconds if pending else None,
                   max_rank_peak_allocated_bytes=peak,process_seconds=time.monotonic()-started)
        return value if engine.rank==0 else None
    finally:engine.close()


def deepghs(config, items, result_path):
    settings=config['deepghs']
    previous=completed(result_path)
    pending=[item for item in items if item['id'] not in previous]
    started=time.monotonic()
    # Keep this interpreter's CUDA ORT ahead of the shared imgutils dependencies.
    for path in settings.get('extra_python_paths',[]):sys.path.append(path)
    import onnxruntime as ort
    from imgutils.metrics import anime_dbaesthetic
    from imgutils.metrics.dbaesthetic import _MODEL
    if 'CUDAExecutionProvider' not in ort.get_available_providers():raise RuntimeError('CUDA ONNX runtime required')
    model_name=settings['model']
    load_start=time.monotonic()
    from huggingface_hub import hf_hub_download
    model_path=hf_hub_download('deepghs/anime_aesthetic',filename=model_name+'/model.onnx',local_files_only=True)
    options=ort.SessionOptions()
    options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads=settings.get('cpu_threads',2)
    options.inter_op_num_threads=1
    session=ort.InferenceSession(model_path,options,providers=[
        ('CUDAExecutionProvider',{'use_tf32':str(int(settings.get('use_tf32',False)))}),'CPUExecutionProvider'])
    # Reuse the official classifier's preprocessing and aesthetic calibration.
    _MODEL.classifier._models[model_name]=session
    if session.get_providers()[0]!='CUDAExecutionProvider':raise RuntimeError('DeepGHS fell back to CPU')
    load_seconds=time.monotonic()-load_start
    warmup_start=time.monotonic()
    for item in pending[:settings.get('warmup_images',0)]:anime_dbaesthetic(item['image'],model_name=model_name)
    warmup_seconds=time.monotonic()-warmup_start
    batch_barrier(config)
    inference_started_at=time.time()
    inference_start=time.monotonic()
    for index,item in enumerate(pending,1):
        label,percentile,score=anime_dbaesthetic(item['image'],model_name=model_name,fmt=('label','percentile','score'))
        if not math.isfinite(float(score)) or not 0<=float(percentile)<=1:raise ValueError('Invalid DeepGHS score')
        append(result_path,item['id'],dict(status='ok',label=label,percentile=float(percentile),score=float(score)))
        if index%8==0 or index==len(pending):progress('deepghs',len(previous)+index,len(items),inference_start)
    elapsed=time.monotonic()-inference_start
    inference_finished_at=time.time()
    return dict(model='deepghs/anime_aesthetic/'+model_name,backend='official imgutils + CUDA ONNX Runtime',
                ort_version=ort.__version__,providers=session.get_providers(),gpus=settings['devices'],
                cuda_provider_options=session.get_provider_options().get('CUDAExecutionProvider',{}),
                load_seconds=load_seconds,warmup_seconds=warmup_seconds,inference_seconds=elapsed,
                inference_started_at=inference_started_at,inference_finished_at=inference_finished_at,
                images=len(pending),total_images=len(items),images_per_second=len(pending)/elapsed if pending else None,
                process_seconds=time.monotonic()-started)


def tagger(config, items, result_path):
    settings=config['tagger']
    previous=completed(result_path)
    pending=[item for item in items if item['id'] not in previous]
    started=time.monotonic()
    import gc
    import torch
    from PIL import Image
    from transformers import AutoConfig,AutoImageProcessor,AutoModel
    torch.set_num_threads(settings.get('cpu_threads',2))
    dtype=getattr(torch,settings['dtype'])
    load_start=time.monotonic()
    model_dir=settings['model']
    cfg=AutoConfig.from_pretrained(model_dir,trust_remote_code=True,local_files_only=True)
    processor=AutoImageProcessor.from_pretrained(model_dir,trust_remote_code=True,local_files_only=True)
    model=AutoModel.from_pretrained(model_dir,trust_remote_code=True,local_files_only=True,torch_dtype=dtype,
                                    low_cpu_mem_usage=True).eval().to('cuda')
    gc.collect()
    torch.cuda.synchronize()
    load_seconds=time.monotonic()-load_start
    tags=list(cfg.tags)
    positions={tag:index for index,tag in enumerate(tags)}
    thresholds=torch.tensor([float((cfg.category_best_threshold or {}).get(cat,0.2))
                             for cat,count in cfg.tags_split for _ in range(count)],dtype=torch.float32)
    if cfg.tags_best_threshold is not None:thresholds=torch.tensor(cfg.tags_best_threshold,dtype=torch.float32)

    def forward(batch):
        images=[]
        for item in batch:
            with Image.open(item['image']) as image:
                images.append(image.copy())
        tensor=processor(images,return_tensors='pt')['pixel_values'].to('cuda',dtype=dtype)
        with torch.inference_mode():probs=model(tensor).float().sigmoid().cpu()
        return probs

    warmup_start=time.monotonic()
    if pending and settings.get('warmup_images',0):forward(pending[:settings['warmup_images']])
    torch.cuda.synchronize()
    warmup_seconds=time.monotonic()-warmup_start
    batch_barrier(config)
    inference_started_at=time.time()
    inference_start=time.monotonic()
    batch_size=settings.get('batch_size',1)
    for start in range(0,len(pending),batch_size):
        batch=pending[start:start+batch_size]
        probabilities=forward(batch)
        for item,probs in zip(batch,probabilities):
            predictions={}
            offset=0
            for category,count in cfg.tags_split:
                selected=torch.nonzero(probs[offset:offset+count]>thresholds[offset:offset+count]).flatten().tolist()
                predictions[category]={tags[offset+i]:float(probs[offset+i]) for i in selected}
                offset+=count
            canonical=list(dict.fromkeys(tag.strip().lower().replace(' ','_') for tag in item['tags'] if tag.strip()))
            known=[tag for tag in canonical if tag in positions]
            unknown=[tag for tag in canonical if tag not in positions]
            matched=[tag for tag in known if probs[positions[tag]]>thresholds[positions[tag]]]
            append(result_path,item['id'],dict(status='ok',predictions=predictions,
                target_probabilities={tag:float(probs[positions[tag]]) for tag in known},
                target_thresholds={tag:float(thresholds[positions[tag]]) for tag in known},
                matched_target_tags=matched,unsupported_target_tags=unknown,
                known_target_count=len(known),target_tag_recall=len(matched)/len(known) if known else None))
        if (start+len(batch))%8==0 or start+len(batch)==len(pending):
            progress('tagger',len(previous)+start+len(batch),len(items),inference_start)
    torch.cuda.synchronize()
    elapsed=time.monotonic()-inference_start
    inference_finished_at=time.time()
    return dict(model='pixai-labs/pixai-tagger-v1.0',model_dir=model_dir,backend='official PyTorch',
                dtype=settings['dtype'],gpus=settings['devices'],input_size=processor.size,
                batch_size=batch_size,category_thresholds=cfg.category_best_threshold,num_tags=len(tags),
                load_seconds=load_seconds,warmup_seconds=warmup_seconds,inference_seconds=elapsed,
                inference_started_at=inference_started_at,inference_finished_at=inference_finished_at,
                images=len(pending),total_images=len(items),images_per_second=len(pending)/elapsed if pending else None,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),process_seconds=time.monotonic()-started)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=['gemma','qwen','hps','deepghs','tagger'])
    parser.add_argument('--work',required=True)
    args=parser.parse_args()
    work=Path(args.work)
    config=json.loads((work/'config.json').read_text())
    items=json.loads((work/'input.json').read_text())['items']
    result=globals()[args.stage](config,items,work/(args.stage+'.jsonl'))
    if result is not None:
        save(work/(args.stage+'.metrics.json'),result)
        print(json.dumps(dict(event='complete',stage=args.stage,metrics=result)),flush=True)


if __name__=='__main__':main()
