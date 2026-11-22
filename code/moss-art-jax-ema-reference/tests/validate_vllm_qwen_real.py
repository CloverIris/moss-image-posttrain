"""Two isolated processes: vLLM prefill, then original HF MOSS numeric oracle."""
import argparse
import json
from pathlib import Path
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--phase', choices=['vllm','reference'], required=True)
    args = parser.parse_args()
    from moss_art.config import load_config
    from moss_art.vllm_prepare import select_devices
    config = load_config(args.config)
    select_devices(config['preparation']['vllm']['devices'])
    root = Path(config['data']['cache']) / 'numeric_validation'
    root.mkdir(parents=True,exist_ok=True)
    if args.phase == 'vllm':
        from PIL import Image, ImageDraw
        from moss_art.vllm_qwen import VllmPrefill
        for name, color, size in [('blue',(60,90,220),(256,256)),('red',(220,60,80),(384,256))]:
            image=Image.new('RGB',size,'white')
            ImageDraw.Draw(image).ellipse((40,40,size[0]-40,size[1]-40),fill=color)
            image.save(root/f'{name}.png')
        examples=[
            {'id':'empty','prompt':'','images':[]},
            {'id':'short','prompt':'prompt:\nA white cat on a blue chair.\nnegative prompt:\nwatermark','images':[]},
            {'id':'long','prompt':'A quiet landscape with trees, distant hills, and warm sunlight. '*36,'images':[]},
            {'id':'image','prompt':'Describe the blue circle on the white background.','images':[str(root/'blue.png')]},
            {'id':'two_images','prompt':'A blue circle and a red circle, shown side by side.','images':[str(root/'blue.png'),str(root/'red.png')]},
        ]
        (root/'examples.json').write_text(json.dumps(examples,indent=2))
        start=time.perf_counter()
        engine=VllmPrefill(config['model']['path'],config['preparation']['vllm'],root/'engine',
                          max_sequence_length=config['data']['context_length'])
        report={'initialization_seconds':time.perf_counter()-start,'metadata':engine.metadata,'calls':[]}
        for group in [examples[:3],examples[3:4],examples[4:]]:
            start=time.perf_counter()
            features,ids=engine.encode_requests(group)
            for example,feature,tokens in zip(group,features,ids):
                np.savez(root/f"{example['id']}_vllm.npz",embedding=feature,input_ids=tokens)
            report['calls'].append({'ids':[x['id'] for x in group],'seconds':time.perf_counter()-start,
                                    'token_counts':[len(x) for x in ids]})
            (root/'vllm.json').write_text(json.dumps(report,indent=2))
        tokens=np.load(root/'long_vllm.npz')['input_ids']
        for batch_size in [1,8,8]:
            start=time.perf_counter()
            engine.encode_token_lists([tokens]*batch_size)
            report['calls'].append({'benchmark_batch':batch_size,'seconds':time.perf_counter()-start,
                                    'total_prompt_tokens':len(tokens)*batch_size})
        report['completed']=True
        (root/'vllm.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report),flush=True)
        return

    import torch
    from transformers import Qwen3_5Model,AutoTokenizer,AutoImageProcessor,AutoVideoProcessor,Qwen3VLProcessor
    from PIL import Image,ImageOps
    torch.set_num_threads(4)
    model_dir=Path(config['model']['path'])
    tokenizer=AutoTokenizer.from_pretrained(model_dir/'tokenizer',local_files_only=True)
    processor_dir=config['preparation']['vllm']['processor_dir']
    processor=Qwen3VLProcessor(
        image_processor=AutoImageProcessor.from_pretrained(processor_dir,local_files_only=True),
        video_processor=AutoVideoProcessor.from_pretrained(processor_dir,local_files_only=True),
        tokenizer=tokenizer,chat_template=tokenizer.chat_template)
    start=time.perf_counter()
    model=Qwen3_5Model.from_pretrained(model_dir/'text_encoder',dtype=torch.bfloat16,local_files_only=True,
                                      device_map='auto',max_memory={0:'12GiB',1:'12GiB','cpu':'24GiB'},
                                      attn_implementation='sdpa').eval().requires_grad_(False)
    import transformers
    report={'model_load_seconds':time.perf_counter()-start,'examples':[],
            'transformers_version':transformers.__version__,'torch_version':torch.__version__,
            'dtype':'bfloat16','attention':'sdpa','use_cache':False,
            'declared_acceptance':{'exact_token_ids':True,'cosine_minimum':0.999,'relative_l2_maximum':0.05}}
    examples=json.loads((root/'examples.json').read_text())
    for example in examples:
        if example['images']:
            images=[]
            for path in example['images']:
                with Image.open(path) as im:images.append(ImageOps.exif_transpose(im).convert('RGB').copy())
            inputs=processor.apply_chat_template(
                [{'role':'user','content':[*[{'type':'image','image':im} for im in images],
                                          {'type':'text','text':example['prompt']}]}],
                tokenize=True,add_generation_prompt=True,enable_thinking=True,return_dict=True,return_tensors='pt',
                processor_kwargs={'images_kwargs':{'size':{'shortest_edge':65536,'longest_edge':config['preparation']['vllm']['max_reference_pixels']}}})
        else:
            rendered=tokenizer.apply_chat_template([{'role':'user','content':example['prompt'].strip()}],
                         tokenize=False,add_generation_prompt=True,enable_thinking=True)
            inputs=tokenizer(rendered,return_tensors='pt',truncation=False)
        expected=np.load(root/f"{example['id']}_vllm.npz")
        np.testing.assert_array_equal(inputs['input_ids'][0].numpy(),expected['input_ids'])
        inputs=inputs.to(model.language_model.embed_tokens.weight.device)
        start=time.perf_counter()
        with torch.inference_mode():
            actual=model(**inputs,use_cache=False,return_dict=True).last_hidden_state.float().cpu().numpy()
        np.savez(root/f"{example['id']}_reference.npz",embedding=actual)
        reference=actual.reshape(-1).astype(np.float64)
        candidate=expected['embedding'].reshape(-1).astype(np.float64)
        difference=candidate-reference
        cosine=float(np.dot(reference,candidate)/(np.linalg.norm(reference)*np.linalg.norm(candidate)))
        rel=float(np.linalg.norm(difference)/np.linalg.norm(reference))
        result={'id':example['id'],'shape':list(actual.shape),'token_ids_equal':True,
                'cosine':cosine,'relative_l2':rel,'mae':float(np.abs(difference).mean()),
                'max_abs':float(np.abs(difference).max()),'reference_seconds':time.perf_counter()-start,
                'passed':cosine>=.999 and rel<=.05}
        report['examples'].append(result)
        (root/'reference.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(result),flush=True)
    report['passed']=all(x['passed'] for x in report['examples'])
    (root/'reference.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
