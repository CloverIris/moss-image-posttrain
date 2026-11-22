"""CPU-only tokenizer + image-header length scan. No JAX/Torch/Transformers."""
import argparse
import json
import math
from pathlib import Path
from tokenizers import Tokenizer
from PIL import Image,ImageOps
from .prompt_buckets import PromptBuckets,log_drop,validate_lengths
from .data import atomic_json


def reference_grid(path,max_pixels=262144,min_pixels=65536,patch=16,merge=2):
    # Same smart_resize rule as the installed Qwen image processor.
    with Image.open(path) as image:
        w,h=image.size
        if image.getexif().get(274) in (5,6,7,8): h,w=w,h
    if max(h,w)/min(h,w)>200: raise ValueError(f"Reference aspect ratio >200: {path}")
    factor=patch*merge
    hh,ww=round(h/factor)*factor,round(w/factor)*factor
    if hh*ww>max_pixels:
        beta=math.sqrt(h*w/max_pixels)
        hh=max(factor,math.floor(h/beta/factor)*factor)
        ww=max(factor,math.floor(w/beta/factor)*factor)
    elif hh*ww<min_pixels:
        beta=math.sqrt(min_pixels/(h*w))
        hh=math.ceil(h*beta/factor)*factor;ww=math.ceil(w*beta/factor)*factor
    return [1,hh//patch,ww//patch]


class LengthScanner:
    def __init__(self,model,max_reference_pixels=262144):
        self.tokenizer=Tokenizer.from_file(str(Path(model)/"tokenizer/tokenizer.json"))
        self.tokenizer.no_truncation();self.tokenizer.no_padding()
        self.max_reference_pixels=max_reference_pixels
        config=json.loads((Path(model)/"text_encoder/config.json").read_text())
        v=config["vision_config"]
        self.patch=v["patch_size"];self.merge=v["spatial_merge_size"]
        self.image_token_id=config["image_token_id"]
        if self.tokenizer.token_to_id("<|image_pad|>")!=self.image_token_id:
            raise ValueError("Unsupported image token/template")

    def tokens(self,prompt,refs=()):
        grids=[reference_grid(p,self.max_reference_pixels,patch=self.patch,merge=self.merge) for p in refs]
        content="".join("<|vision_start|><|image_pad|><|vision_end|>" for _ in refs)+(prompt if refs else prompt.strip())
        rendered="<|im_start|>user\n"+content+"<|im_end|>\n<|im_start|>assistant\n<think>\n"
        ids=self.tokenizer.encode(rendered,add_special_tokens=True).ids
        # Expand only known reference placeholders. Literal modality tokens in
        # user text are rejected to avoid ambiguously binding images.
        locations=[i for i,v in enumerate(ids) if v==self.image_token_id]
        if len(locations)!=len(refs): raise ValueError("Prompt contains unexpected image placeholders")
        visual=[math.prod(g)//self.merge**2 for g in grids]
        length=len(ids)+sum(n-1 for n in visual)
        return dict(tokens=length,visual_tokens=sum(visual),reference_tokens=visual,
                    image_grid_thw=grids,text_and_template_tokens=length-sum(visual))


def analyze(config):
    from .sft_data import load_manifest
    from .data import load_prompts
    if config["kind"]=="sft":
        rows=load_manifest(config["manifest"],conditioning_only=True)
    elif config["kind"]=="art":
        rows=[dict(id=r.id,prompt=r.prompt,negative_prompt=r.negative_prompt,conditioning_images=[])
              for r in load_prompts(config["manifest"],default_negative_prompt=config["negative_prompt"])]
    else: raise ValueError("kind must be sft or art")
    validate_lengths(config["short"],config["full"])
    scanner=LengthScanner(config["model"],config["max_reference_pixels"])
    details=[];dropped=[]
    for row in rows:
        item=dict(id=row["id"],**scanner.tokens(row["prompt"],row["conditioning_images"]))
        negative=scanner.tokens(row.get("negative_prompt",""))["tokens"]
        item["negative_tokens"]=negative
        item["required_tokens"]=max(item["tokens"],negative)
        if item["required_tokens"]>config["full"]:
            dropped.append(log_drop(row["id"],item["required_tokens"],config["full"],"tokenizer_scan"))
        details.append(item)
    lengths=sorted(x["required_tokens"] for x in details)
    accepted=[n for n in lengths if n<=config["full"]]
    buckets=PromptBuckets.resolve(lengths,config["short"],config["full"],
        minimum=scanner.tokens("")["tokens"]) if accepted else None
    report=dict(manifest=str(Path(config["manifest"]).resolve()),model=config["model"],
        total=len(rows),accepted=len(accepted),dropped=len(dropped),
        short=buckets.short if buckets else None,full=config["full"],
        short_coverage_all=sum(n<=buckets.short for n in lengths)/len(lengths) if buckets else 0,
        short_coverage_accepted=sum(n<=buckets.short for n in accepted)/len(accepted) if buckets else 0,
        percentiles={str(q):lengths[math.ceil(q/100*len(lengths))-1] for q in (50,70,90,95,99,100)},
        max_reference_pixels=config["max_reference_pixels"],records=details,dropped_records=dropped,
        method="tokenizer and image headers only; no neural model or GPU")
    output=Path(config["output"]);atomic_json(output,report)
    output.with_suffix(".dropped.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in dropped))
    print(json.dumps({k:v for k,v in report.items() if k not in ("records","dropped_records")},ensure_ascii=False,indent=2))
    return report


def main(argv=None):
    from .config_io import parse_file_arguments
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind",choices=("sft","art"),default="sft")
    p.add_argument("--manifest",required=True)
    p.add_argument("--model",default="$MODEL_ROOT/MOSS-Image-SFT")
    p.add_argument("--max-reference-pixels",type=int,default=262144)
    p.add_argument("--negative-prompt",default="")
    p.add_argument("--short",type=int,default=0)
    p.add_argument("--full",type=int,default=4096)
    p.add_argument("--output",default="reports/prompt_lengths.json")
    c=vars(parse_file_arguments(p,argv))
    if c["max_reference_pixels"]<65536: raise ValueError("max_reference_pixels must be >=65536")
    return analyze(c)

if __name__=="__main__": main()
