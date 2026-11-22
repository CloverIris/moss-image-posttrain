#!/usr/bin/env python3
"""Clean TAG prompt files against the PixAI vocabulary (no model inference)."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from moss_art.tag_reward import preprocess_file

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--config",required=True);args=parser.parse_args()
    c=json.loads(Path(args.config).read_text())
    required={"input","output","report","metadata_dir"}
    optional={"max_removed_ratio","min_tags","min_chars","auto_add_prompt_header"}
    if not isinstance(c,dict) or not required<=set(c) or set(c)-required-optional:
        raise ValueError(f"Expected {sorted(required)} with optional {sorted(optional)}")
    options={key:c[key] for key in optional if key in c}
    report=preprocess_file(c["input"],c["output"],c["report"],c["metadata_dir"],**options)
    print(json.dumps(dict(input_prompts=len(report),kept_prompts=sum(r["accepted"] for r in report),
        dropped_prompts=sum(not r["accepted"] for r in report),
        kept_tags=sum(len(r["target_tags"]) for r in report if r["accepted"]),
        removed_tags=sum(len(r["dropped"]) for r in report),
        added_prompt_headers=sum(r["prompt_header_added"] for r in report if r["accepted"]),
        output=c["output"],report=c["report"]),ensure_ascii=False))

if __name__=="__main__":main()
