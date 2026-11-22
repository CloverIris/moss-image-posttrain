"""Metadata-only tag normalization, prompt cleaning and calibrated hard rewards."""
import json
import math
import re
from pathlib import Path
import numpy as np

DEFAULT_METADATA = Path(__file__).resolve().parents[1]/"vendor/pixai_tagger/metadata"


def vocabulary(metadata_dir=DEFAULT_METADATA):
    config=json.loads((Path(metadata_dir)/"config.json").read_text())
    tags=config["tags"]; categories=[]
    for category,count in config["tags_split"]:categories.extend([category]*count)
    if len(categories)!=len(tags) or len(set(tags))!=len(tags):raise ValueError("Invalid tag vocabulary")
    thresholds=config.get("tags_best_threshold")
    if thresholds is None:
        values=config.get("category_best_threshold") or {}
        thresholds=[values.get(category,.5) for category in categories]
    elif isinstance(thresholds,dict):
        thresholds=[thresholds[tag] for tag in tags]
    thresholds=np.asarray(thresholds,np.float32)
    if thresholds.shape!=(len(tags),) or not np.isfinite(thresholds).all() or not ((thresholds>=0)&(thresholds<=1)).all():raise ValueError("Invalid tag thresholds")
    return config,{tag:i for i,tag in enumerate(tags)},np.asarray(categories),thresholds


def normalize_tag(token):
    token=token.strip()
    match=re.fullmatch(r"\((.+):([+-]?(?:\d+(?:\.\d*)?|\.\d+))\)",token)
    return (match.group(1) if match else token).strip().replace(" ","_")


def clean_prompt(prompt, tag_ids, *, allow_empty=False, auto_add_prompt_header=True):
    """Preserve prompt header/line layout and optional negative body; clean TAGs only."""
    match=re.match(r"\A(\s*prompt\s*:\s*)",prompt,re.I)
    header=match.group(1) if match else "";body=prompt[len(header):]
    negative=re.search(r"(?im)^\s*negative\s+prompt\s*:",body)
    tail=body[negative.start():] if negative else ""
    if negative:body=body[:negative.start()]
    used=[];dropped=[];lines=[]
    for line in body.splitlines():
        kept=[]
        for token in line.split(","):
            token=token.strip()
            if not token:continue
            tag=normalize_tag(token)
            if tag not in tag_ids:dropped.append({"text":token,"normalized":tag,"reason":"unknown"});continue
            if tag in used:dropped.append({"text":token,"normalized":tag,"reason":"duplicate"});continue
            used.append(tag);kept.append(token)
        if kept:lines.append(", ".join(kept)+("," if line.rstrip().endswith(",") else ""))
    if not used and not allow_empty:raise ValueError("No supported target tags remain")
    if not header and auto_add_prompt_header:header="prompt:\n"
    cleaned=header+"\n".join(lines)
    if tail:cleaned+="\n"+tail
    return cleaned,used,dropped


def source_records(path):
    from .rollout_only import unescape_prompt
    path=Path(path)
    if path.suffix.lower() in (".jsonl",".ndjson"):
        rows=[];seen=set()
        for line_number,line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(),1):
            if not line.strip():continue
            value=json.loads(line)
            if not isinstance(value,dict):raise ValueError(f"Line {line_number}: expected JSON object")
            identifier=value.get("id",value.get("sample_id",f"line-{line_number:06d}"))
            if "id" in value and "sample_id" in value and value["id"]!=value["sample_id"]:raise ValueError("Conflicting id/sample_id")
            if not isinstance(identifier,str) or not identifier or identifier in seen:raise ValueError("Invalid/duplicate prompt id")
            prompt=value.get("prompt");negative=value.get("negative_prompt","");metadata=value.get("metadata",{})
            if not isinstance(prompt,str) or not isinstance(negative,str) or not isinstance(metadata,dict):raise ValueError(f"Line {line_number}: invalid prompt, negative_prompt or metadata type")
            seen.add(identifier)
            rows.append(dict(id=identifier,prompt=unescape_prompt(prompt),negative_prompt=unescape_prompt(negative),metadata=metadata))
        if not rows:raise ValueError("Empty prompt file")
        return rows
    if path.suffix.lower()!=".txt":raise ValueError("Input must be txt or jsonl")
    text=path.read_text(encoding="utf-8-sig")
    physical=[line for line in text.splitlines() if line.strip()]
    if physical and all(re.match(r"\s*prompt\s*:",line,re.I) for line in physical):parts=physical
    else:
        starts=[m.start() for m in re.finditer(r"(?im)^\s*prompt\s*:",text)]
        if starts and text[:starts[0]].strip():raise ValueError("Text before first prompt header")
        parts=[text[a:b] for a,b in zip(starts,starts[1:]+[len(text)])] if starts else physical
    if not parts:raise ValueError("Empty prompt file")
    return [dict(id=f"line-{i:06d}",prompt=unescape_prompt(part.rstrip()),negative_prompt="",metadata={}) for i,part in enumerate(parts,1)]


def preprocess_file(input_path,output_path,report_path,metadata_dir=DEFAULT_METADATA, *,
                    max_removed_ratio=1.0,min_tags=1,min_chars=0,auto_add_prompt_header=True):
    """Write accepted prompts only; rejected records and reasons remain in report.

    Removal ratio includes both unknown and duplicate positive TAG occurrences.
    Character length excludes the prompt header and any negative-prompt body.
    Threshold equality is accepted; an empty target list is always rejected.
    """
    if type(max_removed_ratio) not in (int,float) or not math.isfinite(max_removed_ratio) or not 0<=max_removed_ratio<=1:
        raise ValueError("max_removed_ratio must be a finite number in [0,1]")
    if type(min_tags) is not int or min_tags<1:raise ValueError("min_tags must be a positive integer")
    if type(min_chars) is not int or min_chars<0:raise ValueError("min_chars must be a nonnegative integer")
    if type(auto_add_prompt_header) is not bool:raise ValueError("auto_add_prompt_header must be boolean")
    input_path,output_path,report_path=map(Path,(input_path,output_path,report_path))
    resolved=[p.resolve() for p in (input_path,output_path,report_path)]
    if len(set(resolved))!=3:raise ValueError("Input, output and report must be different paths")
    for path in (output_path,report_path):
        if path.exists():raise FileExistsError(path)
    _,ids,_,thresholds=vocabulary(metadata_dir);records=source_records(input_path);rows=[];reports=[]
    for record in records:
        original=record["prompt"]
        clean,tags,dropped=clean_prompt(original,ids,allow_empty=True,auto_add_prompt_header=auto_add_prompt_header)
        total=len(tags)+len(dropped)
        ratio=len(dropped)/total if total else 0.0
        positive=re.sub(r"\A\s*prompt\s*:\s*","",clean,flags=re.I)
        negative=re.search(r"(?im)^\s*negative\s+prompt\s*:",positive)
        if negative:positive=positive[:negative.start()]
        chars=len(positive.strip())
        reasons=[]
        if not tags:reasons.append("no_supported_tags")
        if ratio>max_removed_ratio:reasons.append("removed_ratio_exceeded")
        if len(tags)<min_tags:reasons.append("too_few_tags")
        if chars<min_chars:reasons.append("too_short_chars")
        accepted=not reasons
        if accepted:
            meta=dict(record.get("metadata",{}),target_tags=tags,tag_preprocessed=True)
            rows.append(dict(id=record["id"],prompt=clean,negative_prompt=record["negative_prompt"],metadata=meta))
        reports.append(dict(id=record["id"],original_prompt=original,cleaned_prompt=clean,target_tags=tags,
            accepted=accepted,rejection_reasons=reasons,original_tag_count=total,retained_tag_count=len(tags),
            removed_tag_count=len(dropped),unknown_tag_count=sum(x["reason"]=="unknown" for x in dropped),
            duplicate_tag_count=sum(x["reason"]=="duplicate" for x in dropped),removed_ratio=ratio,
            cleaned_positive_chars=chars,
            prompt_header_added=auto_add_prompt_header and not bool(re.match(r"\A\s*prompt\s*:",original,re.I)),
            thresholds={tag:float(thresholds[ids[tag]]) for tag in tags},dropped=dropped))
    policy=dict(max_removed_ratio=max_removed_ratio,min_tags=min_tags,min_chars=min_chars,
                auto_add_prompt_header=auto_add_prompt_header,ratio_counts_duplicates=True)
    summary=dict(input_prompts=len(records),kept_prompts=len(rows),dropped_prompts=len(records)-len(rows),
                 kept_tags=sum(len(row["metadata"]["target_tags"]) for row in rows))
    output_path.parent.mkdir(parents=True,exist_ok=True);report_path.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive creation also guards against another process winning the check.
    with output_path.open("x",encoding="utf-8") as output:
        output.write("".join(json.dumps(row,ensure_ascii=False)+"\n" for row in rows))
    with report_path.open("x",encoding="utf-8") as report:
        report.write(json.dumps(dict(input=str(input_path),output=str(output_path),policy=policy,summary=summary,records=reports),ensure_ascii=False,indent=2)+"\n")
    return reports


def hard_scores(probabilities,targets,metadata_dir=DEFAULT_METADATA,metric="coverage"):
    if metric not in ("coverage","iou"):raise ValueError("metric must be coverage or iou")
    config,ids,categories,thresholds=vocabulary(metadata_dir)
    probs=np.asarray(probabilities,np.float32)
    if probs.ndim!=2 or probs.shape!=(len(targets),len(ids)) or not np.isfinite(probs).all() or not ((probs>=0)&(probs<=1)).all():raise ValueError("Invalid probability matrix")
    result=[]
    for row,tags in zip(probs,targets):
        if not isinstance(tags,list) or not tags or any(tag not in ids for tag in tags):raise ValueError("Explicit supported target_tags required")
        tags=list(dict.fromkeys(tags));target={ids[tag] for tag in tags}
        scope=set(categories[list(target)]);eligible=np.isin(categories,list(scope))
        # Strict greater-than matches PixAI's official postprocessor.
        predicted=set(np.flatnonzero((row>thresholds)&eligible).tolist())
        shared=target&predicted;union=target|predicted
        coverage=len(shared)/len(target);iou=len(shared)/len(union)
        result.append(dict(reward=coverage if metric=="coverage" else iou,coverage=coverage,iou=iou,
            matched=len(shared),target_count=len(target),predicted_count=len(predicted),union_count=len(union),
            scope_categories=sorted(scope),matched_tags=[tag for tag in tags if ids[tag] in shared],
            missing_tags=[tag for tag in tags if ids[tag] not in shared],
            predicted_tags=[config["tags"][i] for i in sorted(predicted)],
            target_scores={tag:float(row[ids[tag]]) for tag in tags},
            target_thresholds={tag:float(thresholds[ids[tag]]) for tag in tags}))
    return result
