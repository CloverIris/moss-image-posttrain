"""Shared strict JSON configuration reader; legacy YAML files remain readable."""
import argparse
import json
import math
from pathlib import Path

RUN_DEFAULTS = {"resume":"", "init_adapter":"", "checkpoint":"", "adapter":"",
                "sample_index":0, "seed":42, "decode":False}

def _object(pairs):
    result={}
    for key,value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key]=value
    return result

def _bad_constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")

def read_mapping(path):
    path=Path(path)
    source=path.read_text(encoding="utf-8-sig")
    if path.suffix.lower()==".json":
        content=json.loads(source,object_pairs_hook=_object,parse_constant=_bad_constant)
    else:
        import yaml
        content=yaml.safe_load(source)
        if content is None:
            content={}
    if not isinstance(content,dict):
        raise ValueError("Configuration must be a JSON object / mapping")
    def finite(value):
        if isinstance(value,float) and not math.isfinite(value):
            raise ValueError("Configuration numbers must be finite")
        if isinstance(value,dict):
            for child in value.values(): finite(child)
        elif isinstance(value,list):
            for child in value: finite(child)
    finite(content)
    return content

def validate_run(run, command=None, *, sft=False):
    for key in ("resume","init_adapter","checkpoint","adapter"):
        if not isinstance(run[key],str) or "\x00" in run[key]:
            raise ValueError(f"run.{key} must be a path string or empty")
    if type(run["sample_index"]) is not int or run["sample_index"]<0:
        raise ValueError("run.sample_index must be a nonnegative integer")
    if type(run["seed"]) is not int or not 0<=run["seed"]<2**32:
        raise ValueError("run.seed must be uint32")
    if type(run["decode"]) is not bool:
        raise ValueError("run.decode must be boolean")
    sources=[k for k in ("resume","init_adapter","checkpoint","adapter") if run[k]]
    if len(sources)>1:
        raise ValueError("run model-state paths are mutually exclusive")
    allowed={"train":{"resume","init_adapter"},"sample":{"adapter","checkpoint"},
             "evaluate":{"adapter","checkpoint"},"doctor":set(),"prepare-text":set()}
    if command is not None and any(k not in allowed[command] for k in sources):
        raise ValueError(f"run state is not applicable to {command}")
    if command=="evaluate" and not sources:
        raise ValueError("evaluate requires run.checkpoint or run.adapter")
    if command not in (None,"sample") and (run["sample_index"]!=0 or run["seed"]!=42):
        raise ValueError("run.sample_index and run.seed are only used by sample; training uses training.seed")
    if command not in (None,"evaluate") and run["decode"]:
        raise ValueError("run.decode is only used by evaluate")
    return run

def parse_file_arguments(schema, argv=None):
    """Reuse an argparse schema to validate a flat preprocessing/decode JSON file.

    The actual command line accepts ONLY --config. Original argument declarations
    are the field schema, so required values and choices cannot silently drift.
    """
    cli=argparse.ArgumentParser(description=schema.description)
    cli.add_argument("--config",required=True,help="JSON configuration file")
    options=cli.parse_args(argv)
    values=read_mapping(options.config)
    actions={a.dest:a for a in schema._actions if a.dest!="help"}
    unknown=set(values)-set(actions)
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    arguments=[]
    for key,value in values.items():
        a=actions[key]
        if value is None and a.default is None and not a.required:
            continue
        flag=a.option_strings[0]
        if isinstance(a,argparse._StoreTrueAction):
            if type(value) is not bool: raise ValueError(f"{key} must be boolean")
            if value: arguments.append(flag)
            continue
        sequence=a.nargs in ("*","+")
        if sequence and not isinstance(value,list): raise ValueError(f"{key} must be an array")
        items=value if sequence else [value]
        for item in items:
            expected=a.type or str
            if expected is int and type(item) is not int: raise ValueError(f"{key} must be integer")
            if expected is str and not isinstance(item,str): raise ValueError(f"{key} must be string")
            if isinstance(item,str) and "\x00" in item: raise ValueError(f"{key} contains NUL")
        if sequence:
            arguments.append(flag)
            arguments.extend(str(x) for x in items)
        else:
            arguments.append(flag+"="+str(value))
    return schema.parse_args(arguments)
