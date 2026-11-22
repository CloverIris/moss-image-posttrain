"""Host-only two-shape length policy. Never truncate a condition."""
import json
import math
from dataclasses import dataclass

@dataclass(frozen=True)
class PromptBuckets:
    short: int
    full: int

    def capacity(self, length):
        if not 0 < length <= self.full:
            raise ValueError(f"Condition length {length} exceeds full={self.full}")
        return self.short if length <= self.short else self.full

    @classmethod
    def resolve(cls, lengths, short=0, full=4096, minimum=1, emit=True):
        validate_lengths(short, full)
        if minimum > full:
            raise ValueError("Empty/negative condition exceeds full length")
        lengths = list(lengths)
        kept = sorted(n for n in lengths if 0 < n <= full)
        if not kept:
            raise ValueError("No prompts remain within full length")
        if short == 0:
            # Nearest-rank percentile guarantees >=70% of accepted requests.
            quantile = kept[math.ceil(.7 * len(kept))-1]
            short = min(full, math.ceil(max(quantile, minimum)/64)*64)
        if short < minimum:
            raise ValueError("short length must also fit the empty/negative condition")
        result = cls(short, full)
        if emit:
            print(json.dumps(dict(event="prompt_buckets",short=short,full=full,
                retained=len(kept),dropped=len(lengths)-len(kept),
                short_fraction=sum(n<=short for n in kept)/len(kept),
                short_fraction_all=sum(n<=short for n in kept)/len(lengths))),flush=True)
        return result


def validate_lengths(short, full):
    if type(short) is not int or type(full) is not int or full < 1 or not 0 <= short <= full:
        raise ValueError("Require 0 <= short <= full, full > 0; short=0 selects the 70th percentile")


def log_drop(identifier, length, full, stage):
    event=dict(event="prompt_dropped",id=identifier,tokens=int(length),full=int(full),
               reason="exceeds_full_length",stage=stage)
    print(json.dumps(event,ensure_ascii=False),flush=True)
    return event


def configure_sft_cache(cache, data, emit=True):
    full=data["context_length"]
    lengths=[row["sequence_length"] for row in cache.rows]
    kept=[]
    for row in cache.rows:
        if row["sequence_length"]>full:
            if emit: log_drop(row["id"],row["sequence_length"],full,"sft")
        else: kept.append(row)
    buckets=PromptBuckets.resolve(lengths,data.get("short_context_length",0),full,
                                  minimum=cache.empty_context().shape[1],emit=emit)
    cache.rows=kept
    return buckets
