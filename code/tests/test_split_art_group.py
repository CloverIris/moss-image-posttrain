import copy
import numpy as np
import pytest
from moss_art.config import load_config,validate
from moss_art.runtime import WorkerPool
from moss_art.trainer import make_payloads,aggregate_gradients


def test_split_groups_match_unsharded_art_gradients():
    from moss_art.art import covariance,prompt_windows
    c=load_config("configs/smoke_cpu.yaml")
    c["runtime"].update(devices=[0],cpu_threads=1)
    c["sampling"].update(prompts_per_rollout=1,group_size=4,split_group_across_workers=True,
        height=32,width=32,steps=2,train_timesteps=[0,1],skip_layers=[])
    records=[dict(sample_id="a",prompt="a red bird",negative_prompt="",metadata={"target_tags":["1girl"]})]
    rewards=np.array([0,.25,.75,1.],np.float32);groups=np.zeros(4,np.int32)
    windows=prompt_windows(groups,2,seed=42)
    reference=None;endpoints=None
    for devices in ([0],[0,1]):
        config=copy.deepcopy(c);config["runtime"]["devices"]=devices;validate(config)
        with WorkerPool(config) as pool:
            outputs=pool.call("rollout",make_payloads(config,records,0))
            current=np.concatenate([x["endpoints"] for x in outputs])
            assert sum([x["target_tags"] for x in outputs],[])==[["1girl"]]*4
            if endpoints is None:endpoints=current
            else:np.testing.assert_array_equal(endpoints,current)
            stats={k:np.asarray(v) for k,v in covariance(current,rewards,groups,group_size=4).items()}
            payloads=[];offset=0
            for output in outputs:
                lo=offset;hi=lo+len(output["endpoints"]);offset=hi
                local={key:value[np.unique(groups[lo:hi])] if key.startswith("group_") else value[lo:hi] if key.startswith("row_") else value for key,value in stats.items()}
                local_windows=[np.asarray([(int(row)-lo,int(slot)) for row,slot in w if lo<=row<hi],np.int64).reshape(-1,2) for w in windows]
                payloads.append(dict(stats=local,windows=local_windows))
            pool.call("prepare_update",payloads)
            gradients=[aggregate_gradients(pool.call("gradient",[dict(window=i) for _ in devices]))[0] for i in range(2)]
            if reference is None:reference=gradients
            else:
                for a,b in zip(reference,gradients):
                    for key in a:np.testing.assert_allclose(a[key],b[key],rtol=2e-5,atol=2e-7)


def test_async_condition_buffer_broadcasts_one_prompt_to_four_consumers():
    from moss_art.prefetch import DoubleBuffer
    c=load_config("configs/smoke_cpu.yaml")
    c["preparation"]["async_prefetch"]=True
    c["data"].update(short_context_length=4,context_length=8)
    c["sampling"].update(prompts_per_rollout=1,group_size=4,split_group_across_workers=True)
    records=[dict(sample_id="a",prompt="red bird",negative_prompt="",metadata={},condition_token_counts=[3,2])]
    class Consumers:
        workers=[None]*4
        def call(self,op,payloads):
            assert op=="bind_conditions" and len(payloads)==4
            for payload in payloads:
                assert payload["records"]==records
                assert payload["descriptor"]["positions"]==[0]
            return [dict(generation=p["descriptor"]["generation"],copied=1) for p in payloads]
    with DoubleBuffer(c,records) as buffer:
        event=buffer.bind(Consumers(),records)
        assert event["event"]=="qwen_buffer_acquired"
