"""Detached native-JAX thresholded PixAI reward, distributed over free GPUs."""
import traceback
import numpy as np


def worker_main(conn,c,rank,device):
    from .runtime import configure_process
    configure_process(c,device)
    model=None
    try:
        from .pixai_tagger import PixAITagger
        from .tag_reward import hard_scores
        reward=c["reward"]
        model=PixAITagger.from_pretrained(reward["model_path"],metadata_dir=reward["metadata_dir"],
            dtype=reward["dtype"],attention_backend=reward["attention_backend"])
        conn.send({"result":{"rank":rank,"phase":"pixai"}})
        while True:
            request=conn.recv();op=request.pop("op")
            if op=="close":break
            if op!="score":raise ValueError(op)
            images=request["images"];tags=request["target_tags"]
            if not len(images):result=[]
            else:
                probabilities=model.probabilities(images,batch_size=reward["batch_size"])
                result=hard_scores(probabilities,tags,reward["metadata_dir"],reward["metric"])
            conn.send({"result":result})
    except EOFError:pass
    except BaseException as exc:
        try:conn.send({"error":str(exc),"traceback":traceback.format_exc()})
        except (BrokenPipeError,OSError):pass
    finally:
        if model is not None:model.close()
        conn.close()


def score_round(c,images,target_tags):
    from .runtime import WorkerPool
    n=len(c["runtime"]["devices"])
    indices=np.array_split(np.arange(len(images)),n)
    payloads=[dict(images=np.asarray(images)[part],target_tags=[target_tags[i] for i in part]) for part in indices]
    with WorkerPool(c,worker_main=worker_main,cache_base=False) as pool:
        values=pool.call("score",payloads)
    return [item for chunk in values for item in chunk]
