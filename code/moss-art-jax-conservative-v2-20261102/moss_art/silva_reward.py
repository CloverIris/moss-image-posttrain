"""Legacy worker fallback; the Mesh runtime scores in its existing process."""
import traceback
import numpy as np


def worker_main(conn,c,rank,device):
    from .runtime import configure_process
    configure_process(c,device)
    model=None
    try:
        from .silva_jax import SilvaScorerJax
        reward=c["reward"]
        model=SilvaScorerJax.from_pretrained(reward["model_path"],reward["backbone_path"],dtype=reward["dtype"],attention_backend=reward["attention_backend"])
        conn.send({"result":{"rank":rank,"phase":"silva"}})
        while True:
            request=conn.recv();op=request.pop("op")
            if op=="close":break
            if op!="score":raise ValueError(op)
            conn.send({"result":model.score(request["images"],batch_size=reward["batch_size"])})
    except EOFError:pass
    except BaseException as exc:
        try:conn.send({"error":str(exc),"traceback":traceback.format_exc()})
        except (BrokenPipeError,OSError):pass
    finally:
        if model is not None:model.close()
        conn.close()


def score_round(c,images):
    from .runtime import WorkerPool
    indices=np.array_split(np.arange(len(images)),len(c["runtime"]["devices"]))
    with WorkerPool(c,worker_main=worker_main,cache_base=False) as pool:
        values=pool.call("score",[dict(images=np.asarray(images)[part]) for part in indices])
    return [item for chunk in values for item in chunk]
