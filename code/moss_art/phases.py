"""VAE-only workers; no transformer is constructed in a decode phase."""
import traceback


def vae_worker_main(conn, config, rank, device):
    from .runtime import configure_process
    configure_process(config, device)
    decoder = None
    try:
        from .evaluate import LatentDecoder
        import os
        conn.send({"result": {"rank": rank, "pid": os.getpid(), "phase": "vae", "dit_loaded": False}})
        while True:
            request = conn.recv()
            op = request.pop("op")
            if op == "close": break
            if op != "decode": raise ValueError(op)
            endpoints = request["endpoints"]
            if endpoints is None:
                result = None
            else:
                m = config["model"]
                if decoder is None:
                    decoder = LatentDecoder(m["path"], precision=m["vae_precision"],
                        low_precision_stages=m["vae_low_precision_stages"],
                        conv_backend=m["vae_conv_backend"], conv_workspace_mib=m["vae_conv_workspace_mib"])
                result = {"images": decoder(endpoints), "memory_stats": decoder.device.memory_stats() or {}}
            conn.send({"result": result})
    except EOFError: pass
    except BaseException as exc:
        try: conn.send({"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
        except (BrokenPipeError, OSError): pass
    finally:
        if decoder is not None: decoder.close()
        conn.close()


def decode_phase(config, endpoints):
    """All input arrays are host-owned; one decode wave, with no live DiT pool."""
    from .runtime import WorkerPool
    with WorkerPool(config, worker_main=vae_worker_main, cache_base=False) as decoder:
        return decoder.call("decode", [{"endpoints": x} for x in endpoints])
