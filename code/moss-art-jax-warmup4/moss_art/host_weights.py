"""One shared CPU BF16 base retained across DiT process restarts."""
from multiprocessing.shared_memory import SharedMemory
import numpy as np


class HostWeights:
    def __init__(self, model_dir):
        import ml_dtypes
        from vendor.moss_jax.weights import load_host_params
        source = load_host_params(model_dir, "transformer")
        entries = {}; offset = 0
        for key, value in source.items():
            dtype = value.dtype if key == "rope.bands" else np.dtype(ml_dtypes.bfloat16)
            offset = (offset + 63) // 64 * 64
            entries[key] = (offset, value.shape, str(dtype))
            offset += value.size * dtype.itemsize
        # Fail before touching pages if /dev/shm cannot hold the complete base.
        import os
        stat = os.statvfs("/dev/shm")
        if offset > stat.f_bavail * stat.f_frsize:
            raise MemoryError(f"Shared CPU base requires {offset} bytes in /dev/shm")
        self.memory = SharedMemory(create=True, size=offset)
        self.descriptor = {"name": self.memory.name, "entries": entries, "bytes": offset}
        try:
            for key, value in source.items():
                at, shape, dtype = entries[key]
                target = np.ndarray(shape, dtype=np.dtype(dtype), buffer=self.memory.buf, offset=at)
                np.copyto(target, value, casting="unsafe")
            del target
        except BaseException:
            self.close(); raise

    def close(self):
        self.memory.close()
        self.memory.unlink()


def attach(descriptor):
    import ml_dtypes  # register BF16 dtype
    memory = SharedMemory(name=descriptor["name"])
    values = {key: np.ndarray(shape, dtype=np.dtype(dtype), buffer=memory.buf, offset=offset)
              for key, (offset, shape, dtype) in descriptor["entries"].items()}
    for value in values.values(): value.flags.writeable = False
    return memory, values
