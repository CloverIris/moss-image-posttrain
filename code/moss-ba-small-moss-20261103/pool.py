"""Small process and lock helpers for the fresh run."""
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path

GPUS=(0,1,2,3)

@contextmanager
def lock(path,nonblocking=False):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|(fcntl.LOCK_NB if nonblocking else 0))
        yield

def identity(pid):
    p=Path(f"/proc/{pid}")
    try:
        s=(p/"stat").read_text().rsplit(") ",1)[1].split()
        if s[0]=="Z":return None
        return s[19],(p/"cmdline").read_bytes().rstrip(b"\0").split(b"\0")
    except (FileNotFoundError,ProcessLookupError):return None

def matches(m):
    return identity(m["pid"])==(str(m["birth"]),[os.fsencode(v) for v in m["command"]])
