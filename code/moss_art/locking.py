"""Linux advisory lock: one writer per experiment output directory."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import socket

@contextmanager
def output_lock(directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ".run.lock"
    # Keep the inode after release: unlinking permits a second lock inode race.
    with path.open("a+") as file:
        try:
            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            file.seek(0)
            raise RuntimeError(f"Output already has an active writer: {directory}; {file.read()}") from error
        try:
            file.seek(0)
            file.truncate()
            file.write(json.dumps({"pid": os.getpid(), "host": socket.gethostname()}))
            file.flush()
            yield directory
        finally:
            fcntl.flock(file, fcntl.LOCK_UN)
