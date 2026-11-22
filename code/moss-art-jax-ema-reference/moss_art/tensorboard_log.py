"""Torch-free scalar TensorBoard events with explicit flushes."""
from pathlib import Path
import time

class ScalarLogger:
    def __init__(self,config,output):
        self.writer=None
        if config.get("logging",{}).get("tensorboard",False):
            import os,socket,uuid
            from tensorboard.summary.writer.record_writer import RecordWriter
            from tensorboard.compat.proto.event_pb2 import Event
            directory=Path(config["logging"].get("log_dir") or Path(output)/"tensorboard")
            directory.mkdir(parents=True,exist_ok=True)
            filename=f"events.out.tfevents.{int(time.time())}.{socket.gethostname()}.{os.getpid()}.{uuid.uuid4().hex[:8]}"
            self.writer=RecordWriter((directory/filename).open("wb"))
            self.writer.write(Event(wall_time=time.time(),file_version="brain.Event:2").SerializeToString())
    def write(self,namespace,step,values):
        if self.writer is None:return
        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.compat.proto.summary_pb2 import Summary
        entries=[Summary.Value(tag=namespace+"/"+key,simple_value=float(value)) for key,value in values.items() if isinstance(value,(int,float)) and not isinstance(value,bool)]
        if entries:self.writer.write(Event(wall_time=time.time(),step=int(step),summary=Summary(value=entries)).SerializeToString())
        self.writer.flush()
    def close(self):
        if self.writer is not None:self.writer.close();self.writer=None
    def __enter__(self):return self
    def __exit__(self,*exc):self.close()
