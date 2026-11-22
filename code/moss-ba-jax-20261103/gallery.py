"""Decode evaluation endpoints using one identical VAE for every column."""
import fcntl
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from common import NATIVE,RUN,read

def main():
    folder=Path(sys.argv[1])
    with (folder/"decode.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (folder/"gallery.jpg").exists():return
        if hasattr(os,"sched_getaffinity"):
            os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:6])
        sys.path.insert(0,str(NATIVE))
        from moss_art.evaluate import LatentDecoder
        cfg=read(RUN/"config.json")
        cols=["euler48","euler8","ba8","euler12","ba12"]
        sheet=Image.new("RGB",(len(cols)*224,4*250),"white")
        pen=ImageDraw.Draw(sheet)
        with LatentDecoder(cfg["model"],precision="fp32",conv_backend="xla") as decoder:
            for i in range(4):
                with np.load(folder/f"{i:02d}.npz") as data:
                    for j,col in enumerate(cols):
                        output=folder/f"prompt_{i}_{col}.png"
                        if not output.exists():
                            Image.fromarray(decoder(data[col])[0]).save(output)
                        with Image.open(output) as image:
                            sheet.paste(image.resize((224,224),Image.Resampling.LANCZOS),(j*224,i*250))
                        pen.text((j*224+5,i*250+226),f"P{i+1}   {col}",fill="black")
        tmp=folder/"gallery.part"
        sheet.save(tmp,format="JPEG",quality=94)
        os.replace(tmp,folder/"gallery.jpg")
if __name__=="__main__":main()

