"""Read-only inspection of committed experiment outputs, with a human-readable report."""
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from common import RUN,read,save

def alive(role):
    path=RUN/(role+"_process.json")
    if not path.exists():return False
    marker=read(path)
    try:
        fields=Path(f"/proc/{marker['pid']}/stat").read_text().split(") ")[1].split()
        return fields[0]!="Z" and fields[19]==marker["birth"]
    except FileNotFoundError:return False

def report():
    cfg=read(RUN/"config.json")
    latest=read(RUN/"latest.json") if (RUN/"latest.json").exists() else {}
    dp=RUN/"dp4"
    target=len(cfg["order"])*cfg["updates_per_prompt"]
    if (dp/"handoff.json").is_file():
        target=read(dp/"handoff.json")["total_updates"]
        if (dp/"latest.json").is_file():latest=read(dp/"latest.json")
    learner_up=alive("learner")
    if (dp/"handoff.json").is_file() and (dp/"coordinator.json").is_file():
        from pool import matches
        learner_up=matches(read(dp/"coordinator.json"))
    count=len(list((RUN/"latents/train").glob("*.npz")))
    lines=[datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M 北京时间"),
           "MOSS JAX 少步蒸馏，CNN／Attention SideNet",
           f"48步参考已生成 {count}/{len(cfg['order'])} 条",
           f"已提交优化器更新 {latest.get('step',0)}/{target} 次",
           f"已训练 {latest.get('processed_prompts',0)} 条不同提示词",
           f"参考生成进程存活 {alive('producer')}，训练进程存活 {learner_up}"]
    evaluations={}
    for path in sorted([*(RUN/"evaluation").glob("*/metrics.json"),*(dp/"evaluation").glob("*/metrics.json")],key=lambda p:int(p.parent.name)):
        rows=read(path)
        lines.append("固定留出评估，更新 "+path.parent.name)
        groups={}
        for n in cfg["nfe"]:
            batch=[r for r in rows if r["nfe"]==n]
            avg=lambda field:sum(r[field] for r in batch)/len(batch)
            avg_ba,avg_euler,avg_zero=map(avg,("ba_mse","euler_mse","zero_side_ba_mse"))
            groups[str(n)]=dict(pairs=len(batch),ba=avg_ba,euler=avg_euler,zero_side=avg_zero,
                                wins_vs_euler=sum(r["ba_mse"]<r["euler_mse"] for r in batch),
                                wins_vs_zero=sum(r["ba_mse"]<r["zero_side_ba_mse"] for r in batch))
            lines.append(f"  {n}次主干调用，对48步教师的平均latent MSE：BA {avg_ba:.6f}，Euler {avg_euler:.6f}，零残差BA {avg_zero:.6f}。")
        evaluations[path.parent.name]=groups
    for path in sorted(RUN.glob("benchmark*/result.json")):
        measurement=read(path)
        if "observed_denoising_speedup" in measurement:
            lines.append(f"实测同种子去噪速度倍率 {measurement['observed_denoising_speedup']:.2f}，计时与单图数据见 {path}")
    for role in ("producer","learner"):
        if (RUN/(role+"_failure.json")).exists():lines.append(f"存在异常记录 {(RUN/(role+'_failure.json'))}")
    lines.append(f"完整训练状态 {latest.get('checkpoint','')}，各阶段权重和配对原图位于对应阶段的目录。")
    text="\n".join(lines)+"\n"
    tmp=RUN/"progress.txt.part"
    tmp.write_text(text,encoding="utf-8");os.replace(tmp,RUN/"progress.txt")
    save(RUN/"progress.json",dict(reference_prompts=count,latest=latest,evaluations=evaluations,
                                  producer_alive=alive("producer"),learner_alive=learner_up,
                                  finished=(RUN/"complete.json").exists(),checked_at=time.time()))
    return text

if __name__=="__main__":
    if "--watch" not in sys.argv:print(report())
    else:
        import fcntl
        with (RUN/"report.lock").open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            while True:
                report()
                if (RUN/"complete.json").exists() or not (alive("producer") or alive("learner")):break
                time.sleep(60)

