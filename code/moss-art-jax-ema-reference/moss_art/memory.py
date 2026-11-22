"""Aggregate Linux process/system memory diagnostics; no data contents."""
from pathlib import Path
import resource


def memory_snapshot(devices=()):
    result={"peak_rss_gib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20}
    for filename,prefix,fields in (
        ("/proc/self/smaps_rollup","",{"Rss":"rss_gib","Pss":"pss_gib","Pss_Anon":"pss_anon_gib","Pss_File":"pss_file_gib","Swap":"process_swap_gib"}),
        ("/proc/meminfo","",{"MemAvailable":"system_available_gib"})):
        try:lines=Path(filename).read_text().splitlines()
        except OSError:continue
        for line in lines:
            key,_,value=line.partition(":")
            if key in fields:result[fields[key]]=int(value.strip().split()[0])/2**20
    result["gpu_allocated_gib"]=[(d.memory_stats() or {}).get("bytes_in_use",0)/2**30 for d in devices]
    return result
