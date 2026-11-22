"""Record physical PCI ancestry and verify a requested CUDA device order.

Run before JAX imports: PYTHONPATH=. $CONDA_PREFIX/envs/jax_env/bin/python scripts/diagnostics/inspect_gpu_topology.py
This is read-only; it does not guess switch membership from the PXB label.
"""
import argparse
import copy
import ctypes
import json
import os
from pathlib import Path
import subprocess
from moss_art.runtime import gpu_inventory, configure_process, device_groups
from moss_art.sft import DEFAULTS


def pci_path(bus_id):
    domain, bus, device = bus_id.lower().split(":")
    return Path("/sys/bus/pci/devices") / f"{int(domain,16):04x}:{bus}:{device}"


def link_info(path):
    result={"bdf":path.name}
    for name in ("vendor","device","current_link_speed","current_link_width",
                 "max_link_speed","max_link_width"):
        try:
            result[name]=(path/name).read_text().strip()
        except OSError:
            pass
    return result


def cuda_mapping(config, requested):
    configure_process(config,requested)
    driver=ctypes.CDLL("libcuda.so.1")
    def check(status):
        if status:
            raise RuntimeError(f"CUDA driver error {status}")
    check(driver.cuInit(0))
    count=ctypes.c_int()
    check(driver.cuDeviceGetCount(ctypes.byref(count)))
    result=[]
    for ordinal in range(count.value):
        device=ctypes.c_int()
        check(driver.cuDeviceGet(ctypes.byref(device),ordinal))
        bdf=ctypes.create_string_buffer(32)
        check(driver.cuDeviceGetPCIBusId(bdf,len(bdf),device))
        result.append({"ordinal":ordinal,"pci_bus_id":bdf.value.decode()})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices",default="0,3,1,2",help="nvidia-smi indices, in TP pair order")
    parser.add_argument("--output",default="reports/pcie_topology.json")
    args=parser.parse_args()
    devices=[int(x) for x in args.devices.split(",")]
    inventory=gpu_inventory()
    config=copy.deepcopy(DEFAULTS)
    config["runtime"].update(devices=devices,tensor_parallel_size=2)
    groups=device_groups(config)
    for index in devices:
        entry=inventory[index]
        path=pci_path(entry["pci_bus_id"]).resolve(strict=True)
        entry["sysfs_path"]=str(path)
        entry["pci_ancestry"]=[link_info(p) for p in reversed([path,*path.parents]) if (p/"vendor").is_file()]
    actual=cuda_mapping(config,devices)
    expected=[pci_path(inventory[index]["pci_bus_id"]).name for index in devices]
    observed=[pci_path(item["pci_bus_id"]).name for item in actual]
    if expected!=observed:
        raise RuntimeError(f"CUDA mapping differs: expected {expected}, got {observed}")
    result={"physical_groups":groups,"inventory":inventory,
            "cuda_visible_devices":os.environ["CUDA_VISIBLE_DEVICES"],
            "cuda_mapping":actual,"mapping_verified":True}
    for label,cmd in (("lspci_tree",["lspci","-tv"]),
                      ("nvidia_topology",["nvidia-smi","topo","-m"]),
                      ("p2p_read",["nvidia-smi","topo","-p2p","r"]),
                      ("p2p_write",["nvidia-smi","topo","-p2p","w"])):
        result[label]=subprocess.run(cmd,text=True,capture_output=True,check=True).stdout
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"output":str(output),"physical_groups":groups,"cuda_mapping":actual,"mapping_verified":True},indent=2))


if __name__=="__main__":
    main()
