"""Host-only PCIe topology discovery and deterministic GPU-pair selection.

Physical GPU indices identify devices only. Proximity comes from resolved Linux
sysfs PCI paths, not index order or nvidia-smi's coarse PXB labels. Pair routes
are chosen together so independent pipeline replicas avoid sharing uplinks.
No accelerator framework is imported or initialized.
"""
from __future__ import annotations

from collections import Counter
import csv
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path, PurePosixPath
import re
import subprocess
from typing import Sequence


_BDF = re.compile(r"^(?P<domain>[0-9a-f]{4,8}):(?P<bus>[0-9a-f]{2}):(?P<device>[0-9a-f]{2})\.(?P<function>[0-7])$", re.I)
_ROOT = re.compile(r"^pci[0-9a-f]{4,8}:[0-9a-f]{2}$", re.I)


def normalize_bus_id(value: str) -> str:
    """Normalize nvidia-smi's eight-digit domain to the Linux sysfs spelling."""
    match = _BDF.fullmatch(str(value).strip())
    if match is None or int(match["device"], 16) > 31:
        raise ValueError(f"Invalid PCI bus ID: {value!r}")
    return (f'{int(match["domain"], 16):04x}:{int(match["bus"], 16):02x}:'
            f'{int(match["device"], 16):02x}.{match["function"]}')


def _physical_indices(values: Sequence[int], label: str) -> list[int]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{label} must be a nonempty list of physical GPU indices")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        raise ValueError(f"{label} must contain nonnegative integer GPU indices")
    if len(values) != len(set(values)):
        raise ValueError(f"{label} contains duplicate GPU indices")
    return list(values)


@dataclass(frozen=True)
class GPUDevice:
    index: int
    uuid: str
    bus_id: str
    pci_path: str
    name: str = ""

    def __post_init__(self):
        _physical_indices([self.index], "GPU index")
        if not isinstance(self.uuid, str) or not self.uuid.startswith("GPU-") or "," in self.uuid:
            raise ValueError(f"Invalid GPU UUID for physical GPU {self.index}: {self.uuid!r}")
        bus = normalize_bus_id(self.bus_id)
        path = PurePosixPath(str(self.pci_path).replace("\\", "/"))
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError(f"GPU {self.index} requires an absolute resolved PCI path")
        if normalize_bus_id(path.name) != bus:
            raise ValueError(f"GPU {self.index} PCI path endpoint does not match bus {bus}")
        object.__setattr__(self, "bus_id", bus)
        object.__setattr__(self, "pci_path", str(path))
        if not self.ancestor_paths:
            raise ValueError(f"GPU {self.index} has no PCI ancestors in {path}")

    @property
    def ancestor_paths(self) -> tuple[str, ...]:
        path = PurePosixPath(self.pci_path)
        return tuple(str(PurePosixPath(*path.parts[:i+1])) for i, part in enumerate(path.parts)
                     if _BDF.fullmatch(part))

    @property
    def root_complex(self) -> str | None:
        path = PurePosixPath(self.pci_path)
        return next((str(PurePosixPath(*path.parts[:i+1])) for i, part in enumerate(path.parts)
                     if _ROOT.fullmatch(part)), None)

    def as_dict(self) -> dict:
        return {"index": self.index, "uuid": self.uuid, "bus_id": self.bus_id,
                "name": self.name, "pci_path": self.pci_path,
                "ancestor_paths": list(self.ancestor_paths), "root_complex": self.root_complex}


def parse_nvidia_smi_csv(text: str) -> list[dict]:
    """Parse the explicit index,uuid,pci.bus_id,name query without CUDA imports."""
    devices = []
    for line, row in enumerate(csv.reader(str(text).splitlines()), 1):
        if not row or not any(value.strip() for value in row):
            continue
        if len(row) != 4:
            raise ValueError(f"Invalid nvidia-smi CSV at line {line}: expected four fields")
        index, uuid, bus, name = (value.strip() for value in row)
        if not index.isdecimal():
            raise ValueError(f"Invalid physical GPU index at CSV line {line}: {index!r}")
        devices.append({"index": int(index), "uuid": uuid, "bus_id": normalize_bus_id(bus), "name": name})
    if not devices:
        raise ValueError("nvidia-smi returned no GPU devices")
    return devices


def _sysfs_pci_path(sysfs_root: str | Path, bus_id: str) -> str:
    path = Path(sysfs_root) / bus_id
    try:
        return str(path.resolve(strict=True))
    except OSError as error:
        raise RuntimeError(f"Cannot resolve PCI sysfs device {path}: {error}") from error


def _validate_inventory(devices: Sequence[GPUDevice]) -> dict[int, GPUDevice]:
    by_index = {device.index: device for device in devices}
    if len(by_index) != len(devices):
        raise ValueError("GPU inventory contains duplicate physical indices")
    for attribute in ("uuid", "bus_id"):
        if len({getattr(device, attribute) for device in devices}) != len(devices):
            raise ValueError(f"GPU inventory contains duplicate {attribute} values")
    return by_index


def validate_groups(groups, device_indices: Sequence[int], known_indices=None) -> list[list[int]]:
    selected = _physical_indices(device_indices, "device_indices")
    if not isinstance(groups, (list, tuple)) or not groups:
        raise ValueError("explicit_groups must be a nonempty list of two-GPU groups")
    pairs = []
    for index, group in enumerate(groups):
        pair = _physical_indices(group, f"explicit_groups[{index}]")
        if len(pair) != 2:
            raise ValueError(f"explicit_groups[{index}] must contain exactly two GPUs")
        pairs.append(sorted(pair))
    flat = [value for pair in pairs for value in pair]
    if len(flat) != len(set(flat)):
        raise ValueError("explicit_groups must use every selected GPU exactly once; duplicates found")
    if known_indices is not None:
        unknown = sorted(set(flat)-set(known_indices))
        if unknown:
            raise ValueError(f"Unknown GPU indices in explicit_groups: {unknown}; available: {sorted(known_indices)}")
    if set(flat) != set(selected):
        raise ValueError(f"explicit_groups must cover exactly device_indices {sorted(selected)}; got {sorted(flat)}")
    return pairs


def _common_path(first: str, second: str) -> str:
    common = []
    for left, right in zip(PurePosixPath(first).parts, PurePosixPath(second).parts):
        if left != right:
            break
        common.append(left)
    return str(PurePosixPath(*common)) if common else "/"


def _pair_route(first: GPUDevice, second: GPUDevice) -> dict:
    lca = _common_path(first.pci_path, second.pci_path)
    shared = [path for path in first.ancestor_paths if path in second.ancestor_paths]
    edges = []
    for device in (first, second):
        nodes = ([device.root_complex] if device.root_complex else []) + list(device.ancestor_paths)
        branch = [lca] + [node for node in nodes if node != lca and PurePosixPath(lca) in PurePosixPath(node).parents]
        edges.extend(tuple(sorted((left, right))) for left, right in zip(branch, branch[1:]))
    return {"indices": sorted((first.index, second.index)), "lca_path": lca,
            "lca_pci_depth": len(shared), "shared_ancestor_paths": shared,
            "route_hops": len(edges), "route_edges": edges,
            "crosses_root_complex": first.root_complex != second.root_complex}


def _matchings(indices: tuple[int, ...]):
    if not indices:
        yield []
        return
    for offset in range(1, len(indices)):
        pair = [indices[0], indices[offset]]
        remaining = indices[1:offset]+indices[offset+1:]
        for rest in _matchings(remaining):
            yield [pair]+rest


def build_topology_report(devices: Sequence[GPUDevice], device_indices: Sequence[int], *, explicit_groups=None) -> dict:
    """Rank perfect matchings by shared route links, hops and upstream cost.

Upstream links receive greater contention cost than deeper switch-local links.
No current link speed is used: idle PCIe power management can downshift it.
Explicit group membership overrides selection, while canonical group ordering
still places the deepest shared switch first and physical IDs ascend in pairs.
"""
    by_index = _validate_inventory(devices)
    indices = _physical_indices(device_indices, "device_indices")
    unknown = sorted(set(indices)-set(by_index))
    if unknown:
        raise ValueError(f"Unknown GPU indices {unknown}; available: {sorted(by_index)}")
    if len(indices) % 2:
        raise ValueError("Two-GPU pipeline groups require an even number of selected GPUs")
    pairs = {tuple(pair): _pair_route(by_index[pair[0]], by_index[pair[1]])
             for pair in combinations(sorted(indices), 2)}
    max_depth = max(len(device.ancestor_paths) for device in devices)

    def edge_cost(edge):
        child = max(edge, key=lambda value: len(PurePosixPath(value).parts))
        depth = sum(bool(_BDF.fullmatch(part)) for part in PurePosixPath(child).parts)
        return max(1, max_depth-depth+1)

    def canonical(groups):
        normalized = [sorted(group) for group in groups]
        return sorted(normalized, key=lambda group: (-pairs[tuple(group)]["lca_pci_depth"],
                                                     pairs[tuple(group)]["route_hops"], tuple(group)))

    def evaluate(groups):
        groups = canonical(groups)
        routes = [pairs[tuple(group)] for group in groups]
        counts = Counter(edge for route in routes for edge in route["route_edges"])
        shared = [{"edge": list(edge), "groups_using_link": count, "upstream_weight": edge_cost(edge)}
                  for edge, count in sorted(counts.items()) if count > 1]
        cost = {"cross_root_pairs": sum(route["crosses_root_complex"] for route in routes),
                "shared_link_cost": sum((count-1)*edge_cost(edge) for edge, count in counts.items()),
                "total_route_hops": sum(route["route_hops"] for route in routes),
                "upstream_route_cost": sum(count*edge_cost(edge) for edge, count in counts.items())}
        return {"groups": groups, "cost": cost, "shared_route_links": shared}

    candidates = [evaluate(grouping) for grouping in _matchings(tuple(sorted(indices)))]
    candidates.sort(key=lambda item: (*item["cost"].values(), item["groups"]))
    chosen = candidates[0]
    if explicit_groups is not None:
        chosen = evaluate(validate_groups(explicit_groups, indices, by_index))
    groups = chosen["groups"]
    flat = [index for group in groups for index in group]
    uuids = [by_index[index].uuid for index in flat]
    return {"strategy": "sysfs PCI LCA; minimize root crossings, cross-group shared links, route hops, upstream cost",
            "grouping_source": "explicit" if explicit_groups is not None else "automatic",
            "selected_indices": sorted(indices), "groups": groups,
            "group_uuids": [[by_index[index].uuid for index in group] for group in groups],
            "flattened_indices": flat, "flattened_uuids": uuids,
            "cuda_visible_devices": ",".join(uuids),
            "index_to_uuid": {index: by_index[index].uuid for index in sorted(by_index)},
            "devices": [by_index[index].as_dict() for index in sorted(indices)],
            "pair_routes": [pairs[pair] for pair in sorted(pairs)],
            "selected_cost": chosen["cost"], "shared_route_links": chosen["shared_route_links"],
            "candidate_matchings": candidates}


def discover_gpu_topology(device_indices: Sequence[int], *, explicit_groups=None,
                          sysfs_root: str | Path = "/sys/bus/pci/devices", query_output: str | None = None) -> dict:
    """Query physical GPU index/UUID/BDF and resolve the Linux PCI tree.

``query_output`` injects the exact nvidia-smi CSV for host-only tests. Existing
CUDA_VISIBLE_DEVICES does not change physical NVML/nvidia-smi GPU indices.
"""
    _physical_indices(device_indices, "device_indices")
    if query_output is None:
        try:
            query_output = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id,name", "--format=csv,noheader,nounits"],
                check=True, capture_output=True, text=True).stdout
        except (OSError, subprocess.CalledProcessError) as error:
            raise RuntimeError(f"Cannot query physical GPU topology with nvidia-smi: {error}") from error
    inventory = parse_nvidia_smi_csv(query_output)
    available = [item["index"] for item in inventory]
    unknown = sorted(set(device_indices)-set(available))
    if unknown:
        raise ValueError(f"Unknown GPU indices {unknown}; available: {sorted(available)}")
    if explicit_groups is not None:
        validate_groups(explicit_groups, device_indices, available)
    devices = [GPUDevice(**item, pci_path=_sysfs_pci_path(sysfs_root, item["bus_id"])) for item in inventory]
    return build_topology_report(devices, device_indices, explicit_groups=explicit_groups)
