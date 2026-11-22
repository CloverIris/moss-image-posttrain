"""Topology pairing is host-only and determined by PCI routes, not GPU IDs."""
import json
from unittest.mock import patch

import pytest

from moss_art.gpu_topology import (
    GPUDevice, build_topology_report, discover_gpu_topology,
    normalize_bus_id, parse_nvidia_smi_csv, validate_groups,
)


ROOT = "/sys/devices/pci0000:00/0000:00:06.0/0000:01:00.0"
PATHS = {
    0: ROOT+"/0000:02:04.0/0000:06:00.0/0000:07:00.0/0000:08:00.0",
    1: ROOT+"/0000:02:04.0/0000:06:00.0/0000:07:10.0/0000:09:00.0/0000:0a:04.0/0000:0e:00.0/0000:0f:00.0/0000:10:00.0",
    2: ROOT+"/0000:02:04.0/0000:06:00.0/0000:07:10.0/0000:09:00.0/0000:0a:08.0/0000:12:00.0/0000:13:00.0/0000:14:00.0",
    3: ROOT+"/0000:02:08.0/0000:1b:00.0/0000:1c:00.0/0000:1d:00.0",
    4: "/sys/devices/pci0000:00/0000:00:06.3/0000:25:00.0",
}


def inventory(remap=None):
    remap = remap or {}
    return [GPUDevice(remap.get(index, index), f"GPU-device-{index}", path.rsplit("/", 1)[-1], path, "test GPU")
            for index, path in PATHS.items()]


def test_real_server_tree_prefers_inner_switch_pair_and_avoids_overlap():
    report = build_topology_report(inventory(), [0, 1, 2, 3])
    assert report["groups"] == [[1, 2], [0, 3]]
    assert report["flattened_indices"] == [1, 2, 0, 3]
    assert report["flattened_uuids"] == ["GPU-device-1", "GPU-device-2", "GPU-device-0", "GPU-device-3"]
    assert report["index_to_uuid"][4] == "GPU-device-4"
    assert report["selected_cost"]["shared_link_cost"] == 0
    assert report["shared_route_links"] == []
    assert report["candidate_matchings"][1]["cost"]["shared_link_cost"] > 0
    assert report["candidate_matchings"][2]["cost"]["shared_link_cost"] > 0
    pair = next(row for row in report["pair_routes"] if row["indices"] == [1, 2])
    assert pair["lca_path"].endswith("0000:09:00.0")
    assert report["devices"][1]["ancestor_paths"][-1] == PATHS[1]
    json.dumps(report, allow_nan=False)


def test_renumbered_devices_follow_hardware_tree_and_input_order_is_irrelevant():
    devices = inventory({0: 9, 1: 3, 2: 8, 3: 1, 4: 12})
    first = build_topology_report(devices, [9, 1, 8, 3])
    second = build_topology_report(list(reversed(devices)), [3, 8, 1, 9])
    assert first["groups"] == [[3, 8], [1, 9]]
    assert first["groups"] == second["groups"]
    assert first["flattened_uuids"] == ["GPU-device-1", "GPU-device-2", "GPU-device-3", "GPU-device-0"]


def test_explicit_groups_override_membership_but_are_canonicalized():
    report = build_topology_report(inventory(), [0, 1, 2, 3], explicit_groups=[[3, 1], [2, 0]])
    assert report["grouping_source"] == "explicit"
    assert report["groups"] == [[0, 2], [1, 3]]
    assert report["selected_cost"]["shared_link_cost"] > 0
    assert validate_groups([[2, 1], [3, 0]], [0, 1, 2, 3]) == [[1, 2], [0, 3]]


@pytest.mark.parametrize("indices, groups, message", [
    ([0, 1, 2, 7], None, "Unknown GPU indices.*7"),
    ([0, 1, 1, 3], None, "duplicate"),
    ([0, 1, True, 3], None, "nonnegative integer"),
    ([0, 1, 2], None, "even number"),
    ([0, 1, 2, 3], [[0, 1], [1, 3]], "duplicates"),
    ([0, 1, 2, 3], [[0, 1], [2, 4]], "exactly device_indices"),
    ([0, 1, 2, 3], [[0, 1], [2, 7]], "Unknown GPU indices.*7"),
    ([0, 1, 2, 3], [[0, 1, 2], [3]], "exactly two"),
])
def test_invalid_selected_devices_and_explicit_groups_fail_clearly(indices, groups, message):
    with pytest.raises(ValueError, match=message):
        build_topology_report(inventory(), indices, explicit_groups=groups)


def test_csv_discovery_resolves_actual_sysfs_paths_without_gpu_frameworks():
    csv = "\n".join(f"{i}, GPU-device-{i}, 00000000:{path.rsplit('/', 1)[-1].split(':', 1)[1].upper()}, Test GPU"
                    for i, path in PATHS.items())
    by_bus = {path.rsplit("/", 1)[-1]: path for path in PATHS.values()}
    with patch("moss_art.gpu_topology._sysfs_pci_path", side_effect=lambda root, bus: by_bus[bus]) as resolve, \
            patch("moss_art.gpu_topology.subprocess.run") as run:
        report = discover_gpu_topology([3, 2, 1, 0], query_output=csv)
    run.assert_not_called()
    assert resolve.call_count == 5
    assert report["groups"] == [[1, 2], [0, 3]]
    assert report["devices"][3]["bus_id"] == "0000:1d:00.0"


def test_nvidia_smi_command_is_read_only_and_failure_is_explanatory():
    with patch("moss_art.gpu_topology.subprocess.run", side_effect=FileNotFoundError("missing")):
        with pytest.raises(RuntimeError, match="nvidia-smi"):
            discover_gpu_topology([0, 1, 2, 3])


def test_unknown_gpu_fails_before_reading_any_sysfs_path():
    with patch("moss_art.gpu_topology._sysfs_pci_path") as resolve:
        with pytest.raises(ValueError, match="Unknown GPU indices.*7"):
            discover_gpu_topology([0, 1, 2, 7], query_output="0, GPU-one, 0000:08:00.0, test")
    resolve.assert_not_called()


def test_inventory_uuid_and_bus_duplicates_are_rejected():
    devices = inventory()
    duplicate = GPUDevice(6, devices[0].uuid, devices[4].bus_id, devices[4].pci_path)
    with pytest.raises(ValueError, match="duplicate uuid"):
        build_topology_report([*devices, duplicate], [0, 1, 2, 3])


@pytest.mark.parametrize("value", ["", "0000:zz:00.0", "0000:00:ff.0", "0000:01:00.8"])
def test_invalid_bus_ids(value):
    with pytest.raises(ValueError, match="PCI bus ID"):
        normalize_bus_id(value)


def test_domain_case_normalization_and_malformed_csv():
    assert normalize_bus_id("00000000:1D:00.0") == "0000:1d:00.0"
    with pytest.raises(ValueError, match="no GPU"):
        parse_nvidia_smi_csv("")
    with pytest.raises(ValueError, match="four fields"):
        parse_nvidia_smi_csv("0,GPU-one,0000:01:00.0")
    with pytest.raises(ValueError, match="physical GPU index"):
        parse_nvidia_smi_csv("index,GPU-one,0000:01:00.0,GPU")


def test_different_root_complex_is_visible_in_report():
    first = GPUDevice(0, "GPU-left", "0000:01:00.0", "/sys/devices/pci0000:00/0000:01:00.0")
    second = GPUDevice(1, "GPU-right", "0001:01:00.0", "/sys/devices/pci0001:00/0001:01:00.0")
    report = build_topology_report([first, second], [0, 1])
    assert report["pair_routes"][0]["crosses_root_complex"]
    assert report["selected_cost"]["cross_root_pairs"] == 1
