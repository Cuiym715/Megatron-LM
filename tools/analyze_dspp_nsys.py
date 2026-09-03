#!/usr/bin/env python3
"""Extract compact DSPP overlap evidence from an Nsight Systems SQLite export."""

import argparse
import json
import sqlite3
from collections import defaultdict


def _union(intervals):
    merged = []
    for begin, end in sorted(intervals):
        if merged and begin <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([begin, end])
    return merged


def _intersection_ms(left, right):
    left, right = _union(left), _union(right)
    i = j = 0
    total = 0
    while i < len(left) and j < len(right):
        total += max(0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return total / 1e6


def _concurrent_ms(per_stream):
    points = []
    for intervals in per_stream.values():
        for begin, end in _union(intervals):
            points.extend(((begin, 1), (end, -1)))
    active = 0
    previous = None
    total = 0
    for timestamp, delta in sorted(points, key=lambda item: (item[0], item[1])):
        if previous is not None and active >= 2:
            total += timestamp - previous
        active += delta
        previous = timestamp
    return total / 1e6


def _percentile(values, fraction):
    if not values:
        return 0.0
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _task_kernel_summary(connection):
    """Attribute non-NCCL kernels launched during each outer DSPP task range."""

    rows = connection.execute(
        """select n.rowid, n.text, k.deviceId, k.start, k.end
           from NVTX_EVENTS n
           join CUPTI_ACTIVITY_KIND_RUNTIME r
             on (r.globalTid >> 24) = (n.globalTid >> 24)
            and r.start between n.start and n.end
           join CUPTI_ACTIVITY_KIND_KERNEL k
             on k.correlationId = r.correlationId
            and (k.globalPid >> 24) = (r.globalTid >> 24)
           join StringIds s on s.id = k.shortName
           where n.text like 'DSPP/%'
             and n.text not like 'DSPP/P2P/%'
             and s.value not like 'ncclDevKernel%'"""
    )
    per_task = defaultdict(lambda: {"kernel_count": 0, "kernel_ns": 0})
    for row_id, label, device, begin, end in rows:
        key = (int(row_id), str(label), int(device))
        per_task[key]["kernel_count"] += 1
        per_task[key]["kernel_ns"] += int(end) - int(begin)

    grouped = defaultdict(lambda: {"kernel_count": [], "kernel_sum_ms": []})
    for (_row_id, label, device), values in per_task.items():
        kind = label.split("/", 2)[1]
        target = grouped[(device, kind)]
        target["kernel_count"].append(values["kernel_count"])
        target["kernel_sum_ms"].append(values["kernel_ns"] / 1e6)
    result = defaultdict(dict)
    for (device, kind), values in sorted(grouped.items()):
        result[str(device)][kind] = {
            "event_count": len(values["kernel_count"]),
            "kernel_count_median": _percentile(values["kernel_count"], 0.5),
            "kernel_count_p95": _percentile(values["kernel_count"], 0.95),
            "kernel_sum_ms_median": _percentile(values["kernel_sum_ms"], 0.5),
            "kernel_sum_ms_p95": _percentile(values["kernel_sum_ms"], 0.95),
        }
    return dict(result)


def analyze(path):
    connection = sqlite3.connect(path)
    rows = connection.execute(
        """select k.deviceId, k.streamId, k.start, k.end, s.value
           from CUPTI_ACTIVITY_KIND_KERNEL k
           join StringIds s on s.id = k.shortName"""
    )
    per_device = defaultdict(lambda: {"p2p": [], "compute": [], "streams": defaultdict(list)})
    for device, stream, begin, end, name in rows:
        target = per_device[int(device)]
        if name == "ncclDevKernel_SendRecv":
            target["p2p"].append((begin, end))
            target["streams"][int(stream)].append((begin, end))
        elif not name.startswith("ncclDevKernel"):
            target["compute"].append((begin, end))

    directions = defaultdict(int)
    for (text,) in connection.execute(
        "select text from NVTX_EVENTS where text like '%DSPP/P2P/%'"
    ):
        label = text.lstrip(":").split("/", 3)[2]
        direction = label.split("/", 1)[0]
        directions[direction] += 1
    return {
        "devices": {
            str(device): {
                "p2p_kernel_count": len(values["p2p"]),
                "p2p_stream_ids": sorted(values["streams"]),
                "p2p_stream_count": len(values["streams"]),
                "compute_p2p_overlap_ms": _intersection_ms(
                    values["compute"], values["p2p"]
                ),
                "concurrent_p2p_stream_ms": _concurrent_ms(values["streams"]),
            }
            for device, values in sorted(per_device.items())
        },
        "directional_nvtx_instances": dict(sorted(directions.items())),
        "task_non_nccl_kernels": _task_kernel_summary(connection),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite", help="Nsight Systems SQLite export")
    parser.add_argument("--output", help="optional JSON output path")
    args = parser.parse_args()
    result = analyze(args.sqlite)
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as stream:
            stream.write(payload + "\n")


if __name__ == "__main__":
    main()
