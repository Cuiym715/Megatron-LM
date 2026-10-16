#!/usr/bin/env python3
"""Probe one DSPP physical shape in an isolated CUDA process."""

from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from megatron.core.datasets.dspp_compute_profile import (
    DsppCapacityProbeResult,
    DsppComputeShape,
    load_adapter,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--adapter-kwargs-json", default="{}")
    parser.add_argument("--capacity", required=True, type=int)
    parser.add_argument("--history-ratio", type=int, default=4)
    parser.add_argument("--operation", choices=("F", "B", "W"), default="B")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    result = DsppCapacityProbeResult(capacity=args.capacity, feasible=False)
    try:
        device = torch.device(args.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for capacity probing")
        kwargs = json.loads(args.adapter_kwargs_json)
        adapter = partial(load_adapter(args.adapter), **kwargs)
        shape = DsppComputeShape(
            stage=0,
            operation=args.operation,
            capacity=args.capacity,
            query_lengths=(args.capacity,),
            history_lengths=(args.history_ratio * args.capacity,),
        )
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        built = adapter(shape=shape, device=args.device)
        run = built if callable(built) else built["run"]
        use_cuda_graph = False if callable(built) else bool(built.get("use_cuda_graph", True))
        run()
        torch.cuda.synchronize(device)
        graph = None
        if use_cuda_graph:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            graph.replay()
            torch.cuda.synchronize(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        del free_bytes
        result = DsppCapacityProbeResult(
            capacity=args.capacity,
            feasible=True,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
            total_memory_bytes=total_bytes,
        )
    except (RuntimeError, ValueError) as error:
        reason = str(error).splitlines()[0][:500]
        total_bytes = 0
        if torch.cuda.is_available():
            try:
                _free, total_bytes = torch.cuda.mem_get_info(args.device)
            except RuntimeError:
                pass
        result = DsppCapacityProbeResult(
            capacity=args.capacity,
            feasible=False,
            total_memory_bytes=total_bytes,
            reason=reason,
        )
    print(json.dumps(result.to_dict()), flush=True)


if __name__ == "__main__":
    main()
