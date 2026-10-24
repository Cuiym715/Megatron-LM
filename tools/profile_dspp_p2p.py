#!/usr/bin/env python3
"""Small directed NCCL calibration. Run under torchrun, one rank per GPU.

CUDA event interval includes dispatch-to-completion on each endpoint, excluding
the setup barrier. With --overlap compute, independent GEMMs share the GPU with
communication; this is a specified probe workload, not every training overlap.
"""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from megatron.core.datasets.dspp_calibration import hardware_identity
from megatron.core.datasets.dspp_p2p import DsppP2PCostModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--bytes', nargs='+', type=int, default=[65536, 131072])
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--overlap', nargs='+', choices=['isolated', 'compute'], default=['isolated', 'compute'])
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 1 or any(size <= 0 for size in args.bytes):
        raise ValueError('positive sizes, warmup and repeats required')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    # Materialize event handles, GEMM kernels and operands before samples.
    compute_stream = torch.cuda.Stream()
    x = torch.ones((1024, 1024), dtype=torch.bfloat16, device='cuda')
    y = torch.empty_like(x)
    torch.mm(x, x, out=y)
    begin.record(); end.record(); torch.cuda.synchronize()
    rows = []
    for sender in range(world):
        for receiver in range(world):
            if sender == receiver:
                continue
            for size in args.bytes:
                tensor = torch.empty(size, dtype=torch.uint8, device='cuda')
                for overlap in args.overlap:
                    for iteration in range(args.warmup + args.repeats):
                        dist.barrier()
                        torch.cuda.synchronize()
                        duration = torch.zeros(1, dtype=torch.float64, device='cuda')
                        if rank in (sender, receiver):
                            begin.record()
                            if overlap == 'compute':
                                compute_stream.wait_event(begin)
                                with torch.cuda.stream(compute_stream):
                                    for _ in range(8):
                                        torch.mm(x, x, out=y)
                            op = (dist.P2POp(dist.isend, tensor, receiver) if rank == sender
                                  else dist.P2POp(dist.irecv, tensor, sender))
                            work = dist.batch_isend_irecv([op])[0]
                            work.wait()
                            end.record()
                            torch.cuda.synchronize()
                            duration.fill_(begin.elapsed_time(end))
                        dist.all_reduce(duration, op=dist.ReduceOp.MAX)
                        if rank == 0 and iteration >= args.warmup:
                            rows.append(dict(bytes=size, direction=f'{sender}->{receiver}',
                                             overlap=overlap, duration_ms=duration.item()))
    if rank == 0:
        model = DsppP2PCostModel.from_records(rows).to_dict()
        model.update(records=rows, hardware=hardware_identity(), nccl=list(torch.cuda.nccl.version()),
                     measurement='max endpoint CUDA dispatch-to-completion envelope',
                     compute_probe='8 BF16 1024x1024 GEMMs on independent CUDA stream')
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(model, indent=2))
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
