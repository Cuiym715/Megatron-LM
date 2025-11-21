import os
import sys
import torch
import unittest

from megatron import get_args
from megatron import get_timers
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.initialize import initialize_megatron
from megatron.training import print_datetime
from megatron.training import setup_model_and_optimizer
from pretrain_llama import model_provider


class TestOptimizerOverlap(unittest.TestCase):
    def test_optimizer_overlap(self):
        here = os.path.dirname(os.path.abspath(__file__))
        with unittest.mock.patch.object(sys, "argv", [
                sys.argv[0],
                "--accumulate-allreduce-grads-in-fp32",
                "--bf16",
                "--encoder-seq-length=256",
                "--hidden-size=256",
                "--kaimm-overlap-optimizer-communication",
                "--lr=0.0001",
                "--max-position-embeddings=256",
                "--merge-file=" + os.path.join(here, "..", "data", "gpt2-merges.txt"),
                "--micro-batch-size=1",
                "--no-masked-softmax-fusion",
                "--num-attention-heads=8",
                "--num-layers=12",
                "--num-layers-per-virtual-pipeline-stage=1",
                "--pipeline-model-parallel-size=4",
                "--train-iters=1",
                "--untie-embeddings-and-output-weights",
                "--use-distributed-optimizer",
                "--vocab-file=" + os.path.join(here, "..", "data", "gpt2-vocab.json")]):
            initialize_megatron(args_defaults={"tokenizer_type": "GPT2BPETokenizer"})
        print_datetime('after megatron is initialized')
        model_type = ModelType.encoder_or_decoder
        models, optimizer, opt_param_scheduler = setup_model_and_optimizer(model_provider, model_type)
        check_once(models, optimizer)


def check_once(models, optimizer):
    args = get_args()
    timers = get_timers()

    DP = mpu.get_data_parallel_world_size()
    PP = mpu.get_pipeline_model_parallel_world_size()
    data_parallel_rank = mpu.get_data_parallel_rank()
    seed = args.seed + args.rank * 12345 + 54321
    if DP < 2:
        raise ValueError("no DP")
    if args.virtual_pipeline_model_parallel_size is None:
        raise ValueError("no virtual PP")
    if not args.kaimm_overlap_optimizer_communication:
        raise ValueError("no DP overlap")
    if not args.untie_embeddings_and_output_weights:
        raise ValueError("tied weights")

    # partition
    gather_ratio = args.kaimm_overlap_gather_ratio
    reduce_ratio = args.kaimm_overlap_reduce_ratio
    overlap_gather = bool(gather_ratio)
    overlap_reduce = bool(reduce_ratio)
    gather_fast_jobs, gather_slow_jobs, reduce_slow_jobs, reduce_fast_jobs = \
        optimizer.partition_into_sub_divisions(overlap_gather, overlap_reduce, gather_ratio, reduce_ratio)

    if overlap_gather:
        pbuf_view_items = optimizer.get_model_param_buffer_dp_views()

        # init
        generator = torch.Generator(device="cuda")
        generator.manual_seed(seed)
        for model_index, dtype, pbuf, pbuf_views in pbuf_view_items:
            pbuf.normal_(generator=generator)
        optimizer.params_may_be_dirty = True

        # reference
        params_list_ref = []
        optimizer.gather_model_params(args, timers)
        for model in models:
            params_list_ref.append([param.clone() for name, param in sorted(model.named_parameters())])

        # clear
        for model_index, dtype, pbuf, pbuf_views in pbuf_view_items:
            pbuf.zero_()
        optimizer.params_may_be_dirty = True
        optimizer.gather_model_params(args, timers)

        # re-init
        generator = torch.Generator(device="cuda")
        generator.manual_seed(seed)
        for model_index, dtype, pbuf, pbuf_views in pbuf_view_items:
            pbuf.normal_(generator=generator)
        optimizer.params_may_be_dirty = True

        # overlap
        params_list_overlap = []
        for job in gather_fast_jobs:
            optimizer.gather_sub_division(args, timers, *job)
            torch.distributed.barrier()  # check there is no dead lock using barrier
            optimizer.post_gather_sub_division(args, timers, *job)
        for model in models:
            params_list_overlap.append([param.clone() for name, param in sorted(model.named_parameters())])
            for _ in range(PP):
                if gather_slow_jobs:
                    job = gather_slow_jobs.pop(0)
                    optimizer.gather_sub_division(args, timers, *job)
                    torch.distributed.barrier()  # check there is no dead lock using barrier
                    optimizer.post_gather_sub_division(args, timers, *job)

        # assert equal
        for params_ref, params_overlap in zip(params_list_ref, params_list_overlap):
            for param_ref, param_overlap in zip(params_ref, params_overlap):
                assert (param_ref == param_overlap).all()

        # assert zero grad
        gbuf_view_items = optimizer.get_model_grad_buffer_dp_views()
        for model_index, dtype, gbuf, gbuf_views in gbuf_view_items:
            assert (gbuf == torch.zeros_like(gbuf)).all()

        del params_list_ref
        del params_list_overlap
        torch.cuda.empty_cache()

    if overlap_reduce:
        gbuf_view_items = optimizer.get_model_grad_buffer_dp_views()
        if len(models) != len(gbuf_view_items):
            raise ValueError("don't know how to init model[n-1]..model[0]")

        # init
        generator = torch.Generator(device="cuda")
        generator.manual_seed(seed)
        for model_index, dtype, gbuf, gbuf_views in reversed(gbuf_view_items):
            gbuf.normal_(generator=generator)

        # reference
        gbufs_ref = []
        optimizer.reduce_model_grads(args, timers)
        for model_index, dtype, gbuf, gbuf_views in gbuf_view_items:
            gbufs_ref.append(gbuf.view(DP, -1)[data_parallel_rank].clone())

        # clear
        for model_index, dtype, gbuf, gbuf_views in gbuf_view_items:
            gbuf.zero_()

        # overlap
        gbufs_overlap = []
        num_backward_to_skip = PP * (len(models) - 1) - len(reduce_slow_jobs)
        generator = torch.Generator(device="cuda")
        generator.manual_seed(seed)
        for model_index, dtype, gbuf, gbuf_view in reversed(gbuf_view_items):
            gbuf.normal_(generator=generator)
            for _ in range(PP):
                if reduce_slow_jobs and num_backward_to_skip <= 0:
                    job = reduce_slow_jobs.pop(0)
                    optimizer.pre_reduce_sub_division(args, timers, *job)
                    optimizer.reduce_sub_division(args, timers, *job)
                    torch.distributed.barrier()  # check there is no dead lock using barrier
                num_backward_to_skip -= 1
        for job in reduce_fast_jobs:
            optimizer.pre_reduce_sub_division(args, timers, *job)
            optimizer.reduce_sub_division(args, timers, *job)
            torch.distributed.barrier()  # check there is no dead lock using barrier
        for model_index, dtype, gbuf, gbuf_views in gbuf_view_items:
            gbufs_overlap.append(gbuf.view(DP, -1)[data_parallel_rank])

        # assert equal
        for gbuf_overlap, gbuf_ref in zip(gbufs_overlap, gbufs_ref):
            slice_size = 2 ** 28
            for i in range(0, slice_size, len(gbuf_ref)):
                gbuf_overlap_slice = gbuf_overlap[i:i + slice_size]
                gbuf_ref_slice = gbuf_ref[i:i + slice_size]
                torch.testing.assert_close(gbuf_overlap_slice, gbuf_ref_slice)

        del gbufs_ref
        del gbufs_overlap
        torch.cuda.empty_cache()


# Test way 1: using mock args
#   torchrun --nproc-per-node 8 -m unittest path/to/test_xxx.py
#   OR
#   mpirun --allow-run-as-root -np 8 python3 -m unittest path/to/test_xxx.py
#
# Test way 2: using real args
#   Use the pretrain script, but change pretrain_xxx.py to path/to/test_xxx.py

if __name__ == "__main__":
    initialize_megatron(args_defaults={"tokenizer_type": "GPT2BPETokenizer"})
    print_datetime('after megatron is initialized')
    model_type = ModelType.encoder_or_decoder
    models, optimizer, opt_param_scheduler = setup_model_and_optimizer(model_provider, model_type)
    check_once(models, optimizer)
