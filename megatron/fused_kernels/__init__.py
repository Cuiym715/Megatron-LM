# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import os
import pathlib
import subprocess

import torch
from torch.utils import cpp_extension

def load(args):
    # Build path
    srcpath = pathlib.Path(__file__).parent.absolute()
    torch_abi = f"torch{torch.__version__}cxx11abi{torch._C._GLIBCXX_USE_CXX11_ABI}"
    compute_capability = "sm_" + "".join([str(x) for x in torch.cuda.get_device_capability()])
    buildpath = srcpath / 'build' / torch_abi / compute_capability
    _create_build_dir(buildpath)

    # Helper function to build the kernels.
    def _cpp_extention_load_helper(name, sources, extra_cuda_flags):
        return cpp_extension.load(
            name=name,
            sources=sources,
            build_directory=buildpath,
            extra_cflags=['-O3',],
            extra_cuda_cflags=['-O3',
                               '--use_fast_math'] + extra_cuda_flags,
            verbose=(args.rank == 0)
        )

    # ==============
    # Fused softmax.
    # ==============

    if args.masked_softmax_fusion:
        extra_cuda_flags = ['-U__CUDA_NO_HALF_OPERATORS__',
                            '-U__CUDA_NO_HALF_CONVERSIONS__',
                            '--expt-relaxed-constexpr',
                            '--expt-extended-lambda']

        # Upper triangular softmax.
        sources=[srcpath / 'scaled_upper_triang_masked_softmax.cpp',
                 srcpath / 'scaled_upper_triang_masked_softmax_cuda.cu']
        scaled_upper_triang_masked_softmax_cuda = _cpp_extention_load_helper(
            "scaled_upper_triang_masked_softmax_cuda",
            sources, extra_cuda_flags)

        # Masked softmax.
        sources=[srcpath / 'scaled_masked_softmax.cpp',
                 srcpath / 'scaled_masked_softmax_cuda.cu']
        scaled_masked_softmax_cuda = _cpp_extention_load_helper(
            "scaled_masked_softmax_cuda", sources, extra_cuda_flags)

        # Softmax
        sources=[srcpath / 'scaled_softmax.cpp',
                 srcpath / 'scaled_softmax_cuda.cu']
        scaled_softmax_cuda = _cpp_extention_load_helper(
            "scaled_softmax_cuda", sources, extra_cuda_flags)

    # ==========
    # Fast RoPE.
    # ==========

    if args.use_fast_rope:
        extra_cuda_flags = ['-U__CUDA_NO_HALF_OPERATORS__',
                            '-U__CUDA_NO_HALF_CONVERSIONS__',
                            '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
                            '--fmad=false']

        # Rotary pos emb
        sources=[srcpath / 'fast_rotary_pos_emb.cpp',
                 srcpath / 'fast_rotary_pos_emb.cu']
        fast_rotary_pos_emb = _cpp_extention_load_helper(
            "fast_rotary_pos_emb", sources, extra_cuda_flags)

    # =================
    # Context parallel.
    # =================

    if args.context_parallel_size >= 2 \
        or args.kaimm_offload_activation_ratio > 0 \
        or args.kaimm_kv_cache_impl == 'chunked':

        extra_cuda_flags = []

        # Fast flip
        sources=[srcpath / 'fast_flip.cpp',
                 srcpath / 'fast_flip_cuda.cu']
        fast_rotary_pos_emb = _cpp_extention_load_helper(
            "fast_flip_cuda", sources, extra_cuda_flags)

        # Fused mma
        sources=[srcpath / 'wrap_gemm.cpp',
                 srcpath / 'wrap_gemm_cuda.cu']
        fast_rotary_pos_emb = _cpp_extention_load_helper(
            "wrap_gemm_cuda", sources, extra_cuda_flags)

        # Fast cat
        sources=[srcpath / 'fast_cat.cpp',
                 srcpath / 'fast_cat_cuda.cu']
        fast_rotary_pos_emb = _cpp_extention_load_helper(
            "fast_cat_cuda", sources, extra_cuda_flags)


def _get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"],
                                         universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]

    return raw_output, bare_metal_major, bare_metal_minor


def _create_build_dir(buildpath):
    try:
        os.makedirs(buildpath)
    except OSError:
        if not os.path.isdir(buildpath):
            print(f"Creation of the build directory {buildpath} failed")
