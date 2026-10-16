// Uses FA2 v2.7.4.post1 headers under their BSD-3-Clause license.
// See THIRD_PARTY.md and flash_bwd_split.patch for upstream attribution/changes.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>
#include <limits>

#define FLASH_NAMESPACE dspp_flash_split
#include "flash.h"
#include "flash_bwd_preprocess_kernel.h"
#include "dspp_flash_bwd_kernel.h"

namespace dspp_split_detail {
using dspp_flash_split::Flash_bwd_params;
// This is upstream FA2's BF16/head-dim-128 configuration for SMs with <144 KiB
// opt-in shared memory (including L40). No tile/precision retuning in stage A.
using Traits = Flash_bwd_kernel_traits<
    128, 64, 64, 8, 4, 2, 2, true, false, cutlass::bfloat16_t>;

__global__ void prepare_kernel(const Flash_bwd_params p) {
    dspp_flash_split::compute_dot_do_o<false, Traits>(p);
}

template<bool ComputeDq>
__global__ void split_kernel(const Flash_bwd_params p, int first_k_block) {
    // Is_first/Is_last=false, Seq_parallel=true: delta is already prepared;
    // dQ-only atomically accumulates FP32 partials; dKV-only owns each KV tile.
    dspp_flash_split::compute_dq_dk_dv_1colblock<
        Traits, false, true, false, false, false, true, false,
        false, false, true, ComputeDq, !ComputeDq>(
            p, blockIdx.y, blockIdx.z, blockIdx.x + first_k_block);
}

__global__ void dq_current_kernel(const Flash_bwd_params p, int history) {
    // Tiles wholly within history need only dQ. A boundary tile also computes
    // dKV, but its surplus history rows are discarded by the current-only
    // output slice. H later computes/routes those rows exactly once.
    if ((int(blockIdx.x) + 1) * 64 <= history) {
        dspp_flash_split::compute_dq_dk_dv_1colblock<
            Traits, false, true, false, false, false, true, false,
            false, false, true, true, false>(p, blockIdx.y, blockIdx.z, blockIdx.x);
    } else {
        dspp_flash_split::compute_dq_dk_dv_1colblock<
            Traits, false, true, false, false, false, true, false,
            false, false, true, true, true>(p, blockIdx.y, blockIdx.z, blockIdx.x);
    }
}

__global__ void convert_dq_kernel(const Flash_bwd_params p) {
    dspp_flash_split::convert_dQ<Traits>(p, 1);
}

void check_inputs(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                  const at::Tensor& o, const at::Tensor& dout, const at::Tensor& lse,
                  double scale) {
    TORCH_CHECK(q.is_cuda(), "split backward requires CUDA tensors");
    TORCH_CHECK(q.dim() == 3 && q.size(0) > 0 && q.size(1) > 0 && q.size(2) == 128,
                "split backward requires nonempty [sequence, heads, 128] tensors");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16, "split backward requires BF16");
    TORCH_CHECK(q.is_contiguous(), "split backward requires contiguous Q");
    // Keep all index/rounding arithmetic in the supported upstream range.
    TORCH_CHECK(q.numel() < std::numeric_limits<int>::max() / 4 && q.size(1) <= 65535,
                "tensor exceeds split backward indexing/grid limits");
    for (const auto& t : {k, v, o, dout}) {
        TORCH_CHECK(t.device() == q.device() && t.scalar_type() == q.scalar_type(),
                    "Q/K/V/O/dO must share device and BF16 dtype");
        TORCH_CHECK(t.dim() == 3 && t.numel() < std::numeric_limits<int>::max() / 4,
                    "expected rank-3 tensor within indexing limits");
        TORCH_CHECK(t.is_contiguous(), "split backward requires contiguous Q/K/V/O/dO");
    }
    TORCH_CHECK(o.sizes() == q.sizes() && dout.sizes() == q.sizes(),
                "O/dO must have Q shape");
    TORCH_CHECK(k.sizes() == v.sizes() && k.size(0) >= q.size(0) &&
                k.size(1) > 0 && k.size(2) == 128 && q.size(1) % k.size(1) == 0,
                "requires matching K/V, Sk >= Sq, and Hq divisible by Hkv");
    TORCH_CHECK(k.size(0) * q.size(1) * 128 < std::numeric_limits<int>::max() / 4,
                "GQA gradient workspace exceeds indexing limits");
    TORCH_CHECK(lse.device() == q.device() && lse.scalar_type() == at::kFloat &&
                lse.dim() == 2 && lse.size(0) == q.size(1) &&
                lse.size(1) == q.size(0) && lse.is_contiguous(),
                "LSE must be contiguous FP32 [heads, sequence] on Q's device");
    TORCH_CHECK(std::isfinite(scale) && scale > 0 &&
                scale <= std::numeric_limits<float>::max(), "invalid softmax scale");
    auto* props = at::cuda::getDeviceProperties(q.get_device());
    TORCH_CHECK(props->major == 8 || props->major == 9,
                "split backward is built for SM80-SM90 only");
    TORCH_CHECK(props->major * 10 + props->minor == DSPP_FA_SPLIT_ARCH,
                "device architecture differs from this split backward build target");
    TORCH_CHECK(props->sharedMemPerBlockOptin >= Traits::kSmemSize1colblock,
                "insufficient shared memory for split backward kernel");
}

Flash_bwd_params params(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                        const at::Tensor& o, const at::Tensor& dout,
                        const at::Tensor& lse, const at::Tensor& delta, double scale) {
    Flash_bwd_params p{};
    p.b = 1;
    p.h = q.size(1); p.h_k = k.size(1);
    p.h_h_k_ratio = p.h / p.h_k;
    p.d = p.d_rounded = 128;
    p.seqlen_q = p.total_q = q.size(0); p.seqlen_k = k.size(0);
    p.seqlen_q_rounded = (p.seqlen_q + 127) / 128 * 128;
    p.seqlen_k_rounded = (p.seqlen_k + 127) / 128 * 128;
    p.q_ptr = q.data_ptr(); p.k_ptr = k.data_ptr(); p.v_ptr = v.data_ptr();
    p.o_ptr = o.data_ptr(); p.do_ptr = dout.data_ptr();
    p.softmax_lse_ptr = lse.data_ptr(); p.dsoftmax_sum = delta.data_ptr();
    p.q_row_stride = p.o_row_stride = p.do_row_stride = q.stride(0);
    p.k_row_stride = k.stride(0); p.v_row_stride = v.stride(0);
    p.q_head_stride = p.k_head_stride = p.v_head_stride = p.o_head_stride = p.do_head_stride = 128;
    p.q_batch_stride = p.o_batch_stride = p.do_batch_stride = q.numel();
    p.k_batch_stride = k.numel(); p.v_batch_stride = v.numel();
    p.scale_softmax = scale;
    p.scale_softmax_log2 = scale * M_LOG2E;
    p.p_dropout = p.rp_dropout = 1.f;
    p.p_dropout_in_uint8_t = 255;
    p.scale_softmax_rp_dropout = scale;
    p.is_bf16 = p.is_causal = p.is_seqlens_k_cumulative = true;
    p.window_size_left = p.seqlen_k;
    p.window_size_right = 0;
    return p;
}

void check_delta(const at::Tensor& q, const at::Tensor& delta) {
    TORCH_CHECK(delta.device() == q.device() && delta.scalar_type() == at::kFloat &&
                delta.dim() == 2 && delta.size(0) == q.size(1) &&
                delta.size(1) == (q.size(0) + 127) / 128 * 128 && delta.is_contiguous(),
                "delta must be contiguous FP32 [heads, rounded_sequence] on Q's device");
}

template<bool ComputeDq>
void launch(const Flash_bwd_params& p, cudaStream_t stream, int begin, int end) {
    if (begin == end) { return; }
    auto kernel = split_kernel<ComputeDq>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        Traits::kSmemSize1colblock));
    const int first_block = begin / 64;
    const int blocks = (end + 63) / 64 - first_block;
    kernel<<<dim3(blocks, 1, p.h), Traits::kNThreads,
              Traits::kSmemSize1colblock, stream>>>(p, first_block);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace dspp_split_detail

using namespace dspp_split_detail;

at::Tensor split_prepare(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                          const at::Tensor& o, const at::Tensor& dout,
                          const at::Tensor& lse, double scale) {
    check_inputs(q, k, v, o, dout, lse, scale);
    const c10::cuda::CUDAGuard guard(q.device());
    // Zero padded rows too; kernels only use valid rows but state is reproducible.
    auto delta = at::zeros({q.size(1), (q.size(0) + 127) / 128 * 128}, q.options().dtype(at::kFloat));
    auto p = params(q, k, v, o, dout, lse, delta, scale);
    prepare_kernel<<<dim3((p.seqlen_q + 63) / 64, 1, p.h), Traits::kNThreads, 0,
                     at::cuda::getCurrentCUDAStream()>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return delta;
}

at::Tensor split_dq(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                     const at::Tensor& o, const at::Tensor& dout, const at::Tensor& lse,
                     const at::Tensor& delta, double scale) {
    check_inputs(q, k, v, o, dout, lse, scale);
    check_delta(q, delta);
    const c10::cuda::CUDAGuard guard(q.device());
    auto dq = at::empty_like(q);
    auto accum = at::zeros({(q.size(0) + 127) / 128 * 128, q.size(1), 128}, q.options().dtype(at::kFloat));
    auto p = params(q, k, v, o, dout, lse, delta, scale);
    p.dq_ptr = dq.data_ptr(); p.dq_accum_ptr = accum.data_ptr();
    p.dq_row_stride = dq.stride(0); p.dq_head_stride = 128; p.dq_batch_stride = dq.numel();
    auto stream = at::cuda::getCurrentCUDAStream();
    launch<true>(p, stream, 0, p.seqlen_k);
    convert_dq_kernel<<<dim3((p.seqlen_q + 63) / 64, 1, p.h), Traits::kNThreads,
                         Traits::kSmemdQSize, stream>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return dq;
}

std::vector<at::Tensor> split_dkv(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                 const at::Tensor& o, const at::Tensor& dout, const at::Tensor& lse,
                                 const at::Tensor& delta, double scale, int64_t begin, int64_t end,
                                 bool compact) {
    check_inputs(q, k, v, o, dout, lse, scale);
    check_delta(q, delta);
    const c10::cuda::CUDAGuard guard(q.device());
    TORCH_CHECK(begin >= 0 && begin <= end && end <= k.size(0), "invalid KV region");
    // Defined zero outside the selected region; no uninitialized gradient is
    // exposed to the caller. An empty history launches no main kernel.
    auto dk = compact ? at::empty({end - begin, k.size(1), 128}, k.options()) : at::zeros_like(k);
    auto dv = compact ? at::empty_like(dk) : at::zeros_like(v);
    if (begin == end) { return {dk, dv}; }
    // Upstream FA2 GQA uses one BF16 partial per Q head, accumulated in FP32
    // inside the main kernel, then sums groups into compact KV-head outputs.
    // Inputs K/V remain compact. Only selected KV tiles write this workspace.
    auto partial_k = at::empty({k.size(0), q.size(1), 128}, q.options());
    auto partial_v = at::empty_like(partial_k);
    auto p = params(q, k, v, o, dout, lse, delta, scale);
    p.dk_ptr = partial_k.data_ptr(); p.dv_ptr = partial_v.data_ptr();
    p.dk_row_stride = partial_k.stride(0); p.dv_row_stride = partial_v.stride(0);
    p.dk_head_stride = p.dv_head_stride = 128;
    p.dk_batch_stride = partial_k.numel(); p.dv_batch_stride = partial_v.numel();
    launch<false>(p, at::cuda::getCurrentCUDAStream(), begin, end);
    // Read only the requested rows, which the selected tiles fully wrote.
    // Boundary-tile surplus is never returned or included in the reduction.
    auto dk_region = compact ? dk : dk.slice(0, begin, end);
    auto dv_region = compact ? dv : dv.slice(0, begin, end);
    if (p.h_h_k_ratio == 1) {
        dk_region.copy_(partial_k.slice(0, begin, end));
        dv_region.copy_(partial_v.slice(0, begin, end));
    } else {
        // ATen CUDA BF16 sum uses FP32 accumulation and casts once to BF16,
        // matching the original FA2 API's GQA reduction/rounding contract.
        at::sum_out(dk_region, partial_k.slice(0, begin, end).view(
            {end - begin, p.h_k, p.h_h_k_ratio, 128}), {2});
        at::sum_out(dv_region, partial_v.slice(0, begin, end).view(
            {end - begin, p.h_k, p.h_h_k_ratio, 128}), {2});
    }
    return {dk, dv};
}

std::vector<at::Tensor> split_dq_current(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& o, const at::Tensor& dout, const at::Tensor& lse,
    const at::Tensor& delta, double scale) {
    check_inputs(q, k, v, o, dout, lse, scale);
    check_delta(q, delta);
    const c10::cuda::CUDAGuard guard(q.device());
    const int history = k.size(0) - q.size(0);
    auto dq = at::empty_like(q);
    auto dk = at::empty({q.size(0), k.size(1), 128}, k.options());
    auto dv = at::empty_like(dk);
    auto accum = at::zeros({(q.size(0) + 127) / 128 * 128, q.size(1), 128}, q.options().dtype(at::kFloat));
    // Keep upstream full-address per-Q-head partials and rounding semantics.
    // Only current-intersecting tiles write dKV; only current rows are read.
    auto partial_k = at::empty({k.size(0), q.size(1), 128}, q.options());
    auto partial_v = at::empty_like(partial_k);
    auto p = params(q, k, v, o, dout, lse, delta, scale);
    p.dq_ptr = dq.data_ptr(); p.dq_accum_ptr = accum.data_ptr();
    p.dq_row_stride = dq.stride(0); p.dq_head_stride = 128; p.dq_batch_stride = dq.numel();
    p.dk_ptr = partial_k.data_ptr(); p.dv_ptr = partial_v.data_ptr();
    p.dk_row_stride = partial_k.stride(0); p.dv_row_stride = partial_v.stride(0);
    p.dk_head_stride = p.dv_head_stride = 128;
    p.dk_batch_stride = partial_k.numel(); p.dv_batch_stride = partial_v.numel();
    auto stream = at::cuda::getCurrentCUDAStream();
    C10_CUDA_CHECK(cudaFuncSetAttribute(dq_current_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, Traits::kSmemSize1colblock));
    dq_current_kernel<<<dim3((p.seqlen_k + 63) / 64, 1, p.h), Traits::kNThreads,
        Traits::kSmemSize1colblock, stream>>>(p, history);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    convert_dq_kernel<<<dim3((p.seqlen_q + 63) / 64, 1, p.h), Traits::kNThreads,
        Traits::kSmemdQSize, stream>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (p.h_h_k_ratio == 1) {
        dk.copy_(partial_k.slice(0, history));
        dv.copy_(partial_v.slice(0, history));
    } else {
        at::sum_out(dk, partial_k.slice(0, history).view(
            {q.size(0), p.h_k, p.h_h_k_ratio, 128}), {2});
        at::sum_out(dv, partial_v.slice(0, history).view(
            {q.size(0), p.h_k, p.h_h_k_ratio, 128}), {2});
    }
    return {dq, dk, dv};
}
