// Uses FA2 v2.7.4.post1 headers under their BSD-3-Clause license.
// See THIRD_PARTY.md and flash_bwd_split.patch for upstream attribution/changes.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <mutex>
#include <atomic>
#include <memory>

#define FLASH_NAMESPACE dspp_flash_split
#include "flash.h"
#include "flash_bwd_preprocess_kernel.h"
#include "dspp_flash_bwd_kernel.h"

namespace dspp_split_detail {
using dspp_flash_split::Flash_bwd_params;
using Tile64Traits = Flash_bwd_kernel_traits<
    128, 64, 64, 8, 4, 2, 2, true, false, cutlass::bfloat16_t>;

// Pinned FA2 head-dim-128 BF16 configurations, selected by device resources.
using Tile128Traits = Flash_bwd_kernel_traits<
    128, 64, 128, 8, 2, 4, 2, false, false, cutlass::bfloat16_t>;

// H-only dKV has a different resource/performance balance from full backward.
// Existing SM80 measurements favor 64x64; it is also the validated SM89 path.
// Keep this operator policy independent of FA2's D/full-backward dispatch.
using HTraits = Tile64Traits;

struct DeviceConfig {
    int max_smem, sms;
    bool wide;
    std::shared_ptr<std::once_flag> partial = std::make_shared<std::once_flag>();
};
std::atomic<int64_t> property_queries{0};
std::atomic<int64_t> attribute_setups{0};
template<typename Traits> void configure_kernels(int max_smem);

const DeviceConfig& device_config(int device) {
    // Cache by CUDA device, not current free VRAM. A hot call on the same host
    // thread needs only this integer comparison: no mutex or property query.
    static std::mutex mutex;
    static std::map<int, DeviceConfig> configs;
    thread_local int last_device = -1;
    thread_local const DeviceConfig* last = nullptr;
    if (last != nullptr && device == last_device) { return *last; }
    std::lock_guard<std::mutex> lock(mutex);
    auto it = configs.find(device);
    if (it == configs.end()) {
        const auto* props = at::cuda::getDeviceProperties(device);
        property_queries.fetch_add(1, std::memory_order_relaxed);
        TORCH_CHECK(props->major == 8 || props->major == 9,
                    "split backward is built for SM80-SM90 only");
        TORCH_CHECK(props->major * 10 + props->minor == DSPP_FA_SPLIT_ARCH,
                    "device architecture differs from this split backward build target");
        DeviceConfig config{int(props->sharedMemPerBlockOptin), props->multiProcessorCount,
                            props->sharedMemPerBlockOptin >= 144 * 1024};
        const c10::cuda::CUDAGuard guard(device);
        if (config.wide) { configure_kernels<Tile128Traits>(config.max_smem); }
        else { configure_kernels<Tile64Traits>(config.max_smem); }
        it = configs.emplace(device, config).first;
    }
    last = &it->second;
    last_device = device;
    return *last;
}

template<typename Traits> struct TileTag { using type = Traits; };
template<typename Fn> auto dispatch_tile(int device, Fn&& fn) {
    TORCH_CHECK(device >= 0, "split backward requires CUDA tensors");
    if (device_config(device).wide) { return fn(TileTag<Tile128Traits>{}); }
    return fn(TileTag<Tile64Traits>{});
}

template<typename DTraits>
__global__ void prepare_kernel(const Flash_bwd_params p) {
    dspp_flash_split::compute_dot_do_o<false, DTraits>(p);
}

template<typename KernelTraits, bool ComputeDq>
__global__ void split_kernel(const Flash_bwd_params p, int first_k_block) {
    // Is_first/Is_last=false, Seq_parallel=true: delta is already prepared;
    // dQ-only atomically accumulates FP32 partials; dKV-only owns each KV tile.
    dspp_flash_split::compute_dq_dk_dv_1colblock<
        KernelTraits, false, true, false, false, false, true, false,
        false, false, true, ComputeDq, !ComputeDq>(
            p, blockIdx.y, blockIdx.z, blockIdx.x + first_k_block);
}

template<typename DTraits>
__device__ __forceinline__ void dq_current_column(
    const Flash_bwd_params& p, int history, int n_block) {
    // Tiles wholly within history need only dQ. A boundary tile also computes
    // dKV, but its surplus history rows are discarded by the current-only
    // output slice. H later computes/routes those rows exactly once.
    if ((n_block + 1) * DTraits::kBlockN <= history) {
        dspp_flash_split::compute_dq_dk_dv_1colblock<
            DTraits, false, true, false, false, false, true, false,
            false, false, true, true, false>(p, blockIdx.y, blockIdx.z, n_block);
    } else {
        dspp_flash_split::compute_dq_dk_dv_1colblock<
            DTraits, false, true, false, false, false, true, false,
            false, false, true, true, true>(p, blockIdx.y, blockIdx.z, n_block);
    }
}

template<typename DTraits>
__global__ void dq_current_kernel(const Flash_bwd_params p, int history) {
    dq_current_column<DTraits>(p, history, blockIdx.x);
}

// Match FA2's deterministic seq-K traversal: each CTA owns a separate FP32
// dQ partition, visits its KV columns in order, then convert_dQ sums partitions.
// Default kernels above keep the original one-CTA-per-KV-column launch.
template<typename DTraits, bool CurrentDkv>
__global__ void deterministic_dq_kernel(const Flash_bwd_params p, int history) {
    for (int n_block = blockIdx.x; n_block < (p.seqlen_k + DTraits::kBlockN - 1) / DTraits::kBlockN;
         n_block += gridDim.x) {
        if constexpr (CurrentDkv) {
            dq_current_column<DTraits>(p, history, n_block);
        } else {
            dspp_flash_split::compute_dq_dk_dv_1colblock<
                DTraits, false, true, false, false, false, true, false,
                false, false, true, true, false>(p, blockIdx.y, blockIdx.z, n_block);
        }
    }
}

// Separate noncausal specializations: legacy causal GPU kernels above are
// unchanged, including their register and shared-memory resource selection.
template<typename Traits, bool ComputeDq, bool ComputeDkv>
__device__ __forceinline__ void partial_column(const Flash_bwd_params& p, int n_block) {
    dspp_flash_split::compute_dq_dk_dv_1colblock<
        Traits, false, false, false, false, false, true, false,
        false, false, true, ComputeDq, ComputeDkv>(p, blockIdx.y, blockIdx.z, n_block);
}
template<typename Traits, bool ComputeDq>
__global__ void partial_split_kernel(const Flash_bwd_params p, int first) {
    partial_column<Traits,ComputeDq,!ComputeDq>(p,blockIdx.x+first);
}
template<typename Traits>
__global__ void partial_current_kernel(const Flash_bwd_params p, int history) {
    if ((blockIdx.x+1)*Traits::kBlockN <= history) {
        partial_column<Traits,true,false>(p,blockIdx.x);
    } else {
        partial_column<Traits,true,true>(p,blockIdx.x);
    }
}

template<typename DTraits>
__global__ void convert_dq_kernel(const Flash_bwd_params p) {
    dspp_flash_split::convert_dQ<DTraits>(p, 1);
}

template<typename DTraits>
__global__ void convert_dq_deterministic_kernel(const Flash_bwd_params p, int nsplits) {
    dspp_flash_split::convert_dQ<DTraits>(p, nsplits);
}

at::Tensor dq_accumulator(Flash_bwd_params& p, const at::Tensor& q,
                          bool deterministic, int& nsplits) {
    p.deterministic = deterministic;
    nsplits = 1;
    if (!deterministic) {
        return at::zeros({p.seqlen_q_rounded, p.h, 128}, q.options().dtype(at::kFloat));
    }
    // Same SM/head partition count and layout as pinned FA2 (batch=1 here).
    const int sms = device_config(q.get_device()).sms;
    nsplits = (sms + p.h - 1) / p.h;
    p.dq_accum_split_stride = int64_t(p.seqlen_q_rounded) * p.h * 128;
    return at::zeros({nsplits, p.seqlen_q_rounded, p.h, 128}, q.options().dtype(at::kFloat));
}

template<typename DTraits, bool CurrentDkv>
void launch_deterministic_dq(const Flash_bwd_params& p, cudaStream_t stream,
                              int nsplits, int history) {
    auto kernel = deterministic_dq_kernel<DTraits, CurrentDkv>;
    kernel<<<dim3(nsplits, 1, p.h), DTraits::kNThreads,
              DTraits::kSmemSize1colblock, stream>>>(p, history);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename DTraits>
void finish_dq(const Flash_bwd_params& p, cudaStream_t stream, int nsplits) {
    const dim3 grid((p.seqlen_q + DTraits::kBlockM - 1) / DTraits::kBlockM, 1, p.h);
    if (p.deterministic) {
        convert_dq_deterministic_kernel<DTraits><<<grid, DTraits::kNThreads,
            DTraits::kSmemdQSize, stream>>>(p, nsplits);
    } else {
        convert_dq_kernel<DTraits><<<grid, DTraits::kNThreads, DTraits::kSmemdQSize, stream>>>(p);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Joint K/V output pass. Each thread owns one [token, KV-head, dim]
// element and sums the Q-head group's BF16 partials in FP32, casting once.
// The source remains full-address; begin/rows exclude boundary-tile surplus.
__global__ void reduce_dkv_output_kernel(
    const cutlass::bfloat16_t* __restrict__ partial_k,
    const cutlass::bfloat16_t* __restrict__ partial_v,
    cutlass::bfloat16_t* __restrict__ dk,
    cutlass::bfloat16_t* __restrict__ dv,
    int64_t elements, int64_t begin, int kv_heads, int ratio) {
    for (int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         index < elements; index += int64_t(blockDim.x) * gridDim.x) {
        const int dim = index % 128;
        const int64_t token_head = index / 128;
        const int64_t token = token_head / kv_heads;
        const int head = token_head % kv_heads;
        const int64_t source = ((begin + token) * kv_heads + head) * ratio * 128 + dim;
        if (ratio == 1) {
            dk[index] = partial_k[source];
            dv[index] = partial_v[source];
        } else {
            float sum_k = 0.f, sum_v = 0.f;
            for (int qhead = 0; qhead < ratio; ++qhead) {
                sum_k += float(partial_k[source + qhead * 128]);
                sum_v += float(partial_v[source + qhead * 128]);
            }
            dk[index] = cutlass::bfloat16_t(sum_k);
            dv[index] = cutlass::bfloat16_t(sum_v);
        }
    }
}

void write_dkv_output(const at::Tensor& partial_k, const at::Tensor& partial_v,
                      at::Tensor& dk, at::Tensor& dv, int64_t begin, int ratio) {
    const int64_t elements = dk.numel();
    if (!elements) { return; }
    const int blocks = int(std::min<int64_t>((elements + 255) / 256, 65535));
    reduce_dkv_output_kernel<<<blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const cutlass::bfloat16_t*>(partial_k.data_ptr()),
        reinterpret_cast<const cutlass::bfloat16_t*>(partial_v.data_ptr()),
        reinterpret_cast<cutlass::bfloat16_t*>(dk.data_ptr()),
        reinterpret_cast<cutlass::bfloat16_t*>(dv.data_ptr()),
        elements, begin, dk.size(1), ratio);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename Traits> void configure_partial_kernels(int max_smem) {
    TORCH_CHECK(max_smem >= Traits::kSmemSize1colblock && max_smem >= HTraits::kSmemSize1colblock,
                "insufficient shared memory for partial backward");
    auto enable = [](auto kernel, int bytes) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
        attribute_setups.fetch_add(1,std::memory_order_relaxed);
    };
    enable(partial_split_kernel<Traits,true>,Traits::kSmemSize1colblock);
    enable(partial_split_kernel<HTraits,false>,HTraits::kSmemSize1colblock);
    enable(partial_current_kernel<Traits>,Traits::kSmemSize1colblock);
}
void ensure_partial_config(int device) {
    const auto& config = device_config(device);
    std::call_once(*config.partial,[&] {
        const c10::cuda::CUDAGuard guard(device);
        if (config.wide) { configure_partial_kernels<Tile128Traits>(config.max_smem); }
        else { configure_partial_kernels<Tile64Traits>(config.max_smem); }
    });
}

void check_inputs(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                  const at::Tensor& o, const at::Tensor& dout, const at::Tensor& lse,
                  double scale, bool causal) {
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
    TORCH_CHECK(k.sizes() == v.sizes() && k.size(0) > 0 && (!causal || k.size(0) >= q.size(0)) &&
                k.size(1) > 0 && k.size(2) == 128 && q.size(1) % k.size(1) == 0,
                "requires matching nonempty K/V, causal Sk >= Sq, and Hq divisible by Hkv");
    TORCH_CHECK(k.size(0) * q.size(1) * 128 < std::numeric_limits<int>::max() / 4,
                "GQA gradient workspace exceeds indexing limits");
    TORCH_CHECK(lse.device() == q.device() && lse.scalar_type() == at::kFloat &&
                lse.dim() == 2 && lse.size(0) == q.size(1) &&
                lse.size(1) == q.size(0) && lse.is_contiguous(),
                "LSE must be contiguous FP32 [heads, sequence] on Q's device");
    TORCH_CHECK(std::isfinite(scale) && scale > 0 &&
                scale <= std::numeric_limits<float>::max(), "invalid softmax scale");
    (void)device_config(q.get_device());
    if (!causal) { ensure_partial_config(q.get_device()); }
}

Flash_bwd_params params(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                        const at::Tensor& o, const at::Tensor& dout,
                        const at::Tensor& lse, const at::Tensor& delta, double scale, bool causal) {
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
    p.is_bf16 = p.is_seqlens_k_cumulative = true;
    p.is_causal = causal;
    p.window_size_left = p.seqlen_k;
    p.window_size_right = causal ? 0 : p.seqlen_k;
    return p;
}

void check_delta(const at::Tensor& q, const at::Tensor& delta) {
    TORCH_CHECK(delta.device() == q.device() && delta.scalar_type() == at::kFloat &&
                delta.dim() == 2 && delta.size(0) == q.size(1) &&
                delta.size(1) == (q.size(0) + 127) / 128 * 128 && delta.is_contiguous(),
                "delta must be contiguous FP32 [heads, rounded_sequence] on Q's device");
}

template<typename KernelTraits, bool ComputeDq>
void launch(const Flash_bwd_params& p, cudaStream_t stream, int begin, int end) {
    if (begin == end) { return; }
    auto kernel = split_kernel<KernelTraits, ComputeDq>;
    const int first_block = begin / KernelTraits::kBlockN;
    const int blocks = (end + KernelTraits::kBlockN - 1) / KernelTraits::kBlockN - first_block;
    if (p.is_causal) {
        kernel<<<dim3(blocks, 1, p.h), KernelTraits::kNThreads,
                  KernelTraits::kSmemSize1colblock, stream>>>(p, first_block);
    } else {
        partial_split_kernel<KernelTraits,ComputeDq><<<dim3(blocks,1,p.h),KernelTraits::kNThreads,
                  KernelTraits::kSmemSize1colblock,stream>>>(p,first_block);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
template<typename Traits> void configure_kernels(int max_smem) {
    // device_config already obtained the device properties. Validate the exact
    // selected specialization before enabling opt-in dynamic shared memory.
    TORCH_CHECK(max_smem >= Traits::kSmemSize1colblock && max_smem >= Traits::kSmemdQSize &&
                max_smem >= HTraits::kSmemSize1colblock,
                "insufficient shared memory for split backward kernel");
    auto enable = [](auto kernel, int bytes) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
        attribute_setups.fetch_add(1, std::memory_order_relaxed);
    };
    enable(split_kernel<Traits, true>, Traits::kSmemSize1colblock);
    enable(split_kernel<HTraits, false>, HTraits::kSmemSize1colblock);
    enable(dq_current_kernel<Traits>, Traits::kSmemSize1colblock);
    enable(deterministic_dq_kernel<Traits, false>, Traits::kSmemSize1colblock);
    enable(deterministic_dq_kernel<Traits, true>, Traits::kSmemSize1colblock);
    enable(convert_dq_kernel<Traits>, Traits::kSmemdQSize);
    enable(convert_dq_deterministic_kernel<Traits>, Traits::kSmemdQSize);
}
}  // namespace dspp_split_detail

using namespace dspp_split_detail;

// Diagnostics only: counters do not initialize the device or query properties.
std::vector<int64_t> split_dispatch_counters() {
    return {property_queries.load(std::memory_order_relaxed),
            attribute_setups.load(std::memory_order_relaxed)};
}
std::vector<int64_t> split_dispatch_info(int device) {
    const auto& c = device_config(device);
    return {c.max_smem, c.sms, 64, c.wide ? 128 : 64, HTraits::kBlockM, HTraits::kBlockN};
}

template<typename DTraits>
at::Tensor split_prepare_impl(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                          const at::Tensor& o, const at::Tensor& dout,
                          const at::Tensor& lse, double scale, bool causal) {
    check_inputs(q, k, v, o, dout, lse, scale, causal);
    const c10::cuda::CUDAGuard guard(q.device());
    // Zero padded rows too; kernels only use valid rows but state is reproducible.
    auto delta = at::zeros({q.size(1), (q.size(0) + 127) / 128 * 128}, q.options().dtype(at::kFloat));
    auto p = params(q, k, v, o, dout, lse, delta, scale, causal);
    prepare_kernel<DTraits><<<dim3((p.seqlen_q + DTraits::kBlockM - 1) / DTraits::kBlockM, 1, p.h), DTraits::kNThreads, 0,
                     at::cuda::getCurrentCUDAStream()>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return delta;
}

template<typename DTraits>
at::Tensor split_dq_impl(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                     const at::Tensor& o, const at::Tensor& dout, const at::Tensor& lse,
                     const at::Tensor& delta, double scale, bool deterministic, bool causal) {
    check_inputs(q, k, v, o, dout, lse, scale, causal);
    check_delta(q, delta);
    const c10::cuda::CUDAGuard guard(q.device());
    auto dq = at::empty_like(q);
    auto p = params(q, k, v, o, dout, lse, delta, scale, causal);
    TORCH_CHECK(causal || !deterministic,"partial backward currently requires default nondeterministic kernels");
    int nsplits;
    auto accum = dq_accumulator(p, q, deterministic, nsplits);
    p.dq_ptr = dq.data_ptr(); p.dq_accum_ptr = accum.data_ptr();
    p.dq_row_stride = dq.stride(0); p.dq_head_stride = 128; p.dq_batch_stride = dq.numel();
    auto stream = at::cuda::getCurrentCUDAStream();
    if (deterministic) {
        launch_deterministic_dq<DTraits, false>(p, stream, nsplits, 0);
    } else {
        launch<DTraits, true>(p, stream, 0, p.seqlen_k);
    }
    finish_dq<DTraits>(p, stream, nsplits);
    return dq;
}

std::vector<at::Tensor> split_dkv_impl(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                 const at::Tensor& o, const at::Tensor& dout, const at::Tensor& lse,
                                 const at::Tensor& delta, double scale, int64_t begin, int64_t end,
                                 bool compact, bool causal) {
    check_inputs(q, k, v, o, dout, lse, scale, causal);
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
    auto p = params(q, k, v, o, dout, lse, delta, scale, causal);
    p.dk_ptr = partial_k.data_ptr(); p.dv_ptr = partial_v.data_ptr();
    p.dk_row_stride = partial_k.stride(0); p.dv_row_stride = partial_v.stride(0);
    p.dk_head_stride = p.dv_head_stride = 128;
    p.dk_batch_stride = partial_k.numel(); p.dv_batch_stride = partial_v.numel();
    launch<HTraits, false>(p, at::cuda::getCurrentCUDAStream(), begin, end);
    // Read only the requested rows, which the selected tiles fully wrote.
    // Boundary-tile surplus is never returned or included in the reduction.
    auto dk_region = compact ? dk : dk.slice(0, begin, end);
    auto dv_region = compact ? dv : dv.slice(0, begin, end);
    write_dkv_output(partial_k, partial_v, dk_region, dv_region, begin, p.h_h_k_ratio);
    return {dk, dv};
}

template<typename DTraits>
std::vector<at::Tensor> split_dq_current_impl(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& o, const at::Tensor& dout, const at::Tensor& lse,
    const at::Tensor& delta, double scale, bool deterministic, int64_t current_begin, bool causal) {
    check_inputs(q, k, v, o, dout, lse, scale, causal);
    check_delta(q, delta);
    const c10::cuda::CUDAGuard guard(q.device());
    const int history = current_begin < 0 ? k.size(0) - q.size(0) : current_begin;
    TORCH_CHECK(history >= 0 && history <= k.size(0), "invalid current KV boundary");
    auto dq = at::empty_like(q);
    auto dk = at::empty({k.size(0) - history, k.size(1), 128}, k.options());
    auto dv = at::empty_like(dk);
    auto p = params(q, k, v, o, dout, lse, delta, scale, causal);
    TORCH_CHECK(causal || !deterministic,"partial backward currently requires default nondeterministic kernels");
    int nsplits;
    auto accum = dq_accumulator(p, q, deterministic, nsplits);
    // Keep upstream full-address per-Q-head partials and rounding semantics.
    // Only current-intersecting tiles write dKV; only current rows are read.
    auto partial_k = at::empty({k.size(0), q.size(1), 128}, q.options());
    auto partial_v = at::empty_like(partial_k);
    p.dq_ptr = dq.data_ptr(); p.dq_accum_ptr = accum.data_ptr();
    p.dq_row_stride = dq.stride(0); p.dq_head_stride = 128; p.dq_batch_stride = dq.numel();
    p.dk_ptr = partial_k.data_ptr(); p.dv_ptr = partial_v.data_ptr();
    p.dk_row_stride = partial_k.stride(0); p.dv_row_stride = partial_v.stride(0);
    p.dk_head_stride = p.dv_head_stride = 128;
    p.dk_batch_stride = partial_k.numel(); p.dv_batch_stride = partial_v.numel();
    auto stream = at::cuda::getCurrentCUDAStream();
    if (deterministic) {
        launch_deterministic_dq<DTraits, true>(p, stream, nsplits, history);
    } else {
        const dim3 grid((p.seqlen_k+DTraits::kBlockN-1)/DTraits::kBlockN,1,p.h);
        if (causal) {
            dq_current_kernel<DTraits><<<grid,DTraits::kNThreads,DTraits::kSmemSize1colblock,stream>>>(p,history);
        } else {
            partial_current_kernel<DTraits><<<grid,DTraits::kNThreads,DTraits::kSmemSize1colblock,stream>>>(p,history);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    finish_dq<DTraits>(p, stream, nsplits);
    write_dkv_output(partial_k, partial_v, dk, dv, history, p.h_h_k_ratio);
    return {dq, dk, dv};
}

at::Tensor split_prepare(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                          const at::Tensor& o, const at::Tensor& dout,
                          const at::Tensor& lse, double scale, bool causal) {
    return dispatch_tile(q.get_device(), [&](auto tag) {
        using Traits = typename decltype(tag)::type;
        return split_prepare_impl<Traits>(q, k, v, o, dout, lse, scale, causal);
    });
}

at::Tensor split_dq(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                     const at::Tensor& o, const at::Tensor& dout, const at::Tensor& lse,
                     const at::Tensor& delta, double scale, bool deterministic, bool causal) {
    return dispatch_tile(q.get_device(), [&](auto tag) {
        using Traits = typename decltype(tag)::type;
        return split_dq_impl<Traits>(q, k, v, o, dout, lse, delta, scale, deterministic, causal);
    });
}

std::vector<at::Tensor> split_dkv(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                 const at::Tensor& o, const at::Tensor& dout, const at::Tensor& lse,
                                 const at::Tensor& delta, double scale, int64_t begin, int64_t end,
                                 bool compact, bool causal) {
    return split_dkv_impl(q, k, v, o, dout, lse, delta, scale, begin, end, compact, causal);
}

std::vector<at::Tensor> split_dq_current(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& o, const at::Tensor& dout, const at::Tensor& lse,
    const at::Tensor& delta, double scale, bool deterministic, int64_t current_begin, bool causal) {
    return dispatch_tile(q.get_device(), [&](auto tag) {
        using Traits = typename decltype(tag)::type;
        return split_dq_current_impl<Traits>(q, k, v, o, dout, lse, delta, scale, deterministic, current_begin, causal);
    });
}
