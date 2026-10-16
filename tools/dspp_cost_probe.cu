// Diagnostic ablation only: not imported by production training.
// Reuse the exact pinned FA2 split implementation and precision contract.
#include "../megatron/core/kv_cache/fa_split_src/backward.cu"

__global__ void diagnostic_combined_kernel(Flash_bwd_params p, int history) {
    // Limited to tile-aligned boundaries in this diagnostic.
    if (int(blockIdx.x) * 64 < history) {
        dspp_flash_split::compute_dq_dk_dv_1colblock<
            Traits, false, true, false, false, false, true, false,
            false, false, true, true, false>(p, blockIdx.y, blockIdx.z, blockIdx.x);
    } else {
        dspp_flash_split::compute_dq_dk_dv_1colblock<
            Traits, false, true, false, false, false, true, false,
            false, false, true, true, true>(p, blockIdx.y, blockIdx.z, blockIdx.x);
    }
}

std::vector<at::Tensor> diagnostic_combined(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& o, const at::Tensor& dout, const at::Tensor& lse,
    const at::Tensor& delta, double scale) {
    check_inputs(q, k, v, o, dout, lse, scale);
    check_delta(q, delta);
    const c10::cuda::CUDAGuard guard(q.device());
    const int history = k.size(0) - q.size(0);
    TORCH_CHECK(history % 64 == 0 && q.size(0) % 64 == 0,
                "Diagnostic probe only supports aligned current/history boundaries");
    auto dq = at::empty_like(q);
    auto dk = at::empty({q.size(0), k.size(1), 128}, k.options());
    auto dv = at::empty_like(dk);
    auto accum = at::zeros({(q.size(0)+127)/128*128, q.size(1), 128}, q.options().dtype(at::kFloat));
    auto pk = at::empty({k.size(0), q.size(1), 128}, q.options());
    auto pv = at::empty_like(pk);
    auto p = params(q, k, v, o, dout, lse, delta, scale);
    p.dq_ptr=dq.data_ptr(); p.dq_accum_ptr=accum.data_ptr();
    p.dq_row_stride=dq.stride(0); p.dq_head_stride=128; p.dq_batch_stride=dq.numel();
    p.dk_ptr=pk.data_ptr(); p.dv_ptr=pv.data_ptr();
    p.dk_row_stride=pk.stride(0); p.dv_row_stride=pv.stride(0);
    p.dk_head_stride=p.dv_head_stride=128;
    p.dk_batch_stride=pk.numel(); p.dv_batch_stride=pv.numel();
    auto stream=at::cuda::getCurrentCUDAStream();
    C10_CUDA_CHECK(cudaFuncSetAttribute(diagnostic_combined_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, Traits::kSmemSize1colblock));
    diagnostic_combined_kernel<<<dim3(p.seqlen_k/64, 1, p.h), Traits::kNThreads,
        Traits::kSmemSize1colblock, stream>>>(p, history);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    convert_dq_kernel<<<dim3((p.seqlen_q+63)/64, 1, p.h), Traits::kNThreads,
        Traits::kSmemdQSize, stream>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    at::sum_out(dk, pk.slice(0,history).view({q.size(0),p.h_k,p.h_h_k_ratio,128}), {2});
    at::sum_out(dv, pv.slice(0,history).view({q.size(0),p.h_k,p.h_h_k_ratio,128}), {2});
    return {dq,dk,dv};
}
