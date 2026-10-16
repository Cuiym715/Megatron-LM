"""Built-in PP-free Transformer compute adapter for DSPP calibration.

The adapter measures one physical local Transformer stage.  It contains no
pipeline communication, dataloader, optimizer, or iteration loop.  Model
dimensions are runtime parameters so the same code is rerun on each target
model/GPU/software environment.

F and B execute a Llama-style RMSNorm/attention/SwiGLU block.  B freezes
weights and differentiates activations, excluding weight-gradient GEMMs.  W
replays only the linear weight-gradient GEMM shapes.  This mirrors DSPP's
deferred B/W split without requiring PP merely to collect a compute profile.
"""

from __future__ import annotations

import os
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dspp_compute_profile import DsppComputeShape

try:
    from megatron.model.fused_layer_norm import MixedFusedRMSNorm
except ImportError:  # pragma: no cover - target environment decides support.
    MixedFusedRMSNorm = None

try:
    import fused_weight_gradient_mlp_cuda
except ImportError:  # pragma: no cover - target environment decides support.
    fused_weight_gradient_mlp_cuda = None

try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
except ImportError:  # pragma: no cover - depends on the target environment.
    flash_attn_varlen_func = None


def _dtype(name: str) -> torch.dtype:
    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return aliases[name.lower()]
    except KeyError as error:
        raise ValueError(f"unsupported profile dtype: {name}") from error


class _LocalLlamaBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        num_heads: int,
        num_query_groups: int,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if num_heads % num_query_groups:
            raise ValueError("num_heads must be divisible by num_query_groups")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_query_groups = num_query_groups
        self.head_dim = hidden_size // num_heads
        if MixedFusedRMSNorm is None:
            raise RuntimeError("Megatron fused RMSNorm is required by the local adapter")
        self.input_norm = MixedFusedRMSNorm(hidden_size, eps=1e-5)
        self.post_attention_norm = MixedFusedRMSNorm(hidden_size, eps=1e-5)
        qkv_width = (num_heads + 2 * num_query_groups) * self.head_dim
        self.qkv = nn.Linear(hidden_size, qkv_width, bias=False, dtype=dtype)
        self.o = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)
        self.gate_up = nn.Linear(hidden_size, 2 * ffn_hidden_size, bias=False, dtype=dtype)
        self.down = nn.Linear(ffn_hidden_size, hidden_size, bias=False, dtype=dtype)

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        query_lengths: Sequence[int],
        history_lengths: Sequence[int],
        history_kv: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        cu_q: torch.Tensor,
        cu_k: torch.Tensor,
        rope_freqs: torch.Tensor,
    ) -> torch.Tensor:
        capacity = hidden.size(0)
        valid_tokens = sum(query_lengths)
        normalized = self.input_norm(hidden)
        mixed = self.qkv(normalized).view(
            capacity,
            self.num_query_groups,
            self.num_heads // self.num_query_groups + 2,
            self.head_dim,
        )
        query = mixed[:, :, : self.num_heads // self.num_query_groups].reshape(
            capacity, self.num_heads, self.head_dim
        )
        current_key = mixed[:, :, -2]
        current_value = mixed[:, :, -1]
        query = _apply_rope(query, rope_freqs)
        current_key = _apply_rope(current_key, rope_freqs)

        key_parts: List[torch.Tensor] = []
        value_parts: List[torch.Tensor] = []
        offset = 0
        for query_length, (cached_key, cached_value) in zip(query_lengths, history_kv):
            end = offset + query_length
            if cached_key.numel():
                key_parts.append(cached_key)
                value_parts.append(cached_value)
            key_parts.append(current_key[offset:end])
            value_parts.append(current_value[offset:end])
            offset = end
        packed_key = torch.cat(key_parts, dim=0) if len(key_parts) > 1 else key_parts[0]
        packed_value = torch.cat(value_parts, dim=0) if len(value_parts) > 1 else value_parts[0]
        if flash_attn_varlen_func is None:
            raise RuntimeError(
                "FlashAttention is required for the built-in DSPP compute adapter; "
                "install it or pass another --adapter"
            )
        attended = flash_attn_varlen_func(
            query[:valid_tokens],
            packed_key,
            packed_value,
            cu_q,
            cu_k,
            max(query_lengths),
            max(q + h for q, h in zip(query_lengths, history_lengths)),
            0.0,
            softmax_scale=self.head_dim ** -0.5,
            causal=True,
        ).reshape(valid_tokens, self.hidden_size)
        padded = hidden.new_zeros((capacity, self.hidden_size))
        padded[:valid_tokens] = attended
        hidden = hidden + self.o(padded)
        normalized = self.post_attention_norm(hidden)
        gate, up = self.gate_up(normalized).chunk(2, dim=-1)
        return hidden + self.down(F.silu(gate) * up)


def _apply_rope(tensor: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Match the non-fast Megatron RoPE arithmetic used by current DSPP runs."""

    half = tensor.size(-1) // 2
    first, second = tensor[..., :half], tensor[..., half:]
    rotated = torch.cat((-second, first), dim=-1)
    return (
        tensor * freqs.cos().to(tensor.dtype)
        + rotated * freqs.sin().to(tensor.dtype)
    )


def _linear_work(
    *, hidden_size: int, ffn_hidden_size: int, num_heads: int,
    num_query_groups: int, capacity: int,
) -> float:
    head_dim = hidden_size // num_heads
    qkv_width = (num_heads + 2 * num_query_groups) * head_dim
    return 2.0 * capacity * (
        hidden_size * qkv_width
        + hidden_size * hidden_size
        + hidden_size * (2 * ffn_hidden_size)
        + ffn_hidden_size * hidden_size
    )


def local_llama_adapter(
    *,
    shape: DsppComputeShape,
    device: str = "cuda",
    hidden_size: int | None = None,
    ffn_hidden_size: int | None = None,
    num_heads: int | None = None,
    num_query_groups: int | None = None,
    layers_per_stage: int | Sequence[int] | None = None,
    dtype: str | None = None,
) -> Dict[str, object]:
    """Create a CUDA-graph-repeatable local F, B, or W task."""

    shape.validate()
    hidden_size = int(hidden_size or os.getenv("DSPP_PROFILE_HIDDEN_SIZE", 2048))
    ffn_hidden_size = int(
        ffn_hidden_size or os.getenv("DSPP_PROFILE_FFN_HIDDEN_SIZE", hidden_size * 4)
    )
    num_heads = int(num_heads or os.getenv("DSPP_PROFILE_NUM_HEADS", 16))
    num_query_groups = int(
        num_query_groups or os.getenv("DSPP_PROFILE_NUM_QUERY_GROUPS", num_heads)
    )
    configured_layers = layers_per_stage or os.getenv("DSPP_PROFILE_LAYERS_PER_STAGE", 1)
    if isinstance(configured_layers, Sequence) and not isinstance(configured_layers, str):
        if shape.stage >= len(configured_layers):
            raise ValueError("layers_per_stage does not cover requested stage")
        local_layer_count = int(configured_layers[shape.stage])
    else:
        local_layer_count = int(configured_layers)
    dtype_obj = _dtype(str(dtype or os.getenv("DSPP_PROFILE_DTYPE", "bf16")))
    target = torch.device(device)
    if target.type != "cuda":
        raise ValueError("the built-in local Llama adapter requires CUDA")

    blocks = nn.ModuleList([
        _LocalLlamaBlock(
            hidden_size, ffn_hidden_size, num_heads, num_query_groups, dtype_obj
        )
        for _ in range(max(1, local_layer_count))
    ]).to(device=target, dtype=dtype_obj)
    blocks.train()
    capacity = shape.capacity
    hidden = torch.randn(capacity, hidden_size, device=target, dtype=dtype_obj)
    cu_q = torch.tensor(
        [0] + list(torch.tensor(shape.query_lengths).cumsum(0).tolist()),
        device=target,
        dtype=torch.int32,
    )
    key_lengths = [q + h for q, h in zip(shape.query_lengths, shape.history_lengths)]
    cu_k = torch.tensor(
        [0] + list(torch.tensor(key_lengths).cumsum(0).tolist()),
        device=target,
        dtype=torch.int32,
    )
    histories = []
    for _layer in blocks:
        per_item = []
        for history_length in shape.history_lengths:
            key = torch.randn(
                history_length, num_query_groups, hidden_size // num_heads,
                device=target, dtype=dtype_obj,
            )
            value = torch.randn_like(key)
            per_item.append((key, value))
        histories.append(tuple(per_item))
    inv_freq = 1.0 / (
        10000.0 ** (
            torch.arange(0, hidden_size // num_heads, 2, device=target).float()
            / (hidden_size // num_heads)
        )
    )
    positions = torch.arange(capacity, device=target, dtype=torch.float32)
    rope_half = torch.outer(positions, inv_freq)
    rope_freqs = torch.cat((rope_half, rope_half), dim=-1).unsqueeze(1)

    def forward_once(input_hidden: torch.Tensor) -> torch.Tensor:
        value = input_hidden
        for block, history in zip(blocks, histories):
            value = block(
                value,
                query_lengths=shape.query_lengths,
                history_lengths=shape.history_lengths,
                history_kv=history,
                cu_q=cu_q,
                cu_k=cu_k,
                rope_freqs=rope_freqs,
            )
        return value

    holder: List[torch.Tensor] = []
    if shape.operation == "F":
        for parameter in blocks.parameters():
            parameter.requires_grad_(False)

        def run() -> None:
            value = forward_once(hidden)
            if holder:
                holder[0] = value
            else:
                holder.append(value)

        multiplier = 1.0
    elif shape.operation == "B":
        norm_parameters = []
        for parameter in blocks.parameters():
            parameter.requires_grad_(False)
        for block in blocks:
            for parameter in (
                block.input_norm.weight,
                block.post_attention_norm.weight,
            ):
                parameter.requires_grad_(True)
                norm_parameters.append(parameter)
        hidden.requires_grad_(True)
        history_targets = []
        for per_layer in histories:
            for key, value in per_layer:
                if key.numel():
                    key.requires_grad_(True)
                    value.requires_grad_(True)
                    history_targets.extend((key, value))
        output = forward_once(hidden)
        output_grad = torch.randn_like(output)
        backward_targets = (hidden, *history_targets, *norm_parameters)

        def run() -> None:
            grads = torch.autograd.grad(
                output, backward_targets, output_grad,
                retain_graph=True, create_graph=False, allow_unused=False,
            )
            if holder:
                holder[0] = grads[0]
            else:
                holder.append(grads[0])

        multiplier = 2.0
    else:
        # W is the deferred collection of weight-gradient GEMMs.  Inputs and
        # output gradients are persistent buffers, matching the shapes emitted
        # by qkv/o/gate-up/down projections without rerunning attention B.
        gemms = []
        head_dim = hidden_size // num_heads
        qkv_width = (num_heads + 2 * num_query_groups) * head_dim
        for _ in blocks:
            gemms.extend((
                (torch.randn(qkv_width, capacity, device=target, dtype=dtype_obj),
                 torch.randn(capacity, hidden_size, device=target, dtype=dtype_obj)),
                (torch.randn(hidden_size, capacity, device=target, dtype=dtype_obj),
                 torch.randn(capacity, hidden_size, device=target, dtype=dtype_obj)),
                (torch.randn(2 * ffn_hidden_size, capacity, device=target, dtype=dtype_obj),
                 torch.randn(capacity, hidden_size, device=target, dtype=dtype_obj)),
                (torch.randn(hidden_size, capacity, device=target, dtype=dtype_obj),
                 torch.randn(capacity, ffn_hidden_size, device=target, dtype=dtype_obj)),
            ))

        if fused_weight_gradient_mlp_cuda is not None:
            fused_gemms = []
            for grad_output_t, input_tensor in gemms:
                grad_output = grad_output_t.t().contiguous()
                main_grad = torch.zeros(
                    grad_output.size(1), input_tensor.size(1),
                    device=target, dtype=torch.float32,
                )
                fused_gemms.append((input_tensor, grad_output, main_grad))

            def run() -> None:
                for input_tensor, grad_output, main_grad in fused_gemms:
                    fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
                        input_tensor, grad_output, main_grad
                    )
                if holder:
                    holder[0] = fused_gemms[-1][2]
                else:
                    holder.append(fused_gemms[-1][2])
        else:
            def run() -> None:
                values = [left @ right for left, right in gemms]
                if holder:
                    holder[0] = values[-1]
                else:
                    holder.append(values[-1])

        multiplier = 1.0

    linear = _linear_work(
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        num_heads=num_heads,
        num_query_groups=num_query_groups,
        capacity=capacity,
    )
    attention = 4.0 * num_heads * (hidden_size // num_heads) * sum(
        q * (q + h) for q, h in zip(shape.query_lengths, shape.history_lengths)
    )
    if shape.operation == "W":
        work_units = local_layer_count * linear
    else:
        work_units = local_layer_count * multiplier * (linear + attention)
    return {
        "run": run,
        "work_units": float(work_units),
        # FlashAttention backward owns RNG/autograd state created by the
        # forward outside the timed region and cannot be safely captured as a
        # standalone graph. CUDA events still isolate it from PP/P2P/iteration.
        "use_cuda_graph": shape.operation != "B",
    }
