"""Sequence-aware FlashAttention for DSPP residual-packed microbatches.

The pipeline-facing tensor remains `[chunk_size, 1, heads, head_dim]`, while
only valid query tokens are passed to FlashAttention.  A physical microbatch
may contain one continuation segment followed by independent short sequences.

The custom autograd function intentionally keeps cached K/V tensors outside
the autograd graph.  Backward is expected to run in reverse segment order.  It
accumulates gradients for historical K/V in :class:`DsppSequenceKVState` and
returns them when the corresponding earlier segment is backpropagated.  This
matches pipeline runtimes that execute each physical microbatch backward as a
separate task.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from megatron.core.datasets.dspp_batch_plan import DsppMicrobatchMeta, DsppSegment

try:
    from flash_attn.flash_attn_interface import (
        _wrapped_flash_attn_varlen_backward,
        _wrapped_flash_attn_varlen_forward,
    )
except ImportError:  # pragma: no cover - exercised only in CPU-only environments.
    _wrapped_flash_attn_varlen_backward = None
    _wrapped_flash_attn_varlen_forward = None


@dataclass
class _CachedSegment:
    segment_id: int
    key: torch.Tensor
    value: torch.Tensor


@dataclass
class _SequenceLayerEntry:
    next_segment_id: int = 0
    segments: List[_CachedSegment] = field(default_factory=list)
    pending_key_grads: Dict[int, torch.Tensor] = field(default_factory=dict)
    pending_value_grads: Dict[int, torch.Tensor] = field(default_factory=dict)


@dataclass(frozen=True)
class _ItemLayout:
    item: DsppSegment
    history: Tuple[Tuple[int, int], ...]


class DsppSequenceKVState:
    """Iteration-scoped per-layer, per-logical-sequence K/V state."""

    def __init__(self, *, validate_runtime: bool = False,
                 split_attention_backward: bool = False,
                 history_dkv_after_dx: bool = False) -> None:
        self._entries: Dict[Tuple[int, int], _SequenceLayerEntry] = {}
        self._current_meta: Optional[DsppMicrobatchMeta] = None
        self._prepared_cu_seqlens: Dict[
            Tuple[int, torch.device], Tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self.validate_runtime = bool(validate_runtime)
        # Legacy after-dx drains H immediately. Explicit plans freeze a mode
        # for each physical task before forward and may retain independent Hs.
        self.split_attention_backward = bool(split_attention_backward)
        if history_dkv_after_dx and not split_attention_backward:
            raise ValueError("history dKV after dx requires split attention backward")
        self.history_dkv_after_dx = bool(history_dkv_after_dx)
        self._history_task_id = None
        self._history_task_ready = False
        self._history_records = {}
        self._task_modes = None
        self._required_history = {}
        self._completed_history = set()

    def configure_task_modes(self, metas, deferred_tasks) -> None:
        """Freeze selective paths and all history consumers before any F.

        IDs are physical tasks within this logical batch/model chunk. Every
        later segment contributes to each earlier segment, not just its direct
        predecessor. A physical H completes all layers before releasing gates.
        """
        if (not self.history_dkv_after_dx or self._task_modes is not None
                or self._current_meta is not None or self._entries):
            raise RuntimeError('task modes must be frozen once before forward')
        by_id = {m.microbatch_id: m for m in metas}
        deferred = set(deferred_tasks)
        if len(by_id) != len(metas) or not deferred <= by_id.keys():
            raise ValueError('invalid selective backward task IDs')
        chains = {}
        for task, meta in by_id.items():
            if task in deferred and not any(i.has_history for i in meta.items):
                raise ValueError('cannot defer history-free task')
            for item in meta.items:
                chain = chains.setdefault(item.sequence_id, {})
                if item.segment_id in chain:
                    raise ValueError('duplicate sequence segment')
                chain[item.segment_id] = task
        required = {task: set() for task in by_id}
        for chain in chains.values():
            if sorted(chain) != list(range(len(chain))):
                raise ValueError('incomplete sequence segment chain')
            for segment, source in chain.items():
                if source in deferred:
                    for target_segment in range(segment):
                        required[chain[target_segment]].add(source)
        self._task_modes = {task: task in deferred for task in by_id}
        self._required_history = required

    def task_after_dx(self, microbatch_id):
        return (self.history_dkv_after_dx if self._task_modes is None
                else self._task_modes[microbatch_id])

    def check_history_consumers(self, microbatch_id):
        missing = self._required_history.get(microbatch_id, set()) - self._completed_history
        if missing:
            raise RuntimeError(f'backward consumer requires unfinished H tasks: {sorted(missing)}')

    @property
    def _history_jobs(self):
        # Preserve the existing opt-in audit/diagnostic view without retaining
        # an additional copy of any tensors. During backward append to its job list.
        if self._history_task_id is not None:
            return self._history_records[self._history_task_id]['jobs']
        return [job for record in self._history_records.values() for job in record['jobs']]

    @property
    def pending_history_count(self) -> int:
        return sum(len(record['jobs']) for record in self._history_records.values())

    def history_job_count(self, microbatch_id):
        return len(self._history_records[microbatch_id]['jobs'])

    def begin_backward_task(self, microbatch_id: int) -> None:
        """Begin a D; configured independent tasks can retain multiple Hs."""
        if not self.task_after_dx(microbatch_id):
            raise RuntimeError("after-dx task mode is not enabled")
        if (self._history_task_id is not None
                or (self._task_modes is None and self._history_records)):
            raise RuntimeError("previous history task has not completed")
        if microbatch_id in self._history_records or microbatch_id in self._completed_history:
            raise RuntimeError('duplicate backward task')
        self.check_history_consumers(microbatch_id)
        self._history_task_id = microbatch_id
        self._history_task_ready = False
        self._history_records[microbatch_id] = dict(ready=False, jobs=[])

    def mark_dx_ready(self, microbatch_id: int) -> None:
        """Called only after the complete stage backward has returned."""
        if self._history_task_id != microbatch_id or self._history_task_ready:
            raise RuntimeError("invalid or duplicate dx-ready boundary")
        self._history_records[microbatch_id]['ready'] = True
        self._history_task_id = None

    def run_history_task(self, microbatch_id: int) -> int:
        """Execute this task's layers after dx publication on the compute stream.

        This state owns tensor references until kernels and routing have been
        submitted on the same stream used for B. No autograd graph is retained.
        A failed task stays blocked rather than being silently retried.
        """
        record = self._history_records.get(microbatch_id)
        if record is None or not record['ready'] or self._history_task_id is not None:
            raise RuntimeError("history task requires its completed stage dx")
        record['ready'] = False  # A failed H cannot be retried with partial contributions.
        from .fa_split import backward_dkv
        count = len(record['jobs'])
        for split_state, layer_idx, layouts, current_shape in record['jobs']:
            dk, dv = backward_dkv(split_state, "history", compact=True)
            self._route_key_value_grads(
                layer_idx, layouts, dk, dv, current_shape,
                route_history=True, return_current=False, compact_region="history",
            )
        record['jobs'].clear()
        del self._history_records[microbatch_id]
        self._completed_history.add(microbatch_id)
        return count

    def __bool__(self) -> bool:
        # An empty state is still an active cache for the first segment.
        return True

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def current_meta(self) -> DsppMicrobatchMeta:
        if self._current_meta is None:
            raise RuntimeError("set_microbatch() must be called before model forward")
        return self._current_meta

    def set_microbatch(self, meta: DsppMicrobatchMeta) -> None:
        """Select metadata for the next synchronous model forward."""

        if self.validate_runtime:
            meta.validate()
        self._current_meta = meta

    def prepared_cu_seqlens(
        self, meta: DsppMicrobatchMeta, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cache device metadata once per microbatch rather than once per layer."""

        cache_key = (meta.microbatch_id, device)
        prepared = self._prepared_cu_seqlens.get(cache_key)
        if prepared is None:
            prepared = (
                torch.tensor(meta.cu_seqlens_q, dtype=torch.int32, device=device),
                torch.tensor(meta.cu_seqlens_k, dtype=torch.int32, device=device),
            )
            self._prepared_cu_seqlens[cache_key] = prepared
        return prepared

    def exchange_ctx(self, *args, **kwargs) -> None:
        """Compatibility no-op; DSPP MVP excludes attention offloading."""

        del args, kwargs

    def exchange_ctx_for_backward(
        self, output: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        """Compatibility no-op; packed-attention backward owns KV routing."""

        del args, kwargs
        return output

    def _entry_key(self, layer_idx: int, sequence_id: int) -> Tuple[int, int]:
        return int(layer_idx), int(sequence_id)

    def _get_history(
        self,
        layer_idx: int,
        item: DsppSegment,
    ) -> Tuple[Optional[_SequenceLayerEntry], List[_CachedSegment]]:
        key = self._entry_key(layer_idx, item.sequence_id)
        entry = self._entries.get(key)
        if item.segment_id == 0:
            if entry is not None:
                raise RuntimeError(
                    f"sequence {item.sequence_id} layer {layer_idx} was started twice"
                )
            return None, []
        if entry is None:
            raise RuntimeError(
                f"missing KV history for sequence {item.sequence_id}, "
                f"layer {layer_idx}, segment {item.segment_id}"
            )
        if entry.next_segment_id != item.segment_id:
            raise RuntimeError(
                f"out-of-order forward for sequence {item.sequence_id}, layer {layer_idx}: "
                f"got segment {item.segment_id}, expected {entry.next_segment_id}"
            )
        history_length = sum(segment.key.size(0) for segment in entry.segments)
        if history_length != item.token_offset:
            raise RuntimeError(
                f"KV history for sequence {item.sequence_id}, layer {layer_idx} has "
                f"{history_length} tokens; expected {item.token_offset}"
            )
        return entry, list(entry.segments)

    def build_key_value(
        self,
        layer_idx: int,
        meta: DsppMicrobatchMeta,
        current_key: torch.Tensor,
        current_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[_ItemLayout, ...]]:
        """Assemble varlen K/V and retain non-final sequence segments."""

        if self.validate_runtime:
            meta.validate()
        if current_key.size(0) != meta.valid_token_count:
            raise ValueError("current K/V token count does not match DSPP metadata")
        if current_key.shape != current_value.shape:
            raise ValueError("current key and value shapes must match")

        key_parts: List[torch.Tensor] = []
        value_parts: List[torch.Tensor] = []
        layouts: List[_ItemLayout] = []
        direct_current = bool(meta.items) and all(
            not item.has_history and item.is_last_segment for item in meta.items
        )
        current_offset = 0
        for item in meta.items:
            current_end = current_offset + item.token_length
            entry, history = self._get_history(layer_idx, item)
            layouts.append(
                _ItemLayout(
                    item=item,
                    history=tuple(
                        (segment.segment_id, segment.key.size(0)) for segment in history
                    ),
                )
            )
            if direct_current:
                current_offset = current_end
                continue

            item_key = current_key[current_offset:current_end]
            item_value = current_value[current_offset:current_end]
            key_parts.extend(segment.key for segment in history)
            value_parts.extend(segment.value for segment in history)
            key_parts.append(item_key)
            value_parts.append(item_value)

            if not item.is_last_segment:
                if len(meta.items) != 1:
                    raise RuntimeError(
                        "a non-final long-sequence segment must occupy its own microbatch"
                    )
                if entry is None:
                    entry = _SequenceLayerEntry()
                    self._entries[self._entry_key(layer_idx, item.sequence_id)] = entry
                entry.segments.append(
                    _CachedSegment(item.segment_id, item_key, item_value)
                )
                entry.next_segment_id = item.segment_id + 1
            current_offset = current_end

        if direct_current:
            # Independent final sequences already occupy current K/V in
            # ``meta`` order. Avoid per-item tensor views and materialization.
            assembled_key = current_key
            assembled_value = current_value
        elif len(key_parts) == 1:
            assembled_key = key_parts[0]
            assembled_value = value_parts[0]
        else:
            assembled_key = torch.cat(key_parts, dim=0).contiguous()
            assembled_value = torch.cat(value_parts, dim=0).contiguous()
        return assembled_key, assembled_value, tuple(layouts)

    @staticmethod
    def _accumulate(
        target: Dict[int, torch.Tensor], key: int, grad: torch.Tensor
    ) -> None:
        previous = target.get(key)
        target[key] = grad if previous is None else previous + grad

    def distribute_key_value_grads(
        self,
        layer_idx: int,
        layouts: Tuple[_ItemLayout, ...],
        assembled_key_grad: torch.Tensor,
        assembled_value_grad: torch.Tensor,
        current_shape: torch.Size,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Route history gradients backward and return current-segment gradients."""

        return self._route_key_value_grads(
            layer_idx, layouts, assembled_key_grad, assembled_value_grad,
            current_shape, route_history=True, return_current=True,
        )

    def _route_key_value_grads(
        self, layer_idx, layouts, assembled_key_grad, assembled_value_grad,
        current_shape, *, route_history, return_current, compact_region=None,
    ):

        direct_current = bool(layouts) and all(
            not layout.history and layout.item.is_last_segment for layout in layouts
        )
        if direct_current and return_current:
            if (
                assembled_key_grad.shape != current_shape
                or assembled_value_grad.shape != current_shape
            ):
                raise RuntimeError(
                    "current K/V gradient shape does not match physical task"
                )
            # For an all-independent pack, FlashAttention's dK/dV layout is
            # already the physical current-K/V layout.  No allocation or
            # per-item scatter is needed.
            return assembled_key_grad, assembled_value_grad

        single_item = len(layouts) == 1
        if return_current and not single_item:
            current_key_grad = assembled_key_grad.new_empty(current_shape)
            current_value_grad = assembled_value_grad.new_empty(current_shape)
        assembled_offset = 0
        current_offset = 0
        for layout in layouts:
            item = layout.item
            key = self._entry_key(layer_idx, item.sequence_id)
            entry = self._entries.get(key)
            for segment_id, segment_length in layout.history:
                if compact_region == "current":
                    continue
                end = assembled_offset + segment_length
                if route_history:
                    if entry is None or (not return_current and not any(
                        segment.segment_id == segment_id for segment in entry.segments
                    )):
                        raise RuntimeError("history gradient consumer is no longer retained")
                    self._accumulate(
                        entry.pending_key_grads, segment_id,
                        assembled_key_grad[assembled_offset:end],
                    )
                    self._accumulate(
                        entry.pending_value_grads, segment_id,
                        assembled_value_grad[assembled_offset:end],
                    )
                assembled_offset = end

            assembled_end = assembled_offset + (0 if compact_region == "history" else item.token_length)
            if not return_current:
                assembled_offset = assembled_end
                continue
            key_grad = assembled_key_grad[assembled_offset:assembled_end]
            value_grad = assembled_value_grad[assembled_offset:assembled_end]
            if not item.is_last_segment:
                if entry is None or not entry.segments:
                    raise RuntimeError(
                        "missing retained current segment during backward"
                    )
                cached = entry.segments[-1]
                if cached.segment_id != item.segment_id:
                    raise RuntimeError(
                        f"out-of-order backward for sequence {item.sequence_id}: "
                        f"got segment {item.segment_id}, retained segment is {cached.segment_id}"
                    )
                if item.segment_id not in entry.pending_key_grads:
                    raise RuntimeError(
                        f"segment {item.segment_id} of sequence {item.sequence_id} "
                        "was backpropagated before its successor"
                    )
                key_grad = key_grad + entry.pending_key_grads.pop(item.segment_id)
                value_grad = value_grad + entry.pending_value_grads.pop(item.segment_id)
                entry.segments.pop()
                entry.next_segment_id = item.segment_id

            if single_item:
                current_key_grad = key_grad
                current_value_grad = value_grad
            else:
                current_end = current_offset + item.token_length
                current_key_grad[current_offset:current_end] = key_grad
                current_value_grad[current_offset:current_end] = value_grad
                current_offset = current_end
            assembled_offset = assembled_end

            if not item.is_last_segment and item.segment_id == 0:
                if entry is None or entry.segments:
                    raise RuntimeError(
                        "sequence KV cache was not drained at segment zero"
                    )
                if entry.pending_key_grads or entry.pending_value_grads:
                    raise RuntimeError("sequence KV gradient cache was not drained")
                del self._entries[key]

        if assembled_offset != assembled_key_grad.size(0):
            raise RuntimeError("assembled K/V gradient layout was not fully consumed")
        if not return_current:
            return None
        if current_key_grad.shape != current_shape:
            raise RuntimeError(
                "current K/V gradient shape does not match physical task"
            )
        return current_key_grad, current_value_grad

    def clear(self) -> None:
        if self._history_task_id is not None or self._history_records:
            raise RuntimeError("cannot clear an unfinished history task")
        self._entries.clear()
        self._current_meta = None
        self._prepared_cu_seqlens.clear()
        self._completed_history.clear()
        self._task_modes = None
        self._required_history.clear()

    def finish_iteration(self) -> None:
        """Verify gradients drained all KV state, then release device metadata."""

        self.assert_empty()
        self._current_meta = None
        self._prepared_cu_seqlens.clear()
        self._completed_history.clear()
        self._task_modes = None
        self._required_history.clear()

    def assert_empty(self) -> None:
        if self._history_task_id is not None or self._history_records:
            raise RuntimeError("DSPP history task has not completed")
        if self._entries:
            summary = {
                key: {
                    "segments": [segment.segment_id for segment in entry.segments],
                    "pending": sorted(entry.pending_key_grads),
                }
                for key, entry in self._entries.items()
            }
            raise RuntimeError(f"DSPP sequence KV state is not empty: {summary}")


class _DsppPackedFlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        current_key: torch.Tensor,
        current_value: torch.Tensor,
        state: DsppSequenceKVState,
        meta: DsppMicrobatchMeta,
        layer_idx: int,
        softmax_scale: Optional[float],
    ) -> torch.Tensor:
        if _wrapped_flash_attn_varlen_forward is None:
            raise RuntimeError("FlashAttention is required for DSPP packed attention")
        if not query.is_cuda or query.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                "DSPP packed FlashAttention requires CUDA FP16/BF16 tensors"
            )
        if query.ndim != 4 or query.size(1) != 1:
            raise ValueError(
                "DSPP Q/K/V must have shape [chunk_size, 1, heads, head_dim]"
            )
        if current_key.shape != current_value.shape:
            raise ValueError("current key and value shapes must match")
        if (
            query.size(0) != current_key.size(0)
            or query.size(1) != current_key.size(1)
            or query.size(-1) != current_key.size(-1)
        ):
            raise ValueError(
                "query and K/V sequence, batch, and head dimensions must match"
            )
        ctx.history_dkv_after_dx = state.task_after_dx(meta.microbatch_id)
        ctx.split_attention_backward = (state.split_attention_backward
            and (state._task_modes is None or ctx.history_dkv_after_dx))
        if ctx.split_attention_backward:
            # Reject unsupported configurations before mutating retained KV state.
            if query.dtype != torch.bfloat16 or query.size(-1) != 128:
                raise ValueError("DSPP split attention backward requires BF16 and head dimension 128")
            if current_key.size(2) <= 0 or query.size(2) % current_key.size(2):
                raise ValueError("DSPP split attention backward requires Hq divisible by Hkv")
        valid_tokens = meta.valid_token_count
        query_valid = query[:valid_tokens, 0].contiguous()
        current_key_valid = current_key[:valid_tokens, 0].contiguous()
        current_value_valid = current_value[:valid_tokens, 0].contiguous()

        assembled_key, assembled_value, layouts = state.build_key_value(
            int(layer_idx), meta, current_key_valid, current_value_valid
        )
        cu_q, cu_k = state.prepared_cu_seqlens(meta, query.device)
        if assembled_key.size(0) != meta.cu_seqlens_k[-1]:
            raise RuntimeError("assembled K/V length does not match cu_seqlens_k")

        original_head_dim = query_valid.size(-1)
        padded_head_dim = (original_head_dim + 7) // 8 * 8
        if padded_head_dim != original_head_dim:
            query_padded = torch.nn.functional.pad(
                query_valid, (0, padded_head_dim - original_head_dim)
            )
            key_padded = torch.nn.functional.pad(
                assembled_key, (0, padded_head_dim - original_head_dim)
            )
            value_padded = torch.nn.functional.pad(
                assembled_value, (0, padded_head_dim - original_head_dim)
            )
        else:
            query_padded = query_valid
            key_padded = assembled_key
            value_padded = assembled_value
        scale = float(softmax_scale or original_head_dim**-0.5)
        output, softmax_lse, _, rng_state = _wrapped_flash_attn_varlen_forward(
            query_padded,
            key_padded,
            value_padded,
            cu_q,
            cu_k,
            meta.max_seqlen_q,
            meta.max_seqlen_k,
            0.0,
            scale,
            causal=True,
            window_size_left=-1,
            window_size_right=-1,
            softcap=0.0,
            alibi_slopes=None,
            return_softmax=False,
            block_table=None,
            leftpad_k=None,
            seqused_k=None,
            zero_tensors=False,
        )
        ctx.save_for_backward(
            query_padded,
            key_padded,
            value_padded,
            output,
            softmax_lse,
            cu_q,
            cu_k,
            rng_state,
        )
        ctx.state = state
        ctx.meta = meta
        ctx.layouts = layouts
        ctx.has_history = any(layout.history for layout in layouts)
        ctx.layer_idx = int(layer_idx)
        ctx.scale = scale
        ctx.original_head_dim = original_head_dim
        ctx.current_shape = current_key_valid.shape
        ctx.query_physical_shape = query.shape
        ctx.kv_physical_shape = current_key.shape
        ctx.valid_tokens = valid_tokens
        output_valid = output[..., :original_head_dim]
        if valid_tokens == query.size(0):
            return output_valid.unsqueeze(1)
        physical_output = query.new_zeros(query.shape)
        physical_output[:valid_tokens, 0] = output_valid
        return physical_output

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor):
        (
            query,
            key,
            value,
            output,
            softmax_lse,
            cu_q,
            cu_k,
            rng_state,
        ) = ctx.saved_tensors
        output_grad_valid = output_grad[: ctx.valid_tokens, 0]
        if output_grad_valid.size(-1) != query.size(-1):
            output_grad_valid = torch.nn.functional.pad(
                output_grad_valid, (0, query.size(-1) - output_grad_valid.size(-1))
            )
        output_grad_valid = output_grad_valid.contiguous()
        if ctx.history_dkv_after_dx and (
            ctx.state._history_task_id != ctx.meta.microbatch_id
            or ctx.state._history_task_ready
        ):
            raise RuntimeError("after-dx backward requires its active physical task")
        # In after-dx mode an entirely history-free physical task needs no H.
        # Its full KV output has exactly the compact-current layout expected by
        # the route below. Keep task ownership/consumer handling unchanged.
        if ctx.split_attention_backward and not (
            ctx.history_dkv_after_dx and not ctx.has_history
        ):
            from .fa_split import prepare_backward, backward_dq, backward_dkv, backward_dq_current

            # Frozen forward metadata: never consult the state's currently
            # selected microbatch during backward. Host tuples avoid a CUDA
            # metadata round trip for the per-sequence prototype dispatcher.
            split_inputs = (query, key, value, output, output_grad_valid, softmax_lse)
            if ctx.history_dkv_after_dx:
                split_inputs = tuple(t.detach() for t in split_inputs)
            split_state = prepare_backward(
                *split_inputs,
                softmax_scale=ctx.scale,
                cu_seqlens_q=ctx.meta.cu_seqlens_q,
                cu_seqlens_k=ctx.meta.cu_seqlens_k,
            )
            if ctx.history_dkv_after_dx:
                query_grad, key_grad, value_grad = backward_dq_current(split_state)
                if ctx.has_history:
                    ctx.state._history_jobs.append(
                        (split_state, ctx.layer_idx, ctx.layouts, ctx.current_shape)
                    )
            else:
                query_grad = backward_dq(split_state)
                key_grad, value_grad = backward_dkv(split_state, "current")
                history_key_grad, history_value_grad = backward_dkv(split_state, "history")
                key_grad.add_(history_key_grad)
                value_grad.add_(history_value_grad)
        else:
            query_grad = torch.empty_like(query)
            key_grad = torch.empty_like(key)
            value_grad = torch.empty_like(value)
            _wrapped_flash_attn_varlen_backward(
                output_grad_valid,
                query,
                key,
                value,
                output,
                softmax_lse,
                query_grad,
                key_grad,
                value_grad,
                cu_q,
                cu_k,
                ctx.meta.max_seqlen_q,
                ctx.meta.max_seqlen_k,
                0.0,
                ctx.scale,
                True,
                -1,
                -1,
                0.0,
                None,
                False,
                rng_state,
                False,
            )
        query_grad = query_grad[..., : ctx.original_head_dim]
        key_grad = key_grad[..., : ctx.original_head_dim]
        value_grad = value_grad[..., : ctx.original_head_dim]
        if ctx.history_dkv_after_dx:
            current_key_grad, current_value_grad = ctx.state._route_key_value_grads(
                ctx.layer_idx, ctx.layouts, key_grad, value_grad, ctx.current_shape,
                route_history=False, return_current=True, compact_region="current",
            )
        else:
            current_key_grad, current_value_grad = ctx.state.distribute_key_value_grads(
                ctx.layer_idx, ctx.layouts, key_grad, value_grad, ctx.current_shape,
            )
        if ctx.valid_tokens == ctx.query_physical_shape[0]:
            query_grad = query_grad.unsqueeze(1)
            current_key_grad = current_key_grad.unsqueeze(1)
            current_value_grad = current_value_grad.unsqueeze(1)
        else:
            physical_query_grad = query_grad.new_zeros(ctx.query_physical_shape)
            physical_key_grad = current_key_grad.new_zeros(ctx.kv_physical_shape)
            physical_value_grad = current_value_grad.new_zeros(ctx.kv_physical_shape)
            physical_query_grad[: ctx.valid_tokens, 0] = query_grad
            physical_key_grad[: ctx.valid_tokens, 0] = current_key_grad
            physical_value_grad[: ctx.valid_tokens, 0] = current_value_grad
            query_grad = physical_query_grad
            current_key_grad = physical_key_grad
            current_value_grad = physical_value_grad
        return (
            query_grad,
            current_key_grad,
            current_value_grad,
            None,
            None,
            None,
            None,
        )


def dspp_packed_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    state: DsppSequenceKVState,
    meta: DsppMicrobatchMeta,
    layer_idx: int,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """Run packed causal attention and return a fixed-shape padded output.

    Inputs and output use Megatron's `[chunk_size, 1, heads, head_dim]`
    layout.  Valid items must occupy the prefix described by ``meta``.
    """

    if state.validate_runtime:
        meta.validate()
    if query.ndim != 4 or query.size(1) != 1:
        raise ValueError(
            "DSPP packed attention requires [chunk_size, 1, heads, head_dim]"
        )
    if key.shape != value.shape:
        raise ValueError("key and value shapes must match")
    if (
        query.size(0) != key.size(0)
        or query.size(1) != key.size(1)
        or query.size(-1) != key.size(-1)
    ):
        raise ValueError(
            "query and K/V sequence, batch, and head dimensions must match"
        )
    if query.size(0) != meta.chunk_size:
        raise ValueError("Q/K/V sequence dimension must equal metadata chunk_size")
    return _DsppPackedFlashAttention.apply(
        query,
        key,
        value,
        state,
        meta,
        int(layer_idx),
        softmax_scale,
    )
