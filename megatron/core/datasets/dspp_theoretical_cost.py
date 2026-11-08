"""CPU-only assumed costs for correctness runs, never measured calibration.

Dominant matrix FLOPs / assumed throughput + ring send bytes / assumed
bandwidth. No fitted coefficients, latency intercept, timing files or CUDA.
Pointwise kernels, tiling, memory bandwidth and launch overhead are omitted.
"""
from dataclasses import dataclass
import math

from megatron.core.dspp_layout import physical_layer_range
from .dspp_cp_layout import cp_query_runs, hybrid_cp_ownership, owned_query_runs


@dataclass(frozen=True)
class TheoreticalPrediction:
    duration_ms: float
    flops: int
    communication_bytes: int
    source: str = 'theoretical-matmul-v1'
    error_ms: float = 0.
    confidence: str = 'assumed'
    extrapolated: bool = False


class TheoreticalCostModel:
    def __init__(self, args, metadata):
        self.args = args
        self.cp = getattr(args, 'context_parallel_size', 1)
        self.qo = getattr(args, 'kaimm_context_parallel_impl', 'key-value') == 'query-out'
        self.tflops = args.dspp_assumed_tflops
        self.bandwidth_gbps = args.dspp_assumed_bandwidth_gbps
        if any(not math.isfinite(v) or v <= 0 for v in (self.tflops, self.bandwidth_gbps)):
            raise ValueError('assumed throughput and bandwidth must be finite and positive')
        self.flops_per_ms = self.tflops * 1e9
        self.bytes_per_ms = self.bandwidth_gbps * 1e6
        self.hq = args.num_attention_heads
        self.dim = args.kv_channels or args.hidden_size // self.hq
        self.hk = args.num_query_groups if getattr(args, 'group_query_attention', False) else self.hq
        self.rows = {}
        history = {}
        for (logical, _), meta in sorted(metadata.items()):
            ownership = (hybrid_cp_ownership(meta, self.cp)
                         if getattr(args, 'dspp_cp_layout', 'paired') == 'hybrid' else None)
            shared = set(ownership.shared_items) if ownership else set()
            if self.qo:
                shared.update(i for i, item in enumerate(meta.items) if item.segment_count > 1)
            by_rank, widths = [], []
            for rank in range(self.cp):
                runs = (owned_query_runs(meta, ownership.rank_indices[rank]) if ownership
                        else cp_query_runs(meta, self.cp, rank))
                hist_pairs = current_pairs = 0
                widths.append(sum(r.local_end-r.local_begin for r in runs if r.item_index in shared)
                              if ownership else meta.chunk_size // self.cp)
                for i, item in enumerate(meta.items):
                    own = [r for r in runs if r.item_index == i]
                    n = sum(r.local_end-r.local_begin for r in own)
                    key = (logical, rank, item.sequence_id)
                    if self.qo:
                        hist_pairs += item.token_length * history.get(key, 0)
                        end = item.token_offset + item.token_length
                        current_pairs += sum((r.local_end-r.local_begin)*(end-r.position_begin)
                            - (r.local_end-r.local_begin)*(r.local_end-r.local_begin-1)//2 for r in own)
                    else:
                        hist_pairs += n * item.token_offset
                        current_pairs += sum((r.local_end-r.local_begin)*(r.position_begin-item.token_offset+1)
                            + (r.local_end-r.local_begin)*(r.local_end-r.local_begin-1)//2 for r in own)
                    history[key] = history.get(key, 0) + n
                by_rank.append((hist_pairs, current_pairs))
            self.rows[id(meta)] = (tuple(by_rank), max(widths))

    def predict_detailed(self, context, op, meta):
        args = self.args
        stage, chunk = context['stage'], context['virtual_chunk']
        begin, end = physical_layer_range(args.num_layers, args.pipeline_model_parallel_size,
            getattr(args, 'dspp_v_layer_layout', 'balanced'), stage, chunk)
        layers = end-begin
        local = meta.chunk_size // self.cp
        hidden, ffn, key = args.hidden_size, args.ffn_hidden_size, self.hk*self.dim
        dense = layers*(2*local*hidden*(hidden+2*key)+2*local*hidden*hidden+6*local*hidden*ffn)
        if stage == 0 and chunk == 1:
            dense += 2*local*hidden*args.padded_vocab_size
        pairs, width = self.rows[id(meta)]
        flops = []
        for history, current in pairs:
            # FA backward recomputes QK. D omits history dK/dV; H recomputes
            # QK/dP and computes only history dK/dV, with no dQ.
            attention = {'F': 4*(history+current), 'B_fused': 10*(history+current),
                         'D': 6*history+10*current, 'H': 8*history, 'W': 0}[op]
            flops.append((0 if op == 'H' else dense)+layers*attention*self.hq*self.dim)
        cp_bytes = 0
        if self.cp > 1 and op in ('F', 'B_fused', 'D'):
            if self.qo:
                # F: AG Q + AG LSE + RS O. B/D: AG Q/O/dO + AG LSE + RS dQ.
                per_token = (2 if op == 'F' else 4)*hidden*2+self.hq*4
            else:
                per_token = 2*key*2  # AG KV or RS dKV, one packed call.
            cp_bytes = layers*(self.cp-1)*width*per_token
            if op == 'F' and stage == 0 and chunk == 1:
                cp_bytes += (self.cp-1)*local*4  # Gather FP32 per-token loss.
        work = max(flops)
        return TheoreticalPrediction(work/self.flops_per_ms+cp_bytes/self.bytes_per_ms,
                                     work, cp_bytes)

    def p2p_ms(self, payload_bytes):
        return payload_bytes/self.bytes_per_ms

    def assumptions(self):
        return dict(cost_source='theoretical-matmul-v1', assumed_tflops=self.tflops,
                    assumed_bandwidth_gbps=self.bandwidth_gbps,
                    timing_profiles_used=False, communication='per-rank ring send bytes; PP payload bytes',
                    limits='Correctness only; no pointwise/tiling/launch/memory-bandwidth model or measured speedup.')
