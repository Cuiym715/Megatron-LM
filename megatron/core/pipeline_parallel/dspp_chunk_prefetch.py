"""Opt-in adaptive CP1 prefetch, using the existing publication/commit path."""
from dataclasses import replace
from functools import partial
import json
import pickle
import time

import torch

from megatron.core.datasets.dspp_chunk_cost import ChunkCostModel
from megatron.core.datasets.dspp_chunk_search import SearchOptions
from megatron.core.datasets.dspp_training import DsppCpuBatch, _source_sequences
from .dspp_chunk_planner import ChunkPlanner
from .dspp_prefetch import CpuPlanner, CpuIteration, DsppIterationPrefetch, NoPlanner
from .dspp_plan import DsppSchedulePlan


def choose_chunk_plan(packet, result, reference=None):
    reason = packet.metrics.get('submission_error') or 'planner-not-ready'
    if packet.metrics.get('rule_only'):
        return reference, ''
    if result is not None:
        reason = result.get('error') or ''
        if result['iteration'] != packet.iteration:
            reason = 'worker-input-mismatch'
        if not reason:
            candidate = DsppSchedulePlan.from_dict(result['plan'])
            return candidate, ''
    return replace(reference, fallback_reason='prefetch-'+reason), reason


def chunk_worker(jobs, results, model_payload, cfg, options, budget_fraction):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    planner = ChunkPlanner(ChunkCostModel(model_payload), cfg, options)
    while True:
        blob = jobs.get()
        if blob is None:
            return
        iteration, input_key, request = pickle.loads(blob)
        started = time.perf_counter()
        try:
            plan, report = planner.optimize(request, budget_fraction)
            result = dict(iteration=iteration, input_hash=input_key, plan=plan.to_dict(),
                          memory_report=report, b_calls=1, error=None)
        except Exception as exc:
            result = dict(iteration=iteration,input_hash=input_key,plan=None,error=f'{type(exc).__name__}: {exc}')
        result.update(worker_started_at=started, worker_verified_at=time.perf_counter(),
                      worker_cuda_initialized=torch.cuda.is_initialized())
        results.put(json.dumps(result).encode())


class AdaptiveChunkPrefetch(DsppIterationPrefetch):
    def __init__(self, args, iterator, get_batch_func):
        model = ChunkCostModel.load(args.dspp_chunk_model)
        if model.payload['execution'] != 'native-default-bf16-v1':
            raise ValueError('re-export native performance profiles; diagnostic-mode costs cannot drive this executor')
        options = SearchOptions(args.dspp_chunk_points,args.dspp_chunk_step,
                                args.dspp_chunk_spacing,args.dspp_chunk_score)
        self.selector = ChunkPlanner(model,args,options)
        if torch.cuda.get_device_name() != model.payload['memory']['device_name']:
            raise ValueError('chunk model hardware mismatch')
        self.fixed_capacity = args.dspp_chunk_fixed or None
        self.run_b = args.dspp_schedule == 'b'
        factory = (partial(CpuPlanner,target=chunk_worker,
            target_args=(model.payload,self.selector.cfg,options,args.dspp_b_budget_fraction))
            if self.run_b else NoPlanner)
        super().__init__(args,iterator,get_batch_func,planner_factory=factory,plan_selector=choose_chunk_plan)

    def _prepare(self, iteration, consumed, num_microbatches):
        started = time.perf_counter()
        cfg = self.cfg
        sources, lengths, identities, host_bytes = [], [], [], 0
        for logical in range(num_microbatches):
            data = next(self.iterator)
            seqs, group = _source_sequences(data['text'],data['lengths'],cfg.seq_length)
            # Own one compact copy; release the loader's padded matrix.
            seqs = tuple(t.clone() for t in seqs)
            host_bytes += sum(t.numel()*t.element_size() for t in seqs)
            if host_bytes > cfg.dspp_prefetch_max_bytes:
                raise ValueError('future CPU input exceeds prefetch byte limit')
            first = consumed+logical*cfg.micro_batch_size
            identities.append(dict(sample_ids=list(range(first,first+cfg.micro_batch_size)),
                                   raw_lengths=data['lengths'].tolist()))
            sources.append(seqs); lengths.append(tuple(group))
        fetched = time.perf_counter()
        selected, selection = self.selector.prepare(tuple(lengths),tuple(identities),iteration,
            consumed,fixed=self.fixed_capacity)
        self.prepared_reference = iteration, selected.reference
        batches = tuple(DsppCpuBatch(b.plan,s,identity,max(cfg.variable_seq_pad_token_id,0))
                        for b,s,identity in zip(selected.batches,sources,identities))
        ready = time.perf_counter()
        error = ''
        if self.run_b:
            if self.closed or self.planner is None:
                error = 'worker-unavailable'
            else:
                error = self.planner.submit(iteration,selected.reference.binding.input_hash,
                                            self.selector.b_request(selected,started))
        metrics = dict(requested_at=started,fetched_at=fetched,metadata_ready_at=ready,
            enqueued_at=time.perf_counter(),host_bytes=host_bytes,submission_error=error,
            rule_only=not self.run_b,selection=selection,selected_chunk=selected.capacity,
            memory_report=selected.memory_report,states=['fetched','metadata_ready','selected-base'])
        return CpuIteration(iteration,consumed,num_microbatches,batches,{},
            selected.reference.binding.input_hash,metrics,selected.ordering,selected.counts)

    def acquire_iteration(self, *args, **kwargs):
        batches, plan = super().acquire_iteration(*args, **kwargs)
        self.args.micro_seq_length = plan.wire_shape[0]
        return batches, plan
