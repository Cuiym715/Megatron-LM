"""One future CPU iteration and one spawn planner; training owns all collectives.

Only the coordinator's input thread advances the training iterator. The child
receives metadata, frozen contracts and hardware identities, never CUDA tensors
or model state. Publication uses a separate Gloo group at iteration boundaries.
"""
import atexit
from concurrent.futures import Future
from copy import deepcopy
from dataclasses import dataclass, replace
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import queue
import threading
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist

from megatron.core.datasets.dspp_training import (
    DsppTrainingBatch, prepare_dspp_cpu_batch,
)
from megatron.core.datasets.dspp_ordering import build_dspp_ordering_plan
from megatron.core.datasets.dspp_profile_snapshot import profile_snapshot, freeze_runtime_profiles
from .dspp_plan import DsppSchedulePlan, validate_plan, digest, preserves_retention_envelope
from .dspp_plan_runtime import build_runtime_plan, runtime_inputs
from .dspp_memory_counts import memory_layout
from .slice_v import build_slice_v_schedule


def _metadata_inputs(args, batches):
    # Reuse the execution template/padding rule, rather than a second schedule.
    from .schedules import _pad_dspp_v_split_counts
    pp = args.pipeline_model_parallel_size
    ordering = build_dspp_ordering_plan(batches, pp_degree=pp,
        mode=args.dspp_microbatch_order, profile_path=getattr(args, 'dspp_order_profile', None))
    args._dspp_cpu_ordering = ordering
    groups = ordering.schedule_groups
    counts = _pad_dspp_v_split_counts([len(g) for g in groups], pp)
    args._dspp_cpu_counts = counts
    schedules, _ = build_slice_v_schedule(pp, len(counts), counts)
    return schedules, counts, groups


def _build(request, optimize):
    cfg, batches, contexts, started, *cached = request
    if cached:
        ordering, counts = cached[0]
        groups = ordering.schedule_groups
        schedules, _ = build_slice_v_schedule(cfg.pipeline_model_parallel_size, len(counts), counts)
    else:
        schedules, counts, groups = _metadata_inputs(cfg, batches)
    return build_runtime_plan(cfg, schedules, counts, groups, batches,
        cfg.params_dtype, (cfg.seq_length, cfg.micro_batch_size, cfg.hidden_size),
        cfg.micro_seq_length, optimize=optimize, calibration_contexts=contexts,
        planning_started=started)


def _planner_main(jobs, results):
    """Spawn target. No global training args, CUDA context or process groups."""
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    while True:
        blob = jobs.get()
        if blob is None:
            return
        job = pickle.loads(blob)
        iteration, input_hash, request = job
        started = time.perf_counter()
        try:
            reference, plan = _build(request, True)
            result = dict(iteration=iteration, input_hash=input_hash, plan=plan.to_dict(),
                worker_started_at=started, worker_verified_at=time.perf_counter(),
                worker_cuda_initialized=torch.cuda.is_initialized(), error=None)
        except Exception as exc:
            result = dict(iteration=iteration, input_hash=input_hash, plan=None,
                worker_started_at=started, worker_verified_at=time.perf_counter(),
                worker_cuda_initialized=torch.cuda.is_initialized(),
                error=f'{type(exc).__name__}: {exc}')
        results.put(json.dumps(result).encode())


class CpuPlanner:
    def __init__(self):
        ctx = mp.get_context('spawn')
        self.jobs, self.results = ctx.Queue(maxsize=1), ctx.Queue(maxsize=2)
        self.process = ctx.Process(target=_planner_main, args=(self.jobs, self.results), daemon=True)
        # Set these before spawn imports torch/numpy, then restore the parent.
        names = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'CUDA_VISIBLE_DEVICES')
        old = {name: os.environ.get(name) for name in names}
        try:
            os.environ.update({name: '1' for name in names[:-1]})
            # Some upstream imports probe GPU properties. Hide devices before
            # the child imports Megatron, not merely at the target function.
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            self.process.start()
        except Exception:
            self.jobs.close()
            self.results.close()
            raise
        finally:
            for name, value in old.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        self.closed = False
        self.received = queue.Queue(maxsize=2)
        def receive():
            # A killed child can leave a partial multiprocessing frame. Only
            # this daemon reads that pipe; a training boundary never blocks on it.
            while not self.closed:
                try:
                    value = json.loads(self.results.get(timeout=.05))
                    while not self.closed:
                        try:
                            self.received.put(value, timeout=.05)
                            break
                        except queue.Full:
                            pass
                except queue.Empty:
                    continue
                except (EOFError, OSError, ValueError):
                    return
        self.receiver = threading.Thread(target=receive, name='dspp-plan-receiver', daemon=True)
        self.receiver.start()

    def submit(self, iteration, input_hash, request):
        if not self.process.is_alive():
            return 'worker-unavailable'
        try:
            self.jobs.put_nowait(pickle.dumps((iteration, input_hash, request)))
            return ''
        except queue.Full:
            return 'worker-backlog'

    def poll(self):
        try:
            return self.received.get_nowait()
        except queue.Empty:
            return None

    def close(self):
        self.closed = True
        try:
            self.jobs.put_nowait(None)
        except queue.Full:
            pass
        self.process.join(timeout=2)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=2)
        for q in (self.jobs, self.results):
            q.cancel_join_thread()
            q.close()
        self.receiver.join(timeout=.2)


def verify_sources(batches):
    return None


@dataclass
class CpuIteration:
    iteration: int
    consumed_samples: int
    num_microbatches: int
    batches: tuple
    reference: dict
    input_hash: str
    metrics: dict
    ordering: object = None
    schedule_counts: tuple = ()


@dataclass(frozen=True)
class CheckedExecution:
    """Coordinator's checked plan/counts, bound to immutable input and profiles."""
    reference: DsppSchedulePlan
    plan_hash: str
    memory_profile: dict | None
    memory_report: dict | None
    checksum: str


def check_execution_inputs(*args, **kwargs):
    return True


def check_live_capacity(*args, **kwargs):
    return None


def choose_plan(packet, result, reference=None):
    """An absent/stale/invalid result always keeps this input's verified fallback."""
    reference = reference if reference is not None else DsppSchedulePlan.from_dict(packet.reference)
    reason = packet.metrics.get('submission_error') or 'planner-not-ready'
    if result is not None:
        reason = result.get('error') or ''
        if result['iteration'] != packet.iteration:
            reason = 'worker-input-mismatch'
        if not reason:
            try:
                candidate = DsppSchedulePlan.from_dict(result['plan'])
                return candidate, ''
            except (ValueError, TypeError, KeyError) as exc:
                reason = f'worker-plan-invalid: {exc}'
    return replace(reference, fallback_reason=f'prefetch-{reason}'), reason


class DsppIterationPrefetch:
    """Opt-in training iterator wrapper. It is never stored in checkpoint args."""

    def __init__(self, args, data_iterator, get_batch_func):
        from megatron.core.datasets.dspp_calibration import stage_context
        from megatron.microbatches import build_num_microbatches_calculator
        freeze_runtime_profiles(args)
        self.args, self.materialize = args, get_batch_func.dspp_materialize_cpu
        self.rank = dist.get_rank()
        # DP=TP=CP=1: the world is exactly one PP replica. This group is not
        # shared with current-iteration ready/credit/ack or training NCCL.
        self.group = dist.new_group(ranks=list(range(dist.get_world_size())), backend='gloo')
        self.source = self.rank == 0
        self.closed, self.future, self.active = False, None, None
        self.planner, self.iterator = None, None
        self.worker_startup_error = ''
        self.records, self.late_results = [], 0
        self.gpu_events = None
        self.prepared_memory = None
        self.prepared_reference = None
        self.checked_execution = None
        if self.source:
            self.iterator = data_iterator[0] if isinstance(data_iterator, list) else data_iterator
            self.cfg = SimpleNamespace(**deepcopy({k: v for k, v in vars(args).items()
                                                   if not k.startswith('_')}))
            self.contexts = {(r, c): stage_context(args, r, c)
                for r in range(args.pipeline_model_parallel_size) for c in range(2)}
            self.calculator = build_num_microbatches_calculator(
                SimpleNamespace(**{**vars(self.cfg), 'rank': 1}))
            try:
                self.planner = CpuPlanner()
            except Exception as exc:
                self.worker_startup_error = f'worker-startup: {type(exc).__name__}: {exc}'
        atexit.register(self.close)

    def _prepare(self, iteration, consumed, num_microbatches):
        started = time.perf_counter()
        cfg = deepcopy(self.cfg)
        cfg.curr_iteration = iteration - 1
        batches, host_bytes = [], 0
        for logical in range(num_microbatches):
            try:
                data = next(self.iterator)
            except StopIteration as exc:
                raise RuntimeError(f'incomplete DSPP iteration {iteration}: '
                    f'{logical}/{num_microbatches} logical batches') from exc
            # IDs are sampler positions, replayable from consumed_train_samples
            # for the supported deterministic single-pass document dataset.
            first = consumed + logical * cfg.micro_batch_size
            batch = prepare_dspp_cpu_batch(data, range(first, first + cfg.micro_batch_size),
                chunk_size=cfg.micro_seq_length, max_sequence_length=cfg.seq_length,
                pad_token_id=max(cfg.variable_seq_pad_token_id, 0),
                pack_residuals=not getattr(cfg, 'dspp_disable_residual_packing', False),
                packing_strategy=cfg.dspp_packing_strategy)
            host_bytes += batch.host_bytes
            if host_bytes > cfg.dspp_prefetch_max_bytes:
                raise ValueError('DSPP future CPU inputs exceed --dspp-prefetch-max-bytes')
            batches.append(batch)
        fetched = time.perf_counter()
        metadata = tuple(DsppTrainingBatch(b.plan, (), input_identity=b.input_identity) for b in batches)
        request = (cfg, metadata, self.contexts, started)
        reference, _ = _build(request, False)
        cpu_ordering = cfg._dspp_cpu_ordering
        cpu_counts = tuple(cfg._dspp_cpu_counts)
        self.prepared_reference = (iteration, reference)
        ready = time.perf_counter()
        # build_runtime_plan caches counters on its args. The worker rebuilds
        # those from the frozen profile; do not send a second accounting graph.
        for name in list(vars(cfg)):
            if name.startswith('_'):
                delattr(cfg, name)
        metadata_bytes = None
        request = (*request, (cpu_ordering, cpu_counts))
        if self.closed:
            error = 'closed'
        elif self.planner is None:
            error = self.worker_startup_error
        else:
            error = self.planner.submit(iteration, reference.binding.input_hash, request)
        metrics = dict(requested_at=started, fetched_at=fetched, metadata_ready_at=ready,
            enqueued_at=time.perf_counter(), host_bytes=host_bytes, metadata_bytes=metadata_bytes,
            submission_error=error, states=['fetched', 'metadata_ready', 'prepared-fallback', 'planning'])
        return CpuIteration(iteration, consumed, num_microbatches, tuple(batches),
                            {}, reference.binding.input_hash, metrics, cpu_ordering, cpu_counts)

    def _request(self, iteration, consumed, count):
        if self.future is not None:
            raise RuntimeError('only one future DSPP iteration may be queued')
        self.future = Future()
        future = self.future
        def run():
            try:
                future.set_result(self._prepare(iteration, consumed, count))
            except Exception as exc:
                future.set_exception(exc)
        threading.Thread(target=run, name='dspp-input-owner', daemon=True).start()

    def acquire_iteration(self, iteration, count, consumed, *, defer_validation=False):
        begin = time.perf_counter()
        value = [None]
        if self.source:
            try:
                cold = self.future is None
                if cold:
                    self._request(iteration, consumed, count)
                packet = self.future.result(timeout=self.args.dspp_prefetch_timeout)
                input_ready = time.perf_counter()
                self.future = None
                if (packet.iteration, packet.num_microbatches, packet.consumed_samples) != (iteration, count, consumed):
                    raise ValueError('prefetch iteration/sample position/microbatch count mismatch')
                result = None
                # Only cold start waits for optimization. Later boundaries use
                # an available verified result or the same-input fallback.
                until = begin + self.args.dspp_prefetch_timeout if cold else time.perf_counter()
                while True:
                    if self.planner is None or not self.planner.process.is_alive():
                        packet.metrics['submission_error'] = self.worker_startup_error or 'worker-unavailable'
                        break
                    item = self.planner.poll()
                    if item is not None:
                        if item['iteration'] == iteration:
                            result = item
                            break
                        self.late_results += 1
                    elif not cold or not self.planner.process.is_alive() or time.perf_counter() >= until:
                        break
                    else:
                        time.sleep(.002)
                reference = self.prepared_reference[1]
                plan, reason = choose_plan(packet, result, reference=reference)
                checked = None
                packet.metrics.update(cold=cold, fallback_reason=reason, late_results=self.late_results,
                    boundary_started_at=begin, input_wait_ms=(input_ready-begin)*1000,
                    optimization_wait_ms=(time.perf_counter()-input_ready)*1000 if cold else 0.,
                    source_wait_ms=(time.perf_counter()-begin)*1000,
                    publish_started_at=time.perf_counter())
                if result:
                    packet.metrics.update({k: v for k, v in result.items() if k.startswith('worker_')})
                # The worker wire payload was already checked by from_dict.
                # Publish the immutable record, then validate its graph and
                # input binding on every rank; don't decode that payload twice.
                # The fast path carries the reference once as an immutable
                # record, without exporting its redundant task/message lists.
                value[0] = (None, replace(packet, reference={}) if defer_validation else packet,
                            plan, checked)
            except Exception as exc:
                value[0] = (f'{type(exc).__name__}: {exc}', None, None, None)
        dist.broadcast_object_list(value, src=0, group=self.group)
        error, packet, plan, checked = value[0]
        if error:
            raise RuntimeError(f'DSPP CPU prefetch failed: {error}')
        self.checked_execution = checked
        self._active_cpu_ordering = packet.ordering
        self._active_cpu_counts = packet.schedule_counts
        local_error = None
        plan_hash = None
        if not defer_validation:
            dist.barrier(group=self.group)
        self.active = dict(iteration=iteration, consumed_samples=consumed, count=count,
            input_hash=packet.input_hash, plan_hash=plan_hash,
            sample_ids=[b.input_identity['sample_ids'] for b in packet.batches],
            valid_tokens=sum(sum(b.plan.sequence_lengths) for b in packet.batches),
            **packet.metrics, received_at=time.perf_counter(),
            boundary_wait_ms=(time.perf_counter()-begin)*1000)
        self.active['states'].append('prepared')
        return packet.batches, plan

    def acquire_execution(self, iteration, count, consumed, dtype, shape, chunk_size):
        """Materialize this iteration and rendezvous, without runtime audit passes."""
        batches, plan = self.acquire_iteration(iteration, count, consumed, defer_validation=True)
        error = None
        try:
            self.active['h2d_started_at'] = time.perf_counter()
            logical = [self.materialize(batch) for batch in batches]
            self.active['h2d_queued_at'] = time.perf_counter()
            ordering = self._active_cpu_ordering
            counts = self._active_cpu_counts
            self.args._dspp_counted_memory = None
            self.args._dspp_memory_plan = None
            self.args._dspp_last_schedule_plan = plan.to_dict()
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
        # This is the executor rendezvous, not a plan/hash agreement check.
        failed = torch.tensor([int(error is not None)], dtype=torch.int32, device='cpu')
        dist.all_reduce(failed, group=self.group)
        if failed.item():
            raise RuntimeError(error or 'DSPP peer input materialization failed')
        self.active['commit_collectives'] = 1
        self.active['validation_path'] = 'no-runtime-fingerprint-or-audit'
        self.commit(plan)
        return logical, ordering, counts, plan, None

    def commit(self, plan, plan_hash=None):
        self.active.update(committed_at=time.perf_counter(), plan_hash=plan_hash or plan.plan_hash)
        self.active['commit_check_ms'] = (
            (self.active['committed_at'] - self.active['h2d_queued_at']) * 1000
            if 'h2d_queued_at' in self.active else 0.)
        self.active['states'].extend(('committed', 'executing'))
        if self.source and self.active['iteration'] < self.args.train_iters:
            consumed = self.active['consumed_samples'] + self.active['count'] * self.args.micro_batch_size
            self.calculator.update(consumed, True)
            self._request(self.active['iteration'] + 1, consumed, self.calculator.get())

    def mark_compute(self, finished=False):
        if not finished:
            self.gpu_events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        self.gpu_events[int(finished)].record()
        self.active['compute_last_queued_at' if finished else 'compute_first_queued_at'] = time.perf_counter()

    def retire(self):
        self.active.update(retired_at=time.perf_counter(), optimizer_completed_at=time.perf_counter())
        self.active['states'].append('retired')
        if self.gpu_events and self.gpu_events[1].query():
            self.active['gpu_compute_ms'] = self.gpu_events[0].elapsed_time(self.gpu_events[1])
        self.args._dspp_prefetch_metrics = dict(self.active)
        if getattr(self.args, 'dspp_prefetch_report_dir', None):
            self.records.append(dict(self.active))
        self.active, self.gpu_events = None, None

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.planner is not None:
            self.planner.close()
        directory = getattr(self.args, 'dspp_prefetch_report_dir', None)
        if directory:
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            pending = None
            if self.future is not None and self.future.done() and not self.future.exception():
                packet = self.future.result()
                pending = dict(iteration=packet.iteration, consumed_samples=packet.consumed_samples,
                               input_hash=packet.input_hash)
            (path / f'rank{self.rank}.json').write_text(json.dumps(dict(records=self.records,
                unconsumed_future_discarded=self.future is not None,
                unconsumed_future_identity=pending,
                replay_from_consumed_train_samples=True), indent=2))
        # No collective in atexit/error cleanup. The group lives until teardown.
        atexit.unregister(self.close)
