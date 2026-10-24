"""Opt-in storage ownership audit and order-based capacity contracts.

The audit never retains tensors, waits for CUDA, or changes execution order.
Boundary snapshots cannot see every transient workspace/copy. The allocator
peak and uncovered residual are reported separately; incomplete audits cannot
be promoted into a certified capacity contract.
"""
from dataclasses import asdict, dataclass
import functools
import json
import math
from pathlib import Path
import types

import torch

from megatron.core.datasets.dspp_calibration import fingerprint


def storage_owners(roots):
    """Follow containers, closures, module state and autograd saved tensors.

    Count full allocations once, retaining each view's offset/extent and all
    owner labels. Python object ids/storage handles remain local to this audit.
    """
    storages, visited, keepalive = {}, set(), []
    def visit(value, owner):
        if value is None or isinstance(value, (str, bytes, int, float, bool, type)):
            return
        marker = (id(value), owner)
        if marker in visited:
            return
        visited.add(marker)
        keepalive.append(value)  # avoid id reuse of temporary autograd wrappers
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            key = f'{value.device}:{storage._cdata}'
            row = storages.setdefault(key, dict(bytes=storage.nbytes(), device=str(value.device),
                                                address=storage.data_ptr(),
                                                pinned=value.is_pinned(), owners=set(), views=set()))
            row['owners'].add(owner)
            row['views'].add((value.storage_offset(), tuple(value.shape), tuple(value.stride()), str(value.dtype)))
            if value.grad_fn is not None:
                visit(value.grad_fn, owner)
            if hasattr(value, 'main_grad'):
                visit(value.main_grad, 'resident-main-grad')
            return
        if isinstance(value, dict):
            children = value.values()
        elif isinstance(value, (list, tuple, set)) or type(value).__name__ == 'deque':
            children = value
        elif isinstance(value, functools.partial):
            children = (value.func, value.args, value.keywords)
        elif isinstance(value, types.FunctionType):
            children = [value.__defaults__, value.__kwdefaults__]
            for cell in value.__closure__ or ():
                try:
                    children.append(cell.cell_contents)
                except ValueError:
                    pass
        elif isinstance(value, torch.nn.Module):
            children = (value._parameters, value._buffers, value._modules)
        elif hasattr(value, 'next_functions'):
            children = [node for node, _ in value.next_functions]
            for name in dir(value):
                if name.startswith('_saved_') or name == 'saved_tensors':
                    try:
                        children.append(getattr(value, name))
                    except RuntimeError:  # graph already released at this boundary
                        pass
        elif hasattr(value, '__dict__') and type(value).__module__.startswith('megatron.'):
            children = vars(value).values()
        else:
            return
        for child in children:
            visit(child, owner)
    for owner, root in roots.items():
        visit(root, owner)
    result = {key: dict(row, owners=sorted(row['owners']), views=[list(v) for v in sorted(row['views'])])
              for key, row in storages.items()}
    # Recursive visit has a self-referencing closure. Empty the temporary
    # keepalive explicitly: waiting for cyclic GC would prolong GPU lifetimes.
    keepalive.clear()
    return result


@dataclass(frozen=True)
class OwnerLifetime:
    owner: str
    storage: str
    bytes: int
    acquire: int
    release: int  # exclusive; actual last user, including asynchronous consumers
    pool: str = 'cuda'


def conservative_peak(lifetimes, *, reserve_bytes=0, pool='cuda'):
    """Per-card submission order; no dependence on predicted milliseconds."""
    if reserve_bytes < 0:
        raise ValueError('negative reserve')
    rows = [row for row in lifetimes if row.pool == pool]
    if any(row.bytes < 0 or row.acquire < 0 or row.release <= row.acquire for row in rows):
        raise ValueError('invalid storage lifetime')
    peak = reserve_bytes
    for boundary in sorted({r.acquire for r in rows} | {r.release for r in rows}):
        active = {}
        for row in rows:
            if row.acquire <= boundary < row.release:
                if row.storage in active and active[row.storage] != row.bytes:
                    raise ValueError('conflicting sizes for shared storage')
                active[row.storage] = row.bytes
        peak = max(peak, reserve_bytes + sum(active.values()))
    return peak


@dataclass(frozen=True)
class MemoryContract:
    physical_bytes: int
    resident_bytes: int
    physical_reserve_bytes: int
    baseline_dynamic_bytes: int
    dynamic_growth: float = .05
    workspace_bytes: int = 0
    communication_bytes: int = 0
    submission_bytes: int = 0
    pinned_host_limit_bytes: int = 0
    certified: bool = False
    identity: str = ''

    def __post_init__(self):
        values = asdict(self)
        if any(v < 0 for k, v in values.items() if isinstance(v, (int, float))):
            raise ValueError('negative memory capacity')
        if not math.isfinite(self.dynamic_growth):
            raise ValueError('nonfinite dynamic growth')

    @property
    def version(self):
        return fingerprint(asdict(self))

    def validate(self, lifetimes, *, version, identity, observed_allocated=0, observed_reserved=0):
        if version != self.version or identity != self.identity:
            raise ValueError('stale memory contract or layout')
        if not self.certified:
            raise ValueError('uncertified owner/workspace bounds')
        dynamic = conservative_peak(lifetimes, reserve_bytes=(self.workspace_bytes +
            self.communication_bytes + self.submission_bytes))
        total = self.resident_bytes + dynamic
        if total > self.physical_bytes - self.physical_reserve_bytes:
            raise ValueError('physical memory capacity exceeded')
        if observed_reserved > self.physical_bytes - self.physical_reserve_bytes:
            raise ValueError('reserved memory capacity exceeded')
        if dynamic > self.baseline_dynamic_bytes * (1 + self.dynamic_growth):
            raise ValueError('dynamic memory growth exceeded')
        if observed_allocated > total:
            raise ValueError('observed allocator peak exceeds owner bound')
        if conservative_peak(lifetimes, pool='pinned') > self.pinned_host_limit_bytes:
            raise ValueError('pinned host capacity exceeded')
        return dict(dynamic_bytes=dynamic, total_bytes=total)


class MemoryAudit:
    def __init__(self, directory, stage, iteration):
        self.directory, self.stage, self.iteration = Path(directory), stage, iteration
        self.rows = []
        self.baseline_allocated = torch.cuda.memory_allocated()
        self.peak_at_start = torch.cuda.max_memory_allocated()
        self.baseline_keys = None
        self.events = []
        # Diagnostic-only allocator history includes allocation/free_requested/
        # free_completed events and transient copy/workspace allocations missed
        # by boundary references. It is evidence, not a future workspace bound.
        torch.cuda.memory._record_memory_history(enabled='all', max_entries=100000)

    def snapshot(self, label, roots):
        owners = storage_owners(roots)
        cuda = {key for key, row in owners.items() if row['device'].startswith('cuda')}
        if self.baseline_keys is None:
            self.baseline_keys = cuda
        dynamic = sum(owners[key]['bytes'] for key in cuda - self.baseline_keys)
        completed = [index for index, event in enumerate(self.events) if event.query()]
        event = torch.cuda.Event()
        event.record()
        self.events.append(event)
        self.rows.append(dict(boundary=len(self.rows), label=label, storages=owners,
                              completed_compute_boundaries=completed,
                              tracked_dynamic_bytes=dynamic,
                              allocated_bytes=torch.cuda.memory_allocated(),
                              reserved_bytes=torch.cuda.memory_reserved(),
                              allocator_peak_bytes=torch.cuda.max_memory_allocated()))

    def finish(self):
        peak = torch.cuda.max_memory_allocated()
        tracked = max((r['tracked_dynamic_bytes'] for r in self.rows), default=0)
        value = dict(schema='dspp-owner-audit-v1', stage=self.stage, iteration=self.iteration,
                     baseline_allocated_bytes=self.baseline_allocated,
                     peak_at_start_bytes=self.peak_at_start, allocated_peak_bytes=peak,
                     reserved_peak_bytes=torch.cuda.max_memory_reserved(),
                     tracked_dynamic_peak_bytes=tracked,
                     uncovered_peak_increment_bytes=max(0, peak-self.baseline_allocated-tracked),
                     certified=False,
                     limitations=['boundary snapshots miss transient copy/workspace peaks',
                                  'logical references do not certify CUDA stream retirement',
                                  'optimizer/resident allocations included in allocator baseline',
                                  'allocator peak is process watermark unless caller reset it'],
                     boundaries=self.rows)
        self.directory.mkdir(parents=True, exist_ok=True)
        trace_path = self.directory / f'iteration_{self.iteration:06d}_stage_{self.stage}_allocator.pickle'
        torch.cuda.memory._dump_snapshot(str(trace_path))
        torch.cuda.memory._record_memory_history(enabled=None)
        value['allocator_trace'] = str(trace_path)
        path = self.directory / f'iteration_{self.iteration:06d}_stage_{self.stage}.json'
        path.write_text(json.dumps(value, indent=2))
        return path
