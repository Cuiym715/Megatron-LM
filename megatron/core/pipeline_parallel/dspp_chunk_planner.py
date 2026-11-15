"""One metadata preparation per fine candidate, one B call on its winner."""
from copy import copy
from dataclasses import dataclass, replace
import math
from types import SimpleNamespace

from megatron.core.datasets.dspp_batch_plan import build_dspp_batch_plan
from megatron.core.datasets.dspp_training import DsppTrainingBatch
from megatron.core.datasets.dspp_ordering import build_dspp_ordering_plan
from megatron.core.datasets.dspp_p2p import DsppP2PCostModel, DsppP2PPoint
from megatron.core.datasets.dspp_chunk_search import ChunkSearch
from .slice_v import build_slice_v_schedule
from .dspp_plan import PlanBinding, template_plan, history_dependencies, node_key, receive_orders, replay_plan
from .dspp_memory_counts import CountedMemory


@dataclass
class PreparedChunk:
    capacity: int
    batches: tuple
    ordering: object
    counts: tuple
    reference: object
    slots: dict
    compute: dict
    messages: dict
    memory_report: dict
    predicted_ms: float


class ChunkPlanner:
    def __init__(self, model, cfg, options):
        names = (*model.payload['layout']['model'], 'params_dtype','pipeline_model_parallel_size',
                 'dspp_packing_strategy','dspp_microbatch_order','dspp_order_profile',
                 'dspp_disable_residual_packing','dspp_recv_lookahead','dspp_send_inflight_depth',
                 'dspp_history_dkv_after_dx','dspp_split_attention_backward','dspp_planner_backend')
        if getattr(cfg, 'dspp_planner_backend', 'cpp') == 'cpp':
            from .dspp_native import library
            library()
        self.model = model
        self.cfg = SimpleNamespace(**{k:getattr(cfg,k) for k in names if hasattr(cfg,k)})
        self.fused_cfg = copy(self.cfg)
        self.fused_cfg.dspp_history_dkv_after_dx = False
        self.fused_cfg.dspp_split_attention_backward = False
        self.search = ChunkSearch(model, options)
        self.p2p = DsppP2PCostModel(tuple(DsppP2PPoint(**p) for p in model.payload['p2p']['points']))
        if cfg.pipeline_model_parallel_size != model.pp or getattr(cfg,'context_parallel_size',1) != 1:
            raise ValueError('chunk model requires matching PP and CP1')
        for name, value in model.payload['layout']['model'].items():
            if getattr(cfg, name, None) != value:
                raise ValueError('chunk model configuration mismatch: '+name)
        if str(cfg.params_dtype) != model.payload['layout']['dtype']:
            raise ValueError('chunk model dtype mismatch')

    def memory_profile(self, capacity):
        memory = self.model.payload['memory']
        rows = []
        for source in memory['stages']:
            row = dict(source)
            for field in memory['scaled_fields']:
                row[field] = [math.ceil(v*capacity) for v in source[field]]
            rows.append(row)
        return dict(schema=memory['schema'], accounting=memory['accounting'],
                    stages=rows, dynamic_growth=memory['dynamic_growth'])

    def pack(self, lengths, capacity):
        if not self.model.domain[0] <= capacity <= self.model.domain[1]:
            raise ValueError('capacity outside model search domain')
        return tuple(
            build_dspp_batch_plan(
                group, chunk_size=capacity,
                packing_strategy=self.cfg.dspp_packing_strategy,
                pack_residuals=not getattr(self.cfg, 'dspp_disable_residual_packing', False),
                validate=False,
            )
            for group in lengths
        )

    def fine(self, lengths, identities, iteration, input_key, capacity, *, plans=None):
        cfg, pp = self.cfg, self.model.pp
        if plans is None:
            plans = self.pack(lengths, capacity)
        batches = tuple(DsppTrainingBatch(plan, (), input_identity=identity)
                        for plan, identity in zip(plans, identities))
        ordering = build_dspp_ordering_plan(batches, pp_degree=pp, mode=cfg.dspp_microbatch_order,
                                           profile_path=getattr(cfg,'dspp_order_profile',None))
        groups = ordering.schedule_groups
        counts = tuple(len(g) for g in groups)
        schedules, _ = build_slice_v_schedule(pp, len(counts), counts)
        task_map = tuple((g,s,*task) for g,group in enumerate(groups) for s,task in enumerate(group))
        metadata = {(l,p):meta for l,b in enumerate(batches) for p,meta in enumerate(b.plan.microbatches)}
        slots = {(g,s):metadata[l,p] for g,s,l,p in task_map}
        binding = PlanBinding(iteration, input_key, f'PP{pp}/CP1/c{capacity}',
            cost_version='resident-cross-capacity-v1', capacity_version='resident-memory-envelope-v1',
            sample_ids=tuple(x for identity in identities for x in identity['sample_ids']))
        reference = template_plan(schedules, counts, task_map, binding,
                                  history_dependencies=history_dependencies(task_map,metadata))
        reference = replace(reference, wire_shape=(capacity,1,cfg.hidden_size),
                            wire_dtype=str(cfg.params_dtype), wire_bytes=capacity*cfg.hidden_size*2)
        memory = CountedMemory(self.memory_profile(capacity), self.fused_cfg, reference,
                               slots, cfg.params_dtype, capacity)
        report = memory.check(reference)
        cache, compute = {}, {}
        shapes = {slot:tuple((i.token_length,i.token_offset) for i in meta.items) for slot,meta in slots.items()}
        for seq in schedules:
            for node in seq:
                shape = shapes.get((node.microbatch,node.split))
                if shape is None:
                    compute[node_key(node)] = 0.
                    continue
                op = 'B_fused' if node.kind=='B' else node.kind
                key = node.stage, node.chunk, op, shape if op!='W' else ()
                if key not in cache:
                    cache[key] = self.model.predict(node.stage,node.chunk,op,capacity,shape)
                compute[node_key(node)] = cache[key]
        messages = {}
        for (sender,receiver), lane in receive_orders(schedules).items():
            pred = self.p2p.predict_detailed(reference.wire_bytes, direction=f'{sender}->{receiver}',overlap='compute')
            if pred.extrapolated:
                raise ValueError('chunk wire exceeds measured P2P domain')
            for message in lane:
                messages[('message',sender,receiver,*message)] = pred.duration_ms
        replay = replay_plan(reference, compute, messages)
        reference = replace(reference,reference_ms=replay['makespan_ms'],predicted_ms=replay['makespan_ms'])
        return PreparedChunk(capacity,batches,ordering,counts,reference,slots,compute,messages,report,replay['makespan_ms'])

    def prepare(self, lengths, identities, iteration, consumed, *, fixed=None):
        input_key = f'{iteration}:{consumed}:{len(lengths)}'
        return self.search.select(
            (n for group in lengths for n in group),
            lambda capacity: self.pack(lengths, capacity),
            lambda capacity, plans: self.fine(
                lengths, identities, iteration, input_key, capacity, plans=plans,
            ),
            fixed=fixed,
        )

    def b_request(self, prepared, started):
        # Static models/configuration never enter this per-iteration request.
        return dict(reference=prepared.reference, slots=prepared.slots, compute=prepared.compute,
                    messages=prepared.messages, capacity=prepared.capacity, planning_started=started)

    def optimize(self, request, budget_fraction):
        from .dspp_plan_b import plan_configuration_b
        reference, capacity = request['reference'], request['capacity']
        slots, split, tokens, cache = request['slots'], {}, {}, {}
        for seq in reference.schedules:
            for node in seq:
                meta = slots.get((node.microbatch,node.split))
                if meta is None:
                    continue
                if node.kind == 'F':
                    tokens[node_key(node)] = meta.valid_token_count
                if node.kind != 'B' or not any(i.has_history for i in meta.items):
                    continue
                shape = tuple((i.token_length,i.token_offset) for i in meta.items)
                key = node.stage,node.chunk,shape
                if key not in cache:
                    cache[key] = tuple(self.model.predict(node.stage,node.chunk,op,capacity,shape) for op in ('D','H'))
                split[node_key(node)] = cache[key]
        memory = CountedMemory(self.memory_profile(capacity), self.cfg, reference, slots,
                               self.cfg.params_dtype, capacity)
        plan = plan_configuration_b(reference,request['compute'],request['messages'],split,
            owners=memory.owners,capacities=memory.capacities,tokens=tokens,memory=memory,
            planning_started=request['planning_started'],budget_fraction=budget_fraction,
            backend=getattr(self.cfg, 'dspp_planner_backend', 'cpp'), slot_meta=slots)
        return plan, memory.check(plan)
