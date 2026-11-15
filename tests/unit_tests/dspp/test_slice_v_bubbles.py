"""CPU-only scarce-work V schedule tests; no fake compute or P2P tasks."""
import ast
from dataclasses import asdict
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[3]
if __name__ == '__main__':
    for name in ('megatron','megatron.core','megatron.core.datasets','megatron.core.pipeline_parallel'):
        module = types.ModuleType(name)
        module.__path__ = [str(ROOT/name.replace('.','/'))]
        sys.modules[name] = module
    if importlib.util.find_spec('torch') is None:
        sys.modules['torch'] = types.ModuleType('torch')

from megatron.core.pipeline_parallel.slice_v import build_slice_v_schedule, build_slice_v_execution_plan
from megatron.core.pipeline_parallel.dspp_plan import (
    PlanBinding, template_plan, validate_plan, receive_orders, incoming_sender,
    message_key, node_key, replay_plan,
)
from megatron.core.datasets.dspp_batch_plan import build_dspp_batch_plan
from megatron.core.datasets.dspp_ordering import build_dspp_ordering_plan

# Reuse the existing bounded credit/ready protocol simulator without importing
# the CUDA training package or requiring pytest for this standalone CPU test.
path = ROOT/'tests/unit_tests/dspp/test_schedule_plan.py'
function = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n,ast.FunctionDef) and n.name == 'simulate_protocol')
exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'))


def plan_for(pp,counts):
    schedules, meta = build_slice_v_schedule(pp,len(counts),counts)
    mapping = [(g,s,g,s) for g,count in enumerate(counts) for s in range(count)]
    return template_plan(schedules,meta.split_counts,mapping,PlanBinding(1,'input','layout'))


class BubbleTests(unittest.TestCase):
    def test_small_batches_dependencies_and_exact_wire_counts(self):
        for pp in (1,2,3,4,8):
            for counts in [(1,),(8,),(16,),(1,1),(8,4),(2,2,2),(1,8,1),
                           *itertools.product(range(1,5),repeat=3)]:
                with self.subTest(pp=pp,counts=counts):
                    plan = plan_for(pp,counts)
                    validate_plan(plan)
                    build_slice_v_execution_plan(plan.schedules)
                    self.assertEqual(plan.split_counts,tuple(counts))
                    self.assertTrue(all(len(seq)==6*sum(counts) for seq in plan.schedules))
                    self.assertEqual(sum(map(len,receive_orders(plan.schedules).values())),
                                     4*(pp-1)*sum(counts))

    def test_bounded_communication_progress(self):
        for pp in (2,3,4,8):
            for counts in ((1,),(3,),(1,1),(8,1),(1,1,1,1,1,1,1,1,4)):
                plan=plan_for(pp,counts)
                for lookahead in (0,1,2):
                    for depth in (1,2):
                        simulate_protocol(plan,lookahead,depth)

    def test_real_4096_workload_has_twelve_tasks_not_eighteen(self):
        batch=types.SimpleNamespace(plan=build_dspp_batch_plan([2048]*16+[16384],chunk_size=4096))
        order=build_dspp_ordering_plan([batch],pp_degree=8)
        counts=tuple(map(len,order.schedule_groups))
        self.assertEqual(sum(counts),12)
        plan=plan_for(8,counts)
        self.assertEqual(len(plan.schedules[0]),72)  # F/B/W × VPP2 × 12
        self.assertEqual(sum(map(len,receive_orders(plan.schedules).values())),336)
        simulate_protocol(plan,2,2)
        costs={node_key(n):1. for seq in plan.schedules for n in seq}
        messages={('message',a,b,*m):.1 for (a,b),lane in receive_orders(plan.schedules).items() for m in lane}
        self.assertGreater(replay_plan(plan,costs,messages)['makespan_ms'],0)

    def test_saturated_schedules_are_unchanged(self):
        # Golden hashes from the pre-change committed generator, all node fields.
        cases=[(3,[6,5],'62c9cbf1f0a045d73344e89b0b03d9bfbd168d19cb5ed276fac2041d40cc057b'),
               (8,[8,8,8],'ba6f03ffa8d6d33ce22ecb30cb584b88eda0af8603661a2792e9f612252437b7'),
               (2,[2,4],'a0be00366bcf53afd40b27be93e1eeec9ad9b0d85cb06b3d547eeba3c8b8d868')]
        for pp,counts,expected in cases:
            schedules,_=build_slice_v_schedule(pp,len(counts),counts)
            value=json.dumps([[asdict(n) for n in seq] for seq in schedules],sort_keys=True).encode()
            self.assertEqual(hashlib.sha256(value).hexdigest(),expected)


if __name__=='__main__':unittest.main()
