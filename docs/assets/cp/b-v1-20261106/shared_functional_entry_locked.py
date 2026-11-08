"""Serialize NCCL host submissions; preserve Gloo credit/ack progress.

No GPU synchronization, tensor/math changes, or per-operation logging.
Keep native isend/irecv identities: P2POp validates those functions.
"""
import functools,inspect,json,os,runpy,threading
from pathlib import Path
import torch.distributed as dist

launch_lock=threading.RLock()
counts={}
patched=[]

def is_nccl(group):
    if not dist.is_initialized() or group == dist.GroupMember.NON_GROUP_MEMBER:
        return False
    return str(dist.get_backend(group)) == 'nccl'

def wrap(name):
    native=getattr(dist,name)
    signature=inspect.signature(native)
    @functools.wraps(native)
    def call(*args,**kwargs):
        if name=='batch_isend_irecv':
            operations=args[0] if args else kwargs['p2p_op_list']
            nccl=any(is_nccl(op.group) for op in operations)
        else:
            group=signature.bind_partial(*args,**kwargs).arguments.get('group')
            nccl=is_nccl(group)
        if not nccl:
            return native(*args,**kwargs)
        with launch_lock:
            counts[name]=counts.get(name,0)+1
            return native(*args,**kwargs)
    setattr(dist,name,call)
    patched.append(name)

for name in ('batch_isend_irecv','all_reduce','all_reduce_coalesced',
             'all_gather','all_gather_into_tensor','_all_gather_base',
             'reduce_scatter','reduce_scatter_tensor','_reduce_scatter_base',
             'broadcast','reduce','barrier','all_gather_object',
             'broadcast_object_list','all_to_all','all_to_all_single'):
    if hasattr(dist,name):wrap(name)

runpy.run_path(str(Path(__file__).with_name('shared_functional_entry.py')),run_name='__main__')
out=Path(os.environ['DSPP_CP_BENCH_OUTPUT'])/f'launch-serialization-rank{os.environ["RANK"]}.json'
assert not out.exists()
out.write_text(json.dumps(dict(status='completed',patched=patched,nccl_calls=counts,
    gpu_synchronization_added=False,gloo_cpu_operations_locked=False,
    math_or_source_changed=False,timing_eligible=False),indent=2)+'\n')
