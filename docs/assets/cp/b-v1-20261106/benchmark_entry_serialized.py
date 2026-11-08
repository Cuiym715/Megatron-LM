"""Timed entry: validated NCCL enqueue mutex, without diagnostic sampling."""
import functools,inspect,json,os,runpy,threading
from pathlib import Path
import torch
import torch.distributed as dist
assert os.environ['NCCL_LAUNCH_ORDER_IMPLICIT']=='0'
assert os.environ['NCCL_LAUNCH_RACE_FATAL']=='1'
launch_lock=threading.RLock()
patched=[]
def is_nccl(group):
    if not dist.is_initialized() or group==dist.GroupMember.NON_GROUP_MEMBER:return False
    return str(dist.get_backend(group))=='nccl'
def wrap(name):
    native=getattr(dist,name);signature=inspect.signature(native)
    @functools.wraps(native)
    def call(*args,**kwargs):
        bound=signature.bind_partial(*args,**kwargs)
        if name=='batch_isend_irecv':
            operations=args[0] if args else kwargs['p2p_op_list']
            nccl=any(is_nccl(op.group) for op in operations)
        else:nccl=is_nccl(bound.arguments.get('group'))
        if not nccl:return native(*args,**kwargs)
        synchronous='async_op' in signature.parameters and not bound.arguments.get('async_op',False)
        if synchronous:bound.arguments['async_op']=True
        with launch_lock:work=native(*bound.args,**bound.kwargs)
        if synchronous:
            work.wait()
            return None
        return work
    setattr(dist,name,call);patched.append(name)
for name in ('batch_isend_irecv','all_reduce','all_reduce_coalesced','all_gather',
             'all_gather_into_tensor','_all_gather_base','reduce_scatter',
             'reduce_scatter_tensor','_reduce_scatter_base','broadcast','reduce',
             'barrier','all_gather_object','broadcast_object_list','all_to_all','all_to_all_single'):
    if hasattr(dist,name):wrap(name)
runpy.run_path(str(Path(__file__).with_name('benchmark_entry_warmup.py')),run_name='__main__')
out=Path(os.environ['DSPP_CP_BENCH_OUTPUT'])/f'launch-policy-rank{os.environ["RANK"]}.json'
assert not out.exists()
out.write_text(json.dumps(dict(policy='nccl-enqueue-mutex-native-order-wait-outside-v1',
    nccl_version=torch.cuda.nccl.version(),implicit_order='0',race_fatal='1',
    patched=patched,per_operation_logging=False,CPU_wait_sampling=False),indent=2)+'\n')
