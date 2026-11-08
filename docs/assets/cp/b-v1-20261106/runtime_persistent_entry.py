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

# Read-only background sampler: no main-thread per-update IO or tensor hooks.
import sys,time
monitor_done=threading.Event()
def sample_wait():
    while not monitor_done.wait(10):
        row=dict(time=time.time(),threads=[])
        try:
            from megatron import get_args
            row['iteration']=get_args().curr_iteration+1
        except Exception:pass
        for tid,frame in list(sys._current_frames().items()):
            stack=[];schedule=None
            while frame is not None:
                if len(stack)<8:stack.append(dict(function=frame.f_code.co_name,line=frame.f_lineno))
                if frame.f_code.co_name=='pipelining_with_dspp_stage_local_v':schedule=dict(frame.f_locals)
                frame=frame.f_back
            item=dict(tid=tid,stack=stack)
            if schedule is not None:
                v=schedule;item['stage']=v.get('stage');item['node']=repr(v.get('node'))
                for field in ('send_cursor','recv_cursor','ack_counts','credit_counts'):
                    item[field]=dict(v.get(field,{}))
                item['ready_sends']={str(k):[repr(x) for x in list(z.keys())[:12]] for k,z in v.get('ready_sends',{}).items()}
                for field,cursor in [('expected_sends','send_cursor'),('expected_receives','recv_cursor')]:
                    item[field]={str(k):[repr(x) for x in z[v.get(cursor,{}).get(k,0):v.get(cursor,{}).get(k,0)+5]] for k,z in v.get(field,{}).items()}
                item['inflight']={str(k):len(z) for k,z in v.get('inflight_sends',{}).items()}
            row['threads'].append(item)
        path=Path(os.environ['DSPP_CP_BENCH_OUTPUT'])/f'wait-state-rank{os.environ["RANK"]}.jsonl'
        with path.open('a') as f:f.write(json.dumps(row)+'\n')
threading.Thread(target=sample_wait,daemon=True,name='read-only-background-sampler').start()

# Isolated source-derived runtime correction: reuse PP streams per rank/edge.
sys.path.insert(0,os.environ['CP_V1_SOURCE'])
import hashlib,difflib
from megatron.core.pipeline_parallel import schedules
original_schedule=schedules.pipelining_with_dspp_stage_local_v
source=inspect.getsource(original_schedule)
old_send='send_streams = {peer: torch.cuda.Stream() for peer in expected_sends} if cp_p2p else {}'
old_recv='recv_streams = {peer: torch.cuda.Stream() for peer in expected_sends} if cp_p2p else {}'
assert source.count(old_send)==1 and source.count(old_recv)==1
changed=source.replace(old_send,'send_streams = {peer: _persistent_pp_stream(stage, peer, "send") for peer in expected_sends} if cp_p2p else {}').replace(old_recv,'recv_streams = {peer: _persistent_pp_stream(stage, peer, "recv") for peer in expected_sends} if cp_p2p else {}')
stream_cache={}
def persistent_pp_stream(stage,peer,direction):
    key=(torch.cuda.current_device(),stage,peer,direction)
    if key not in stream_cache:stream_cache[key]=torch.cuda.Stream()
    return stream_cache[key]
schedules.__dict__['_persistent_pp_stream']=persistent_pp_stream
exec(compile(changed,original_schedule.__code__.co_filename,'exec'),schedules.__dict__)
root=Path(os.environ['DSPP_CP_BENCH_OUTPUT'])
(root/f'stream-cache-patch-rank{os.environ["RANK"]}.json').write_text(json.dumps(dict(
    original_sha256=hashlib.sha256(source.encode()).hexdigest(),patched_sha256=hashlib.sha256(changed.encode()).hexdigest(),
    delta='reusePPsend/recvCUDAstreams perdevice/stage/peer/direction; nootherfunctionchange',production_modified=False,
    hypothesis='RepeatedperiterationStream allocations cyclethroughsharedpool andmayalias NCCL streams; hypothesisnotproven'))+'\n')
(root/f'stream-cache-patch-rank{os.environ["RANK"]}.diff').write_text(''.join(difflib.unified_diff(source.splitlines(True),changed.splitlines(True))))
runpy.run_path(str(Path(__file__).with_name('benchmark_entry_warmup.py')),run_name='__main__')
monitor_done.set()
out=Path(os.environ['DSPP_CP_BENCH_OUTPUT'])/f'launch-policy-rank{os.environ["RANK"]}.json'
assert not out.exists()
out.write_text(json.dumps(dict(policy='nccl-enqueue-mutex-native-order-wait-outside-v1',
    nccl_version=torch.cuda.nccl.version(),implicit_order='0',race_fatal='1',
    patched=patched,per_operation_logging=False,CPU_wait_sampling=False),indent=2)+'\n')

(root/f'stream-cache-final-rank{os.environ["RANK"]}.json').write_text(json.dumps({str(k):v.cuda_stream for k,v in stream_cache.items()})+"\n")
