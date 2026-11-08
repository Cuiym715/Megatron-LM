"""Excluded runtime-hang diagnostic; CPU progress and stack traces, no tensors."""
import faulthandler,json,os,runpy,sys,time,threading
from pathlib import Path
SRC=Path(os.environ['CP_V1_SOURCE']);sys.path.insert(0,str(SRC))
from megatron import training,get_args
root=Path(os.environ['DSPP_CP_BENCH_OUTPUT']);root.mkdir(exist_ok=True)
rank=os.environ['RANK']
main_ident=threading.get_ident()
def inspect_wait():
    while True:
        time.sleep(10)
        frame=sys._current_frames().get(main_ident)
        while frame is not None:
            if frame.f_code.co_name=='pipelining_with_dspp_stage_local_v':
                v=frame.f_locals
                row=dict(time=time.time(),stage=v.get('stage'),node=repr(v.get('node')),physical_id=v.get('physical_id'))
                for field in ['send_cursor','recv_cursor','ack_counts','credit_counts']:
                    row[field]=dict(v.get(field,{}))
                row['ready_sends']={str(k):[repr(x) for x in list(z.keys())[:12]] for k,z in v.get('ready_sends',{}).items()}
                for field,cursor in [('expected_sends','send_cursor'),('expected_receives','recv_cursor')]:
                    row[field]={str(k):[repr(x) for x in z[v.get(cursor,{}).get(k,0):v.get(cursor,{}).get(k,0)+5]] for k,z in v.get(field,{}).items()}
                row['inflight']={str(k):len(z) for k,z in v.get('inflight_sends',{}).items()}
                with (root/f'wait-state-rank{rank}.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                break
            frame=frame.f_back
threading.Thread(target=inspect_wait,daemon=True,name='read-only-frame-sampler').start()
original=training.train_step
def progress(*args,**kwargs):
    iteration=get_args().curr_iteration+1
    with (root/f'progress-rank{rank}.jsonl').open('a') as f:
        f.write(json.dumps(dict(event='begin',iteration=iteration,time=time.time()))+'\n')
    result=original(*args,**kwargs)
    with (root/f'progress-rank{rank}.jsonl').open('a') as f:
        f.write(json.dumps(dict(event='end',iteration=iteration,time=time.time(),skipped=int(result[1])))+'\n')
    return result
training.train_step=progress
runpy.run_path(str(Path(__file__).with_name('benchmark_entry.py')),run_name='__main__')
