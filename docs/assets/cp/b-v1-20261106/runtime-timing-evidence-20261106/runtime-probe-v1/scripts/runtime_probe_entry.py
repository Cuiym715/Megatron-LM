"""Excluded runtime-hang diagnostic; CPU progress and stack traces, no tensors."""
import faulthandler,json,os,runpy,sys,time
from pathlib import Path
SRC=Path(os.environ['CP_V1_SOURCE']);sys.path.insert(0,str(SRC))
from megatron import training,get_args
root=Path(os.environ['DSPP_CP_BENCH_OUTPUT']);root.mkdir(exist_ok=True)
rank=os.environ['RANK']
stack=(root/f'stacks-rank{rank}.log').open('x')
faulthandler.enable(file=stack,all_threads=True)
faulthandler.dump_traceback_later(80,repeat=True,file=stack)
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
faulthandler.cancel_dump_traceback_later()
stack.close()
