"""29L PP4 endpoint adaptation; base/production source files remain unchanged."""
import os,sys,runpy,hashlib,json
from pathlib import Path
sys.path.insert(0,os.environ['CP_V1_SOURCE'])
from megatron.core import dspp_layout
replacement=runpy.run_path(str(Path(__file__).with_name('dspp_layout29.py')))
# Update the same function object, preserving aliases already imported by arguments.
dspp_layout.logical_layer_ranges.__code__=replacement['logical_layer_ranges'].__code__
assert dspp_layout.logical_layer_ranges(29,4,'endpoint-3-2')==((0,3),(3,7),(7,11),(11,15),(15,19),(19,23),(23,27),(27,29))
from megatron import training,get_args
from megatron.core import parallel_state
native_step=training.train_step
ownership_recorded=False
def checked_step(*a,**kw):
    global ownership_recorded
    if not ownership_recorded:
        cfg=get_args();stage=parallel_state.get_pipeline_model_parallel_rank()
        assert (cfg.num_layers,cfg.hidden_size,cfg.ffn_hidden_size,cfg.num_attention_heads,cfg.vocab_size)==(29,2048,5504,16,32128)
        models=kw.get('model',a[3] if len(a)>3 else None)
        assert isinstance(models,list) and len(models)==2
        chunks=[]
        for chunk,model in enumerate(models):
            begin,end=dspp_layout.physical_layer_range(29,4,'endpoint-3-2',stage,chunk)
            actual=sorted(int(m.layer_number) for m in model.modules() if type(m).__name__=='ParallelTransformerLayer')
            assert actual==list(range(begin+1,end+1)),(stage,chunk,actual,begin,end)
            chunks.append(dict(chunk=chunk,global_layers=actual,parameters=sum(p.numel() for p in model.parameters())))
        target=Path(os.environ['DSPP_CP_BENCH_OUTPUT'])/f'model29-ownership-rank{os.environ["RANK"]}.json'
        target.write_text(json.dumps(dict(stage=stage,cp_rank=parallel_state.get_context_parallel_rank(),chunks=chunks,
            model=dict(layers=29,H=2048,FFN=5504,Qheads=16,vocab=32128,PP=4,CP=2,VPP=2),
            layout_sha256=hashlib.sha256(Path(__file__).with_name('dspp_layout29.py').read_bytes()).hexdigest()))+'\n')
        ownership_recorded=True
    return native_step(*a,**kw)
training.train_step=checked_step
runpy.run_path(str(Path(__file__).with_name('benchmark_entry_persistent.py')),run_name='__main__')
