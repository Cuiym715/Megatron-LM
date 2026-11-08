"""Common exact-p0 path for the CP baseline; no diagnostic tensor hooks."""
import json,os,runpy,sys
from pathlib import Path
SRC=Path(os.environ['CP_V1_SOURCE']);sys.path.insert(0,str(SRC))
import torch
if os.environ.get('CP_V1_FULL_REDUCTION')=='1':
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
nccl_version=tuple(torch.cuda.nccl.version())
assert nccl_version >= (2,26,0), 'launch-order option requires NCCL2.26+'
assert os.environ['NCCL_LAUNCH_ORDER_IMPLICIT']=='0' and os.environ['NCCL_LAUNCH_RACE_FATAL']=='1'
from zero_dropout import install
stats,_,_=install()
deterministic_fa=os.environ.get('CP_V1_DETERMINISTIC_FA')=='1'
if deterministic_fa:
    from megatron.core.kv_cache import dspp_packed_attention as packed
    native_fa=packed._wrapped_flash_attn_varlen_backward
    assert native_fa is not None
    def deterministic_backward(*args,**kwargs):
        if len(args)>20:
            args=list(args);assert isinstance(args[20],bool);args[20]=True
        else:kwargs['deterministic']=True
        return native_fa(*args,**kwargs)
    packed._wrapped_flash_attn_varlen_backward=deterministic_backward
from megatron import training,get_args
native_pretrain=training.pretrain
input_records=[]
remembered_models=[]
observed_losses=[]
class InputIterator:
    def __init__(self,iterator):self.iterator=iterator
    def __iter__(self):return self
    def __next__(self):
        data=next(self.iterator)
        lengths=data['lengths'];assert lengths.device.type=='cpu'
        cfg=get_args()
        input_records.append(dict(iteration=cfg.curr_iteration+1,
            consumed_train_samples=cfg.consumed_train_samples,
            raw_lengths=lengths.tolist(),text_shape=list(data['text'].shape)))
        return data
def pretrain(*args,**kwargs):
    args=list(args);batch=kwargs.get('get_batch_func',args[4] if len(args)>4 else None)
    assert batch is not None
    benchmark_step=training.train_step
    def remember(*a,**kw):
        if not remembered_models:
            model=kw.get('model',a[3] if len(a)>3 else None)
            assert model is not None
            remembered_models.extend(model if isinstance(model,list) else [model])
        progress=Path(os.environ['DSPP_CP_BENCH_OUTPUT'])/f'progress-rank{os.environ["RANK"]}.jsonl'
        with progress.open('a') as f:f.write(json.dumps(dict(event='begin',iteration=get_args().curr_iteration+1))+'\n')
        result=benchmark_step(*a,**kw)
        with progress.open('a') as f:f.write(json.dumps(dict(event='end',iteration=get_args().curr_iteration+1,skipped=int(result[1])))+'\n')
        # One common first-update warmup fence, before the timed interval.
        if get_args().curr_iteration==0:
            torch.cuda.synchronize()
            torch.distributed.barrier()
        observed_losses.extend(v.detach() if isinstance(v,torch.Tensor) else torch.as_tensor(v) for v in result[0].values())
        return result
    training.train_step=remember
    def traced(iterator):return batch(InputIterator(iterator) if iterator is not None else None)
    traced.__dict__.update(batch.__dict__)
    if 'get_batch_func' in kwargs:kwargs['get_batch_func']=traced
    else:args[4]=traced
    return native_pretrain(*args,**kwargs)
training.pretrain=pretrain
runpy.run_path(str(SRC/'tools/benchmark_dspp_cp.py'),run_name='__main__')
# Outside the timed interval. Keep this separate from the native raw summary.
import torch
path=Path(os.environ['DSPP_CP_BENCH_OUTPUT'])/f'wrapper-rank{os.environ["RANK"]}.json'
assert not path.exists()
# Full tensor checks happen after the native timer, summary and barrier close.
assert remembered_models
parameters_finite=all(bool(torch.isfinite(p).all().item()) for m in remembered_models for p in m.parameters())
losses_finite=all(bool(torch.isfinite(v).all().item()) for v in observed_losses)
path.write_text(json.dumps(dict(source=str(SRC),zero_dropout=stats,
    torch=str(torch.__version__),cuda=torch.version.cuda,
    deterministic_fa_backward=deterministic_fa,
    bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
    all_parameters_finite=parameters_finite,all_losses_finite=losses_finite,loss_records=len(observed_losses),
    dtype=str(get_args().params_dtype),diagnostic_tensor_hooks=False,
    nccl_version=nccl_version,nccl_launch_order_implicit=os.environ['NCCL_LAUNCH_ORDER_IMPLICIT'],nccl_launch_race_fatal=os.environ['NCCL_LAUNCH_RACE_FATAL'],
    timing_eligible=False,validation_scope='shared functional8updates only; not numericalequivalence',
    warmup_fence='one first-update fence, outside raw5warm3window; alltimings excluded',
    inputs=input_records,input_identity='CPU sampler positions, raw lengths and shapes, bound to frozen dataset SHA/seed/order; no runtime content hash or GPU synchronization'),indent=2)+'\n')
assert parameters_finite and losses_finite,'nonfinite training result'
