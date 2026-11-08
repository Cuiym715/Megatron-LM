"""Paired CP2 transport-only windows, including packing and allocation costs."""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from megatron.core.datasets.dspp_batch_plan import build_dspp_batch_plan
from megatron.core.kv_cache.dspp_packed_attention import DsppSequenceKVState
from megatron.core.kv_cache.dspp_cp_attention import _prepared,_gather,_scatter


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    cfg=p.parse_args();assert not cfg.output.exists()
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']));dist.init_process_group('nccl')
    assert dist.get_world_size()==2
    rank=dist.get_rank();rows=[]
    for label,lengths,index in [('shared',(8193,),0),('mixed',(4608,768,768),2),('private',(800,224,600,424),0)]:
        meta=build_dspp_batch_plan(lengths,2048).microbatches[index]
        state=DsppSequenceKVState(cp_layout='hybrid')
        transport=_prepared(state,meta,2,rank,torch.device('cuda',rank))[3]
        k=torch.randn(1024,8,128,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
        dk=torch.randn(meta.valid_token_count,8,128,device='cuda',dtype=torch.bfloat16)
        dv=torch.randn_like(dk)
        def run(fused):
            if fused:
                a,b=_gather(torch.stack((k,v),1),dist.group.WORLD,2,transport).unbind(1)
                c,d=_scatter(torch.stack((dk,dv),1),2048,dist.group.WORLD,2,transport).unbind(1)
            else:
                a=_gather(k,dist.group.WORLD,2,transport);b=_gather(v,dist.group.WORLD,2,transport)
                c=_scatter(dk,2048,dist.group.WORLD,2,transport);d=_scatter(dv,2048,dist.group.WORLD,2,transport)
            return a,b,c,d
        before,after=run(False),run(True)
        for a,b in zip(before,after):torch.testing.assert_close(a,b,atol=0,rtol=0)
        del before,after
        for repeat in range(5):
            for fused in (False,True):
                for _ in range(10):result=run(fused)
                dist.barrier();torch.cuda.synchronize()
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                started=time.perf_counter();begin.record()
                for _ in range(100):result=run(fused)
                end.record();end.synchronize()
                value=dict(wall_ms=(time.perf_counter()-started)*1000/100,
                           cuda_ms=begin.elapsed_time(end)/100)
                both=[None,None];dist.all_gather_object(both,value)
                rows.append(dict(case=label,fused=fused,repeat=repeat,
                    wall_ms=max(r['wall_ms'] for r in both),cuda_ms=max(r['cuda_ms'] for r in both)))
    if rank==0:
        summary=[]
        for label in ('shared','mixed','private'):
            pair=[statistics.median(r['wall_ms'] for r in rows if r['case']==label and r['fused']==f)
                  for f in (False,True)]
            summary.append(dict(case=label,before_ms=pair[0],after_ms=pair[1],speedup=pair[0]/pair[1]-1))
        cfg.output.write_text(json.dumps(dict(rows=rows,summary=summary),indent=2))
        print(json.dumps(summary))
    dist.destroy_process_group()


if __name__=='__main__':main()
