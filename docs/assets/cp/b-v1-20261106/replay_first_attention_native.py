"""Replay existing first-operator inputs on one shared GPU; no model/PP run."""
from pathlib import Path
import fcntl,hashlib,json,os,subprocess,sys,time
ROOT=Path('/workspace/experiments/dspp-cp-v1-20261106')
SRC=ROOT/'source-current';BASE=Path('/workspace/outputs/dspp-cp-v1-20261106')
OUT=BASE/'numeric-v5-first-attention-native-replay-v1'
lock=(ROOT/'numeric.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
prior=json.loads((BASE/'numeric-v5-first-attention-CPU-v1/result.json').read_text())
assert all(v['exact'] for v in prior['QKV'].values())
assert len(prior['meta']['items'])==1 and prior['meta']['items'][0]['token_offset']==0
source=prior['source']
for name,sha in source['required_sha256'].items():assert hashlib.sha256((SRC/name).read_bytes()).hexdigest()==sha
raw=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader'],text=True)
choices=[(int(s.split(',')[1].split()[0]),s.split(',')[0].strip()) for s in raw.splitlines()]
free,uuid=max(choices);assert free>=4096,'shared replay needs4GiB free, not empty GPU'
os.environ['CUDA_VISIBLE_DEVICES']=uuid;os.environ['OMP_NUM_THREADS']='1'
OUT.mkdir(exist_ok=False)
state=dict(status='preparing',pid=os.getpid(),started=time.time(),source=source,gpu_uuids=[uuid],
 resource_mode='shared-functional-single-GPU',timing_eligible=False,numerically_certified=False,
 command=sys.argv,initial_memory_free_MiB=free,resource_snapshot=raw,
 script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 jobs=[dict(name='native-forward-replay',pid=os.getpid(),command=sys.argv)],
 input='Captured same QKV/meta/owners from original16Lfailedinput first2048 of32768. No fresh model, random inputs or backward.',
 scope='Native FA2 calls with original KV/QO partitions and local emulation of two-rank O/LSE combine; no actual collective or PP, no throughput claim')
def save():
 p=OUT/'manifest.tmp';p.write_text(json.dumps(state,indent=2)+'\n');p.replace(OUT/'manifest.json')
save()
try:
 deadline=time.monotonic()+10
 while not (OUT/'host-ownership/manifest.json').exists() and time.monotonic()<deadline:time.sleep(.2)
 assert (OUT/'host-ownership/manifest.json').exists(),'readonly audit must attach first'
 sys.path.insert(0,str(SRC))
 import torch
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
 from megatron.core.datasets.dspp_batch_plan import DsppMicrobatchMeta,DsppSegment
 from megatron.core.kv_cache.dspp_packed_attention import DsppSequenceKVState
 from megatron.core.kv_cache import dspp_packed_attention as packed
 from megatron.core.kv_cache.dspp_cp_attention import _prepared
 from megatron.core.kv_cache.dspp_cp_qo_attention import prepare_qo_metadata
 torch.cuda.set_device(0);device=torch.device('cuda',0)
 state.update(status='running',torch_version=torch.__version__,cuda_version=torch.version.cuda,
              compute_capability=list(torch.cuda.get_device_capability()),native_source='installed pinned FA2 wrapped forward, no compilation/source modifications');save()
 meta_dict=prior['meta'];meta=DsppMicrobatchMeta(meta_dict['microbatch_id'],meta_dict['chunk_size'],tuple(DsppSegment(**x) for x in meta_dict['items']))
 n=meta.valid_token_count;assert n==meta.chunk_size==2048
 indices=prior['indices'];captured={};inputs={}
 for backend in ['kv','qo']:
  tensors={}
  for rank in [0,1]:
   p=BASE/f'numeric-v5-first-attention-{backend}-v1/first-attention/probe-iteration1-rank{rank}.pt'
   snapshot=torch.load(p,map_location='cpu',weights_only=True,mmap=True)['attention']
   for name in ['q','k','v','out']:
    v=snapshot['l0:p0:c0:layer1:'+name][:len(indices[rank]),0]
    if name not in tensors:tensors[name]=v.new_zeros((n,*v.shape[1:]))
    tensors[name][torch.tensor(indices[rank],dtype=torch.long)]=v
  captured[backend]=tensors['out']
  inputs[backend]={k:v for k,v in tensors.items() if k!='out'}
 assert all(torch.equal(inputs['kv'][k],inputs['qo'][k]) for k in ['q','k','v'])
 q,k,v=(inputs['kv'][name].to(device=device,dtype=torch.bfloat16) for name in ['q','k','v'])
 assert all(torch.equal(x.float().cpu(),inputs['kv'][name]) for name,x in [('q',q),('k',k),('v',v)])
 scale=q.shape[-1]**-.5
 def fa(a,b,c,cq,ck,causal):
  out,lse,_,_=packed._wrapped_flash_attn_varlen_forward(a.contiguous(),b.contiguous(),c.contiguous(),cq,ck,a.size(0),b.size(0),0.,scale,causal=causal,window_size_left=-1,window_size_right=-1,softcap=0.,alibi_slopes=None,return_softmax=False,block_table=None,leftpad_k=None,seqused_k=None,zero_tensors=False)
  return out,lse
 kvout=torch.zeros_like(q);kv_calls=[]
 kvstate=DsppSequenceKVState(cp_layout='hybrid',cp_impl='key-value')
 for rank in [0,1]:
  runs,_,cu,_=_prepared(kvstate,meta,2,rank,device)
  idx=torch.tensor(indices[rank],device=device);localq=q.index_select(0,idx);localout=torch.zeros_like(localq)
  for run,(cq,ck) in zip(runs,cu):
   a=localq[run.local_begin:run.local_end];b=k[run.key_begin:run.key_end];c=v[run.key_begin:run.key_end]
   oi,_=fa(a,b,c,cq,ck,True);localout[run.local_begin:run.local_end]=oi
   kv_calls.append(dict(rank=rank,Q=a.size(0),K=b.size(0),causal=True))
  kvout.index_copy_(0,idx,localout)
 qostate=DsppSequenceKVState(cp_layout='hybrid',cp_impl='query-out');outs=[];lses=[];qo_calls=[]
 for rank in [0,1]:
  prepare_qo_metadata(qostate,[meta],2,rank,device)
  items,parts,cu,_,_,transport=qostate._cp_prepared['qo',meta.microbatch_id,rank,device]
  idx=torch.tensor(indices[rank],device=device);local_k=k.index_select(0,idx);local_v=v.index_select(0,idx)
  out=q.new_zeros(q.shape,dtype=torch.float32);lse=q.new_full(q.shape[:2],-torch.inf,dtype=torch.float32)
  assert len(items)==1 and items[0][2]==0
  for j,(i,qa,qb,ka,kb,causal) in enumerate(parts):
   assert i==0
   oi,li=fa(q[qa:qb],local_k[ka:kb],local_v[ka:kb],cu[j,:2],cu[j,2:],causal)
   out[qa:qb]=oi;lse[qa:qb]=li.T
   qo_calls.append(dict(rank=rank,qa=qa,qb=qb,ka=ka,kb=kb,causal=causal))
  outs.append(out);lses.append(lse)
 shared=transport['shared_global'];assert shared.numel()==n
 gathered=torch.stack([lse.index_select(0,shared) for lse in lses])
 merged=torch.logsumexp(gathered,dim=0);global_lse=lses[0].clone();global_lse.index_copy_(0,shared,merged)
 weighted=[out*torch.where(torch.isfinite(lse),torch.exp(lse-global_lse),0.)[...,None] for out,lse in zip(outs,lses)]
 # Two BF16 inputs summed once: replay the two-rank SUM without communication.
 qoout=weighted[0].to(torch.bfloat16)+weighted[1].to(torch.bfloat16)
 fp32sum=(weighted[0]+weighted[1]).to(torch.bfloat16)
 cu=torch.tensor([0,n],device=device,dtype=torch.int32);full,_=fa(q,k,v,cu,cu,True)
 def error(a,b):
  a,b=a.detach().float().cpu(),b.detach().float().cpu();d=a.double()-b.double()
  return dict(exact=torch.equal(a,b),different_elements=int((a!=b).sum()),relative_l2=float(d.norm()/b.double().norm().clamp_min(1e-24)),max_abs=float(d.abs().max()))
 result=dict(status='complete',scope=state['scope'],gpu_uuid=uuid,
  KV_replay_vs_captured=error(kvout,captured['kv']),QO_replay_vs_captured=error(qoout,captured['qo']),
  QO_replay_vs_KV=error(qoout,kvout),FP32_weighted_sum_vs_KV=error(fp32sum,kvout),
  full_FA_vs_KV=error(full,kvout),full_FA_vs_QO=error(full,qoout),
  KV_calls=kv_calls,QO_calls=qo_calls,allocated_peak=torch.cuda.max_memory_allocated(),reserved_peak=torch.cuda.max_memory_reserved(),
  limitation='Only captured first attention replay, no backward/history/fullmodel equivalence or root-cause certification. BF16 partial-FA outputs retained; full operator source unchanged.')
 torch.save(dict(KV=kvout.cpu(),QO=qoout.cpu(),FP32_weighted_sum=fp32sum.cpu(),full_FA=full.cpu()),OUT/'replayed-outputs.pt')
 (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n');state.update(status='complete',ended=time.time(),result='result.json');save();print(json.dumps(result))
except Exception as e:
 state.update(status='failed',error=repr(e),ended=time.time());save();raise
