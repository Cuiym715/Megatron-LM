"""CPU-only first-update replay from existing v5 gradients and FP32 masters."""
import fcntl,hashlib,json,os,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
torch.set_num_threads(1)
ROOT=Path('/workspace/outputs/dspp-cp-v1-20261106/numeric-v5-longmix512')
OUT=ROOT.parent/'numeric-v5-adam-replay-v1'
lock=open('/workspace/experiments/dspp-cp-v1-20261106/numeric.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
formal=json.loads((ROOT.parent/'baseline-persistent-v1/manifest.json').read_text());assert formal['status']=='stopped'
OUT.mkdir(exist_ok=False)
state=dict(status='running',pid=os.getpid(),started=time.time(),cuda_initialized=torch.cuda.is_initialized(),command='CPU Adam replay, one thread; outer timeout300',source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),input=str(ROOT))
def save():
 p=OUT/'manifest.tmp';p.write_text(json.dumps(state,indent=2)+'\n');p.replace(OUT/'manifest.json')
def load(p):return torch.load(p,map_location='cpu',weights_only=True,mmap=True)
def key(t):return (tuple(t.shape),tuple(t.flatten()[:16].tolist()))
def predict(p,g,group,norm):
 g=g.float()*(1/(norm+1e-6))
 b1,b2=group['betas'];m=g*(1-b1);v=g.square()*(1-b2)
 step=(m/(1-b1))/(torch.sqrt(v/(1-b2))+group['eps'])+p*group['weight_decay']
 return (p-group['lr']*step)-p
save();stats=dict(mapped=0,total_master_elements=0,baseline_update_ss=0.,candidate_update_ss=0.,actual_difference_ss=0.,predicted_difference_ss=0.,difference_residual_ss=0.,baseline_replay_residual_ss=0.,candidate_replay_residual_ss=0.);rows=[];skipped=[]
try:
 for rank in [0,2,4,6]:
  base=load(ROOT/f'mha-kv-reference/complete-tensors/iteration1-rank{rank}.pt');cand=load(ROOT/f'mha-qo-fused/complete-tensors/iteration1-rank{rank}.pt')
  ref=load(ROOT/f'mha/snapshots/full-master-iteration1-rank{rank}.pt');actual=load(ROOT/f'mha-qo-fused/full-master-iteration1-rank{rank}.pt');optim=load(ROOT/f'mha/snapshots/optimizer-iteration1-rank{rank}.pt')['optimizer'];assert not optim['state']
  lookup={}
  for name,t in base['params_before'].items():lookup.setdefault(key(t),[]).append(name)
  for ident,before in ref['before'].items():
   stats['total_master_elements']+=before.numel();names=lookup.get(key(before),[])
   if len(names)!=1:skipped.append(dict(rank=rank,master=ident,elements=before.numel(),reason='ambiguousinitialweight'));continue
   name=names[0];assert torch.equal(before,base['params_before'][name]) and torch.equal(before,actual['before'][ident])
   assert torch.equal(base['params_before'][name],cand['params_before'][name]);group=optim['param_groups'][int(ident.split(':')[0])]
   assert group['bias_correction'] and tuple(group['betas'])==(.9,.999) and group['lr']==.001 and group['eps']==1e-8
   p=before.flatten();gb=base['gradients'][name].flatten();gc=cand['gradients'][name].flatten();ub=ref['update'][ident].flatten();uc=actual['update'][ident].flatten();acc={k:0. for k in stats if k not in ['mapped','total_master_elements']}
   for i in range(0,p.numel(),524288):
    sl=slice(i,i+524288);pb=predict(p[sl],gb[sl],group,20.086);pc=predict(p[sl],gc[sl],group,20.085)
    delta=(uc[sl]-ub[sl]).double();pred=(pc-pb).double()
    vals=dict(baseline_update_ss=ub[sl].double().square().sum(),candidate_update_ss=uc[sl].double().square().sum(),actual_difference_ss=delta.square().sum(),predicted_difference_ss=pred.square().sum(),difference_residual_ss=(pred-delta).square().sum(),baseline_replay_residual_ss=(pb.double()-ub[sl].double()).square().sum(),candidate_replay_residual_ss=(pc.double()-uc[sl].double()).square().sum())
    for k,v in vals.items():acc[k]+=float(v)
   for k,v in acc.items():stats[k]+=v
   stats['mapped']+=p.numel();rows.append(dict(rank=rank,name=name,master=ident,elements=p.numel(),**acc))
  state.update(last_completed_rank=rank,mapped_elements=stats['mapped']);save();del base,cand,ref,actual,optim
 result=dict(status='CPU-diagnostic-only-numerical-errors-unresolved',cuda_initialized=torch.cuda.is_initialized(),scope='Unique whole-tensor initialweight mappings; ambiguous identicalweights excluded, not numericalacceptance',mapped_fraction=stats['mapped']/stats['total_master_elements'],mapped_actual_update_relative_l2=(stats['actual_difference_ss']/stats['baseline_update_ss'])**.5,mapped_predicted_update_relative_l2=(stats['predicted_difference_ss']/stats['baseline_update_ss'])**.5,baseline_replay_residual_relative_l2=(stats['baseline_replay_residual_ss']/stats['baseline_update_ss'])**.5,candidate_replay_residual_relative_l2=(stats['candidate_replay_residual_ss']/stats['candidate_update_ss'])**.5,predicted_difference_residual_relative_to_actual_difference=(stats['difference_residual_ss']/stats['actual_difference_ss'])**.5,initial_optimizer_state_empty=True,norms=dict(KV=20.086,QO=20.085,source='originaltraininglogs roundedto3decimals',uncertainty=.0005),formula_source='https://github.com/NVIDIA/apex/blob/master/csrc/multi_tensor_adam.cu',limitations='CPUFP32 formula approximation, printednorms rounded, shared/identicalweight mappings excluded. Doesnotexplainoriginofgradient/forwarddifferences; no thresholds changed or correctnessclaim.',statistics=stats,skipped=skipped,parameters=rows)
 assert not result['cuda_initialized'];(OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n');state.update(status='complete',ended=time.time());save();print(json.dumps({k:v for k,v in result.items() if not isinstance(v,(list,dict))},indent=2))
except Exception as e:state.update(status='failed',error=repr(e),ended=time.time());save();raise
