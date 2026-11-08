"""One shared-GPU control of an existing numerical failure, not performance."""
import fcntl, hashlib, json, os, subprocess, time
from pathlib import Path

ROOT=Path('/workspace/experiments/dspp-cp-v1-20261106')
SRC=ROOT/'source-current'
OLD=Path('/workspace/outputs/dspp-cp-v1-20261106/numeric-v5-longmix512')
OUT=Path('/workspace/outputs/dspp-cp-v1-20261106/numeric-v5-fp32-o-v1')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

lock=(ROOT/'numeric.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
prior=json.loads((OLD/'manifest.json').read_text())
source=prior['source']['sources']['current']
for name,sha in source['required_sha256'].items():assert digest(SRC/name)==sha
uuids=prior['gpu_uuids'];assert len(uuids)==8
for rank in range(8):
 assert (OLD/'mha/snapshots'/f'iteration1-rank{rank}.pt').is_file()
 assert (OLD/'mha/snapshots'/f'full-master-iteration1-rank{rank}.pt').is_file()
OUT.mkdir(exist_ok=False);(OUT/'scripts').mkdir()
files=['run_numeric_fp32_o.py','numeric_fp32_o_entry.py','numeric_entry_v4.py','zero_dropout.py']
for name in files:(OUT/'scripts'/name).write_bytes((ROOT/name).read_bytes())
prod=Path('/workspace/projects/Megatron-LM-kwai')
patch=subprocess.check_output(['git','diff','--binary'],cwd=prod)
(OUT/'production-before.patch').write_bytes(patch)
state=dict(status='preparing',pid=os.getpid(),started=time.time(),jobs=[],gpu_uuids=uuids,
 source=source,input_identity=prior['data_sha256'],resource_mode='shared-functional-memory-only',
 timing_eligible=False,numerically_certified=False,reference=str(OLD/'mha/snapshots'),
 contract='Original failed16L/H1024/Q8/vocab512 PP4CP2 first88064tokens, one aligned update; only weightedO CPsum FP32 castafter; fullGEMM/detFA/exactp0 unchanged. Common runtime NCCLmutex/PPstream reuse. No numerical pass prerequisite for performance.',
 scripts_sha256={n:digest(ROOT/n) for n in files},production_patch_sha256=hashlib.sha256(patch).hexdigest())
def save():
 p=OUT/'manifest.tmp';p.write_text(json.dumps(state,indent=2)+'\n');p.replace(OUT/'manifest.json')
def resources(label,own=False):
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader'],text=True)
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader'],text=True)
 free={line.split(',')[0].strip():int(line.split(',')[1].split()[0]) for line in gpu.splitlines()}
 with (OUT/'resource-checks.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),label=label,gpus=gpu,apps=apps))+'\n')
 assert all(free[u]>=(512 if own else 20480) for u in uuids),'sharedGPU remaining memory insufficient'
 # Foreign utilization/PIDs are expected and never constitute an abort here.
save()
try:
 resources('shared-admission')
 base=next(j for j in prior['jobs'] if j['name']=='mha-qo-fused')
 cmd=list(base['command']);cmd[cmd.index(str(ROOT/'numeric_entry_v4.py'))]=str(ROOT/'numeric_fp32_o_entry.py')
 cmd+=['--distributed-timeout-minutes','2']
 env={k:v for k,v in os.environ.items() if not k.startswith(('DSPP_','CP_V1_'))}
 oldenv=json.loads((OLD/'mha-qo-fused/command.json').read_text())['env'];env.update(oldenv)
 env.update(CUDA_VISIBLE_DEVICES=','.join(uuids),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',CP_V1_SOURCE=str(SRC),
  NCCL_LAUNCH_ORDER_IMPLICIT='0',NCCL_LAUNCH_RACE_FATAL='1',DSPP_CP_BENCH_OUTPUT=str(OUT/'qo-fp32-o'),
  DSPP_PLAN_CHECK_OUTPUT=str(OUT/'qo-fp32-o'),DSPP_PLAN_CHECK_SNAPSHOTS=str(OLD/'mha/snapshots'),
  CP_V1_MASTER_REFERENCE=str(OLD/'mha/snapshots'),DSPP_PLAN_CHECK_EXPORT_REFERENCE=str(OUT/'qo-fp32-o/complete-tensors'),
  DSPP_PLAN_CHECK_MODE='compare',DSPP_PLAN_CHECK_REPORT_FAILURES='1',DSPP_PLAN_CHECK_ALIGN_STATE='1',
  DSPP_PLAN_CHECK_DETERMINISTIC_FA='1',DSPP_PLAN_CHECK_CP_NUMERICS='1')
 path=OUT/'qo-fp32-o';path.mkdir()
 job=dict(name='qo-fp32-o',command=cmd,output=str(path),started=time.time(),timing_eligible=False)
 state['jobs'].append(job);save()
 (path/'command.json').write_text(json.dumps(dict(argv=cmd,env={k:v for k,v in env.items() if k.startswith(('DSPP_','CP_V1_','NCCL_','CUDA_','OMP_','MKL_'))}),indent=2)+'\n')
 # Attach the read-only host auditor before GPU start. Never attach a foreign-load stop guard.
 deadline=time.monotonic()+10
 while not (OUT/'host-ownership/manifest.json').exists() and time.monotonic()<deadline:time.sleep(.2)
 assert (OUT/'host-ownership/manifest.json').exists(),'read-only auditor not attached'
 with (path/'train.log').open('w') as log:
  child=subprocess.Popen(['timeout','--kill-after=15','300',*cmd],cwd=SRC,env=env,stdout=log,stderr=subprocess.STDOUT)
  job['pid']=child.pid;state['status']='running';save()
  while child.poll() is None:
   try:resources('shared-monitor',own=True)
   except Exception as e:
    job['memory_abort']=repr(e);save()
    if child.poll() is None:child.terminate()
   time.sleep(5)
 job.update(exit_code=child.returncode,ended=time.time());save()
 records=[json.loads((path/f'iteration1-rank{r}.json').read_text()) for r in range(8)]
 masters=[json.loads((path/f'full-master-iteration1-rank{r}.json').read_text()) for r in range(8)]
 assert records[0]['metrics']['valid_tokens']==88064 and all(r['diagnostic_deterministic_fa'] for r in records)
 assert all(r['before']['passed'] and r['before']['exact'] for r in masters)
 assert len(list(path.glob('fp32-o-patch-rank*.json')))==8
 failures=[dict(rank=r['rank'],category=k) for r in records+masters for k,v in r.items() if isinstance(v,dict) and v.get('passed') is False]
 state.update(status='complete-diagnostic',ended=time.time(),reports=8,master_reports=8,numerical_failures=failures,
  numerical_criteria_unchanged=True,scope='Shared one-variable numerical control; failed comparisons retained, not throughput or automatic production adoption')
 save()
except Exception as e:
 state.update(status='failed',error=repr(e),ended=time.time());save();raise
finally:
 state['production_patch_unchanged']=hashlib.sha256(subprocess.check_output(['git','diff','--binary'],cwd=prod)).hexdigest()==state['production_patch_sha256'];save()
