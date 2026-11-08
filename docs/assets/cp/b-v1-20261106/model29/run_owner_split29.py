"""Bounded29L profile batch; diagnostic costs/owners, never online throughput."""
import argparse,fcntl,hashlib,json,os,re,subprocess,time
from pathlib import Path
ROOT=Path('/workspace/experiments/dspp-cp-v1-20261106');SRC=ROOT/'source-model29'
p=argparse.ArgumentParser();p.add_argument('--head',choices=['mha','gqa1'],required=True);p.add_argument('--kind',choices=['compute','owners'],required=True);p.add_argument('--backend',choices=['KV','QO','both'],default='both');p.add_argument('--batch',required=True);p.add_argument('--compute-root',type=Path);cfg=p.parse_args();assert re.fullmatch('[a-z0-9-]+',cfg.batch)
OUT=Path('/workspace/outputs/dspp-cp-v1-20261106')/cfg.batch
lock=(ROOT/'numeric.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
review=json.loads((ROOT/'b29-baseline-timing-review.json').read_text());assert review['status']=='timing-reviewed-numerical-errors-unresolved' and len(review['windows'])==16
baseline=json.loads(Path('/workspace/outputs/dspp-cp-v1-20261106/baseline29-v1/manifest.json').read_text());assert baseline['status']=='complete' and baseline['production_patch_unchanged']
source=json.loads((ROOT/'model29-source-proof.json').read_text());assert source['status']=='prepared-isolated-realfile-layout29' and str(SRC)==source['source']
class PerformanceWindowReady(Exception):pass
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert digest(SRC/source['only_changed_file'])==source['new_sha256']
for name,h in baseline['source']['required_sha256'].items():assert digest(SRC/name)==h
protocol=json.loads((ROOT/'owner-split29-protocol.json').read_text());cases=[c for c in protocol['cases'] if c['head']==cfg.head and c['mode'].startswith(cfg.kind) and (cfg.backend=='both' or c['backend']==cfg.backend)]
assert cfg.kind=='owners' and cfg.head=='gqa1' and cfg.backend=='QO' and len(cases)==1
assert cfg.compute_root is None,'fused owner budget comes from alreadymeasured identical baseline'
OUT.mkdir(exist_ok=False);(OUT/'scripts').mkdir()
files=['run_owner_split29.py','benchmark_entry29_owner_split.py','benchmark_entry29_profile.py','benchmark_entry29.py','dspp_layout29.py','benchmark_entry_persistent.py','benchmark_entry_warmup.py','zero_dropout.py','owner-split29-protocol.json']
for name in files:(OUT/'scripts'/name).write_bytes((ROOT/name).read_bytes())
prod=Path('/workspace/projects/Megatron-LM-kwai');patch=subprocess.check_output(['git','diff','--binary'],cwd=prod);(OUT/'production-before.patch').write_bytes(patch)
uuids=baseline['gpu_uuids'];state=dict(status='preparing',pid=os.getpid(),started=time.time(),head=cfg.head,kind=cfg.kind,jobs=[],pending=[],gpu_uuids=uuids,source=source,base_source=baseline['source'],arithmetic=baseline['arithmetic'],data_sha256=baseline['data_sha256'],resource_mode='exclusive-profile' if cfg.kind=='compute' else 'shared-functional',timing_eligible=False,numerically_certified=False,admission='perrank measuredfusedbaseline reservedmax+8GiB provisional split admission; ownunique storage/CPcounts, notcapacitylimit orbandwidth',protocol=protocol,scripts_sha256={n:digest(ROOT/n) for n in files},production_patch_sha256=hashlib.sha256(patch).hexdigest())
def save():
 p=OUT/'manifest.tmp';p.write_text(json.dumps(state,indent=2)+'\n');p.replace(OUT/'manifest.json')
def check(label,own=False,required=None):
 ap=OUT/'host-ownership/manifest.json';assert ap.exists(),'hostauditnotattached';a=json.loads(ap.read_text());assert a['status']=='running','hostauditclosed'
 if cfg.kind=='compute':assert not a['unknown_observed'],'foreignGPUload'
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu,memory.free','--format=csv,noheader'],text=True)
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader'],text=True)
 with (OUT/'resource-checks.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),label=label,gpus=gpu,apps=apps,host_load=os.getloadavg()))+'\n')
 rows=gpu.splitlines();assert len(rows)==8 and all(u in gpu for u in uuids)
 if cfg.kind=='compute':
  if own:assert all(sum(u in line for line in apps.splitlines())<=1 for u in uuids),'extraGPUprocess'
  else:assert not apps.strip() and all(int(row.split(',')[2].split()[0])<=512 and int(row.split(',')[3].split()[0])==0 for row in rows),'notidle'
 else:
  free={row.split(',')[1].strip():int(row.split(',')[-1].split()[0]) for row in rows}
  assert all(free[u]>=(512 if own else required[u]) for u in uuids),'insufficientsharedmemory'
  if not own and not apps.strip() and all(int(row.split(',')[2].split()[0])<=512 and int(row.split(',')[3].split()[0])==0 for row in rows):raise PerformanceWindowReady('all8idle; defer remaining shared work for formal performance')
save();base=list(baseline['jobs'][0]['command']);base[base.index(str(ROOT/'benchmark_entry29.py'))]=str(ROOT/'benchmark_entry29_owner_split.py');base[base.index('--train-iters')+1]='6'
env={k:v for k,v in os.environ.items() if not k.startswith(('DSPP_','CP_V1_'))};env.update(CUDA_VISIBLE_DEVICES=','.join(uuids),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_DEVICE_MAX_CONNECTIONS='32',CP_V1_SOURCE=str(SRC),CP_V1_FULL_REDUCTION='1',CP_V1_DETERMINISTIC_FA='1',DSPP_CP_BENCH_WARMUP='2',DSPP_CP_BENCH_TIMED='4',DSPP_FA_SPLIT_CACHE='/workspace/cache/dspp-fa-split',NCCL_LAUNCH_ORDER_IMPLICIT='0',NCCL_LAUNCH_RACE_FATAL='1')
try:
 for case in cases:
  name=case['name'];path=OUT/name;required=None
  if cfg.kind=='owners':
   candidates=[j for j in baseline['jobs'] if j['name'].startswith(cfg.head+'-') and j['name'].endswith('-'+case['backend'])]
   assert len(candidates)==4
   required={uuids[r]:max(json.loads((Path(j['output'])/f'rank{r}.json').read_text())['reserved_peak'] for j in candidates)//1048576+8192 for r in range(8)}
  deadline=time.monotonic()+10
  while not (OUT/'host-ownership/manifest.json').exists() and time.monotonic()<deadline:time.sleep(.2)
  for n,h in state['scripts_sha256'].items():assert digest(ROOT/n)==h
  if cfg.kind=='compute':
   for i in range(3):
    check(name+'-idle-'+str(i))
    if i<2:time.sleep(30)
  else:
   try:check(name+'-shared-memory',required=required)
   except AssertionError as e:
    state['pending'].append(dict(name=name,reason=repr(e),required_free_MiB=required));save();continue
  path.mkdir();cmd=list(base);cmd[cmd.index('--kaimm-context-parallel-impl')+1]='key-value' if case['backend']=='KV' else 'query-out'
  if cfg.head=='gqa1':cmd+=['--group-query-attention','--num-query-groups','1']
  cmd+=case['extra_argv']
  if cfg.kind=='compute':cmd+=['--dspp-timeline-dir',str(path/'timeline')]
  jobenv=dict(env,DSPP_CP_BENCH_OUTPUT=str(path))
  if cfg.kind=='owners':jobenv.update(DSPP_CP_BENCH_AUDIT=str(path/'owners'),DSPP_CP_BENCH_IO='1')
  job=dict(name=name,command=cmd,output=str(path),started=time.time(),polluted=False,resource_mode=state['resource_mode'],timing_eligible=False)
  state['jobs'].append(job);state['status']='running';save();(path/'command.json').write_text(json.dumps(dict(argv=cmd,env={k:v for k,v in jobenv.items() if k.startswith(('DSPP_','CP_V1_','CUDA_','OMP_','MKL_','NCCL_'))}),indent=2)+'\n')
  with (path/'train.log').open('w') as f:
   child=subprocess.Popen(['timeout','--kill-after=15','600',*cmd],cwd=SRC,env=jobenv,stdout=f,stderr=subprocess.STDOUT);job['pid']=child.pid;save()
   while child.poll() is None:
    try:check(name+'-monitor',own=True,required=required)
    except Exception as e:
     job.update(polluted=True,pollution_reason=repr(e));save()
     if child.poll() is None:child.terminate()
    time.sleep(5)
  job.update(exit_code=child.returncode,ended=time.time());save();assert child.returncode==0 and not job['polluted'],'failedprofilecase'
  ranks=[json.loads((path/f'rank{r}.json').read_text()) for r in range(8)];wrap=[json.loads((path/f'wrapper-rank{r}.json').read_text()) for r in range(8)]
  assert all(len(r['records'])==6 and r['warmup']==2 and r['timed']==4 and r['diagnostic'] for r in ranks)
  assert all(w['all_parameters_finite'] and w['all_losses_finite'] and w['deterministic_fa_backward'] and not w['bf16_reduced_precision_reduction'] for w in wrap)
  assert [r['metrics']['valid_tokens'] for r in ranks[0]['records']]==[88064,37286]*3
  owners=[json.loads((path/f'model29-ownership-rank{r}.json').read_text()) for r in range(8)]
  assert sorted(i for o in owners if o['cp_rank']==0 for c in o['chunks'] for i in c['global_layers'])==list(range(1,30))
  if cfg.kind=='compute':
   traces=[json.loads(f.read_text()) for f in (path/'timeline').glob('iteration_*_rank_*.json')];assert len(traces)==32 and all(t['task_timing_only'] and t['task_timing_scope']=='all' for t in traces)
   assert not list(path.rglob('*_stage_*.json')),'memoryauditintimingprofile'
   if case['mode'].endswith('split'):
    for rank in range(8):assert any(e['kind'].startswith('H') and e['duration_ms']>0 and any(i['token_offset']>0 for i in e['items']) for t in traces if t['rank']==rank for e in t['events']),'missingnonemptyH'
   job['timeline_files']=len(traces)
  else:
   assert not (path/'timeline').exists(),'timingandownerauditmustseparate'
   assert all((path/'owners'/f'cp{cp}'/f'iteration_{it:06d}_stage_{stage}.json').exists() for cp in range(2) for stage in range(4) for it in [3,4]),'missingCPpair storage audits'
  histories=[json.loads((path/f'history-executed-rank{rank}.json').read_text()) for rank in range(8)]
  assert all(h['completed_jobs']>0 and h['nonempty_parts']>0 and h['history_tokens']>0 for h in histories),'missingactualcompletednonemptyH'
  job.update(profile_validated=True,scope='diagnosticcost/owneronly, notonline throughput or numericalequivalence');save()
 state.update(status='complete',ended=time.time());save()
except PerformanceWindowReady as e:state.update(status='stopped',deferred_for_performance=True,reason=str(e),ended=time.time());save()
except Exception as e:state.update(status='stopped',error=repr(e),ended=time.time());save();raise
finally:
 state['production_patch_unchanged']=hashlib.sha256(subprocess.check_output(['git','diff','--binary'],cwd=prod)).hexdigest()==state['production_patch_sha256'];save()
