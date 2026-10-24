import hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
repo=Path('/workspace/src/Megatron-LM-kwai')
root=Path('/workspace/tmp/dspp-a-performance-20261023')
python='/workspace/src/venvs/megatron/bin/python'
old=json.loads(Path('/workspace/tmp/dspp-fixed-w-20261023/commands.json').read_text())
base=list(next(r for r in old if r['label']=='memory/legacy')['command'])
base[base.index('tools/validate_dspp_plan_training.py')]='pretrain_llama.py'
for flag,value in {'--num-layers-per-virtual-pipeline-stage':'4','--num-layers':'24','--hidden-size':'2048','--ffn-hidden-size':'5504','--num-attention-heads':'16','--seq-length':'16384','--max-position-embeddings':'16384','--micro-seq-length':'2048','--global-batch-size':'16','--train-iters':'16','--dataloader-type':'single','--variable-seq-debug-num-batches':'0','--log-interval':'4'}.items():
 base[base.index(flag)+1]=value
base+=['--timing-log-level','0']
commands=[]
def execute(label,cmd,env=None,timeout=480):
 out=root/label;out.mkdir(parents=True,exist_ok=False)
 env=env or {}
 commands.append(dict(label=label,command=cmd,environment=env,started=time.time()))
 (root/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
 print('START',label,flush=True)
 started=time.time()
 with (out/'train.log').open('w') as log:
  proc=subprocess.Popen(cmd,cwd=repo,env={**os.environ,**env},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try: code=proc.wait(timeout=timeout)
  except subprocess.TimeoutExpired:
   os.killpg(proc.pid,signal.SIGUSR1) # workers registered only in benchmark entry; diagnostic best effort
   time.sleep(1)
   os.killpg(proc.pid,signal.SIGTERM)
   try: code=proc.wait(timeout=10)
   except subprocess.TimeoutExpired:
    os.killpg(proc.pid,signal.SIGKILL);code=proc.wait()
   code=124
 (out/'exit.json').write_text(json.dumps(dict(code=code,seconds=time.time()-started))+'\n')
 print('END',label,code,round(time.time()-started,2),flush=True)
 if code:
  print((out/'train.log').read_text()[-7000:],flush=True)
  raise SystemExit(code)
 return out

def command(seed,method,entry='pretrain_llama.py'):
 cmd=list(base)
 cmd[cmd.index('pretrain_llama.py')]=entry
 cmd[cmd.index('--data-path')+1]=str(root/f'data/seed{seed}/data')
 cmd[cmd.index('--dspp-schedule')+1]=method
 cmd[cmd.index('--dspp-execution-profile')+1]=str(root/'profile.json')
 return cmd

mode=sys.argv[1]
if mode=='prepare':
 for seed in (11,19,29):
  path=root/f'data/seed{seed}';path.mkdir(parents=True)
  subprocess.run([python,'/workspace/src/InfiniPipe-code/scripts/generate_dspp_compare_lengths.py','--output',str(path/'lengths.txt'),'--num-sequences','1024','--seed',str(seed),'--max-seq-len','16384','--over-max-policy','drop'],check=True)
  subprocess.run([python,'tools/build_dspp_synthetic_dataset.py','--output-prefix',str(path/'data'),'--raw-lengths-file',str(path/'lengths.txt'),'--repeats','1','--vocab-size','32000','--token-pattern','infinipipe'],check=True)
if mode=='calibrate':
 for seed in (11,19,29):
  out=root/f'calibration/seed{seed}'
  cmd=command(seed,'legacy')+['--dspp-timeline-dir',str(out/'timeline'),'--dspp-timeline-iterations',*[str(i) for i in range(5,17)],'--dspp-task-timing-only','--dspp-task-timing-scope','all']
  execute(f'calibration/seed{seed}',cmd)
 sys.path.insert(0,str(repo))
 from megatron.core.datasets.dspp_execution_cost import DsppExecutionCostModel
 records=[];inputs=[]
 for p in sorted((root/'calibration').glob('seed*/timeline/*rank_*.json')):
  d=json.loads(p.read_text());assert d['backward_mode']=='fused'
  records+=d['events'];inputs.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 assert len(inputs)==3*12*3,len(inputs)
 DsppExecutionCostModel.from_records(records).save(root/'profile.json')
 (root/'profile-provenance.json').write_text(json.dumps(dict(inputs=inputs,events=len(records),frozen_before_comparison=True,same_workload_calibration=True,profile_sha256=hashlib.sha256((root/'profile.json').read_bytes()).hexdigest()),indent=2)+'\n')
 print('PROFILE FROZEN',len(records),flush=True)
if mode in ('benchmark','resume-benchmark'):
 commands[:]=json.loads((root/'commands.json').read_text())
 orders=[('legacy','template','a'),('template','a','legacy'),('a','legacy','template')]
 for repeat in (1,2):
  for index,seed in enumerate((11,19,29)):
   order=orders[(index+repeat-1)%3]
   for method in order:
    label=f'bench/repeat{repeat}/seed{seed}/{method}'
    if mode=='resume-benchmark' and (root/label/'exit.json').exists():
     assert json.loads((root/label/'exit.json').read_text())['code']==0
     continue
    execute(label,command(seed,method,str(root/'bench_entry.py')),dict(DSPP_BENCH_OUTPUT=str(root/label),DSPP_BENCH_WARMUP='4'))
if mode=='diagnostic-after-benchmark':
 deadline=time.time()+1500
 expected=[root/f'bench/repeat{repeat}/seed{seed}/{method}/exit.json' for repeat in (1,2) for seed in (11,19,29) for method in ('legacy','template','a')]
 while True:
  for p in expected:
   if p.exists(): assert json.loads(p.read_text())['code']==0,p
  if all(p.exists() for p in expected): break
  assert time.time()<deadline,'benchmark completion wait timed out'
  time.sleep(2)
 mode='diagnostic'
if mode=='diagnostic':
 commands[:]=json.loads((root/'commands.json').read_text())
 for method in ('legacy','template','a'):
  label=f'diagnostic/seed11/{method}'
  execute(label,command(11,method,str(root/'hash_entry.py'))+['--exit-interval','6'],dict(DSPP_HASH_OUTPUT=str(root/label)),timeout=480)
