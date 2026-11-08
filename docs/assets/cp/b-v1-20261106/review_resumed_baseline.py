"""Local CPU admission across interrupted/resumed windows; never alter raw receipts."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('roots',nargs='+',type=Path);p.add_argument('--head',choices=['mha','gqa1']);p.add_argument('--output',required=True,type=Path);cfg=p.parse_args()
def read(p):return json.loads(p.read_text())
order=['KV','QO','QO','KV','QO','KV','KV','QO'];heads=[cfg.head] if cfg.head else ['mha','gqa1'];expected={f'{h}-{i:02d}-{m}' for h in heads for i,m in enumerate(order,1)}
accepted={};roots=[];common_inputs=common_tokens=common_source=common_entry=common_math=common_patch=None;commands={}
for root in cfg.roots:
 state=read(root/'manifest.json');audit=read(root/'host-ownership/manifest.json');samples=[json.loads(s) for s in (root/'host-ownership/samples.jsonl').read_text().splitlines()];checks=[json.loads(s) for s in (root/'resource-checks.jsonl').read_text().splitlines()]
 if not cfg.head:assert state['status'] in ['complete','stopped'] and audit['status']=='complete' and state['production_patch_unchanged']
 assert samples
 first_foreign=min((s['time'] for s in samples if any(not o['own'] for o in s['owners'])),default=None)
 roots.append(dict(root=str(root),status=state['status'],audit_status=audit['status'],first_foreign_epoch=first_foreign,production_guard=state.get('production_patch_unchanged','pending')))
 for job in state['jobs']:
  name=job['name']
  if name not in expected or not job.get('timing_window_validated'):continue
  assert name not in accepted and job['exit_code']==0 and not job['polluted'] and not job['numerically_certified']
  assert job['started']>=samples[0]['time'] and job['ended']<=samples[-1]['time']
  in_window=[s for s in samples if job['started']<=s['time']<=job['ended']];assert in_window and not any(not o['own'] for s in in_window for o in s['owners'])
  idle=[c for c in checks if c['label'].startswith(name+'-idle-')];assert len(idle)==3 and idle[-1]['time']-idle[0]['time']>=60 and all(not c['apps'].strip() for c in idle)
  for c in idle:
   assert all(int(line.split(',')[2].split()[0])<=512 and int(line.split(',')[3].split()[0])==0 for line in c['gpus'].splitlines())
  folder=root/name;ranks=[read(folder/f'rank{r}.json') for r in range(8)];wrappers=[read(folder/f'wrapper-rank{r}.json') for r in range(8)]
  assert all(len(r['records'])==25 and r['warmup']==5 and r['timed']==20 and not r['diagnostic'] for r in ranks)
  assert all(w['all_parameters_finite'] and w['all_losses_finite'] and w['deterministic_fa_backward'] and not w['bf16_reduced_precision_reduction'] for w in wrappers)
  assert not list(folder.glob('wait-state-rank*')) and all(not r['selected'] for rank in ranks for r in rank['records'])
  patches=[]
  for r in range(8):
   v=read(folder/f'launch-policy-rank{r}.json');assert v['policy']=='nccl-enqueue-mutex-native-order-wait-outside-v1' and v['implicit_order']=='0' and v['race_fatal']=='1' and not v['CPU_wait_sampling'] and not v['per_operation_logging']
   a=read(folder/f'stream-cache-patch-rank{r}.json');patches.append((a['original_sha256'],a['patched_sha256']));cache=read(folder/f'stream-cache-final-rank{r}.json');assert len(cache)==(2 if r//2 in [0,3] else 4)
  assert len(set(patches))==1
  inp=wrappers[0]['inputs'];tokens=[r['metrics']['valid_tokens'] for r in ranks[0]['records']];assert tokens==[88064 if i%2==0 else 37286 for i in range(25)]
  source=state['source'];entry=state['scripts_sha256']['benchmark_entry_persistent.py'];math=state['arithmetic']
  if common_inputs is None:common_inputs,common_tokens,common_source,common_entry,common_math,common_patch=inp,tokens,source,entry,math,patches[0]
  assert inp==common_inputs and tokens==common_tokens and source==common_source and entry==common_entry and math==common_math and patches[0]==common_patch
  h=name.split('-')[0];cmd=list(job['command']);cmd[cmd.index('--kaimm-context-parallel-impl')+1]='BACKEND'
  if h not in commands:commands[h]=cmd
  assert commands[h]==cmd
  sec=max(v['window_ms'] for v in ranks)/1000;valid=sum(tokens[5:]);rate=valid/sec;assert abs(rate-job['tokens_per_second'])<1e-6
  accepted[name]=dict(name=name,root=str(root),valid_tokens=valid,seconds=sec,tokens_per_second=rate,started=job['started'],ended=job['ended'],peak_allocated_bytes=max(v['allocated_peak'] for v in ranks),peak_reserved_bytes=max(v['reserved_peak'] for v in ranks))
assert set(accepted)==expected,(set(accepted),expected)
groups={}
for h in heads:
 windows=[accepted[f'{h}-{i:02d}-{m}'] for i,m in enumerate(order,1)];methods={};blocks=[]
 for m in ['KV','QO']:
  rows=[r for r in windows if r['name'].endswith('-'+m)];methods[m]=dict(tokens_per_second=sum(r['valid_tokens'] for r in rows)/sum(r['seconds'] for r in rows),peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in rows),peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in rows),window_rates=[r['tokens_per_second'] for r in rows])
 for block in [windows[:4],windows[4:]]:
  rates={m:sum(r['valid_tokens'] for r in block if r['name'].endswith('-'+m))/sum(r['seconds'] for r in block if r['name'].endswith('-'+m)) for m in ['KV','QO']};blocks.append(rates['QO']/rates['KV']-1)
 groups[h]=dict(methods=methods,QO_vs_KV=methods['QO']['tokens_per_second']/methods['KV']['tokens_per_second']-1,pair_block_QO_vs_KV=blocks,max_between_window_gap_seconds=max(b['started']-a['ended'] for a,b in zip(windows,windows[1:])))
result=dict(status='partial-group-reviewed-await-global-closure' if cfg.head else 'timing-reviewed-numerical-errors-unresolved',numerically_certified=False,scope='Onlyadmittedcleanwindows:7original+9resumed; pollutedoriginalwindow8excluded. Sameinputs/model/runtimepatch/math. No numericalacceptance.',source=common_source,arithmetic=common_math,runtime_entry_sha256=common_entry,runtime_patch_sha256=common_patch,roots=roots,groups=groups,windows=list(accepted.values()))
assert not cfg.output.exists(),'preserve existing review';cfg.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(groups,indent=2))
