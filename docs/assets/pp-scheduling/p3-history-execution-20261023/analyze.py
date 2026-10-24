import ast, hashlib, json, re, sys
from collections import Counter
from pathlib import Path
root=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).parent

def digest(j):
 return hashlib.sha256(json.dumps(j,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def key(n): return n['stage'],n['kind'],n['chunk'],n['microbatch'],n['split']
def canonical(n):
 k=list(key(n)); k[1]='B' if k[1]=='D' else k[1]; return tuple(k)
def reports(s):
 return {(j['iteration'],j['rank']):j for f in (root/s/'comparison').glob('iteration*.json') for j in [json.loads(f.read_text())]}
base=reports('adjacent'); assert len(base)==9
summary={'training':{}, 'base_plan_hashes':{}, 'gpu_multilayer':{}, 'cpu':{'new_contract_cases':6,'existing_regression_cases':97,'targeted_calibration_cases':2}}
for iteration in (1,2,3):
 plans=[base[iteration,r]['plan'] for r in range(3)]
 assert len({digest(p) for p in plans})==1
 summary['base_plan_hashes'][iteration]=digest(plans[0])
for scenario in ('adjacent','across-w','across-fb'):
 cmd=json.loads((root/scenario/'command.json').read_text()); assert cmd['returncode']==0
 records=reports(scenario); assert len(records)==9
 maxima=Counter(); moves=[]; ranks={}; hashes={}
 for (iteration,rank),j in sorted(records.items()):
  p,b=j['plan'],base[iteration,rank]['plan']
  assert p['binding']==b['binding'] and p['history_dependencies']==b['history_dependencies']
  assert p['algorithm']=='explicit-DH' and p['schema_version']==2
  for seq,old in zip(p['schedules'],b['schedules']):
   assert [canonical(n) for n in seq if n['kind']!='H']==[canonical(n) for n in old if n['kind']!='H']
   assert {key(n) for n in seq if n['kind']=='D'}=={key(n) for n in old if n['kind']=='D'}
   for n in seq:
    if n['kind']!='H': continue
    i=next(i for i,x in enumerate(old) if key(x)==key(n)); newpos=n['slot']
    if i!=newpos and rank==0:
     crossed=[x['kind'] for x in old[i+1:newpos+1]]
     moves.append(dict(iteration=iteration,stage=n['stage'],task=key(n),old_slot=i,new_slot=newpos,crossed=crossed))
  assert p['messages']==b['messages']
  if scenario!='adjacent':
   for category in ('params_before','params_after','gradients','outputs','dx','logits','loss'):
    maxima[category]=max(maxima[category],j[category]['max_abs'])
  hashes.setdefault(iteration,set()).add(digest(p))
  ev=json.loads((root/scenario/'comparison'/f'p3-iteration{iteration}-rank{rank}.json').read_text())
  d=[(e['logical'],e['chunk'],e['physical']) for e in ev['events'] if e['kind']=='D']
  h=[(e['logical'],e['chunk'],e['physical']) for e in ev['events'] if e['kind']=='H']
  assert len(d)==len(set(d)) and Counter(d)==Counter(h)
  assert all(e['history_layers']==1 for e in ev['events'] if e['kind']=='H')
  assert ev['allocated_peak_bytes']<=ev['predicted_total_bytes']
  rr=ranks.setdefault(rank,dict(allocated_peak_mib=0,predicted_total_mib=0,warm_allocated_peak_mib=0))
  rr['allocated_peak_mib']=max(rr['allocated_peak_mib'],ev['allocated_peak_bytes']/2**20)
  rr['predicted_total_mib']=max(rr['predicted_total_mib'],ev['predicted_total_bytes']/2**20)
  if iteration>1: rr['warm_allocated_peak_mib']=max(rr['warm_allocated_peak_mib'],ev['allocated_peak_bytes']/2**20)
 assert all(len(h)==1 for h in hashes.values())
 if scenario!='adjacent':
  assert len(moves)==3
  assert all(any(k in ('W',) if scenario=='across-w' else k in ('F','B') for k in m['crossed']) for m in moves)
 iterations=Counter(); entries={}
 for line in (root/scenario/'train.log').read_text().splitlines():
  for m in re.finditer(r'\[dspp\]\[p2p-debug\]\[stage=(\d+)\]\[seq=(\d+)\] (.*?)(?=\[dspp\]\[p2p-debug\]|$)',line):
   stage,seq=int(m[1]),int(m[2]); info=m[3]
   if seq==0: iterations[stage]+=1
   entries.setdefault((iterations[stage],stage),[]).append((seq,info))
 assert set(iterations.values())=={3}
 counts=Counter()
 for (iteration,stage),logs in entries.items():
  plan=records[iteration,stage]['plan']; seqs=plan['schedules'][stage]
  dh={(n['chunk'],n['microbatch'],n['split']):n for n in seqs if n['kind']=='D'}
  posted={}; hslots={}
  for seq,info in logs:
   m=re.search(r"posted-send peer=\d+ key=(\(.*\))",info)
   if m: posted[ast.literal_eval(m[1])]=seq
   m=re.search(r'node slot=\d+ kind=H(\d) group=(\d+) split=(\d+)',info)
   if m: hslots[tuple(map(int,m.groups()))]=seq
  assert set(hslots)==set(dh)
  for c,g,s in dh:
   if (c==1 and stage<2) or (c==0 and stage>0):
    k=('B',c,g,s); assert k in posted
    counts['D_with_PP_send']+=1
    counts['send_posted_before_H']+=posted[k]<hslots[c,g,s]
   else: counts['D_local_boundary']+=1
  assert not any(k[0]=='H' for k in posted)
 summary['training'][scenario]=dict(returncode=0,steps=3,reports=9,comparisons_max_abs=dict(maxima),moved_h=moves,peaks_by_rank=ranks,communication=dict(counts),plan_hashes={k:next(iter(v)) for k,v in hashes.items()})
line=next(l for l in (root/'gpu-multilayer-final.log').read_text().splitlines() if l.startswith('P3_MULTILAYER '))
summary['gpu_multilayer']=json.loads(line.split(' ',1)[1])
assert summary['gpu_multilayer']['pending_tasks']==2
assert all(v['delayed_vs_adjacent']==0 and v['fused_max_abs']==0 for v in summary['gpu_multilayer']['checks'].values())
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2)[:10500])
