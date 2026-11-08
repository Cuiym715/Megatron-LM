"""Read small native summaries/owner headers; do not reparse9GB boundaries."""
from pathlib import Path
import fcntl,hashlib,json,os,time
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path('/workspace/experiments/dspp-cp-v1-20261106')
lock=(ROOT/'numeric.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
OUT=Path('/workspace/outputs/dspp-cp-v1-20261106/owner-fused29-gqa1-v1')
def read(p):return json.loads(p.read_text())
state=read(OUT/'manifest.json');assert state['status']=='complete' and state['production_patch_unchanged']
assert state['resource_mode']=='shared-functional' and not state['timing_eligible'] and len(state['jobs'])==2
groups={};evidence=[];common_tokens=None
for job in state['jobs']:
 assert job['exit_code']==0 and job['profile_validated'] and not job['polluted']
 path=Path(job['output']);ranks=[read(path/f'rank{r}.json') for r in range(8)]
 wrappers=[read(path/f'wrapper-rank{r}.json') for r in range(8)]
 assert all(x['diagnostic'] and x['warmup']==2 and x['timed']==4 and len(x['records'])==6 for x in ranks)
 assert all(x['all_parameters_finite'] and x['all_losses_finite'] and x['deterministic_fa_backward'] and not x['bf16_reduced_precision_reduction'] for x in wrappers)
 owned=[read(path/f'model29-ownership-rank{r}.json') for r in range(8)]
 assert sorted(i for o in owned if o['cp_rank']==0 for c in o['chunks'] for i in c['global_layers'])==list(range(1,30))
 model=ranks[0]['layout']['model'];assert (model['num_layers'],model['hidden_size'],model['num_attention_heads'],model['num_query_groups'],model['padded_vocab_size'])==(29,2048,16,1,32128)
 tokens=[r['metrics']['valid_tokens'] for r in ranks[0]['records']];assert tokens==[88064,37286]*3
 if common_tokens is None:common_tokens=tokens
 assert tokens==common_tokens
 name='KV' if ranks[0]['cp_impl']=='key-value' else 'QO'
 io={};history=[];headers=[]
 for rank in ranks:
  history.append(dict(rank=rank['rank'],stage=rank['stage'],cp_rank=rank['cp_rank'],**rank['history_kv_peak']))
  for api,values in rank['attention_io'].items():
   target=io.setdefault(api,{k:0 for k in values})
   for k,v in values.items():target[k]+=v
 for cp in range(2):
  for stage in range(4):
   for iteration in [3,4]:
    p=path/'owners'/f'cp{cp}'/f'iteration_{iteration:06d}_stage_{stage}.json'
    # Native producer places all summary fields before the huge boundary array.
    head=[]
    with p.open() as f:
     for line in f:
      if line.startswith('  "boundaries":'):break
      head.append(line)
     else:raise ValueError('missing native boundaries field')
    header=''.join(head);data=json.loads(header.rstrip().rstrip(',')+'\n}')
    assert data['schema']=='dspp-owner-audit-v1' and data['stage']==stage and data['iteration']==iteration and not data['certified']
    headers.append(dict(cp_rank=cp,stage=stage,iteration=iteration,**{k:v for k,v in data.items() if k.endswith('_bytes')}))
    stat=p.stat();evidence.append(dict(path=str(p),size=stat.st_size,mtime_ns=stat.st_mtime_ns,summary_header_sha256=hashlib.sha256(header.encode()).hexdigest()))
 groups[name]=dict(attention_io_all8ranks_all6updates=io,
  ring_equivalent_send_bytes=sum(v['ring_equivalent_send_bytes'] for v in io.values()),
  history_F_boundary_samples=history,history_storage_peak=max(r['storage_bytes'] for r in history),
  pending_dkv_F_boundary_sample_peak=max(r['pending_dkv_logical_bytes'] for r in history),
  allocated_peak=max(r['allocated_peak'] for r in ranks),reserved_peak=max(r['reserved_peak'] for r in ranks),
  owner_headers=headers)
assert groups.keys()=={'KV','QO'}
value=dict(status='complete-shared-owner-summary',pid=os.getpid(),time=time.time(),GPU_used=False,
 scope='Originalb2 fusedGQA29L metadata/unique-cache-owner and communication-count evidence. Not throughput/bandwidth/capacitylimit/numericalequivalence or final B memory calibration.',
 valid_tokens_all6updates=sum(common_tokens),groups=groups,
 QO_vs_KV_ring_equivalent_send_ratio=groups['QO']['ring_equivalent_send_bytes']/groups['KV']['ring_equivalent_send_bytes'],
 QO_vs_KV_sampled_history_storage_ratio=groups['QO']['history_storage_peak']/groups['KV']['history_storage_peak'],
 provenance=evidence,limits=['Communication bytes are tensor-size-based ring-equivalent accounting, not measured physical link bytes or timing.',
  'History/pending peaks are sampled at F boundaries; do not sum independently maximized components or certify future allocator bounds.',
  'Owner header summaries are native unique-storage boundary highwatermarks; this review does not independently walk9GB of full boundary rows. Raw JSON and allocator traces retained onserver.',
  'Only fused29L GQA1 measured; split/pending lifetime allowances and complete-stage costs still need original planned evidence. Shared wall-clock not used.'])
(OUT/'owner-summary.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps({k:v for k,v in value.items() if k not in ['groups','provenance']}))
print(json.dumps({name:{k:v for k,v in data.items() if k in ['ring_equivalent_send_bytes','history_storage_peak','pending_dkv_F_boundary_sample_peak','allocated_peak','reserved_peak']} for name,data in groups.items()}))
