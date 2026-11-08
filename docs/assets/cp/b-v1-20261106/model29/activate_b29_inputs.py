"""One-time CPU data freeze, only after active formal batch closes."""
from pathlib import Path
import argparse,fcntl,hashlib,json,os,sys,time,struct
root=Path('/workspace/experiments/dspp-cp-v1-20261106');out=Path('/workspace/outputs/dspp-cp-v1-20261106/baseline29-v1')
lock=(root/'numeric.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,str(root/'source-current'))
import torch
torch.set_num_threads(1)
from megatron.data.indexed_dataset import MMapIndexedDataset
current=json.loads((out/'manifest.json').read_text());assert current['status'] in ['complete','stopped'], 'no server CPU work during active formal'
parser=argparse.ArgumentParser();parser.add_argument('--case',choices=['short-packed-v512','short-tail-v512','arxiv-native32k','all'],default='all');cfg=parser.parse_args()
meta=json.loads((root/'data-v1/manifest.json').read_text());protocol=json.loads((root/'workload29-protocol.json').read_text())
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while True:
   b=f.read(8*1024*1024)
   if not b:break
   h.update(b)
 return h.hexdigest()
for w in meta['workloads']:
 name=w['name']
 if cfg.case!='all' and name!=cfg.case:continue
 base=root/'data-v1'/name;record=root/'data-v1'/f'{name}-verified.json'
 if record.exists():
  old=json.loads(record.read_text());st=base.with_suffix('.bin').stat()
  assert old['status']=='frozen-verified' and st.st_size==old['bin_size'] and st.st_mtime_ns==old['bin_mtime_ns'] and digest(base.with_suffix('.idx'))==old['idx_sha256']
  print(name,'reused frozen identity');continue
 if name=='arxiv-native32k':
  target=Path(w['source']+'.bin');assert target.is_file()
  if base.with_suffix('.bin').exists():assert base.with_suffix('.bin').resolve()==target.resolve()
  else:base.with_suffix('.bin').symlink_to(target)
 a=base.with_suffix('.bin').stat();idxhash=digest(base.with_suffix('.idx'));assert idxhash==w['idx_sha256']==protocol['cases'][name]['idx_sha256']
 binhash=digest(base.with_suffix('.bin')) if name!='arxiv-native32k' else None
 if 'bin_sha256' in w:assert binhash==w['bin_sha256']
 dataset=MMapIndexedDataset(str(base),skip_warmup=True)
 expected=protocol['cases'][name]['training_lengths']
 assert len(dataset)==256 and [int(n)-1 for n in dataset.sizes]==expected
 assert max(dataset.sizes)<=32769
 low=2**63-1;high=-1;view=hashlib.sha256()
 for i in range(256):
  sample=dataset[i];low=min(low,int(sample.min()));high=max(high,int(sample.max()))
  view.update(struct.pack('<QQ',i,len(sample)));view.update(sample.tobytes())
 assert 0<=low<=high<protocol['cases'][name]['vocab'] and not torch.cuda.is_initialized()
 b=base.with_suffix('.bin').stat();assert (a.st_size,a.st_mtime_ns)==(b.st_size,b.st_mtime_ns)
 d=dict(status='frozen-verified',time=time.time(),workload=name,bin_sha256=binhash,idx_sha256=idxhash,bin_size=b.st_size,bin_mtime_ns=b.st_mtime_ns,bin_inode=b.st_ino,bin_resolved_path=str(base.with_suffix('.bin').resolve()),source_metadata=w,truncated_documents=0,native_reader_verified=True,token_range=[low,high],cuda_initialized=False,view_content_sha256=view.hexdigest(),binary_hash_scope='entire synthetic file' if binhash else 'selected256nativeviewrecords withboundaries; entirecorpusnotrehashed')
 record.write_text(json.dumps(d,indent=2)+'\n');print(name,binhash,idxhash)
