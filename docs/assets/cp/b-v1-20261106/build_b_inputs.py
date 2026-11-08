"""CPU-only preparation of explicit b input domains; no CUDA or model code."""
from pathlib import Path
import array,struct,hashlib,json,sys
assert sys.byteorder=='little'
root=Path(sys.argv[1]);root.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write_idx(path,dtype,sizes,pointers,docs):
 with path.open('xb') as f:
  f.write(b'MMIDIDX\0\0'+struct.pack('<QBQQ',1,dtype,len(sizes),len(docs)))
  for fmt,rows in [('i',sizes),('q',pointers),('q',docs)]:f.write(array.array(fmt,rows).tobytes())
result=dict(status='local-CPU-prepared-not-launched',workloads=[])
for name,lengths in [('short-packed-v512',[800,224,600,424,1500,548,700,700]),('short-tail-v512',[648,4608,768,768,6401,2305,97,31])]:
 base=root/name;sizes=[];pointers=[];offset=0
 with base.with_suffix('.bin').open('xb') as f:
  for repeat in range(32):
   for idx,length in enumerate(lengths):
    values=array.array('H',((t+repeat*313+idx*19)%511+1 for t in range(length+1)))
    assert min(values)>=1 and max(values)<512
    f.write(values.tobytes());sizes.append(length+1);pointers.append(offset);offset+=2*(length+1)
 write_idx(base.with_suffix('.idx'),8,sizes,pointers,list(range(len(sizes)+1)))
 assert len(sizes)==256 and base.with_suffix('.bin').stat().st_size==sum(sizes)*2
 result['workloads'].append(dict(name=name,vocab=512,documents=256,training_lengths_pattern=lengths,repeats=32,single_loader_train_iters=25,documents_consumed=200,bin_sha256=sha(base.with_suffix('.bin')),idx_sha256=sha(base.with_suffix('.idx')),first_update_valid_tokens=sum(lengths),timed_valid_tokens=sum(lengths)*20))
baseidx=Path(sys.argv[2]);raw=baseidx.read_bytes();assert raw[:9]==b'MMIDIDX\0\0'
version,dtype,n,ndoc=struct.unpack_from('<QBQQ',raw,9);assert version==1
sizes=array.array('i');sizes.frombytes(raw[34:34+4*n]);pointers=array.array('q');pointers.frombytes(raw[34+4*n:34+12*n]);docs=array.array('q');docs.frombytes(raw[34+12*n:]);assert len(docs)==ndoc
selected=[];new_sizes=[];new_pointers=[];new_docs=[0];rawlengths=[]
for doc,(a,b) in enumerate(zip(docs,docs[1:])):
 length=sum(sizes[a:b])
 if 2<=length<=32769:
  selected.append(doc);rawlengths.append(length);new_sizes.extend(sizes[a:b]);new_pointers.extend(pointers[a:b]);new_docs.append(len(new_sizes))
 if len(selected)==256:break
assert len(selected)==256
path=root/'arxiv-native32k.idx';write_idx(path,dtype,new_sizes,new_pointers,new_docs)
result['workloads'].append(dict(name='arxiv-native32k',vocab=151936,documents=256,source_index_sha256=sha(baseidx),idx_sha256=sha(path),selected_document_indices=selected,raw_document_lengths=rawlengths,max_raw_length=max(rawlengths),source='/workspace/datasets/slimpajama_arxiv_50k/megatron/slimpajama_arxiv_50k_text_document',bin_rule='remote view .bin symlink to original frozen .bin; original offsets retained',truncated_documents=0,training_lengths=[n-1 for n in rawlengths],single_loader_train_iters=25,documents_consumed=200,timed_valid_tokens=sum(n-1 for n in rawlengths[40:200])))
(root/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps([{k:v for k,v in d.items() if k in ['name','documents','max_raw_length','timed_valid_tokens','truncated_documents']} for d in result['workloads']],indent=2))
