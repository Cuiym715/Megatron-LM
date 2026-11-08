"""CPU-only isolated CP sources; existing validated archives stay unchanged."""
import hashlib,json,os,tarfile,time
from pathlib import Path
BASE=Path('/home/test/test01/tinacui');ROOT=BASE/'experiments/dspp-cp-v1-20261106';OLD=BASE/'outputs/dspp-qo-20261105-exec01'
def main():
 ROOT.mkdir(exist_ok=True);out=ROOT/'source-manifest.json';assert not out.exists()
 start=time.time();sources={}
 for label,name in [('current','candidate-theoretical-validated-source.tar.gz'),('first','baseline-source.tar.gz')]:
  archive=OLD/name;dest=ROOT/('source-'+label);dest.mkdir(exist_ok=False)
  with tarfile.open(archive) as t:t.extractall(dest,filter='data')
  required=['megatron/core/kv_cache/dspp_cp_attention.py','megatron/core/kv_cache/dspp_cp_qo_attention.py','megatron/core/kv_cache/fa_split.py','megatron/core/pipeline_parallel/dspp_prefetch.py','megatron/core/pipeline_parallel/dspp_plan_b.py','megatron/core/pipeline_parallel/schedules.py','tools/validate_dspp_plan_training.py','tools/benchmark_dspp_cp.py','examples/run_dspp_cp.sh']
  info={p:hashlib.sha256((dest/p).read_bytes()).hexdigest() for p in required}
  differences=[]
  if label=='current':
   for path,value in info.items():
    working=OLD/'source-dh'/path
    if not working.exists() or hashlib.sha256(working.read_bytes()).hexdigest()!=value:differences.append(path)
  sources[label]=dict(archive=str(archive),archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),isolated=str(dest),required_sha256=info,current_working_differences=differences,selection='validated archive, never unmeasured local/current working changes')
 result=dict(status='prepared-source-only',pid=os.getpid(),started=start,ended=time.time(),sources=sources,production_files_modified=False,source_note='CP oldqueues andsmall/probe processes were explicitly closed in latestreport. This task uses new copies only, not restoreoldqueue. HEAD26adff provenance belongsoldvalidatedCP snapshot; production d5eadirty unchanged.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
