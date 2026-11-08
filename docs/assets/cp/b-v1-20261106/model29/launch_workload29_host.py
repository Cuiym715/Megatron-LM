"""One-call reviewed-case handoff: CPU data activation then bounded GPU controller."""
from pathlib import Path
import argparse,fcntl,json,re,subprocess,time
r=Path('/home/test/test01/tinacui/experiments/dspp-cp-v1-20261106');outputs=Path('/home/test/test01/tinacui/outputs/dspp-cp-v1-20261106')
p=argparse.ArgumentParser();p.add_argument('--case',required=True,choices=['short-packed-v512','short-tail-v512','arxiv-native32k']);p.add_argument('--batch',required=True);p.add_argument('--after',required=True);p.add_argument('--review',required=True);cfg=p.parse_args()
assert all(re.fullmatch('[a-z0-9-]+',v) for v in [cfg.batch,cfg.after]);reviewpath=Path(cfg.review);assert reviewpath.is_relative_to(r)
lock=(r/f'launch-{cfg.batch}.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
prev=outputs/cfg.after;old=json.loads((prev/'manifest.json').read_text());audit=json.loads((prev/'host-ownership/manifest.json').read_text());review=json.loads(reviewpath.read_text())
assert old['status']=='complete' and old['production_patch_unchanged'] and audit['status']=='complete' and not audit['unknown_observed']
assert review['status']=='timing-reviewed-numerical-errors-unresolved' and len(review['windows'])==16 and review['source']==old['source']
expected=old.get('workload_config',{}).get('timed_valid_tokens',1253500);assert all(w['valid_tokens']==expected for w in review['windows'])
out=outputs/cfg.batch;assert not out.exists();launch=r/f'{cfg.batch}-launch.json';assert not launch.exists()
record=dict(status='data-activation',started=time.time(),previous=str(prev),review=str(reviewpath),case=cfg.case,output=str(out))
launch.write_text(json.dumps(record,indent=2)+'\n')
with (r/f'{cfg.batch}-data-activation.log').open('x') as f:
 subprocess.run(['docker','exec','tinacui-megatron','env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=1','timeout','--kill-after=5','120','/workspace/venvs/megatron/bin/python','-u','/workspace/experiments/dspp-cp-v1-20261106/activate_b29_inputs.py','--case',cfg.case],stdout=f,stderr=subprocess.STDOUT,check=True)
cmd=['docker','exec','tinacui-megatron','/workspace/venvs/megatron/bin/python','-u','/workspace/experiments/dspp-cp-v1-20261106/run_workload29.py','--case',cfg.case,'--batch',cfg.batch]
with (r/f'{cfg.batch}-controller.log').open('x') as f:child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
record.update(status='controller-launch',pid=child.pid,controller_started=time.time(),command=cmd,log=str(r/f'{cfg.batch}-controller.log'));launch.write_text(json.dumps(record,indent=2)+'\n')
for _ in range(50):
 if (out/'manifest.json').exists():break
 if child.poll() is not None:raise RuntimeError((r/f'{cfg.batch}-controller.log').read_text()[-2000:])
 time.sleep(.1)
assert (out/'manifest.json').exists()
with (r/f'{cfg.batch}-audit.log').open('x') as f:a=subprocess.Popen(['python3',str(r.parent/'host_ownership_audit.py'),str(out)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
for _ in range(30):
 if (out/'host-ownership/manifest.json').exists():break
 time.sleep(.1)
with (r/f'{cfg.batch}-own-guard.log').open('x') as f:g=subprocess.Popen(['python3',str(r/'own_performance_guard.py'),str(out)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
record.update(status='running',audit_pid=a.pid,guard_pid=g.pid,container_pid=json.loads((out/'manifest.json').read_text())['pid']);launch.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
