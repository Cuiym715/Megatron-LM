"""Four-GPU PP2CP2 targeted complete-update checks, not performance evidence.

Keep the full16-layer target model. Compare each backend only with itself,
changing NCCL host launch ordering, not diagnosing known cross-backend errors.
"""
import fcntl,hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path('/workspace/experiments/dspp-cp-v1-20261106')
SRC=ROOT/'source-current'
OUT=Path('/workspace/outputs/dspp-cp-v1-20261106/smallcard-checks-v1')
NUM=Path('/workspace/outputs/dspp-cp-v1-20261106/numeric-v5-longmix512')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    lock=(ROOT/'numeric.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    old=json.loads((NUM/'manifest.json').read_text())
    source=old['source']['sources']['current']
    for n,h in source['required_sha256'].items():assert digest(SRC/n)==h
    for n,h in old['data_sha256'].items():assert digest(Path(n))==h
    protocol=json.loads((ROOT/'smallcard-protocol.json').read_text())
    uuids=[old['gpu_uuids'][i] for i in protocol['physical_gpu_indices']]
    assert len(uuids)==4 and protocol['layers']==16 and protocol['pp']==2 and protocol['cp']==2
    OUT.mkdir(exist_ok=False);(OUT/'scripts').mkdir()
    scripts=['run_smallcard_checks.py','smallcard_numeric_entry.py','numeric_entry_v4.py','zero_dropout.py','smallcard-protocol.json']
    for n in scripts:(OUT/'scripts'/n).write_bytes((ROOT/n).read_bytes())
    prod=Path('/workspace/projects/Megatron-LM-kwai')
    patch=subprocess.check_output(['git','diff','--binary'],cwd=prod)
    (OUT/'production-before.patch').write_bytes(patch)
    state=dict(status='checking-idle',pid=os.getpid(),started=time.time(),jobs=[],gpu_uuids=uuids,
        physical_gpu_indices=protocol['physical_gpu_indices'],source=source,data_sha256=old['data_sha256'],protocol=protocol,
        scripts_sha256={n:digest(ROOT/n) for n in scripts},production_patch_sha256=hashlib.sha256(patch).hexdigest(),
        diagnostic=True,scope='same-backend full gradient/update comparison of NCCL setting on PP2CP2; not8GPU certification or throughput; oldKV/QOnumerical investigation deferred')
    def save():
        p=OUT/'manifest.tmp';p.write_text(json.dumps(state,indent=2)+'\n');p.replace(OUT/'manifest.json')
    def check(label,own=False,require_auditor=False):
        ap=OUT/'host-ownership/manifest.json'
        if require_auditor:assert ap.exists(),'selected-GPU auditor must attach before launch'
        if ap.exists():
            a=json.loads(ap.read_text());assert a['status']=='running' and not a['unknown_observed'],'selected-GPU audit unavailable/unclean'
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader'],text=True)
        selected=[r for r in gpu.splitlines() if any(u in r for u in uuids)]
        active=[r for r in apps.splitlines() if any(u in r for u in uuids)]
        with (OUT/'resource-checks.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),label=label,gpus=gpu,apps=apps,selected_gpus=selected,selected_apps=active))+'\n')
        assert len(selected)==4
        if own:assert all(sum(u in r for r in active)<=1 for u in uuids),'extra selected-GPU process'
        else:
            assert not active,'selected GPUs occupied'
            assert all(int(r.split(',')[2].split()[0])<=512 and int(r.split(',')[3].split()[0])==0 for r in selected),'selected GPUs not idle'
    base=list(old['jobs'][0]['command'])
    base[base.index('--nproc-per-node=8')]='--nproc-per-node=4'
    base[base.index(str(ROOT/'numeric_entry_v4.py'))]=str(ROOT/'smallcard_numeric_entry.py')
    for flag,value in {'--pipeline-model-parallel-size':'2','--num-layers-per-virtual-pipeline-stage':'4','--train-iters':'1'}.items():base[base.index(flag)+1]=value
    base+=['--distributed-timeout-minutes','2']
    env={k:v for k,v in os.environ.items() if not k.startswith(('DSPP_','CP_V1_'))}
    env.update(CUDA_VISIBLE_DEVICES=','.join(uuids),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_DEVICE_MAX_CONNECTIONS='32',
        CP_V1_SOURCE=str(SRC),DSPP_FA_SPLIT_CACHE='/workspace/cache/dspp-fa-split',
        DSPP_PLAN_CHECK_CP_REFERENCE='0',DSPP_PLAN_CHECK_CP_NUMERICS='1',DSPP_PLAN_CHECK_ALIGN_STATE='1',DSPP_PLAN_CHECK_REPORT_FAILURES='1',DSPP_PLAN_CHECK_DETERMINISTIC_FA='1',NCCL_LAUNCH_RACE_FATAL='1')
    save()
    try:
        for backend in ['KV','QO']:
            snapshots=OUT/backend/'snapshots'
            for mode in ['reference','compare']:
                name=backend.lower()+'-'+mode
                for n,h in state['scripts_sha256'].items():assert digest(ROOT/n)==h
                for i in range(3):
                    check(name+'-idle-'+str(i),require_auditor=i==2)
                    if i<2:time.sleep(30)
                p=OUT/name;p.mkdir();command=list(base)
                command[command.index('--kaimm-context-parallel-impl')+1]='key-value' if backend=='KV' else 'query-out'
                jobenv=dict(env,NCCL_LAUNCH_ORDER_IMPLICIT='0' if mode=='reference' else '1',DSPP_PLAN_CHECK_MODE=mode,
                    DSPP_PLAN_CHECK_OUTPUT=str(p),DSPP_PLAN_CHECK_SNAPSHOTS=str(snapshots),CP_V1_MASTER_REFERENCE=str(snapshots))
                job=dict(name=name,started=time.time(),command=command,output=str(p),polluted=False)
                state['jobs'].append(job);state['status']='running';save()
                (p/'command.json').write_text(json.dumps(dict(argv=command,env={k:v for k,v in jobenv.items() if k.startswith(('DSPP_','CP_V1_','NCCL_','CUDA_','OMP_','MKL_'))}),indent=2)+'\n')
                with (p/'train.log').open('w') as log:
                    process=subprocess.Popen(['timeout','--kill-after=15','300',*command],cwd=SRC,env=jobenv,stdout=log,stderr=subprocess.STDOUT)
                    job['pid']=process.pid;save()
                    while process.poll() is None:
                        try:check(name+'-monitor',own=True)
                        except Exception as e:
                            job.update(polluted=True,pollution_reason=repr(e));save()
                            if process.poll() is None and not job.get('own_stop_requested'):
                                job['own_stop_requested']=time.time();save();process.terminate()
                        time.sleep(5)
                job.update(exit_code=process.returncode,ended=time.time());save();check(name+'-after')
                assert not job['polluted'] and process.returncode in [0,1],'failed or polluted operation'
                native=[json.loads(f.read_text()) for f in p.glob('iteration*-rank*.json')]
                masters=[json.loads(f.read_text()) for f in p.glob('full-master-iteration*-rank*.json')]
                assert len(native)==len(masters)==4,'missing full-update reports'
                assert next(r for r in native if r['rank']==0)['metrics']['valid_tokens']==88064
                passed=all(v['passed'] for r in native+masters for v in r.values() if isinstance(v,dict) and 'passed' in v)
                assert (process.returncode==0)==passed,'unexplained nonzero exit'
                job.update(complete_update_reports=4,native_and_master_checks_passed=passed,numerical_failure_retained=not passed);save()
        state.update(status='complete',ended=time.time());save()
    except Exception as e:state.update(status='stopped',error=repr(e),ended=time.time());save();raise
    finally:
        state['production_patch_unchanged']=hashlib.sha256(subprocess.check_output(['git','diff','--binary'],cwd=prod)).hexdigest()==state['production_patch_sha256'];save()
if __name__=='__main__':main()
