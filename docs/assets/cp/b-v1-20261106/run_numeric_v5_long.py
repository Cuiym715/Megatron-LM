"""Six one-full-update checks on the actual32K/vocab512 baseline input."""
import fcntl,hashlib,json,os,shlex,subprocess,time
from pathlib import Path
ROOT=Path('/workspace/experiments/dspp-cp-v1-20261106')
SRC=ROOT/'source-current';OUT=Path('/workspace/outputs/dspp-cp-v1-20261106/numeric-v5-longmix512')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    lock=(ROOT/'numeric.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    prior=Path('/workspace/outputs/dspp-cp-v1-20261106/numeric-v4-target')
    assert json.loads((prior/'manifest.json').read_text())['status']=='complete'
    assert json.loads((prior/'host-ownership/manifest.json').read_text())['status']=='complete'
    assert not json.loads((prior/'host-ownership/manifest.json').read_text())['unknown_observed']
    assert json.loads((prior/'target-review.json').read_text())['status']=='accepted-target-smoke','review preceding target proof before GPU'
    source=json.loads((ROOT/'source-manifest.json').read_text());assert source['status']=='prepared-source-only'
    for name,h in source['sources']['current']['required_sha256'].items():assert digest(SRC/name)==h
    OUT.mkdir(parents=True,exist_ok=False);(OUT/'scripts').mkdir()
    for p in ROOT.glob('*.py'):(OUT/'scripts'/p.name).write_bytes(p.read_bytes())
    uuids=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines();assert len(uuids)==8
    state=dict(status='checking-idle',pid=os.getpid(),started=time.time(),jobs=[],gpu_uuids=uuids,
        source=source,scripts_sha256={p.name:digest(p) for p in ROOT.glob('*.py')},
        contract=dict(model='16L H1024 FFN2816 Q8 PP4CP2 TP1 DP1 VPP2 balanced chunk2048',
            input='existing longmix-v512 first8 documents, single loader; one complete aligned full update, seed1234, baseline hybrid; expected88064 validtokens and max32768',
            numerical='native all parameter gradients/outputs/dx/logits/loss plus complete FP32 master updates; atol .004 rtol .01 rL2 .02; forward .03/.03',
            arithmetic='deterministic FA2 backward and BF16 reduced precision GEMM reduction OFF in both reference/candidate, exact p0; diagnostic only, not default-path performance',
            timing=False,scope='same-CP2 backend comparison; previous CP1 routing proofs retained, not a new CP1 target certification'))
    data=Path('/workspace/outputs/dspp-cp-longtask-20261101-1602/data/longmix-v512')
    state['data_sha256']={str(data)+s:digest(Path(str(data)+s)) for s in ['.bin','.idx']}
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import sys
    sys.path.insert(0,str(SRC))
    from megatron.data.indexed_dataset import MMapIndexedDataset
    import torch
    dataset=MMapIndexedDataset(str(data),skip_warmup=True)
    expected=[32768,24576,16384,12288,800,224,600,424]
    actual=[int(n)-1 for n in dataset.sizes[:8]]
    assert actual==expected and sum(actual)==88064 and max(actual)==32768
    assert all(int(dataset[i].min())>=0 and int(dataset[i].max())<512 for i in range(8))
    assert not torch.cuda.is_initialized(),'CPU metadata preflight initialized CUDA'
    state['actual_domain']=dict(sample_ids=list(range(8)),training_lengths=actual,valid_tokens=88064,max_length=32768,vocab=512,cuda_initialized=False)
    prod=Path('/workspace/projects/Megatron-LM-kwai')
    patch=subprocess.check_output(['git','diff','--binary'],cwd=prod)
    (OUT/'production-before.patch').write_bytes(patch);state['production_patch_sha256']=hashlib.sha256(patch).hexdigest()
    def save():
        p=OUT/'manifest.tmp';p.write_text(json.dumps(state,indent=2)+'\n');p.replace(OUT/'manifest.json')
    def check(label,own=False):
        audit=OUT/'host-ownership/manifest.json'
        if audit.exists() and json.loads(audit.read_text())['unknown_observed']:raise RuntimeError('unknown host GPU process observed')
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu,clocks.sm,temperature.gpu,power.draw','--format=csv,noheader'],text=True)
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader'],text=True)
        with (OUT/'resource-checks.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),label=label,gpus=gpu,apps=apps))+'\n')
        assert len(gpu.splitlines())==8 and all(u in gpu for u in uuids)
        if not own:
            assert not apps.strip(),'occupied compute PID'
            assert all(int(r.split(',')[2].split()[0])<=512 and int(r.split(',')[3].split()[0])==0 for r in gpu.splitlines()),'not idle'
        else:
            assert all(sum(u in r for r in apps.splitlines())<=1 for u in uuids),'unexpected extra process'
    save()
    old=Path('/workspace/outputs/dspp-cp-longtask-20261101-1602/update-aligned-v2/cp1/command.sh')
    base=shlex.split(old.read_text());base[base.index('--nproc-per-node=2')]='--nproc-per-node=8'
    base[base.index('tools/validate_dspp_plan_training.py')]=str(ROOT/'numeric_entry_v4.py')
    for flag,value in {'--pipeline-model-parallel-size':'4','--context-parallel-size':'2','--num-layers-per-virtual-pipeline-stage':'2','--num-layers':'16','--hidden-size':'1024','--ffn-hidden-size':'2816','--num-attention-heads':'8','--seq-length':'32768','--max-position-embeddings':'32768','--dspp-recv-lookahead':'0','--dspp-send-inflight-depth':'1'}.items():base[base.index(flag)+1]=value
    for flag,value in {'--train-iters':'1','--data-path':str(data),'--vocab-size':'512','--dataloader-type':'single'}.items():base[base.index(flag)+1]=value
    base+=['--lr-decay-iters','25']
    env={k:v for k,v in os.environ.items() if not k.startswith('DSPP_')}
    env.update(CUDA_VISIBLE_DEVICES=','.join(uuids),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_DEVICE_MAX_CONNECTIONS='32',
        DSPP_FA_SPLIT_CACHE='/workspace/cache/dspp-fa-split',CP_V1_SOURCE=str(SRC),
        DSPP_PLAN_CHECK_CP_REFERENCE='0',DSPP_PLAN_CHECK_CP_NUMERICS='1',DSPP_PLAN_CHECK_ALIGN_STATE='1',DSPP_PLAN_CHECK_REPORT_FAILURES='1')
    try:
        for heads in ['mha','gqa1']:
            for mode in ['kv-reference','qo-fused','qo-immediate']:
                snapshots=OUT/heads/'snapshots';master_reference=snapshots
                if mode=='qo-immediate':snapshots=OUT/(heads+'-qo-fused')/'complete-tensors';master_reference=OUT/(heads+'-qo-fused')
                name=heads+'-'+mode
                for i in range(3):
                    check(name+'-idle-'+str(i))
                    if i<2:time.sleep(30)
                for file,h in source['sources']['current']['required_sha256'].items():assert digest(SRC/file)==h
                path=OUT/name;path.mkdir();cmd=list(base)
                cmd[cmd.index('--kaimm-context-parallel-impl')+1]='key-value' if mode=='kv-reference' else 'query-out'
                if heads=='gqa1':cmd+=['--group-query-attention','--num-query-groups','1']
                if mode=='qo-immediate':cmd+=['--dspp-split-attention-backward','--dspp-history-dkv-after-dx']
                job=dict(name=name,started=time.time(),command=cmd,output=str(path),polluted=False)
                state['jobs'].append(job);state['status']='running';save()
                jobenv=dict(env,DSPP_PLAN_CHECK_OUTPUT=str(path),DSPP_PLAN_CHECK_SNAPSHOTS=str(snapshots),CP_V1_MASTER_REFERENCE=str(master_reference),DSPP_PLAN_CHECK_MODE='reference' if mode=='kv-reference' else 'compare',DSPP_PLAN_CHECK_EXPORT_REFERENCE=str(path/'complete-tensors'),DSPP_PLAN_CHECK_DETERMINISTIC_FA='1')
                (path/'command.json').write_text(json.dumps(dict(argv=cmd,env={k:v for k,v in jobenv.items() if k.startswith(('DSPP_','CP_V1','CUDA_','OMP_','MKL_'))}),indent=2)+'\n')
                with (path/'train.log').open('w') as log:
                    p=subprocess.Popen(['timeout','--kill-after=15','900',*cmd],cwd=SRC,env=jobenv,stdout=log,stderr=subprocess.STDOUT)
                    job['pid']=p.pid;save()
                    while p.poll() is None:
                        try:check(name+'-monitor',own=True)
                        except Exception as e:job.update(polluted=True,pollution_reason=repr(e));save()
                        time.sleep(5)
                job.update(exit_code=p.returncode,ended=time.time());save()
                check(name+'-after')
                assert p.returncode==0 and not job['polluted'],'failed/polluted '+name
                records=[json.loads(p.read_text()) for p in path.glob('iteration*-rank*.json')]
                masters=[json.loads(p.read_text()) for p in path.glob('full-master-iteration*-rank*.json')]
                assert len(records)==len(masters)==8,'missing full update records'
                assert next(r for r in records if r['rank']==0)['metrics']['valid_tokens']==88064,'actual first8 input mismatch'
                assert all(v['passed'] for r in records+masters for v in r.values() if isinstance(v,dict) and 'passed' in v)
                if mode=='qo-immediate':assert all(any(r['history_executed']['nonempty_parts']>0 for r in masters if r['rank']==rank) for rank in range(8)),'missing actual nonempty H execution'
                job.update(reports=len(records),full_master_reports=len(masters),validated=True);save()
        state.update(status='complete',ended=time.time());save()
    except Exception as e:state.update(status='stopped',error=repr(e),ended=time.time());save();raise
    finally:
        after=subprocess.check_output(['git','diff','--binary'],cwd=prod)
        state['production_patch_unchanged']=hashlib.sha256(after).hexdigest()==state['production_patch_sha256'];save()
if __name__=='__main__':main()
