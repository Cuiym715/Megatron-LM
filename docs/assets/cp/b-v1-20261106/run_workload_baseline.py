"""Prepared sixteen clean current-KV/QO complete-update baseline windows.

User authorized timing the current version with unresolved numerical errors.
This does not certify numerical correctness. Never run CPU analysis,
archiving or diagnostics beside this batch. No automatic retry after failure.
"""
import argparse,fcntl,hashlib,json,os,re,subprocess,time
from pathlib import Path
ROOT=Path('/workspace/experiments/dspp-cp-v1-20261106');SRC=ROOT/'source-current'
NUM=Path('/workspace/outputs/dspp-cp-v1-20261106/numeric-v5-longmix512')
parser=argparse.ArgumentParser()
parser.add_argument('--case',required=True,choices=['short-packed-v512','short-tail-v512','arxiv-native32k'])
parser.add_argument('--batch',required=True)
CFG=parser.parse_args();assert re.fullmatch(r'[a-z0-9-]+',CFG.batch)
OUT=Path('/workspace/outputs/dspp-cp-v1-20261106')/CFG.batch
DATA=ROOT/'data-v1'/CFG.case
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    lock=(ROOT/'numeric.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    review=json.loads((ROOT/'b-baseline-timing-review.json').read_text())
    assert review['status']=='timing-reviewed-numerical-errors-unresolved' and len(review['windows'])==16
    authorization=json.loads((ROOT/'performance-authorization-20261106.json').read_text())
    assert authorization['user_deferred_numerical_check'] and not authorization['numerically_certified']
    assert authorization['scope']=='continue-entire-master-plan-a-through-f'
    numeric=json.loads((NUM/'manifest.json').read_text());assert numeric['status']=='stopped'
    full_source=json.loads((ROOT/'current-source-verification.json').read_text());assert full_source['status']=='passed'
    protocol=json.loads((ROOT/'workload-baseline-protocol.json').read_text())
    case=protocol['cases'][CFG.case]
    identity=json.loads((ROOT/'data-v1'/f'{CFG.case}-verified.json').read_text())
    assert identity['status']=='frozen-verified' and identity['idx_sha256']==case['idx_sha256']
    assert digest(DATA.with_suffix('.idx'))==identity['idx_sha256']
    # Binary hashed once in CPU preparation. Cheap stat guard thereafter.
    st=DATA.with_suffix('.bin').stat()
    assert st.st_size==identity['bin_size'] and st.st_mtime_ns==identity['bin_mtime_ns']
    source=numeric['source']['sources']['current'];assert source['archive_sha256']==full_source['archive_sha256']
    assert review['source']==source and review['runtime_entry_sha256']==digest(ROOT/'benchmark_entry_persistent.py')
    for n,h in source['required_sha256'].items():assert digest(SRC/n)==h

    OUT.mkdir(exist_ok=False);(OUT/'scripts').mkdir()
    for n in ['run_workload_baseline.py','benchmark_entry_persistent.py','benchmark_entry_warmup.py','zero_dropout.py','workload-baseline-protocol.json','performance-authorization-20261106.json']:(OUT/'scripts'/n).write_bytes((ROOT/n).read_bytes())
    prod=Path('/workspace/projects/Megatron-LM-kwai');patch=subprocess.check_output(['git','diff','--binary'],cwd=prod)
    (OUT/'production-before.patch').write_bytes(patch)
    uuids=numeric['gpu_uuids']
    state=dict(status='checking-idle',pid=os.getpid(),started=time.time(),jobs=[],gpu_uuids=uuids,
        source=source,arithmetic=protocol['arithmetic'],data_identity=identity,workload=CFG.case,workload_config=case,
        numerical_status='known-errors-unresolved-user-authorized-performance',numerically_certified=False,authorization=authorization,
        actual_domain=dict(training_lengths=case['training_lengths'][:8],valid_tokens=sum(case['training_lengths'][:8]),vocab=case['vocab'],truncated_documents=0),protocol=protocol,protocol_sha256=digest(ROOT/'workload-baseline-protocol.json'),
        scripts_sha256={n:digest(ROOT/n) for n in ['run_workload_baseline.py','benchmark_entry_persistent.py','benchmark_entry_warmup.py','zero_dropout.py']},
        production_patch_sha256=hashlib.sha256(patch).hexdigest(),scope='currentKV/QO fused templates, no D/H/no B/no prefetch, actual online full training updates')
    def save():
        p=OUT/'manifest.tmp';p.write_text(json.dumps(state,indent=2)+'\n');p.replace(OUT/'manifest.json')
    def check(label,own=False,require_auditor=False):
        p=OUT/'host-ownership/manifest.json'
        if require_auditor:assert p.exists(),'host ownership auditor must attach before GPU launch'
        if p.exists():
            a=json.loads(p.read_text());assert not a['unknown_observed'] and a['status']=='running','host ownership audit unavailable/unclean'
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu,clocks.sm,temperature.gpu,power.draw','--format=csv,noheader'],text=True)
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader'],text=True)
        with (OUT/'resource-checks.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),label=label,gpus=gpu,apps=apps,host_load=os.getloadavg()))+'\n')
        assert len(gpu.splitlines())==8 and all(u in gpu for u in uuids)
        if own:assert all(sum(u in r for r in apps.splitlines())<=1 for u in uuids),'extra process'
        else:
            assert not apps.strip(),'occupied compute PID'
            assert all(int(r.split(',')[2].split()[0])<=512 and int(r.split(',')[3].split()[0])==0 for r in gpu.splitlines()),'not idle'
    save();base=list(numeric['jobs'][0]['command'])
    base[base.index(str(ROOT/'numeric_entry_v4.py'))]=str(ROOT/'benchmark_entry_persistent.py')
    base[base.index('--data-path')+1]=str(DATA)
    base[base.index('--vocab-size')+1]=str(case['vocab'])
    base[base.index('--seq-length')+1]='32768'
    base[base.index('--max-position-embeddings')+1]='32768'
    for flag,value in {'--train-iters':'25','--log-interval':'1000'}.items():base[base.index(flag)+1]=value
    assert '--dspp-split-attention-backward' not in base and '--dspp-prefetch' not in base
    env={k:v for k,v in os.environ.items() if not k.startswith(('DSPP_','CP_V1_'))}
    env.update(CUDA_VISIBLE_DEVICES=','.join(uuids),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_DEVICE_MAX_CONNECTIONS='32',
        CP_V1_SOURCE=str(SRC),CP_V1_FULL_REDUCTION='1',CP_V1_DETERMINISTIC_FA='1',
        DSPP_CP_BENCH_WARMUP='5',DSPP_CP_BENCH_TIMED='20',DSPP_FA_SPLIT_CACHE='/workspace/cache/dspp-fa-split')
    base+=['--distributed-timeout-minutes','2']
    env.update(NCCL_LAUNCH_ORDER_IMPLICIT='0',NCCL_LAUNCH_RACE_FATAL='1')
    expected_inputs=None;expected_tokens=None
    try:
        for heads in ['mha','gqa1']:
            for i,backend in enumerate(protocol['first_batch']['order_per_head_group'],1):
                assert backend in ['KV','QO'];name=f'{heads}-{i:02d}-{backend}';p=OUT/name
                assert digest(ROOT/'workload-baseline-protocol.json')==state['protocol_sha256']
                for n,h in state['scripts_sha256'].items():assert digest(ROOT/n)==h
                for n,h in source['required_sha256'].items():assert digest(SRC/n)==h
                for k in range(3):
                    check(name+'-idle-'+str(k),require_auditor=(k==2))
                    if k<2:time.sleep(30)
                p.mkdir();command=list(base);command[command.index('--kaimm-context-parallel-impl')+1]='key-value' if backend=='KV' else 'query-out'
                if heads=='gqa1':command+=['--group-query-attention','--num-query-groups','1']
                job=dict(name=name,started=time.time(),command=command,output=str(p),polluted=False,numerically_certified=False)
                state['jobs'].append(job);state['status']='running';save()
                jobenv=dict(env,DSPP_CP_BENCH_OUTPUT=str(p))
                (p/'command.json').write_text(json.dumps(dict(argv=command,env={k:v for k,v in jobenv.items() if k.startswith(('DSPP_','CP_V1_','CUDA_','OMP_','MKL_','NCCL_'))}),indent=2)+'\n')
                with (p/'train.log').open('w') as log:
                    process=subprocess.Popen(['timeout','--kill-after=15','300',*command],cwd=SRC,env=jobenv,stdout=log,stderr=subprocess.STDOUT)
                    job['pid']=process.pid;save()
                    while process.poll() is None:
                        try:check(name+'-monitor',own=True)
                        except Exception as e:
                            job.update(polluted=True,pollution_reason=repr(e));save()
                            # Only our unreaped direct Popen supervisor.
                            if process.poll() is None:process.terminate()
                        time.sleep(5)
                job.update(exit_code=process.returncode,ended=time.time());save();check(name+'-after')
                assert process.returncode==0 and not job['polluted'],'failed/polluted '+name
                ranks=[json.loads((p/f'rank{r}.json').read_text()) for r in range(8)]
                wrappers=[json.loads((p/f'wrapper-rank{r}.json').read_text()) for r in range(8)]
                assert all(not r['diagnostic'] and len(r['records'])==25 and r['timed']==20 and r['warmup']==5 and r['window_ms']>0 for r in ranks)
                assert all(w['deterministic_fa_backward'] and not w['bf16_reduced_precision_reduction'] and w['all_parameters_finite'] and w['all_losses_finite'] for w in wrappers)
                policies=[json.loads((p/f'launch-policy-rank{r}.json').read_text()) for r in range(8)]
                assert all(v['policy']=='nccl-enqueue-mutex-native-order-wait-outside-v1' and v['implicit_order']=='0' and v['race_fatal']=='1' and not v['CPU_wait_sampling'] for v in policies)
                inputs=wrappers[0]['inputs'];assert len(inputs)>=25
                tokens=[r['metrics']['valid_tokens'] for r in ranks[0]['records']]
                assert len(tokens)==25 and tokens==case['expected_valid_tokens_per_update'] and all(t>0 for t in tokens)
                if expected_inputs is None:expected_inputs=inputs;expected_tokens=tokens
                assert inputs==expected_inputs and tokens==expected_tokens,'input identity changed'
                assert all(not r['selected'] for rank in ranks for r in rank['records']),'unexpected D/H'
                assert sum(tokens[5:])==case['timed_valid_tokens']
                patches=[json.loads((p/f'stream-cache-patch-rank{r}.json').read_text()) for r in range(8)]
                caches=[json.loads((p/f'stream-cache-final-rank{r}.json').read_text()) for r in range(8)]
                assert len({v['original_sha256'] for v in patches})==1 and len({v['patched_sha256'] for v in patches})==1
                assert all(len(v)==(2 if rank//2 in (0,3) else 4) for rank,v in enumerate(caches))
                job.update(timing_window_validated=True,validated_scope='completed online updates and matched inputs, not numerical equivalence',timed_valid_tokens=sum(tokens[5:]),
                    tokens_per_second=sum(tokens[5:])/(max(r['window_ms'] for r in ranks)/1000),
                    peak_allocated_bytes=max(r['allocated_peak'] for r in ranks),peak_reserved_bytes=max(r['reserved_peak'] for r in ranks))
                save()
        state.update(status='complete',ended=time.time());save()
    except Exception as e:state.update(status='stopped',error=repr(e),ended=time.time());save();raise
    finally:
        state['production_patch_unchanged']=hashlib.sha256(subprocess.check_output(['git','diff','--binary'],cwd=prod)).hexdigest()==state['production_patch_sha256'];save()
if __name__=='__main__':main()
