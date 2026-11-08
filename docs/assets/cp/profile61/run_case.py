"""One bounded CP2 replay, with selected-GPU isolation and no background queue."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--batch', required=True)
parser.add_argument('--gpu-uuids', required=True)
parser.add_argument('--stage', type=int, required=True)
parser.add_argument('--chunk', type=int, required=True)
parser.add_argument('--backend', required=True)
parser.add_argument('--lengths', default='4608')
parser.add_argument('--warmup', type=int, default=0)
parser.add_argument('--repeats', type=int, default=1)
parser.add_argument('--verify', action='store_true')
cfg = parser.parse_args()
assert cfg.batch.replace('-', '').isalnum()
uuids = cfg.gpu_uuids.split(',')
assert len(uuids) == 2 and len(set(uuids)) == 2
lock = (ROOT / 'run.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
out = Path('/workspace/outputs/dspp-cp-profile61-20261106') / cfg.batch
out.mkdir(exist_ok=False)
command = ['/workspace/venvs/megatron/bin/python', '-m', 'torch.distributed.run',
    '--standalone', '--nproc-per-node=2', str(ROOT/'stage_profile.py'),
    '--stage', str(cfg.stage), '--chunk', str(cfg.chunk), '--backend', cfg.backend,
    '--output', str(out/'result'), '--lengths', cfg.lengths,
    '--warmup', str(cfg.warmup), '--repeats', str(cfg.repeats)]
if cfg.verify:
    command.append('--verify')
state = dict(status='checking-idle', pid=os.getpid(), started=time.time(), gpu_uuids=uuids,
    jobs=[], sampling_scope='CP2 stage replay only, PP roles are emulated; no PP traffic',
    formal_timing=not cfg.verify, model_layers=61, pipeline_target=4, cp=2)
def save():
    p = out/'manifest.tmp'
    p.write_text(json.dumps(state, indent=2)+'\n')
    p.replace(out/'manifest.json')
def check(label, running=False):
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu,clocks.sm,temperature.gpu,power.draw', '--format=csv,noheader'], text=True)
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory', '--format=csv,noheader'], text=True)
    selected = [line for line in gpu.splitlines() if line.split(',')[1].strip() in uuids]
    active = [line for line in apps.splitlines() if line.split(',')[0].strip() in uuids]
    with (out/'resource-checks.jsonl').open('a') as f:
        f.write(json.dumps(dict(time=time.time(), label=label, selected_gpus=selected,
                              selected_apps=active, host_load=os.getloadavg()))+'\n')
    assert len(selected) == 2
    if not running:
        assert not active and all(int(row.split(',')[2].split()[0]) <= 512 and
            int(row.split(',')[3].split()[0]) == 0 for row in selected), 'selected GPUs occupied'
    audit = json.loads((out/'host-ownership/manifest.json').read_text())
    assert audit['status'] == 'running' and not audit['unknown_observed'], 'foreign selected-GPU load'
save()
process = None
try:
    for _ in range(100):
        if (out/'host-ownership/manifest.json').exists():
            break
        time.sleep(.1)
    for i in range(3):
        check('idle-'+str(i))
        if i < 2:
            time.sleep(30)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=cfg.gpu_uuids, OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1', MAX_JOBS='1', DSPP_FA_SPLIT_CACHE='/workspace/cache/dspp-fa-split',
        CUDA_DEVICE_MAX_CONNECTIONS='32', NCCL_LAUNCH_ORDER_IMPLICIT='0', NCCL_LAUNCH_RACE_FATAL='1')
    job = dict(command=command, started=time.time(), output=str(out/'result'))
    state['jobs'] = [job]
    state['status'] = 'running'
    with (out/'train.log').open('w') as f:
        process = subprocess.Popen(['timeout', '--kill-after=15', '600', *command],
            env=env, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        job['pid'] = process.pid
        save()
        while process.poll() is None:
            check('running', running=True)
            time.sleep(5)
    job.update(exit_code=process.returncode, ended=time.time())
    assert process.returncode == 0, 'stage replay failed; inspect train.log'
    check('finished')
    for rank in range(2):
        result = json.loads((out/'result'/f'rank{rank}.json').read_text())
        assert result['status'] == 'complete'
    state.update(status='complete', ended=time.time())
except Exception as error:
    if process is not None and process.poll() is None:
        process.terminate()  # Direct timeout child only; no NVML PID signaling.
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
    state.update(status='stopped', error=repr(error), ended=time.time())
    raise
finally:
    save()
