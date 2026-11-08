"""Wait for requested GPUs without touching running jobs; poll every 15min."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    cfg=p.parse_args()
    wanted=set(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    cfg.output.parent.mkdir(parents=True,exist_ok=True)
    announced=False
    while True:
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu',
                                     '--format=csv,noheader,nounits'],text=True)
        rows=[dict(zip(('index','uuid','memory_mib','utilization'),map(str.strip,line.split(','))))
              for line in raw.strip().splitlines()]
        raw_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
                                              '--format=csv,noheader,nounits'],text=True)
        processes=[dict(zip(('uuid','pid'),map(str.strip,line.split(','))))
                   for line in raw_processes.strip().splitlines() if line.strip()]
        selected=[r for r in rows if r['index'] in wanted or r['uuid'] in wanted]
        if len(selected)!=len(wanted):
            raise ValueError('CUDA_VISIBLE_DEVICES does not match the physical GPU inventory')
        active={p['uuid'] for p in processes}
        # NVML utilization is a trailing sample and may remain nonzero after
        # our previous process exits. With no process and zero memory it is
        # not an occupancy signal; keep it in the evidence, not the wait gate.
        busy=[r for r in selected if int(r['memory_mib'])>=100 or r['uuid'] in active]
        with cfg.output.open('a') as f:
            f.write(json.dumps(dict(time=time.time(),gpus=selected,processes=processes,
                                   busy=bool(busy)))+'\n')
        if not busy:
            print('Requested GPUs are free.',flush=True)
            return
        if not announced:
            print('Requested GPUs are occupied; check every 15min. Existing jobs are untouched.',flush=True)
            announced=True
        time.sleep(900)


if __name__=='__main__':
    main()
