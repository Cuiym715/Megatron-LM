"""Same split algorithm and workload: committed baseline -> no-history -> both.

Timing-only B/H runs and uninstrumented end-to-end runs are separate. Run on
host with dspp-opt015 alive. Launches are sequential to avoid GPU contention.
"""
import argparse
import json
from pathlib import Path
import subprocess

p=argparse.ArgumentParser();p.add_argument('--phase',choices=['tasks','e2e'],required=True)
p.add_argument('--suffix',default='');p.add_argument('--reverse',action='store_true')
args=p.parse_args()
root=Path(__file__).resolve().parent
croot='/workspace/src/Megatron-LM-kwai/docs/assets/fa-split/'+root.name
versions={'baseline':'/workspace/.cache/dspp-fa-split/opt015-baseline',
          'nohistory':'/workspace/.cache/dspp-fa-split/opt015-nohistory',
          'both':'/workspace/src/Megatron-LM-kwai'}
order=list(versions)
if args.phase=='e2e':order.reverse()
if args.reverse:order.reverse()
settings=dict(TORCH_CUDA_ARCH_LIST='8.9',TRAIN_ITERS='3' if args.phase=='tasks' else '6',
 HIDDEN_SIZE='4096',FFN_HIDDEN_SIZE='8192',NUM_ATTENTION_HEADS='32',NUM_QUERY_GROUPS='16',
 NUM_LAYERS='6',LAYERS_PER_VPP='1',SEQ_LENGTH='16384',MICRO_SEQ_LENGTH='2048',
 MICRO_BATCH_SIZE='4',GLOBAL_BATCH_SIZE='8',DSPP_DEBUG_BATCHES='0',LEARNING_RATE='0',MIN_LEARNING_RATE='0',
 DATA_PATH='/workspace/.cache/dspp-fa-split/workload-scaling-20261016/long-data',
 DSPP_TASK_TIMING_ONLY='1',DSPP_SPLIT_ATTENTION_BACKWARD='1',DSPP_HISTORY_DKV_AFTER_DX='1',
 DSPP_TIMELINE_ITERATION='3' if args.phase=='tasks' else '0',
 PYTHON_BIN='/workspace/src/venvs/megatron/bin/python',TRAIN_ENTRYPOINT='pretrain_llama.py')
runname=args.phase+args.suffix
out=root/runname
out.mkdir(exist_ok=False)
(root/(runname+'-configuration.json')).write_text(json.dumps(dict(phase=args.phase,order=order,versions=versions,settings=settings),indent=2)+'\n')
for name in order:
 d=out/name;d.mkdir()
 env=dict(settings,DSPP_TIMELINE_DIR=croot+'/'+runname+'/'+name+'/timeline' if args.phase=='tasks' else '')
 cmd=['docker','exec','-w',versions[name],'dspp-opt015','env',*[f'{k}={v}' for k,v in env.items()],
      'timeout','360','bash','examples/run_dspp_b23_l40.sh']
 print('Starting',runname,name,flush=True)
 with (d/'train.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
 print('Finished',runname,name,flush=True)
