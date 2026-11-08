"""Offline CPU calibration with the same29L layout; never alongside GPU timing."""
import os,sys,runpy,fcntl
from pathlib import Path
queue_lock=Path('/workspace/experiments/dspp-cp-v1-20261106/numeric.lock').open('w')
fcntl.flock(queue_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
os.environ['CUDA_VISIBLE_DEVICES']=''
root=Path(os.environ.get('CP_V1_SOURCE','/workspace/experiments/dspp-cp-v1-20261106/source-current'))
sys.path.insert(0,str(root))
from megatron.core import dspp_layout
replacement=runpy.run_path(str(Path(__file__).with_name('dspp_layout29.py')))
dspp_layout.logical_layer_ranges.__code__=replacement['logical_layer_ranges'].__code__
allowed={'calibrate_dspp_stage.py','calibrate_dspp_cp_memory.py'}
tool=sys.argv[1];assert tool in allowed
sys.argv=[str(root/'tools'/tool)]+sys.argv[2:]
runpy.run_path(sys.argv[0],run_name='__main__')
import torch
assert not torch.cuda.is_initialized()
