"""Same-backend NCCL-setting comparison; no old KV/QO error investigation."""
import os,runpy
from pathlib import Path
import torch
assert tuple(torch.cuda.nccl.version()) >= (2,26,0)
assert os.environ['NCCL_LAUNCH_ORDER_IMPLICIT'] in ('0','1')
assert os.environ['NCCL_LAUNCH_RACE_FATAL']=='1'
runpy.run_path(str(Path(__file__).with_name('numeric_entry_v4.py')),run_name='__main__')
