"""Measurement bootstrap: load the archived pre-fusion CP operator only.

Run from the experiment source root, through examples/run_dspp_cp.sh with
DSPP_CP_ENTRYPOINT pointing here. No profile is reused across implementations.
"""
import importlib.util
import os
from pathlib import Path
import runpy
import sys
import tarfile

def main():
    root=Path.cwd()
    sys.path.insert(0,str(root))
    archive=Path(os.environ['DSPP_CP_BASELINE_ARCHIVE'])
    name='megatron.core.kv_cache.dspp_cp_attention'
    with tarfile.open(archive) as f:
        source=f.extractfile('megatron/core/kv_cache/dspp_cp_attention.py').read()
    spec=importlib.util.spec_from_loader(name,loader=None,origin=str(archive))
    module=importlib.util.module_from_spec(spec)
    module.__file__=str(archive)
    sys.modules[name]=module
    exec(compile(source,str(archive),'exec'),module.__dict__)
    from megatron.core.datasets import dspp_cp_layout
    dspp_cp_layout.CP_IMPLEMENTATION='owner-partials-send-progress-unfused-v0'
    runpy.run_path(str(root/'tools/benchmark_dspp_cp.py'),run_name='__main__')

if __name__=='__main__':
    main()
