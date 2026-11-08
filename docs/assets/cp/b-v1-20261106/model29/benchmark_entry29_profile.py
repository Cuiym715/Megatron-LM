"""Reuse pinned split-kernel cache paths with the isolated29L source copy."""
import os,sys,runpy,hashlib
from pathlib import Path
if '--warm-cache-only' in sys.argv:
    import fcntl
    queue_lock=Path('/workspace/experiments/dspp-cp-v1-20261106/numeric.lock').open('w')
    fcntl.flock(queue_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
sys.path.insert(0,os.environ['CP_V1_SOURCE'])
from megatron.core.kv_cache import fa_split_build
original=Path('/workspace/experiments/dspp-cp-v1-20261106/source-current/megatron/core/kv_cache/fa_split_src')
for name in ['backward.cu','bindings.cpp','flash_bwd_split.patch','upstream.json']:
    assert hashlib.sha256((fa_split_build.NATIVE/name).read_bytes()).digest()==hashlib.sha256((original/name).read_bytes()).digest()
# Exact same bytes/ABI/cache key; retain original source paths to avoid Ninja
# recompiling unchanged kernels merely because the source directory changed.
fa_split_build.NATIVE=original
if '--warm-cache-only' in sys.argv:
    fa_split_build.load_extension(arch=(8,0))
    import torch
    assert not torch.cuda.is_initialized()
else:
    runpy.run_path(str(Path(__file__).with_name('benchmark_entry29.py')),run_name='__main__')
