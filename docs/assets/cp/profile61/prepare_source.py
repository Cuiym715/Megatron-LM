"""CPU-only preparation, never modify production or prior experiment sources."""
import fcntl
import json
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parent
OLD = Path('/workspace/experiments/dspp-cp-v1-20261106')
C61 = Path('/workspace/experiments/dspp-e2e-first-20261106')
lock = (OLD / 'numeric.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
source = ROOT / 'source'
proof = ROOT / 'preparation.json'
if proof.exists():
    print(proof.read_text())
    raise SystemExit(0)
assert not source.exists(), 'Inspect interrupted preparation before retrying'
shutil.copytree(OLD / 'source-current', source,
                ignore=shutil.ignore_patterns('.git', '__pycache__', '*.pyc'))
shutil.copy2(C61 / 'dspp_layout61.py', source / 'megatron/core/dspp_layout.py')
shutil.copy2(C61 / 'dspp-commands.json', ROOT / 'target-commands.json')
shutil.copy2(C61 / 'zero_dropout.py', ROOT / 'zero_dropout.py')
proof.write_text(json.dumps(dict(status='CPU-prepared', time=time.time(), source=str(source),
    base_source=str(OLD / 'source-current'), changed_source_files=['megatron/core/dspp_layout.py'],
    target_cpu_proof=json.loads((C61 / 'CPU-preflight.json').read_text()),
    GPU_tested=False), indent=2) + '\n')
print(proof.read_text())
