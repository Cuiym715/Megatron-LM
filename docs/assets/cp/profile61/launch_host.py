"""Attach the existing host ancestry auditor, scoped to selected CP GPUs."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
argv = sys.argv[1:]
batch = argv[argv.index('--batch')+1]
assert batch.replace('-', '').isalnum()
out = Path('/home/test/test01/tinacui/outputs/dspp-cp-profile61-20261106') / batch
assert not out.exists(), 'Follow existing batch; never duplicate it'
container = '/workspace/experiments/dspp-cp-profile61-20261106'
command = ['docker', 'exec', 'tinacui-megatron', '/workspace/venvs/megatron/bin/python',
           '-u', container+'/run_case.py', *argv]
with (ROOT/(batch+'-controller.log')).open('x') as log:
    child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
for _ in range(150):
    if (out/'manifest.json').exists():
        break
    if child.poll() is not None:
        raise RuntimeError((ROOT/(batch+'-controller.log')).read_text())
    time.sleep(.1)
assert (out/'manifest.json').exists()
with (ROOT/(batch+'-audit.log')).open('x') as log:
    audit = subprocess.Popen(['python3', str(ROOT/'host_ownership_audit.py'), str(out)],
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
state = dict(host_controller_pid=child.pid, host_audit_pid=audit.pid, started=time.time(),
             command=command, output=str(out))
(ROOT/(batch+'-launch.json')).write_text(json.dumps(state, indent=2)+'\n')
print(json.dumps(state))
