"""One uncompressed metadata transfer after a formal batch closes."""
import argparse, fcntl, json, re, tarfile
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--batch', required=True)
cfg = p.parse_args()
assert re.fullmatch('[a-z0-9-]+', cfg.batch)
root = Path('/home/test/test01/tinacui/experiments/dspp-cp-v1-20261106')
lock = (root/'numeric.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
folder = Path('/home/test/test01/tinacui/outputs/dspp-cp-v1-20261106')/cfg.batch
state = json.loads((folder/'manifest.json').read_text())
audit = json.loads((folder/'host-ownership/manifest.json').read_text())
assert state['status'] == 'complete' and state['production_patch_unchanged']
assert audit['status'] == 'complete' and not audit['unknown_observed']
assert all(j['exit_code'] == 0 and not j['polluted'] for j in state['jobs'])
archive = root/(cfg.batch+'-evidence.tar')
assert not archive.exists(), 'Evidence already packed; reuse existing transfer'
files = [folder/'manifest.json', folder/'resource-checks.jsonl', folder/'host-ownership/manifest.json']
for job in state['jobs']:
    files.extend(sorted((folder/job['name']).rglob('*.json')))
assert all(p.is_file() and p.resolve().is_relative_to(folder.resolve()) for p in files)
with tarfile.open(archive, 'x') as output:
    for path in files:
        output.add(path, arcname=str(path.relative_to(folder)), recursive=False)
print(json.dumps(dict(path=str(archive), files=len(files), bytes=archive.stat().st_size,
                      compression=False, original_logs_retained=str(folder))))
