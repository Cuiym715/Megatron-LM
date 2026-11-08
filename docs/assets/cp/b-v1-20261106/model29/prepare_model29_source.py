"""One-time isolated source materialization for future29L spawn workers."""
from pathlib import Path
import fcntl,hashlib,json,tarfile,time
ROOT=Path('/workspace/experiments/dspp-cp-v1-20261106');DEST=ROOT/'source-model29'
lock=(ROOT/'numeric.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
archive=Path('/workspace/outputs/dspp-qo-20261105-exec01/candidate-theoretical-validated-source.tar.gz')
expected='abae3c76e26dd6e27fb51ee0d2ccddc3f8eee1feed8ce399f4a4c1c5503c807c'
assert hashlib.sha256(archive.read_bytes()).hexdigest()==expected
assert not DEST.exists();DEST.mkdir()
with tarfile.open(archive) as f:f.extractall(DEST,filter='data')
p=DEST/'megatron/core/dspp_layout.py';old=p.read_text();new=(ROOT/'dspp_layout29.py').read_text()
assert hashlib.sha256(old.encode()).hexdigest()=='d7c70e3ffe5b69f30fa784efac60e6dfd69d6bc34b84a434221c51021cfac24f'
assert hashlib.sha256(new.encode()).hexdigest()=='8e39ba56c0c749b33b501533a97f2e4b48f212239f2693ef63b2a6cf2dd77da2'
p.write_text(new)
proof=dict(status='prepared-isolated-realfile-layout29',time=time.time(),source=str(DEST),base_archive_sha256=expected,only_changed_file='megatron/core/dspp_layout.py',old_sha256=hashlib.sha256(old.encode()).hexdigest(),new_sha256=hashlib.sha256(new.encode()).hexdigest(),reason='SpawnedCPUplanner workers neednative29Llayout file, not onlymainprocessfunctionpatch. Effective29Llayerlogicidenticaltoalreadymeasuredbaselineentry.',production_modified=False)
(ROOT/'model29-source-proof.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
