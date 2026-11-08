"""Bounded native reader validation after the active formal batch closes."""
from pathlib import Path
import array, fcntl, hashlib, json, os, sys, time

root = Path('/workspace/experiments/dspp-cp-v1-20261106')
lock = (root/'numeric.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, str(root/'source-current'))
import torch
torch.set_num_threads(1)
from megatron.data.indexed_dataset import MMapIndexedDataset

folder = root/'data-v2'
metadata = json.loads((folder/'manifest.json').read_text())
protocol = json.loads((root/'workload29-target-protocol.json').read_text())
name = 'arxiv-mistral32k'
case = protocol['cases'][name]
assert metadata['model_vocab'] == case['vocab'] == 32128
assert metadata['zero_pad_collisions'] == 0 and metadata['case'] == case
base = folder/name
binary, index = base.with_suffix('.bin'), base.with_suffix('.idx')
record = folder/(name+'-verified.json')
stat = binary.stat()
idx_hash = hashlib.sha256(index.read_bytes()).hexdigest()
assert idx_hash == case['idx_sha256'] == metadata['idx_sha256']
if record.exists():
    old = json.loads(record.read_text())
    assert old['status'] == 'frozen-verified' and old['bin_size'] == stat.st_size and old['bin_mtime_ns'] == stat.st_mtime_ns and old['idx_sha256'] == idx_hash
    assert not torch.cuda.is_initialized()
    print('Reused native-verified target32128 arxiv identity')
    raise SystemExit(0)
bin_hash = hashlib.sha256(binary.read_bytes()).hexdigest()
assert bin_hash == metadata['bin_sha256']
dataset = MMapIndexedDataset(str(base), skip_warmup=True)
assert len(dataset) == 256 and [int(n)-1 for n in dataset.sizes] == case['training_lengths']
assert max(dataset.sizes) <= 32769
low, high = 2**63-1, -1
for i in range(256):
    values = dataset[i]
    assert len(values) == case['training_lengths'][i]+1
    assert values[0] == 1 and values[-1] == 2
    low, high = min(low, int(values.min())), max(high, int(values.max()))
assert 0 < low <= high < 32000 and not torch.cuda.is_initialized()
assert (stat.st_size, stat.st_mtime_ns) == (binary.stat().st_size, binary.stat().st_mtime_ns)
result = dict(status='frozen-verified', time=time.time(), workload=name,
              bin_sha256=bin_hash, idx_sha256=idx_hash, bin_size=stat.st_size,
              bin_mtime_ns=stat.st_mtime_ns, bin_inode=stat.st_ino,
              bin_resolved_path=str(binary.resolve()), source_metadata=metadata,
              native_reader_verified=True, truncated_documents=0,
              token_range=[low, high], cuda_initialized=False,
              tokenizer_sha256=metadata['tokenizer_sha256'], binary_hash_scope='entire selected real-input binary')
record.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(dict(status=result['status'], documents=256, token_range=[low, high], model_vocab=32128, cuda_initialized=False)))
