"""Local CPU preparation: whole real documents, existing Mistral tokenizer."""
import argparse, array, hashlib, json, struct, sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
p.add_argument('--sentencepiece-path', type=Path, required=True)
cfg = p.parse_args()
sys.path.insert(0, str(cfg.sentencepiece_path))
import sentencepiece as spm

assert sys.byteorder == 'little' and spm.__version__ == '0.2.1'
tokenizer = Path('dspp/archive/handoff/dspp-pp8-handoff-20261030/inputs/tokenizer.model')
source = Path('dspp/dspp-data/slimpajama_arxiv_50k.jsonl')
tokenizer_hash = hashlib.sha256(tokenizer.read_bytes()).hexdigest()
assert tokenizer_hash == 'dadfd56d766715c61d2ef780a525ab43b8e6da4de6865bda3d95fdef5e134055'
sp = spm.SentencePieceProcessor(model_file=str(tokenizer))
assert sp.vocab_size() == 32000 and sp.bos_id() >= 0 and sp.eos_id() >= 0
cfg.output.mkdir(exist_ok=False)
base = cfg.output / 'arxiv-mistral32k'
sizes, pointers, selected, excluded = [], [], [], []
raw_hash = hashlib.sha256()
position = 0
with source.open('rb') as raw, base.with_suffix('.bin').open('xb') as binary, (cfg.output/'selected-source.jsonl').open('xb') as evidence:
    for doc_id, line in enumerate(raw):
        doc = json.loads(line)
        ids = sp.encode(doc['text'], out_type=int, add_bos=True, add_eos=True, enable_sampling=False)
        if not 2 <= len(ids) <= 32769:
            excluded.append(dict(document=doc_id, tokens=len(ids)))
            continue
        assert min(ids) >= 0 and max(ids) < 32000
        values = array.array('H', ids)
        binary.write(values.tobytes())
        sizes.append(len(ids))
        pointers.append(position)
        position += 2 * len(ids)
        selected.append(doc_id)
        evidence.write(line)
        raw_hash.update(struct.pack('<QQ', doc_id, len(line)))
        raw_hash.update(line)
        if len(sizes) == 256:
            break
assert len(sizes) == 256
with base.with_suffix('.idx').open('xb') as f:
    f.write(b'MMIDIDX\0\0' + struct.pack('<QBQQ', 1, 8, len(sizes), len(sizes)+1))
    for code, rows in [('i', sizes), ('q', pointers), ('q', range(len(sizes)+1))]:
        f.write(array.array(code, rows).tobytes())
lengths = [n-1 for n in sizes]
case = dict(vocab=32128, input_token_vocabulary=32000, documents=256,
            training_lengths=lengths, expected_valid_tokens_per_update=[sum(lengths[i:i+8]) for i in range(0,200,8)],
            timed_valid_tokens=sum(lengths[40:200]), truncated_documents=0,
            idx_sha256=hashlib.sha256(base.with_suffix('.idx').read_bytes()).hexdigest())
assert len(case['expected_valid_tokens_per_update']) == 25
meta = dict(status='local-real-input-prepared-not-GPU-launched', name=base.name,
            case=case, source=str(source), source_size=source.stat().st_size,
            selected_document_indices=selected, excluded_whole_documents=excluded,
            raw_selected_text_sha256=raw_hash.hexdigest(), raw_hash_scope='docID, rawline length, rawline bytes for selected whole docs',
            tokenizer=str(tokenizer), tokenizer_sha256=tokenizer_hash, sentencepiece=spm.__version__,
            add_bos=True, add_eos=True, sampling=False, model_vocab=32128,
            bin_sha256=hashlib.sha256(base.with_suffix('.bin').read_bytes()).hexdigest(),
            idx_sha256=case['idx_sha256'], bin_size=position, tokens=sum(sizes),
            numerical_equivalence_certified=False)
(cfg.output/'manifest.json').write_text(json.dumps(meta, indent=2)+'\n')
protocol = json.loads(Path(__file__).with_name('workload29-protocol.json').read_text())
protocol.update(status='prepared-target32128-real-arxiv-not-launched', cases={base.name:case},
                actual_domain_gate='Unchanged29L/H2048/FFN5504/Q16/modelvocab32128; whole real arXiv retokenized with pinned existing Mistral32000. No text truncation or ID remapping.',
                after_first_batch='Review16real-input windows, then continue remaining b profiles/capacity/templateB and c-f.')
protocol['first_batch']['kv_heads'] = [16, 1]
(cfg.output/'workload29-target-protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
print(json.dumps(dict(documents=256, max_training_length=max(lengths), total_binary_bytes=position,
                      timed_valid_tokens=case['timed_valid_tokens'], excluded_whole_documents=len(excluded),
                      tokenizer_vocabulary=32000, model_vocabulary=32128)))
