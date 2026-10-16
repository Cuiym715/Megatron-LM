"""Record source/evidence hashes after the plan and summaries are finalized."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
root=Path(__file__).resolve().parent
repo=root.parents[3]
prior=json.loads((root.parent/'011-workload-scaling-20261016/manifest.json').read_text())
def digest(p):
    b=p.read_bytes();return {'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
files=sorted(prior['source_files'])
sources={f:digest(repo/f) for f in files}
with tarfile.open(root/'source-snapshot.tar.gz','w:gz') as tar:
    for f in files:tar.add(repo/f,arcname=f)
inputs={}
for prefix in ('/home/cuiym/.cache/dspp-fa-split/clean-timing-measure-final-20261016/data',
               '/home/cuiym/.cache/dspp-fa-split/workload-scaling-20261016/long-data'):
    for ext in ('.json','.bin','.idx'):
        p=Path(prefix+ext);inputs[str(p)]=digest(p)
        if ext=='.json':(root/(p.parent.name+'-data.json')).write_bytes(p.read_bytes())
unchanged=[f for f in sources if sources[f]==prior['source_files'][f]]
evidence={str(p.relative_to(root)):digest(p) for p in sorted(root.rglob('*'))
          if p.is_file() and p.name!='manifest.json'}
manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),
 git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
 environment=prior['environment'],
 scope='Actual PP3/VPP2 complete B/H scaling with width and longest sequence. No production algorithm changes; no end-to-end speedup claim.',
 source_files=sources,source_files_unchanged_from_011=unchanged,
 external_inputs=inputs,evidence_files=evidence,
 commands=['python3 '+str(root/'run_matrix.py'),'python3 '+str(root/'compare_theory.py'),
           'python3 '+str(root/'summarize_scaling.py'),
           "docker exec dspp-width4096 /workspace/src/venvs/megatron/bin/python -m pytest -q -s -p no:cacheprovider tests/unit_tests/dspp/test_fa_combined_current.py -k 'combined_long and 32-16'"],
 validation=['Two new 32/16-head independent attention numerical comparisons; dQ/current/history gradients all checked.',
 'Only complete B/H boundary events in formal training; profiler and correctness tests separate.',
 'One pair/config; no repeat because all preselected paired B samples improve with separated ranges. Samples in a run are not independent runs.',
 'All matched layouts and B/H ordering checks pass. Per-pair loss matches at log precision, not a substitute for full gradient validation.',
 'All stage/layout samples retained; task work summed across ranks is not iteration time.',
 'No git add/commit/push; archives001–011 preserved.'])
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Archived',len(sources),'source files and',len(evidence),'evidence files')
