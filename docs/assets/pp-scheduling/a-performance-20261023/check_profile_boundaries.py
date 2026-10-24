"""Inspect the existing physical-stage cost key without changing the profile."""
import collections
import json
from pathlib import Path
import statistics

root = Path(__file__).resolve().parent
groups = collections.defaultdict(list)
for path in (root/'calibration').glob('seed*/timeline/*rank_*.json'):
    for event in json.loads(path.read_text())['events']:
        if event.get('items'):
            groups[event['stage'], event['kind'][0], event['signature'], event['chunk']].append(event['duration_ms'])
rows = []
for stage in range(3):
    for op in 'FBW':
        ratios = []
        for s, k, signature, chunk in groups:
            if (s, k, chunk) == (stage, op, 0) and (s, k, signature, 1) in groups:
                ratios.append(statistics.median(groups[s,k,signature,1]) /
                              statistics.median(groups[s,k,signature,0]))
        if ratios:
            rows.append(dict(stage=stage, op=op, matched_signatures=len(ratios),
                             median_chunk1_over_chunk0=statistics.median(ratios),
                             minimum=min(ratios), maximum=max(ratios)))
report = dict(observation='Current from_records groups (physical stage, F/B/W, signature), dropping virtual chunk. Runtime predict_detailed receives physical stage/kind/meta with no virtual chunk. Exact signature confidence does not distinguish embedding/head boundary.',
              paired_same_signature_cost_ratios=rows, p2p_profile=None,
              p2p_fallback=dict(bandwidth_GB_per_second=25, latency_microseconds=20),
              limitation='This proves cost-key aggregation and observed boundary asymmetry; it does not isolate the causal share of wall-time regression. No production/profile changes made during comparison.')
(root/'profile-boundary-diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
