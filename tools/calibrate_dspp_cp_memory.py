"""Freeze conservative CP v3 allowances from warmed owner audits.

Offline only. Max over both CP lanes, analytical KV/capture floors, 15% state
margin and explicit workspace/reserve. These are measured allowances, not a
certified bound on future allocator behavior.
"""
import argparse
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibrate_dspp_memory_lifetimes import audit_rows, partition
from megatron.core.pipeline_parallel.dspp_memory_lifetimes import ACCOUNTING, stage_attention_sizes

MIB = 1 << 20


def rounded(value):
    return max(MIB, math.ceil(value*1.15/MIB)*MIB)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark', type=Path, required=True)
    p.add_argument('--owners', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    cfg = p.parse_args()
    if cfg.output.exists():
        raise FileExistsError(cfg.output)
    bench = json.loads(next(cfg.benchmark.glob('rank*.json')).read_text())
    layout = bench['layout']
    args = SimpleNamespace(**layout['model'])
    hq = args.num_attention_heads
    hk = args.num_query_groups if args.group_query_attention else hq
    physical = bench['contexts']['0']['hardware']['total_memory']
    stages, evidence = [], []
    for stage in range(layout['pp']):
        maxima = {k: [0, 0] for k in ('activation', 'weight_forward', 'weight_backward', 'capture')}
        resident = workspace = 0
        paths = sorted(cfg.owners.rglob(f'*_stage_{stage}.json'))
        if not paths:
            raise ValueError(f'missing stage {stage} owner audit')
        for path in paths:
            data = json.loads(path.read_text())
            resident = max(resident, data['baseline_allocated_bytes'])
            workspace = max(workspace, data['uncovered_peak_increment_bytes'])
            forwards, weights = audit_rows(data)
            for tag, forward in forwards.items():
                values = partition(forward, weights[tag], (hq, hk), layout['chunk_size'])
                for key, value in values.items():
                    maxima[key][int(tag[0])] = max(maxima[key][int(tag[0])], value)
            evidence.append(dict(path=str(path), allocated_peak=data['allocated_peak_bytes'],
                                 uncovered=data['uncovered_peak_increment_bytes']))
        kv = []
        for chunk in range(2):
            capture, coefficient = stage_attention_sizes(args, layout['pp'], stage, chunk, layout['chunk_size'])
            maxima['capture'][chunk] = max(maxima['capture'][chunk], capture)
            kv.append(math.ceil(coefficient*1.15))
        row = {key+'_bytes': [rounded(v) for v in values] for key, values in maxima.items()}
        row.update(resident_bytes=rounded(resident), workspace_bytes=max(256*MIB, rounded(workspace)),
                   physical_bytes=physical, reserve_bytes=2*(1 << 30),
                   attention_bytes_per_key_token=kv, pending_bytes_per_token=kv)
        row['weight_bytes'] = [a+b for a,b in zip(row['weight_forward_bytes'],row['weight_backward_bytes'])]
        stages.append(row)
    value = dict(schema='dspp-counted-memory-v3', accounting=ACCOUNTING, layout=layout,
                 device_name=bench['contexts']['0']['hardware']['gpu'],
                 dynamic_growth=.10, stages=stages, certified=False,
                 source='max CP-lane warmed owner partitions; analytical floors; 15% state margin; '
                        '256MiB minimum workspace and 2GiB reserve; no CP division of global history KV',
                 calibration=dict(evidence=evidence, limits='measured layouts only; prepare a new profile for new configurations'))
    cfg.output.parent.mkdir(parents=True,exist_ok=True)
    cfg.output.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(stages))


if __name__ == '__main__':
    main()
