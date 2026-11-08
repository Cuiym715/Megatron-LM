#!/usr/bin/env python3
"""Freeze versioned real-stage costs and whole-shape holdout diagnostics.

Example: python tools/calibrate_dspp_stage.py --input fused/timeline split/timeline
  --output stage-profile.json --report holdout.json
Use warmed iterations only. Never add current/future measurements to a profile
already bound to a performance comparison. Memory audits remain separate.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from megatron.core.datasets.dspp_calibration import StageCalibration, holdout_report, timeline_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--cp-max', action='store_true',
                        help='Use the slowest CP lane for each real stage task sample.')
    args = parser.parse_args()
    records, sources, lanes = [], [], defaultdict(list)
    for source in args.input:
        path = Path(source)
        paths = sorted(path.glob('iteration_*_rank_*.json')) if path.is_dir() else [path]
        for path in paths:
            payload = json.loads(path.read_text())
            rows = list(timeline_records(payload))
            if args.cp_max:
                for row in rows:
                    key = (str(source), payload['iteration'], row['stage'], row['chunk'],
                           row['kind'], row['logical_microbatch'], row['physical_microbatch'])
                    lanes[key].append(row)
            else:
                records.extend(rows)
            sources.append(dict(path=str(path), records=len(rows)))
    if args.cp_max:
        for rows in lanes.values():
            cp = rows[0]['context']['model']['context_parallel_size']
            if len(rows) != cp:
                raise ValueError(f'incomplete CP timing sample: expected {cp}, got {len(rows)}')
            records.append(max(rows, key=lambda row: row['duration_ms']))
    if not records:
        raise ValueError('no complete-stage calibration samples')
    model = StageCalibration(records)
    for path, value in ((args.output, model.to_dict()), (args.report, dict(
            sources=sources, version=model.version, validation=holdout_report(records)))):
        output = Path(path)
        if output.exists():
            raise FileExistsError(f'refusing to overwrite frozen artifact {output}')
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(value, indent=2, allow_nan=False))
    print(json.dumps(dict(records=len(records), groups=len(model.groups), version=model.version)))


if __name__ == '__main__':
    main()
