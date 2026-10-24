#!/usr/bin/env python3
"""Freeze versioned real-stage costs and whole-shape holdout diagnostics.

Example: python tools/calibrate_dspp_stage.py --input fused/timeline split/timeline
  --output stage-profile.json --report holdout.json
Use warmed iterations only. Never add current/future measurements to a profile
already bound to a performance comparison. Memory audits remain separate.
"""
import argparse
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
    args = parser.parse_args()
    records, sources = [], []
    for source in args.input:
        path = Path(source)
        paths = sorted(path.glob('iteration_*_rank_*.json')) if path.is_dir() else [path]
        for path in paths:
            rows = list(timeline_records(json.loads(path.read_text())))
            records.extend(rows)
            sources.append(dict(path=str(path), records=len(rows)))
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
