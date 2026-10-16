#!/usr/bin/env python3
"""Fit a directed P2P profile from simple JSON records.

Input may be a JSON list or ``{"records": [...]}``, with each row containing
``bytes``, ``duration_ms``, and optional ``direction``/``overlap`` fields.
"""

import argparse
import json
from pathlib import Path

from megatron.core.datasets.dspp_p2p import DsppP2PCostModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = Path(args.input)
    if path.suffix == ".jsonl":
        records = []
        seen = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            snapshot = json.loads(line)
            for row in snapshot.get("p2p", []):
                if "duration_ms" not in row or row.get("token") in seen:
                    continue
                seen.add(row.get("token"))
                records.append({
                    "bytes": row.get("payload_bytes", 0),
                    "duration_ms": row["duration_ms"],
                    "direction": f'{row.get("sender")}->{row.get("receiver")}',
                    "overlap": row.get("overlap", "compute"),
                })
    else:
        payload = json.load(path.open("r", encoding="utf-8"))
        records = payload.get("records", payload) if isinstance(payload, dict) else payload
    DsppP2PCostModel.from_records(records).save(args.output)


if __name__ == "__main__":
    main()
