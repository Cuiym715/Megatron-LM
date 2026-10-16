#!/usr/bin/env python3
"""Generate a compute-efficient DSPP chunk menu from an offline profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from megatron.core.datasets.dspp_compute_profile import (
    generate_candidate_menu,
    load_compute_profile,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--efficiency-threshold", type=float, default=0.90)
    parser.add_argument("--min-coverage", type=float, default=0.75)
    parser.add_argument("--lower-quantile", type=float, default=0.20)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument(
        "--allow-proxy-work",
        action="store_true",
        help="Allow analytical proxy work when an adapter omitted work_units",
    )
    args = parser.parse_args()
    menu = generate_candidate_menu(
        load_compute_profile(args.profile),
        efficiency_threshold=args.efficiency_threshold,
        min_coverage=args.min_coverage,
        lower_quantile=args.lower_quantile,
        max_candidates=args.max_candidates,
        require_work_units=not args.allow_proxy_work,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(menu.to_dict(), indent=2), encoding="utf-8")
    print(f"compute-efficient candidates={list(menu.candidates)}")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
