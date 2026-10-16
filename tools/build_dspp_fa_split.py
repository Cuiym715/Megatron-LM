#!/usr/bin/env python3
"""Explicit source fetch and build entry point for the experimental FA split."""
import argparse
import importlib.util
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true", help="fetch pinned FA2 and CUTLASS sources")
    args = parser.parse_args()
    path = Path(__file__).resolve().parents[1] / "megatron/core/kv_cache/fa_split_build.py"
    spec = importlib.util.spec_from_file_location("fa_split_build", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if args.fetch:
        module.fetch_sources()
    extension = module.load_extension()
    print(f"Built isolated extension: {extension.__file__}")


if __name__ == "__main__":
    main()
