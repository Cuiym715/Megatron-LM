#!/usr/bin/env python3
"""Capture real stage-A GPU launches and verify independent dQ/dKV execution.

This is an isolated attention demonstration, not a pipeline F/B/W trace or a
performance benchmark. No production runtime is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch
import flash_attn
from flash_attn.flash_attn_interface import (
    _wrapped_flash_attn_forward, _wrapped_flash_attn_backward,
)
from megatron.core.kv_cache.fa_split import prepare_backward, backward_dq, backward_dkv


CASES = ("original", "dq_then_dkv", "dkv_then_dq", "dq_only", "dkv_only")


def execute(case, q, k, v, dout, scale):
    def tagged(stage, fn):
        with torch.profiler.record_function(f"SPLIT_EVIDENCE/{case}/{stage}"):
            return fn()

    with torch.profiler.record_function(f"SPLIT_CASE/{case}"):
        out, lse, _, rng = tagged("F_attn", lambda: _wrapped_flash_attn_forward(
            q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), 0.0, scale,
            True, -1, -1, 0.0, None, False,
        ))
        if case == "original":
            grads = [torch.empty_like(t).unsqueeze(0) for t in (q, k, v)]
            tagged("fused_B_attn", lambda: _wrapped_flash_attn_backward(
                dout.unsqueeze(0), q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
                out, lse, *grads, 0.0, scale, True, -1, -1, 0.0, None, False, rng,
            ))
            result = dict(zip(("dq", "dk", "dv"), [t.squeeze(0) for t in grads]))
        else:
            state = tagged("prepare", lambda: prepare_backward(
                q, k, v, out.squeeze(0), dout, lse.squeeze(0), softmax_scale=scale,
            ))
            result = {}
            sequence = {
                "dq_then_dkv": ("dQ", "dKV"), "dkv_then_dq": ("dKV", "dQ"),
                "dq_only": ("dQ",), "dkv_only": ("dKV",),
            }[case]
            for stage in sequence:
                if stage == "dQ":
                    result["dq"] = tagged(stage, lambda: backward_dq(state))
                else:
                    result["dk"], result["dv"] = tagged(stage, lambda: backward_dkv(state))
        # Close each case after GPU completion, without imposing barriers between
        # its phases. Kernel association below uses runtime correlation IDs.
        torch.cuda.synchronize()
        return result


def parse_trace(path):
    events = json.loads(path.read_text())["traceEvents"]
    ranges = [e for e in events if e.get("ph") == "X" and
              e.get("name", "").startswith("SPLIT_EVIDENCE/")]
    runtimes = {}
    for e in events:
        if e.get("cat") == "cuda_runtime" and "correlation" in e.get("args", {}):
            runtimes[e["args"]["correlation"]] = e
    result = {name: [] for name in CASES}
    unmatched = []
    for e in events:
        if e.get("ph") != "X" or e.get("cat") not in ("kernel", "gpu_memset", "gpu_memcpy"):
            continue
        runtime = runtimes.get(e.get("args", {}).get("correlation"))
        owners = [] if runtime is None else [
            r for r in ranges if r["pid"] == runtime["pid"] and r["tid"] == runtime["tid"]
            and r["ts"] <= runtime["ts"] <= r["ts"] + r["dur"]
        ]
        if not owners:
            unmatched.append(e["name"])
            continue
        owner = min(owners, key=lambda x: x["dur"])
        _, case, phase = owner["name"].split("/")
        result[case].append({
            "phase": phase, "name": e["name"], "category": e["cat"],
            "start_us": e["ts"], "duration_us": e["dur"],
            "stream": e.get("args", {}).get("stream"),
            "correlation": e.get("args", {}).get("correlation"),
        })
    if unmatched:
        raise RuntimeError(f"GPU events not attributable to marked runtime launches: {unmatched}")
    for case, rows in result.items():
        if not rows:
            raise RuntimeError(f"No GPU events recorded for {case}")
        rows.sort(key=lambda x: x["start_us"])
        origin = rows[0]["start_us"]
        for row in rows:
            row["start_us"] -= origin
    return result


def check_launches(rows):
    checks = {}
    for case, events in rows.items():
        names = [e["name"] for e in events]
        # Kineto demangles bool template parameters as true/false or (bool)1/0.
        dq = sum(bool(re.search(r"split_kernel<(?:true|\(bool\)1)>", n)) for n in names)
        dkv = sum(bool(re.search(r"split_kernel<(?:false|\(bool\)0)>", n)) for n in names)
        prepare = sum("dspp_split_detail::prepare_kernel" in n for n in names)
        convert = sum("dspp_split_detail::convert_dq_kernel" in n for n in names)
        fused = sum("flash_bwd_dq_dk_dv" in n for n in names)
        want_dq = int(case in ("dq_then_dkv", "dkv_then_dq", "dq_only"))
        want_dkv = int(case in ("dq_then_dkv", "dkv_then_dq", "dkv_only"))
        expected = (want_dq, want_dkv, int(case != "original"), want_dq, int(case == "original"))
        observed = (dq, dkv, prepare, convert, fused)
        if observed != expected:
            raise AssertionError(f"Unexpected kernel launches for {case}: {observed} != {expected}; {names}")
        checks[case] = dict(zip(("dq_main", "dkv_main", "prepare", "dq_convert", "original_fused_main"), observed))
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=16)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    gen = torch.Generator(device="cuda").manual_seed(1234)
    q, k, v, dout = [torch.randn(args.length, args.heads, 128, generator=gen,
                                device="cuda", dtype=torch.bfloat16) for _ in range(4)]
    scale = 128**-0.5
    for _ in range(3):
        for case in CASES:
            execute(case, q, k, v, dout, scale)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        outputs = {case: execute(case, q, k, v, dout, scale) for case in CASES}
    raw = args.output / "execution.pt.trace.json"
    prof.export_chrome_trace(str(raw))
    timeline = parse_trace(raw)
    checks = check_launches(timeline)
    accuracy = {}
    for case in CASES[1:]:
        accuracy[case] = {}
        for grad, actual in outputs[case].items():
            expected = outputs["original"][grad]
            torch.testing.assert_close(actual, expected, atol=0.004, rtol=0.01)
            diff = actual.float() - expected.float()
            accuracy[case][grad] = {
                "max_abs": diff.abs().max().item(),
                "relative_l2": (diff.norm() / expected.float().norm()).item(),
            }
    files = [REPO / "megatron/core/kv_cache/fa_split.py",
             REPO / "megatron/core/kv_cache/fa_split_build.py",
             *sorted((REPO / "megatron/core/kv_cache/fa_split_src").glob("*")), Path(__file__)]
    summary = {
        "scope": "Actual GPU kernel launches from isolated attention; not stage F/B/W, deferred history dKV, or a speed benchmark.",
        "config": {"length": args.length, "heads": args.heads, "head_dim": 128,
                   "dtype": "BF16", "causal": True, "dropout": 0, "seed": 1234},
        "environment": {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                        "cuda": torch.version.cuda, "flash_attention": flash_attn.__version__},
        "raw_trace_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        "source_sha256": {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "launch_checks": checks, "accuracy_vs_original": accuracy, "timeline": timeline,
        "timing_note": "Profiler perturbs host/GPU timing. Start is normalized to the first GPU event of each case. Gaps are observed gaps, not available scheduling slack. No speedup claim.",
    }
    (args.output / "evidence.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"launch_checks": checks, "accuracy_vs_original": accuracy}, indent=2))


if __name__ == "__main__":
    main()
