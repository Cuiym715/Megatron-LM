"""Opt-in non-blocking GPU progress snapshots for diagnosing DSPP stalls.

The tracker is deliberately outside the normal training path.  It is enabled
only when ``DSPP_GPU_PROGRESS_DIR`` is set, never synchronizes CUDA, and writes
periodic per-rank JSONL snapshots so a timed-out process still leaves evidence.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import time

import torch
import torch.distributed as dist


def dspp_communication_buffer_count(
    *, neighbor_count: int, recv_lookahead: int, send_inflight_depth: int
) -> int:
    """Conservative per-rank P2P payload reserve for adaptive chunking."""
    if neighbor_count < 0 or recv_lookahead < 0 or send_inflight_depth <= 0:
        raise ValueError(
            "DSPP neighbor count/lookahead must be non-negative and send depth positive"
        )
    # Reserve every outgoing in-flight send, one receive at the current
    # dependency, and every future preposted receive per neighbor.
    return neighbor_count * (send_inflight_depth + 1 + recv_lookahead)


class DsppGpuProgressTracker:
    _instance = 0

    def __init__(self, stage: int):
        output_dir = os.environ.get("DSPP_GPU_PROGRESS_DIR", "")
        self.enabled = bool(output_dir)
        self.stage = stage
        self.rank = dist.get_rank() if dist.is_initialized() else stage
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._next_id = 0
        self._compute = {}
        self._p2p = {}
        self._current_cpu = None
        self._thread = None
        self._file = None
        if not self.enabled:
            return

        instance = DsppGpuProgressTracker._instance
        DsppGpuProgressTracker._instance += 1
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        self.path = path / f"rank_{self.rank}_iteration_{instance}.jsonl"
        self._file = self.path.open("w", encoding="utf-8", buffering=1)
        self._thread = threading.Thread(
            target=self._watchdog,
            name=f"dspp-progress-rank-{self.rank}",
            daemon=True,
        )
        self._thread.start()

    def _allocate_id(self, prefix: str) -> str:
        with self._lock:
            token = f"{prefix}-{self._next_id}"
            self._next_id += 1
        return token

    def begin_compute(self, kind, node, task_id, *, parent=None, detail=None):
        if not self.enabled:
            return None
        token = self._allocate_id("compute")
        entry = {
            "token": token,
            "kind": kind,
            "slot": getattr(node, "slot", None),
            "phase": getattr(node, "phase", None),
            "group": getattr(node, "microbatch", None),
            "split": getattr(node, "split", None),
            "task_id": list(task_id) if task_id is not None else None,
            "parent": parent,
            "detail": detail,
            "cpu_enter_ns": time.monotonic_ns(),
            "cpu_return_ns": None,
            "begin_event": None,
            "begin_recorded": False,
            "end_event": None,
            "end_recorded": False,
        }
        with self._lock:
            self._compute[token] = entry
            self._current_cpu = token
        detail_suffix = ""
        if detail and "wgrad_index" in detail:
            detail_suffix = f"/wgrad{detail['wgrad_index']}"
        torch.cuda.nvtx.range_push(
            f"DSPP_PROGRESS/{kind}/stage{self.stage}"
            f"/group{entry['group']}/split{entry['split']}{detail_suffix}"
        )
        begin = torch.cuda.Event(enable_timing=False)
        with self._lock:
            entry["begin_event"] = begin
        begin.record(torch.cuda.current_stream())
        with self._lock:
            entry["begin_recorded"] = True
        return token

    def end_compute(self, token):
        if not self.enabled or token is None:
            return
        end = torch.cuda.Event(enable_timing=False)
        with self._lock:
            entry = self._compute[token]
            entry["cpu_return_ns"] = time.monotonic_ns()
            entry["end_event"] = end
        end.record(torch.cuda.current_stream())
        torch.cuda.nvtx.range_pop()
        with self._lock:
            entry["end_recorded"] = True
            if self._current_cpu == token:
                self._current_cpu = entry["parent"]

    def track_p2p(self, *, operation, sender, receiver, key, work,
                  payload_bytes=0, overlap="compute"):
        if not self.enabled:
            return None
        token = self._allocate_id("p2p")
        with self._lock:
            self._p2p[token] = {
                "token": token,
                "operation": operation,
                "sender": sender,
                "receiver": receiver,
                "key": list(key),
                "payload_bytes": int(payload_bytes),
                "overlap": str(overlap),
                "cpu_submit_ns": time.monotonic_ns(),
                "wait_enter_ns": None,
                "wait_return_ns": None,
                "work": work,
            }
        return token

    def wait_enter(self, token):
        if self.enabled and token is not None:
            with self._lock:
                self._p2p[token]["wait_enter_ns"] = time.monotonic_ns()

    def wait_return(self, token):
        if self.enabled and token is not None:
            with self._lock:
                now = time.monotonic_ns()
                entry = self._p2p[token]
                entry["wait_return_ns"] = now
                entry["duration_ms"] = max(
                    0.0, (now - entry["cpu_submit_ns"]) / 1.0e6
                )

    @staticmethod
    def _query_event(event, recorded):
        if event is None or not recorded:
            return False
        try:
            return bool(event.query())
        except Exception as exc:  # diagnostic path must preserve its own failure
            return f"error:{type(exc).__name__}:{exc}"

    @staticmethod
    def _query_work(work):
        try:
            return bool(work.is_completed())
        except Exception as exc:  # diagnostic path must preserve its own failure
            return f"error:{type(exc).__name__}:{exc}"

    def _snapshot(self, reason):
        with self._lock:
            compute = [dict(entry) for entry in self._compute.values()]
            p2p = [dict(entry) for entry in self._p2p.values()]
            current_cpu = self._current_cpu
        payload = {
            "timestamp_ns": time.monotonic_ns(),
            "reason": reason,
            "rank": self.rank,
            "stage": self.stage,
            "current_cpu": current_cpu,
            "compute": [
                {
                    key: value
                    for key, value in entry.items()
                    if key not in {"begin_event", "end_event"}
                }
                | {
                    "cuda_begin_complete": self._query_event(
                        entry["begin_event"], entry["begin_recorded"]
                    ),
                    "cuda_end_complete": self._query_event(
                        entry["end_event"], entry["end_recorded"]
                    ),
                }
                for entry in compute
            ],
            "p2p": [
                {
                    key: value
                    for key, value in entry.items()
                    if key != "work"
                }
                | {"work_completed": self._query_work(entry["work"])}
                for entry in p2p
            ],
        }
        self._file.write(json.dumps(payload, separators=(",", ":")) + "\n")

    def _watchdog(self):
        while not self._stop.wait(0.5):
            self._snapshot("periodic")

    def stop(self):
        if not self.enabled:
            return
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._snapshot("normal-stop")
        self._file.close()
