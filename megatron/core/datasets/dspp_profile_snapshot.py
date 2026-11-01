"""Process-lifetime profile snapshots. Restart training to load edited files.

Input shapes and plans remain iteration-local. Only calibration file bytes and
their identity are pinned; no filesystem stat/read/hash is done on cache hits.
The spawned CPU planner owns its own snapshot, never an IPC copy per request.
"""
from dataclasses import dataclass
from pathlib import Path
import threading


@dataclass(frozen=True)
class ProfileSnapshot:
    contents: bytes
    version: str


_profiles = {}
_lock = threading.RLock()


def profile_snapshot(path):
    key = str(path)
    with _lock:
        if key not in _profiles:
            contents = Path(path).read_bytes()
            _profiles[key] = ProfileSnapshot(contents, key)
        return _profiles[key]


def freeze_runtime_profiles(args):
    """Load before the coordinator starts its asynchronous input thread."""
    for name in ('dspp_execution_profile', 'dspp_p2p_profile', 'dspp_memory_profile'):
        path = getattr(args, name, None)
        if path:
            profile_snapshot(path)
