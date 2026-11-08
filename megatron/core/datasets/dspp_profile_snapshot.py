"""Process-lifetime profile snapshots. Restart training to load edited files.

Input shapes and plans remain iteration-local. Only calibration file bytes and
their identity are pinned; no filesystem stat/read/hash is done on cache hits.
The spawned CPU planner owns its own snapshot, never an IPC copy per request.
"""
from dataclasses import dataclass
from pathlib import Path
import json
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
    if getattr(args, 'context_parallel_size', 1) > 1:
        _check_cp_profiles(args)


def _check_cp_profiles(args):
    """Once at preparation: bind CP layout/model/kernel, never hash each input."""
    from .dspp_calibration import stage_context, hardware_identity
    from megatron.core.pipeline_parallel.dspp_memory_counts import memory_layout
    execution = getattr(args, 'dspp_execution_profile', None)
    hardware = hardware_identity() if any(getattr(args,name,None) for name in
        ('dspp_execution_profile','dspp_memory_profile','dspp_p2p_profile')) else None
    if execution:
        value = json.loads(profile_snapshot(execution).contents)
        expected = {(s, c): stage_context(args, s, c,hardware=hardware)
                    for s in range(args.pipeline_model_parallel_size) for c in range(2)}
        seen = set()
        for row in value['records']:
            key = (row['stage'], row['chunk'])
            if row['context'] != expected[key]:
                raise ValueError(f'CP execution profile context mismatch: {key}')
            seen.add(key)
        if seen != expected.keys():
            raise ValueError('CP execution profile has incomplete stage coverage')
    memory = getattr(args, 'dspp_memory_profile', None)
    if memory:
        value = json.loads(profile_snapshot(memory).contents)
        if value['layout'] != memory_layout(args, args.pipeline_model_parallel_size,
                                           args.params_dtype, args.micro_seq_length):
            raise ValueError('CP memory profile layout mismatch')
        if value['device_name'] != hardware['gpu']:
            raise ValueError('CP memory profile device mismatch')
    p2p = getattr(args, 'dspp_p2p_profile', None)
    if p2p:
        value=json.loads(profile_snapshot(p2p).contents)
        if value.get('cp_size') != args.context_parallel_size or value.get('hardware') != hardware:
            raise ValueError('CP P2P profile requires matching hardware and simultaneous CP lanes')
