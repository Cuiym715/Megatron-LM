"""Frozen file contracts; per-input validation remains covered by test_prefetch."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from megatron.core.datasets.dspp_profile_snapshot import profile_snapshot, freeze_runtime_profiles
from megatron.core.datasets.dspp_calibration import load_runtime_calibration
from megatron.core.pipeline_parallel.dspp_plan_runtime import _file_version
from megatron.core.datasets.dspp_p2p import load_p2p_cost_model
from megatron.core.pipeline_parallel.dspp_memory_counts import CountedMemory


def test_concurrent_read_once_and_pin_bytes(tmp_path, monkeypatch):
    path = tmp_path / 'cost.json'
    path.write_text('{"exact":[],"fitted":[]}')
    contents = path.read_bytes()
    reads = []
    native = Path.read_bytes
    def read(self):
        reads.append(str(self))
        return native(self)
    monkeypatch.setattr(Path, 'read_bytes', read)
    with ThreadPoolExecutor(max_workers=8) as pool:
        snapshots = list(pool.map(profile_snapshot, [str(path)] * 32))
    assert reads == [str(path)]
    assert all(s is snapshots[0] for s in snapshots)
    path.write_text('{"exact":[],"fitted":[1]}')
    assert _file_version(str(path)) == hashlib.sha256(contents).hexdigest()
    assert profile_snapshot(str(path)).contents == contents
    assert load_runtime_calibration(str(path), snapshots[0].version).fitted == {}
    assert reads == [str(path)]
    with pytest.raises(ValueError, match='frozen snapshot'):
        load_runtime_calibration(str(path), 'wrong-version')


def test_frozen_p2p_and_memory_need_no_filesystem_after_startup(tmp_path, monkeypatch):
    cost = tmp_path / 'cost.json'; cost.write_text('{}')
    p2p = tmp_path / 'p2p.json'; p2p.write_text('[]')
    memory = tmp_path / 'memory.json'; memory.write_text('{"test":1}')
    cfg = SimpleNamespace(dspp_execution_profile=str(cost), dspp_p2p_profile=str(p2p),
                          dspp_memory_profile=str(memory))
    freeze_runtime_profiles(cfg)
    for path in (cost,p2p,memory):path.unlink()
    def fail(*a,**kw):raise AssertionError('unexpected filesystem read')
    monkeypatch.setattr(Path, 'read_bytes', fail)
    monkeypatch.setattr(Path, 'read_text', fail)
    assert _file_version(str(cost)) != 'none'
    assert not load_p2p_cost_model(str(p2p)).points
    class CaptureMemory(CountedMemory):
        def __init__(self, profile): self.profile=profile
    assert CaptureMemory.load(str(memory)).profile == {'test':1}


def test_missing_profile_not_silently_cached(tmp_path):
    path=tmp_path/'new.json'
    with pytest.raises(FileNotFoundError):profile_snapshot(str(path))
    path.write_text('{}')
    assert profile_snapshot(str(path)).contents == b'{}'
    assert _file_version(None) == 'none'
