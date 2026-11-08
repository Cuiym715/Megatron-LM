"""A foreign CUDA process prevents launch even with a tiny memory footprint."""
import importlib.util
import json
from pathlib import Path
import sys


def test_busy_process_waits_fifteen_minutes_then_resumes(tmp_path,monkeypatch):
    path=Path(__file__).resolve().parents[3]/'tools/wait_dspp_gpus.py'
    spec=importlib.util.spec_from_file_location('wait_cp_gpus',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    values=iter(['0, GPU-test, 0, 0\n','GPU-test, 77\n','0, GPU-test, 0, 0\n',''])
    monkeypatch.setattr(module.subprocess,'check_output',lambda *a,**kw:next(values))
    slept=[];monkeypatch.setattr(module.time,'sleep',slept.append)
    output=tmp_path/'gpu.jsonl'
    monkeypatch.setattr(sys,'argv',['wait','--output',str(output)])
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    module.main()
    assert slept==[900]
    assert [json.loads(line)['busy'] for line in output.read_text().splitlines()]==[True,False]


def test_trailing_utilization_without_process_or_memory_does_not_wait(tmp_path,monkeypatch):
    path=Path(__file__).resolve().parents[3]/'tools/wait_dspp_gpus.py'
    spec=importlib.util.spec_from_file_location('wait_cp_gpus_stale',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    values=iter(['0, GPU-test, 0, 9\n',''])
    monkeypatch.setattr(module.subprocess,'check_output',lambda *a,**kw:next(values))
    def fail(_):raise AssertionError('idle GPU must resume immediately')
    monkeypatch.setattr(module.time,'sleep',fail)
    output=tmp_path/'gpu.jsonl'
    monkeypatch.setattr(sys,'argv',['wait','--output',str(output)])
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    module.main()
    assert not json.loads(output.read_text())['busy']
