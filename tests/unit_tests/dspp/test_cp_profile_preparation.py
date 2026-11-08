"""Reject incompatible CP profiles once, before asynchronous preparation."""
import json
from types import SimpleNamespace

import pytest
import torch

from megatron.core.datasets import dspp_calibration as calibration
from megatron.core.datasets.dspp_profile_snapshot import freeze_runtime_profiles
from megatron.core.pipeline_parallel.dspp_memory_counts import memory_layout


@pytest.mark.parametrize('changed', ('none', 'cp-layout', 'model', 'memory', 'p2p','p2p-gpu','backend'))
def test_cp_profile_applicability(tmp_path, monkeypatch, changed):
    monkeypatch.setattr(calibration, 'hardware_identity', lambda: {'gpu': 'fixture'})
    cfg = SimpleNamespace(context_parallel_size=2,pipeline_model_parallel_size=2,
        num_layers=4,hidden_size=512,params_dtype=torch.bfloat16,micro_seq_length=2048,
        dspp_cp_layout='hybrid',dspp_history_dkv_after_dx=True)
    records = [dict(stage=s,chunk=c,context=calibration.stage_context(cfg,s,c))
               for s in range(2) for c in range(2)]
    layout = memory_layout(cfg,2,torch.bfloat16,2048)
    if changed == 'cp-layout':
        records[0]['context']['model']['dspp_cp_layout'] = 'paired'
    if changed == 'model':
        records[0]['context']['model']['hidden_size'] = 1024
    if changed == 'backend':
        cfg.kaimm_context_parallel_impl = 'query-out'
    if changed == 'memory':
        layout['chunk_size'] = 4096
    payloads = dict(execution={'records':records},memory={'layout':layout,'device_name':'fixture'},
                    p2p={'cp_size':1 if changed == 'p2p' else 2,
                         'hardware':{'gpu':'other' if changed=='p2p-gpu' else 'fixture'}})
    for key, value in payloads.items():
        path = tmp_path/f'{key}.json'
        path.write_text(json.dumps(value))
        setattr(cfg,f'dspp_{key}_profile',str(path))
    if changed == 'none':
        freeze_runtime_profiles(cfg)
        # Process-pinned snapshots remain usable after files disappear.
        for path in tmp_path.iterdir():
            path.unlink()
        freeze_runtime_profiles(cfg)
    else:
        with pytest.raises(ValueError,match='CP .*profile'):
            freeze_runtime_profiles(cfg)
