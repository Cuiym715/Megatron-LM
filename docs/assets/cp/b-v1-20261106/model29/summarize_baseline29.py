"""Local CPU-only review of completed timing windows, not numerical approval.

Usage: python3 summarize_baseline.py DOWNLOADED_BATCH_DIRECTORY
Run after the bounded controller and its host auditor have closed. Copy the
manifest, resource-checks.jsonl, host-ownership/manifest.json and per-rank JSONs.
"""
import json
from pathlib import Path
import sys


def read(path):
    return json.loads(path.read_text())


def main(root, heads_only=None, external_pids=None):
    state = read(root / 'manifest.json')
    audit = read(root / 'host-ownership/manifest.json')
    cutoff = None
    if external_pids:
        assert not heads_only and state['status']=='stopped' and state['production_patch_unchanged']
        assert audit['status']=='complete' and audit['unknown_observed']
        audit_samples=[json.loads(s) for s in (root/'host-ownership/samples.jsonl').read_text().splitlines()]
        foreign=[s for s in audit_samples if any(o['pid'] in external_pids and not o['own'] and o['ancestry'] for o in s['owners'])]
        assert foreign, 'No confirmed external PID/start identity in raw audit'
        cutoff=min(s['time'] for s in foreign)
        jobs=[j for j in state['jobs'] if j.get('timing_window_validated') and j.get('ended',float('inf'))<cutoff]
        assert jobs
    elif state.get('resume'):
        assert not heads_only and state['status']=='complete' and state['production_patch_unchanged']
        assert audit['status']=='complete' and not audit['unknown_observed']
        jobs=state['jobs']
        assert [j['name'] for j in jobs]==state['resume']['expected_job_names']
    elif heads_only:
        jobs=[j for j in state['jobs'] if j['name'].startswith(heads_only+'-')]
        assert len(jobs)==8 and all(j.get('exit_code')==0 for j in jobs)
        assert audit['status'] in ('running','complete') and not audit['unknown_observed']
    else:
        jobs=state['jobs']
        assert state['status'] == 'complete' and len(jobs) == 16
        assert audit['status'] == 'complete' and not audit['unknown_observed']
        assert state['production_patch_unchanged']
    assert not state['numerically_certified']
    samples = [json.loads(line) for line in (root / 'resource-checks.jsonl').read_text().splitlines()]
    rows = []
    common_inputs = None
    common_tokens = None
    for job in jobs:
        assert job['exit_code'] == 0 and not job['polluted']
        assert job['timing_window_validated'] and not job['numerically_certified']
        checks = [s for s in samples if s['label'].startswith(job['name'] + '-idle-')]
        assert len(checks) == 3 and checks[-1]['time'] - checks[0]['time'] >= 60
        assert all(not s['apps'].strip() for s in checks)
        folder = root / job['name']
        ranks = [read(folder / f'rank{r}.json') for r in range(8)]
        wrappers = [read(folder / f'wrapper-rank{r}.json') for r in range(8)]
        assert all(not r['diagnostic'] and len(r['records']) == 25 and r['warmup'] == 5 and r['timed'] == 20 for r in ranks)
        assert all(w['all_parameters_finite'] and w['all_losses_finite'] and w['deterministic_fa_backward'] and not w['bf16_reduced_precision_reduction'] for w in wrappers)
        if 'warmup_control' in state['protocol']:
            assert all(w.get('warmup_fence') == 'cuda synchronize and world barrier only after first untimed update; timed window still5warm20updates' for w in wrappers)
        if 'runtime_correction' in state['protocol']:
            policies = [read(folder / f'launch-policy-rank{r}.json') for r in range(8)]
            assert all(p['policy'] == 'nccl-enqueue-mutex-native-order-wait-outside-v1' and p['implicit_order'] == '0' and p['race_fatal'] == '1' and not p['CPU_wait_sampling'] and not p['per_operation_logging'] for p in policies)
        if 'persistentPP' in state['protocol'].get('runtime_correction',''):
            patches=[read(folder/f'stream-cache-patch-rank{r}.json') for r in range(8)]
            caches=[read(folder/f'stream-cache-final-rank{r}.json') for r in range(8)]
            assert len({p['original_sha256'] for p in patches})==1 and len({p['patched_sha256'] for p in patches})==1
            assert all(len(c)==(2 if rank//2 in (0,3) else 4) for rank,c in enumerate(caches))
            assert not list(folder.glob('wait-state-rank*')), 'CPU diagnostic present in formal window'
        owners=[read(folder/f'model29-ownership-rank{r}.json') for r in range(8)]
        assert sorted(o['stage'] for o in owners)==[0,0,1,1,2,2,3,3]
        expected_vocab=state.get('workload_config',{}).get('vocab',32128)
        target=dict(layers=29,H=2048,FFN=5504,Qheads=16,vocab=expected_vocab,PP=4,CP=2,VPP=2)
        assert all(o['model']==target for o in owners)
        assert len({o['layout_sha256'] for o in owners})==1
        assert all(sum(len(c['global_layers']) for c in o['chunks'])==[5,8,8,8][o['stage']] for o in owners)
        assert sorted(i for o in owners if o['cp_rank']==0 for c in o['chunks'] for i in c['global_layers'])==list(range(1,30))
        cmd=job['command']
        for flag,value in {'--num-layers':'29','--hidden-size':'2048','--ffn-hidden-size':'5504','--num-attention-heads':'16','--vocab-size':str(expected_vocab),'--num-layers-per-virtual-pipeline-stage':'4','--dspp-v-layer-layout':'endpoint-3-2'}.items():assert cmd[cmd.index(flag)+1]==value
        inputs = wrappers[0]['inputs']
        tokens = [r['metrics']['valid_tokens'] for r in ranks[0]['records']]
        if common_inputs is None:
            common_inputs, common_tokens = inputs, tokens
        assert inputs == common_inputs and tokens == common_tokens
        if 'workload_config' in state:
            assert tokens==state['workload_config']['expected_valid_tokens_per_update']
        else:assert tokens[0] == 88064
        assert all(not r['selected'] for rank in ranks for r in rank['records'])
        seconds = max(rank['window_ms'] for rank in ranks) / 1000
        valid_tokens = sum(tokens[5:])
        assert abs(valid_tokens / seconds - job['tokens_per_second']) < 1e-6
        rows.append(dict(name=job['name'], valid_tokens=valid_tokens,
                         seconds=seconds, tokens_per_second=valid_tokens / seconds,
                         peak_allocated_bytes=max(r['allocated_peak'] for r in ranks),
                         peak_reserved_bytes=max(r['reserved_peak'] for r in ranks)))
    groups = {}
    for heads in ([heads_only] if heads_only else ['mha','gqa1']):
        windows = [r for r in rows if r['name'].startswith(heads + '-')]
        if (external_pids or state.get('resume')) and len(windows)!=8:
            groups[heads]=dict(status='incomplete-no-aggregate-claim',valid_windows=len(windows))
            continue
        assert len(windows) == 8
        methods = {}
        for backend in ('KV', 'QO'):
            selected = [r for r in windows if r['name'].endswith('-' + backend)]
            assert len(selected) == 4
            methods[backend] = dict(
                tokens_per_second=sum(r['valid_tokens'] for r in selected) / sum(r['seconds'] for r in selected),
                peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in selected),
                peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in selected))
        pair_ratios = []
        for block in (windows[:4], windows[4:]):
            rates = {}
            for backend in ('KV', 'QO'):
                selected = [r for r in block if r['name'].endswith('-' + backend)]
                rates[backend] = sum(r['valid_tokens'] for r in selected) / sum(r['seconds'] for r in selected)
            pair_ratios.append(rates['QO'] / rates['KV'] - 1)
        groups[heads] = dict(methods=methods, QO_vs_KV=methods['QO']['tokens_per_second'] / methods['KV']['tokens_per_second'] - 1,
                             pair_block_QO_vs_KV=pair_ratios)
    result = dict(status=('timing-partial-reviewed-external-interruption' if external_pids else 'partial-group-reviewed-await-global-closure' if heads_only else 'timing-reviewed-numerical-errors-unresolved'), numerically_certified=False,
                  scope='Current version performance only; preserve v5/v6 numerical failures. First batch of b; continue b then c–f.',
                  source=state['source'], arithmetic=state['arithmetic'],
                  warmup_control=state['protocol'].get('warmup_control'), groups=groups, windows=rows)
    result['runtime_correction'] = state['protocol'].get('runtime_correction')
    result['runtime_entry_sha256']=state['scripts_sha256'].get('benchmark_entry29.py',state['scripts_sha256'].get('benchmark_entry29_workload.py'))
    result['model']=dict(state['protocol']['model'],vocab=state.get('workload_config',{}).get('vocab',32128))
    result['workload']=state.get('workload','longmix-v512')
    result['input_identity']=state.get('data_identity',state.get('data_sha256'))
    if state.get('resume'):
        result.update(status='timing-resume-subset-reviewed-await-merge',resume=state['resume'])
    if external_pids:
        result.update(first_confirmed_external_epoch=cutoff,external_pids=sorted(external_pids),
                      excluded=[j['name'] for j in state['jobs'] if j not in jobs],
                      valid_job_names=[j['name'] for j in jobs],
                      audit_note='Original interrupted manifest/audit retained unchanged; exit ghosts are not confirmed foreign identities. No aggregate claim for incomplete head group.')
    (root / ('interruption-review.json' if external_pids else f'{heads_only}-partial-review.json' if heads_only else 'timing-review.json')).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(groups, indent=2))


if __name__ == '__main__':
    if len(sys.argv)>2 and sys.argv[2]=='--external-pids':
        main(Path(sys.argv[1]),external_pids={int(x) for x in sys.argv[3].split(',')})
    else:
        main(Path(sys.argv[1]), sys.argv[2] if len(sys.argv)>2 else None)
