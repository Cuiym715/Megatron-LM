"""Archive all matched task samples; aggregated work is not wall-clock makespan."""
import json
from pathlib import Path
import re
import statistics as st

root=Path(__file__).resolve().parent
fields=('fused_B_ms','after_dx_B_ms','H_ms','B_plus_H_ms','B_start_to_H_end_ms')
def describe(rows):
    med={f:st.median(r[f] for r in rows) for f in fields}
    delta=[r['after_dx_B_ms']-r['fused_B_ms'] for r in rows]
    return dict(n=len(rows), median_ms=med,
        range_ms={f:[min(r[f] for r in rows),max(r[f] for r in rows)] for f in fields},
        sum_ms={f:sum(r[f] for r in rows) for f in fields},
        B_change_percent=100*(med['after_dx_B_ms']/med['fused_B_ms']-1),
        paired_B_delta_ms=delta, B_faster_count=sum(d<0 for d in delta),
        raw_samples=[{k:r[k] for k in ('measurement_run','stage','model_chunk','logical','physical','layout_q_history',*fields)} for r in rows])

result={}
for cfg in json.loads((root/'configurations.json').read_text()):
    name=cfg['name']; path=root/name/'task-times.json'
    if not path.exists():continue
    raw=json.loads(path.read_text()); rows=raw['records']
    for row in rows:row['measurement_run']=name
    groups={}
    for r in rows:
        k=json.dumps([r['stage'],r['model_chunk'],r['layout_q_history']])
        groups.setdefault(k,[]).append(r)
    selected={}
    for hist in (4096,12288):
        rr=[r for r in rows if r['stage']==1 and r['model_chunk']==0 and r['layout_q_history']==[[2048,hist]]]
        if rr:selected[str(hist)]=describe(rr)
    losses={}
    flags={}
    for mode in ('fused','after_dx'):
        log=(root/name/mode/'train.log').read_text()
        losses[mode]=re.findall(r'lm loss:\s*([0-9.eE+\-]+)', log)
        flags[mode]={key:re.findall(r'^\s*'+key+r'\s*\.+\s*(\S+)',log,re.M) for key in
            ('hidden_size','ffn_hidden_size','num_attention_heads','group_query_attention','num_query_groups',
             'seq_length','micro_seq_length','dspp_task_timing_only','dspp_history_dkv_after_dx')}
    assert len(losses['fused'])==len(losses['after_dx'])==3,(name,losses)
    result[name]=dict(config=cfg,boundary_checks=raw['boundary_checks'],
        losses=losses,losses_match_log_precision=losses['fused']==losses['after_dx'],flags=flags,
        selected=selected, all_tasks=describe(rows),
        history_tasks=describe([r for r in rows if r['H_ms']>0]),
        no_history_tasks=describe([r for r in rows if r['H_ms']==0]),
        all_shape_stage_groups={k:describe(rr) for k,rr in groups.items()})
    print(name, 'loss_match',result[name]['losses_match_log_precision'])
    for hist,v in selected.items():print('  history',hist,'n',v['n'],v['median_ms'],'B change%',v['B_change_percent'],'faster',v['B_faster_count'])
    sums=result[name]['all_tasks']['sum_ms']
    print('  all-task B sums',sums['fused_B_ms'],sums['after_dx_B_ms'])
pooled={}
if {'wide-long','wide-long-repeat'} <= result.keys():
    rr=[r for name in ('wide-long','wide-long-repeat') for r in result[name]['all_tasks']['raw_samples']]
    pooled['wide-long']=dict(selected={}, all_tasks=describe(rr),
        history_tasks=describe([r for r in rr if r['H_ms']>0]),
        no_history_tasks=describe([r for r in rr if r['H_ms']==0]))
    for history in (4096,12288):
        selected=[r for r in rr if r['stage']==1 and r['model_chunk']==0 and r['layout_q_history']==[[2048,history]]]
        pooled['wide-long']['selected'][str(history)]=describe(selected)
    print('Wide-long pooled selected', {h:dict(n=g['n'],median=g['median_ms'],B_change_percent=g['B_change_percent']) for h,g in pooled['wide-long']['selected'].items()})
(root/'summary.json').write_text(json.dumps(dict(
    scope='Complete B/H boundary events, matched within configuration; all-task sums are GPU task work across ranks, NOT iteration times.',
    sample_limit='One paired run per configuration initially; two warmup iterations, iteration 3 recorded. Samples within one run are not independent runs.',
    results=result, pooled_across_runs=pooled),indent=2)+'\n')
