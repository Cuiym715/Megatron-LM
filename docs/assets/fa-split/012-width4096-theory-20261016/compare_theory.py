"""Useful matmul FLOPs and an explicitly hypothetical equal-rate time reference.

FLOPs exclude separate W, elementwise/softmax/reductions, padding tiles, memory,
launch/host gaps, routing and communication. They are not hardware instruction
counts. Equivalent-rate times are NOT measured split-kernel predictions.
"""
import json
from pathlib import Path
root=Path(__file__).resolve().parent

def model(q, history, hidden, ffn, kv_width):
    ah=q*history; ac=q*(q+1)//2
    linear=2*q*(hidden*(hidden+2*kv_width)+hidden*hidden+3*hidden*ffn)
    attention=10*hidden*(ah+ac)
    removed=4*hidden*ah
    b0=linear+attention
    b=linear+6*hidden*ah+10*hidden*ac
    h=8*hidden*ah
    assert b0-b==removed and b+h-b0==removed and h==2*removed
    return dict(q=q,history=history,hidden=hidden,ffn=ffn,kv_width=kv_width,
        gflops=dict(linear_B=linear/1e9,attention_B=attention/1e9,
                    fused_B=b0/1e9,removed_from_B=removed/1e9,
                    split_B=b/1e9,H=h/1e9,split_B_plus_H=(b+h)/1e9,
                    extra_recomputation=removed/1e9),
        B_reduction_percent=100*removed/b0,
        attention_B_reduction_percent=100*removed/attention,
        B_plus_H_increase_percent=100*removed/b0,
        H_over_fused_B_flops=h/b0)

scales=[model(2048,p,d,2*d,d//2) for d in (512,2048,4096) for p in (4096,12288)]
for v in scales:print(v['hidden'],v['history'],v['gflops'],'B reduction%',v['B_reduction_percent'])
comparisons=[]
summary_path=root/'summary.json'
if summary_path.exists():
    summary=json.loads(summary_path.read_text())
    selections=[(name, r['selected']) for name,r in summary['results'].items()]
    selections += [(name+'-pooled',r['selected']) for name,r in summary['pooled_across_runs'].items()]
    for name, groups in selections:
        for hist, group in groups.items():
            m=model(2048,int(hist),4096,8192,2048)
            t=group['median_ms'];g=m['gflops'];base=t['fused_B_ms']
            # Calibrate one equal throughput to the entire baseline B interval,
            # including its real non-matmul costs. Deliberately just a reference.
            equal_rate={field:base*g[key]/g['fused_B'] for field,key in
                        [('split_B_ms','split_B'),('H_ms','H'),('B_plus_H_ms','split_B_plus_H')]}
            comparison=dict(run=name,history=int(hist),n=group['n'],theory=m,
                measured_median_ms=t,measured_B_reduction_percent=-group['B_change_percent'],
                measured_B_delta_ms=base-t['after_dx_B_ms'],
                hypothetical_equal_rate_ms=equal_rate,
                equal_rate_B_saving_ms=base*m['B_reduction_percent']/100,
                measured_over_equal_rate={key:t[actual]/equal_rate[key] for key,actual in
                                         [('split_B_ms','after_dx_B_ms'),('H_ms','H_ms'),('B_plus_H_ms','B_plus_H_ms')]},
                useful_flops_per_ms_TF_s=dict(fused_B=g['fused_B']/base,
                    split_B=g['split_B']/t['after_dx_B_ms'],H=g['H']/t['H_ms']))
            comparisons.append(comparison)
            print(name,hist,'actual B decrease%',comparison['measured_B_reduction_percent'],
                  'equal-rate ref ms',equal_rate,'measured',t)
result=dict(scope=__doc__,
 assumptions='TP1; dQ width equals hidden; GQA2; SwiGLU FFN=2*hidden; one block per B; full stage input backward excludes separate W.',
 formulas=dict(history_pairs='Ah=q*history',current_pairs='Ac=q*(q+1)/2',
               linear_B='2*q*[D*(D+2*Dkv)+D^2+3*D*F]',fused_attention='10*D*(Ah+Ac)',
               removed_from_B='4*D*Ah',split_attention_B='6*D*Ah+10*D*Ac',H='8*D*Ah',
               extra_recomputation='4*D*Ah'),
 width_scaling=scales,measured_comparisons=comparisons)
(root/'theory-comparison.json').write_text(json.dumps(result,indent=2)+'\n')
