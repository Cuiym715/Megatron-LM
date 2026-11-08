"""CPU-only attribution after the finite GPU batch and auditor have closed."""
import json,os,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
torch.set_num_threads(4)
ROOT=Path('/workspace/outputs/dspp-cp-v1-20261106/numeric-v5-longmix512')
def main():
    s=json.loads((ROOT/'manifest.json').read_text());assert s['status'] in ['complete','stopped']
    audit=json.loads((ROOT/'host-ownership/manifest.json').read_text());assert audit['status']=='complete' and not audit['unknown_observed']
    dest=ROOT/'CPU-attribution';dest.mkdir(exist_ok=False)
    state=dict(status='running',pid=os.getpid(),started=time.time(),cuda_initialized=torch.cuda.is_initialized())
    def save():(dest/'manifest.json').write_text(json.dumps(state,indent=2)+'\n')
    save();rows=[];ss=rr=0.;sign_bad=elements=0;update_ss=update_rr=0.;native_failures=[]
    mapped_elements=0;associated_error=total_mapped_error=0.
    try:
        for f in (ROOT/'mha-qo-fused').glob('iteration*-rank*.json'):
            report=json.loads(f.read_text())
            for category,value in report.items():
                if isinstance(value,dict) and value.get('passed') is False:
                    native_failures.append(dict(file=f.name,category=category,
                        max_relative_l2=value.get('max_relative_l2'),max_abs=value.get('max_abs'),
                        failed_tensors=len(value.get('failures',[])),first_failures=value.get('failures',[])[:2]))
        for rank in [0,2,4,6]:
            candidate=torch.load(ROOT/f'mha-qo-fused/complete-tensors/iteration1-rank{rank}.pt',map_location='cpu',weights_only=True)
            baseline=torch.load(ROOT/f'mha-kv-reference/complete-tensors/iteration1-rank{rank}.pt',map_location='cpu',weights_only=True)
            a=candidate['gradients'];b=baseline['gradients']
            assert a.keys()==b.keys()
            for name,x in a.items():
                y=b[name];d=x.double()-y.double();e=float(d.square().sum());n=float(y.double().square().sum())
                changed=int(((x.sign()!=y.sign())&((x!=0)|(y!=0))).sum())
                rows.append(dict(rank=rank,name=name,elements=x.numel(),gradient_relative_l2=(e/max(n,1e-24))**.5,
                    changed_signs=changed,gradient_error_ss=e,reference_ss=n))
                ss+=e;rr+=n;elements+=x.numel();sign_bad+=changed
            reference=torch.load(ROOT/f'mha/snapshots/full-master-iteration1-rank{rank}.pt',map_location='cpu',weights_only=True)
            n=sum(float(x.double().square().sum()) for x in reference['update'].values())
            report=json.loads((ROOT/f'mha-qo-fused/full-master-iteration1-rank{rank}.json').read_text())['update']
            update_rr+=n;update_ss+=n*report['relative_l2']**2
            actual=torch.load(ROOT/f'mha-qo-fused/full-master-iteration1-rank{rank}.pt',map_location='cpu',weights_only=True)
            def key(t):return (tuple(t.shape),tuple(t.flatten()[:16].tolist()))
            lookup={}
            for name,t in baseline['params_before'].items():lookup.setdefault(key(t),[]).append(name)
            for group,t in reference['before'].items():
                names=lookup.get(key(t),[])
                if len(names)!=1:continue
                name=names[0]
                assert torch.equal(t,baseline['params_before'][name]),'sampled identity is not a proof'
                assert torch.equal(actual['before'][group],t)
                changed=(a[name].sign()!=b[name].sign())&((a[name]!=0)|(b[name]!=0))
                error=(actual['update'][group].double()-reference['update'][group].double()).square()
                mapped_elements+=t.numel();associated_error+=float(error[changed].sum());total_mapped_error+=float(error.sum())
            del a,b,reference,actual,candidate,baseline
        result=dict(scope='first aligned full update, one rank per replicated CP pair, full model; not trajectory',
            elements=elements,gradient_relative_l2=(ss/rr)**.5,gradient_changed_signs=sign_bad,
            changed_sign_fraction=sign_bad/elements,master_update_relative_l2=(update_ss/update_rr)**.5,
            master_update_gate=.02,master_update_pass=(update_ss/update_rr)**.5<=.02,
            native_failures=native_failures,largest_gradient_relative_errors=sorted(rows,key=lambda r:r['gradient_relative_l2'],reverse=True)[:12],
            mapped_update_elements=mapped_elements,sign_changed_associated_update_error_fraction=associated_error/max(total_mapped_error,1e-24),
            mapping_scope='Unique initial-weight match verified over whole tensor; ambiguous equal layernorm tensors excluded from sign/error association only, not full gradient/update gate',
            all_parameter_gradients=rows,cuda_initialized=torch.cuda.is_initialized(),
            interpretation='Small gradient error and sign changes may explain Adam first-update sensitivity, but signs alone do not prove a routing/kernel root cause. No thresholds changed.')
        assert not result['cuda_initialized'];(dest/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        state.update(status='complete',ended=time.time());save()
    except Exception as e:state.update(status='failed',error=repr(e),ended=time.time());save();raise
if __name__=='__main__':main()
