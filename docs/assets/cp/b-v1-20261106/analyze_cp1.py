"""Report whole-model AND per-stage norms; never rewrite original gates."""
import json,os,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
torch.set_num_threads(4)
ROOT=Path('/workspace/outputs/dspp-cp-v1-20261106/numeric-v6-cp1-attribution')
def main():
    state=json.loads((ROOT/'manifest.json').read_text());assert state['status']=='complete'
    audit=json.loads((ROOT/'host-ownership/manifest.json').read_text());assert audit['status']=='complete' and not audit['unknown_observed']
    out=ROOT/'CPU-attribution';out.mkdir(exist_ok=False)
    manifest=dict(status='running',pid=os.getpid(),started=time.time(),cuda_initialized=torch.cuda.is_initialized())
    def save():(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    save();methods=[]
    for method in ['mha-kv-cp2','mha-qo-fused']:
        gss=grr=uss=urr=0.;elements=0;perstage=[]
        p=ROOT/method
        native=[json.loads(f.read_text()) for f in p.glob('iteration*-rank*.json')]
        native_summary={k:dict(passed=all(r[k]['passed'] for r in native),
            max_relative_l2=max(r[k]['max_relative_l2'] for r in native),
            failed_tensors=sum(len(r[k].get('failures',[])) for r in native))
            for k in ['params_before','params_after','gradients','outputs','dx','logits','loss']}
        for stage in range(4):
            ref=torch.load(ROOT/f'mha/snapshots/iteration1-rank{stage}.pt',map_location='cpu',weights_only=True)
            actual=torch.load(p/f'complete-tensors/iteration1-rank{2*stage}.pt',map_location='cpu',weights_only=True)
            for name,x in actual['gradients'].items():
                y=ref['gradients'][name];gss+=float((x.double()-y.double()).square().sum());grr+=float(y.double().square().sum());elements+=x.numel()
            masters=torch.load(ROOT/f'mha/snapshots/full-master-iteration1-rank{stage}.pt',map_location='cpu',weights_only=True)
            n=sum(float(x.double().square().sum()) for x in masters['update'].values())
            report=json.loads((p/f'full-master-iteration1-rank{2*stage}.json').read_text())['update']
            uss+=n*report['relative_l2']**2;urr+=n
            perstage.append(dict(stage=stage,relative_l2=report['relative_l2'],gate_pass=report['passed']))
            del ref,actual,masters
        methods.append(dict(method=method,elements=elements,native=native_summary,
            whole_model_gradient_relative_l2=(gss/grr)**.5,whole_model_actual_update_relative_l2=(uss/urr)**.5,
            per_stage_actual_update=perstage,
            note='Descriptive whole-model norm alongside the previously declared per-stage guard. This analysis does not change acceptance or certify a failed configuration.'))
    result=dict(methods=methods,reference='PP4CP1 same actual88064-token failed input and explicitly injected initial weights/optimizer',
        scope='one aligned complete update, no trajectory/throughput claim',cuda_initialized=torch.cuda.is_initialized())
    assert not result['cuda_initialized'];(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    manifest.update(status='complete',ended=time.time());save()
if __name__=='__main__':main()
