"""Stage C integration: actual packed autograd/cache, RoPE and block gradients.

Criteria declared before running: packed split vs packed fused atol=.004,
rtol=.01; small model vs independent unsliced FP64 atol=.04, rtol=.04 and
relative L2 <= .03. BF16 model accumulation is assessed by norm as well as
pointwise tolerances. FP64 model uses the same BF16-rounded weights/inputs.
"""
import copy
import json

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from megatron.core.datasets.dspp_batch_plan import build_dspp_batch_plan
from megatron.core.kv_cache import dspp_packed_attention as packed
from megatron.model.rotary_pos_embedding import RotaryEmbedding, apply_rotary_pos_emb
from .test_fa_split import errors

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def physical(source,meta):
    values=torch.cat([source[i.sequence_id][i.token_offset:i.token_offset+i.token_length] for i in meta.items])
    return torch.cat([values,values.new_zeros(meta.padding_token_count,*values.shape[1:])]).detach().requires_grad_()


def unpack(plan,values,lengths):
    parts=[[] for _ in lengths]
    for meta,value in zip(plan.microbatches,values):
        cursor=0
        for i in meta.items:
            parts[i.sequence_id].append((i.segment_id,value[cursor:cursor+i.token_length]))
            cursor+=i.token_length
    return [torch.cat([x for _,x in sorted(p)]) for p in parts]


class AuditState(packed.DsppSequenceKVState):
    def __init__(self,**kwargs):
        super().__init__(validate_runtime=True,**kwargs);self.routes=[];self.pending_shapes=[]
    def distribute_key_value_grads(self,layer,layouts,dk,dv,shape):
        key=(layer,tuple((x.item.sequence_id,x.item.segment_id) for x in layouts))
        assert key not in self.routes,'gradient routed twice'
        self.routes.append(key)
        result=super().distribute_key_value_grads(layer,layouts,dk,dv,shape)
        for entry in self._entries.values():
            for t in (*entry.pending_key_grads.values(),*entry.pending_value_grads.values()):
                self.pending_shapes.append(tuple(t.shape))
        return result


def run_qkv(lengths,hq,hk,split,seed=102):
    plan=build_dspp_batch_plan(lengths,chunk_size=64)
    gen=torch.Generator(device='cuda').manual_seed(seed)
    source=[tuple(torch.randn(n,h,128,device='cuda',generator=gen).to(torch.bfloat16) for h in (hq,hk,hk)) for n in lengths]
    upstream=[torch.randn(n,hq,128,device='cuda',generator=gen).to(torch.bfloat16) for n in lengths]
    state=AuditState(split_attention_backward=split)
    inputs=[];outputs=[]
    for meta in plan.microbatches:
        inp=[physical([s[j] for s in source],meta).unsqueeze(1).detach().requires_grad_() for j in range(3)]
        inputs.append(inp)
        outputs.append(packed.dspp_packed_flash_attention(*inp,state=state,meta=meta,layer_idx=3))
    # A wrong mutable current_meta must not affect saved forward layout.
    state.set_microbatch(plan.microbatches[0])
    for meta,out in reversed(list(zip(plan.microbatches,outputs))):
        # Nonzero upstream on padding tests explicit masking of physical padding.
        grad=torch.ones_like(out);cursor=0
        for item in meta.items:
            grad[cursor:cursor+item.token_length,0]=upstream[item.sequence_id][item.token_offset:item.token_offset+item.token_length]
            cursor+=item.token_length
        out.backward(grad)
    state.finish_iteration()
    assert len(state)==0 and not state._prepared_cu_seqlens
    assert len(state.routes)==len(plan.microbatches)
    assert all(shape[1:]==(hk,128) for shape in state.pending_shapes)
    for meta,inp,out in zip(plan.microbatches,inputs,outputs):
        assert torch.count_nonzero(out[meta.valid_token_count:]).item()==0
        for t in inp:assert torch.count_nonzero(t.grad[meta.valid_token_count:]).item()==0
    return [o.detach() for o in outputs],[[t.grad for t in inp] for inp in inputs],state


@pytest.mark.parametrize('hq,hk',[(16,16),(16,8),(16,4),(16,1)])
@pytest.mark.parametrize('lengths',[(145,130,17,3),(17,3)])
def test_packed_split_real_route_and_padding(lengths,hq,hk,monkeypatch):
    fused=run_qkv(lengths,hq,hk,False)
    def fail(*a,**kw):raise AssertionError('split called fused backward')
    monkeypatch.setattr(packed,'_wrapped_flash_attn_varlen_backward',fail)
    split=run_qkv(lengths,hq,hk,True)
    for a,b in zip(split[0],fused[0]):torch.testing.assert_close(a,b,atol=0,rtol=0)
    result={}
    for j,name in enumerate(('dq','dk','dv')):
        a=torch.cat([x[j].flatten() for x in split[1]]);b=torch.cat([x[j].flatten() for x in fused[1]])
        torch.testing.assert_close(a,b,atol=.004,rtol=.01);result[name]=errors(a,b)
    print('PACKED_SPLIT_CHECK '+json.dumps(dict(lengths=lengths,hq=hq,hk=hk,gradients=result,routes=len(split[2].routes),pending_shapes=sorted(set(split[2].pending_shapes)),padding_zero=True)))


class Block(nn.Module):
    def __init__(self,hq,hk,hidden=64):
        super().__init__();self.hq=hq;self.hk=hk
        self.ln1=nn.LayerNorm(hidden);self.ln2=nn.LayerNorm(hidden)
        self.q=nn.Linear(hidden,hq*128,bias=False);self.k=nn.Linear(hidden,hk*128,bias=False)
        self.v=nn.Linear(hidden,hk*128,bias=False);self.o=nn.Linear(hq*128,hidden,bias=False)
        self.up=nn.Linear(hidden,hidden*2);self.down=nn.Linear(hidden*2,hidden)
        self.rope=RotaryEmbedding(128)
    def forward(self,x,positions,state=None,meta=None,layer=0):
        z=self.ln1(x)
        q=self.q(z).view(-1,1,self.hq,128);k=self.k(z).view(-1,1,self.hk,128);v=self.v(z).view(-1,1,self.hk,128)
        if state is not None:
            freq=self.rope(positions)
            q=apply_rotary_pos_emb(q,freq);k=apply_rotary_pos_emb(k,freq)
            att=packed.dspp_packed_flash_attention(q,k,v,state=state,meta=meta,layer_idx=layer)
        else:
            # Independent full-sequence reference, with double-precision RoPE
            # and softmax, and explicit KV expansion for autograd group sums.
            inv=10000.**(-torch.arange(0,128,2,device=x.device,dtype=x.dtype)/128)
            angles=positions.to(x.dtype)[:,None]*inv[None,:]
            freq=torch.cat([angles,angles],-1)[:,None,None,:]
            def rotate(t):
                a,b=t.chunk(2,-1)
                return t*freq.cos()+torch.cat([-b,a],-1)*freq.sin()
            q,k=rotate(q),rotate(k)
            qt=q[:,0].transpose(0,1);kt=k[:,0].repeat_interleave(self.hq//self.hk,1).transpose(0,1)
            vt=v[:,0].repeat_interleave(self.hq//self.hk,1).transpose(0,1)
            score=qt@kt.transpose(-1,-2)*128**-.5
            mask=torch.ones(x.size(0),x.size(0),device=x.device,dtype=torch.bool).tril()
            att=(score.masked_fill(~mask,-torch.inf).softmax(-1)@vt).transpose(0,1).unsqueeze(1)
        x=x+self.o(att.flatten(1))
        return x+self.down(F.gelu(self.up(self.ln2(x))))


def run_blocks(model,source,upstream,plan,state):
    inputs=[];outputs=[]
    for meta in plan.microbatches:
        x=physical(source,meta);inputs.append(x)
        positions=torch.cat([torch.arange(i.token_offset,i.token_offset+i.token_length,device='cuda') for i in meta.items])
        positions=F.pad(positions,(0,meta.padding_token_count))
        y=x
        for layer,block in enumerate(model):y=block(y,positions,state,meta,layer)
        outputs.append(y)
    for meta,y in reversed(list(zip(plan.microbatches,outputs))):
        grad=physical(upstream,meta).detach()
        y.backward(grad)
    state.finish_iteration()
    assert not state._entries and not state._prepared_cu_seqlens
    dx=unpack(plan,[x.grad for x in inputs],[len(s) for s in source])
    outs=unpack(plan,[y.detach() for y in outputs],[len(s) for s in source])
    return outs,dx,{n:p.grad.clone() for n,p in model.named_parameters()}


@pytest.mark.parametrize('hq,hk',[(4,4),(4,2),(4,1)])
@pytest.mark.parametrize('depth',[1,3])
def test_rope_blocks_dx_parameters_and_reused_state(depth,hq,hk):
    torch.manual_seed(301)
    split_model=nn.ModuleList([Block(hq,hk) for _ in range(depth)]).cuda().to(torch.bfloat16)
    fused_model=copy.deepcopy(split_model);ref_model=copy.deepcopy(split_model).double()
    states=[AuditState(split_attention_backward=b) for b in (True,False)]
    records=[]
    # Reuse the SAME state across iterations and logical batches; the sequence
    # ids restart at zero, exposing stale layer/cache/metadata state.
    for iteration in range(2):
        for logical,lengths in enumerate(((145,70,17,3),(81,19,5))):
            plan=build_dspp_batch_plan(lengths,chunk_size=64)
            gen=torch.Generator(device='cuda').manual_seed(700+iteration*2+logical)
            source=[torch.randn(n,64,device='cuda',generator=gen).to(torch.bfloat16) for n in lengths]
            upstream=[torch.randn(n,64,device='cuda',generator=gen).to(torch.bfloat16)/32 for n in lengths]
            actual=[]
            for model,state in zip((split_model,fused_model),states):
                model.zero_grad(set_to_none=True);state.routes.clear()
                actual.append(run_blocks(model,source,upstream,plan,state))
            ref_model.zero_grad(set_to_none=True);refs=[];refdx=[]
            for x,dy in zip(source,upstream):
                x=x.double().requires_grad_();y=x
                for layer,block in enumerate(ref_model):y=block(y,torch.arange(len(x),device='cuda'))
                y.backward(dy.double());refs.append(y.detach());refdx.append(x.grad)
            refparams={n:p.grad for n,p in ref_model.named_parameters()}
            record={'iteration':iteration,'logical':logical,'checks':{}}
            for name,a,b,r in (
                ('output',torch.cat(actual[0][0]),torch.cat(actual[1][0]),torch.cat(refs)),
                ('dx',torch.cat(actual[0][1]),torch.cat(actual[1][1]),torch.cat(refdx)),
                *[(n,actual[0][2][n],actual[1][2][n],r) for n,r in refparams.items()]):
                torch.testing.assert_close(a,b,atol=.004,rtol=.01,msg=lambda m:f'{name}: {m}')
                for value in (a,b):
                    torch.testing.assert_close(value.double(),r,atol=.04,rtol=.04,msg=lambda m:f'{name}: {m}')
                    assert errors(value,r)['relative_l2']<=.03,name
                record['checks'][name]={'split_vs_fused':errors(a,b),'split_vs_fp64':errors(a,r),'fused_vs_fp64':errors(b,r)}
            records.append(record)
    print('BLOCK_SPLIT_CHECK '+json.dumps(dict(depth=depth,hq=hq,hk=hk,records=records)))


@pytest.mark.parametrize('dtype,dim',[(torch.float16,128),(torch.bfloat16,64)])
def test_reject_before_mutating_cache(dtype,dim):
    plan=build_dspp_batch_plan([145],chunk_size=64)
    state=AuditState(split_attention_backward=True)
    q,k,v=[torch.randn(64,1,4,dim,device='cuda',dtype=dtype,requires_grad=True) for _ in range(3)]
    with pytest.raises(ValueError,match='BF16 and head dimension 128'):
        packed.dspp_packed_flash_attention(q,k,v,state=state,meta=plan.microbatches[0],layer_idx=0)
    assert not state._entries


def test_default_path_does_not_load_split_extension(monkeypatch):
    import megatron.core.kv_cache.fa_split as split
    def fail(*a,**kw):raise AssertionError('default path used split')
    monkeypatch.setattr(split,'prepare_backward',fail)
    run_qkv((145,17,3),4,2,False)


@pytest.mark.parametrize('after_dx', [False, True])
def test_single_stage_schedule_propagates_split_flag(monkeypatch, after_dx):
    from types import SimpleNamespace
    from .test_training_runtime import _patch_single_stage_runtime
    from megatron.core.pipeline_parallel import schedules
    from megatron.core.datasets.dspp_training import build_dspp_training_batch
    from megatron.core.enums import ModelType
    _patch_single_stage_runtime(monkeypatch,debug=0)
    monkeypatch.setattr(schedules,'get_args',lambda:SimpleNamespace(
        dspp_split_attention_backward=True,dspp_history_dkv_after_dx=after_dx,
        variable_seq_debug_num_batches=0,
        kaimm_cuda_synchronize_level=0,variable_seq_slicing=False,use_flash_attn=True))
    class Model(nn.Module):
        model_type=ModelType.encoder_or_decoder
        def __init__(self):
            super().__init__();self.embedding=nn.Embedding(256,64);self.block=Block(4,2);self.head=nn.Linear(64,256)
        def set_input_tensor(self,t):self.input_tensor=t
        def forward(self,tokens,positions,state,labels):
            assert state.split_attention_backward
            assert state.history_dkv_after_dx == after_dx
            x=self.embedding(tokens[0]);x=self.block(x,positions[0],state,state.current_meta,0)
            return F.cross_entropy(self.head(x).float(),labels[0],reduction='none')[None]
    model=Model().cuda().to(torch.bfloat16)
    raw=torch.arange(1,151,device='cuda')[None]
    batch=build_dspp_training_batch(raw,[150],chunk_size=64,validate=True)
    def loss(mask,out):
        value=(mask*out).sum()/mask.sum();return value,{'loss':value.detach()}
    data=iter([(batch,lambda out:loss(batch.loss_mask,out))])
    result=schedules.forward_backward_dspp_no_pipelining(
        forward_step_func=lambda b,s,m:m(b[0],b[1],s,b[3]),
        get_batch_func=lambda i:next(i),data_iterator=data,model=model,
        num_microbatches=1,micro_seq_length=64,kv_cache_class=None)
    assert torch.isfinite(result[0]['loss'])
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_split_flag_is_frozen_at_forward(monkeypatch):
    plan=build_dspp_batch_plan([17],chunk_size=64)
    state=AuditState(split_attention_backward=True)
    q,k,v=[torch.randn(64,1,4,128,device='cuda',dtype=torch.bfloat16,requires_grad=True) for _ in range(3)]
    out=packed.dspp_packed_flash_attention(q,k,v,state=state,meta=plan.microbatches[0],layer_idx=0)
    state.split_attention_backward=False
    def fail(*a,**kw):raise AssertionError("forward-selected split path changed")
    monkeypatch.setattr(packed,"_wrapped_flash_attn_varlen_backward",fail)
    out.sum().backward();state.finish_iteration()
    assert all(t.grad is not None for t in (q,k,v))


@pytest.mark.parametrize('bad,match',[
    ('no_dspp','requires --dspp'),('dtype','requires BF16'),('head_dim','head dimension 128'),
    ('impl','local transformer'),('alibi','ALiBi')])
def test_cli_rejects_unsupported_split_configuration(monkeypatch,bad,match):
    import sys
    from megatron.arguments import parse_args,validate_args
    flags=['--dspp-split-attention-backward','--num-layers','1','--hidden-size','512',
           '--num-attention-heads','4','--seq-length','64','--max-position-embeddings','256',
           '--micro-batch-size','1','--global-batch-size','1','--micro-seq-length','64',
           '--use-flash-attn','--hidden-dropout','0','--attention-dropout','0']
    if bad!='no_dspp':flags+=['--dspp']
    if bad!='dtype':flags+=['--bf16']
    if bad=='head_dim':flags+=['--kv-channels','64']
    if bad=='impl':flags+=['--transformer-impl','transformer_engine']
    if bad=='alibi':flags+=['--use-alibi']
    monkeypatch.setattr(sys,'argv',['test',*flags])
    monkeypatch.setenv('WORLD_SIZE','1');monkeypatch.setenv('RANK','0')
    args=parse_args()
    with pytest.raises(AssertionError,match=match):validate_args(args)
