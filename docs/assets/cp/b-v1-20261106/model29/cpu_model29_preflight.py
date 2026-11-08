import os,sys,json,runpy
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['WORLD_SIZE']='8';os.environ['RANK']='0';os.environ['LOCAL_RANK']='0'
sys.path.insert(0,os.environ['CP_V1_SOURCE'])
import torch
from megatron.core import dspp_layout
replacement=runpy.run_path(str(Path(__file__).with_name('dspp_layout29.py')))
dspp_layout.logical_layer_ranges.__code__=replacement['logical_layer_ranges'].__code__
from megatron.arguments import parse_args,validate_args
sys.argv=[__file__]+json.loads(Path(__file__).with_name('target-arguments.json').read_text())
a=parse_args();validate_args(a)
assert a.num_layers==29 and a.hidden_size==2048 and a.ffn_hidden_size==5504 and a.num_attention_heads==16
assert a.virtual_pipeline_model_parallel_size==2 and a.num_layers_per_virtual_pipeline_stage==4 and a.data_parallel_size//a.context_parallel_size==1
assert not torch.cuda.is_initialized()
print('MODEL29_NATIVE_ARGUMENTS_PASSED '+json.dumps(dict(layers=a.num_layers,hidden=a.hidden_size,ffn=a.ffn_hidden_size,qheads=a.num_attention_heads,vocab=a.vocab_size,PP=a.pipeline_model_parallel_size,CP=a.context_parallel_size,VPP=a.virtual_pipeline_model_parallel_size,internal_DP=a.data_parallel_size,sample_DP=a.data_parallel_size//a.context_parallel_size,CUDA_initialized=False)))
