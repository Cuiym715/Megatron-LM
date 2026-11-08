import importlib,os,json,runpy,sys
from pathlib import Path
sys.path.insert(0,os.environ['CP_V1_SOURCE'])
initialize=importlib.import_module('megatron.initialize')
native_validate=initialize.validate_args
def validate_experimental(args,*a,**kw):
    # Keep the original argument validation, then explicitly enter an
    # unvalidated research queue configuration in this isolated entry only.
    assert args.dspp_send_inflight_depth==1 and args.dspp_recv_lookahead==0
    result=native_validate(args,*a,**kw)
    assert args.dspp and args.pipeline_model_parallel_size==4 and args.context_parallel_size==2
    args.dspp_send_inflight_depth=2
    path=Path(os.environ['DSPP_CP_BENCH_OUTPUT'])/f'experimental-queue-rank{os.environ["RANK"]}.json'
    path.write_text(json.dumps(dict(original_validated_depth=1,experimental_depth=2,
        validation_status='unvalidated research override; functional only',production_modified=False))+'\n')
    return result
initialize.validate_args=validate_experimental
runpy.run_path(str(Path(__file__).with_name('runtime_background_entry.py')),run_name='__main__')
