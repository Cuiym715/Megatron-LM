"""Originalb3 split-H functional owner check; no timing or numerical certification."""
import os,sys,json,runpy
from pathlib import Path
sys.path.insert(0,os.environ['CP_V1_SOURCE'])
from megatron.core.kv_cache.dspp_cp_qo_attention import _QOHistoryJob
native=_QOHistoryJob.run
counts=dict(completed_jobs=0,nonempty_parts=0,history_tokens=0)
def checked(self):
 parts=sum(h>0 for _,_,h in self.parts);tokens=sum(h for _,_,h in self.parts)
 result=native(self)
 counts['completed_jobs']+=1;counts['nonempty_parts']+=parts;counts['history_tokens']+=tokens
 return result
_QOHistoryJob.run=checked
try:
 runpy.run_path(str(Path(__file__).with_name('benchmark_entry29_profile.py')),run_name='__main__')
finally:
 (Path(os.environ['DSPP_CP_BENCH_OUTPUT'])/f'history-executed-rank{os.environ["RANK"]}.json').write_text(json.dumps(counts)+'\n')
