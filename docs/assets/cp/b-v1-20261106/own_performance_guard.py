"""Stop only identity-verified direct timeout child after foreign GPU admission."""
import json,os,pathlib,signal,sys,time
root=pathlib.Path(sys.argv[1]);record=root/'own-performance-guard.json'
assert not record.exists()
state=dict(pid=os.getpid(),started=time.time(),status='running',stops=[])
def save():
 p=record.with_suffix('.tmp');p.write_text(json.dumps(state,indent=2)+'\n');p.replace(record)
def proc(pid):
 p=pathlib.Path('/proc')/str(pid)
 try:
  status={r.split(':',1)[0]:r.split(':',1)[1].strip() for r in (p/'status').read_text().splitlines() if ':' in r}
  st=(p/'stat').read_text().rsplit(')',1)[1].split()
  return dict(pid=int(pid),parent=int(status['PPid']),ns=[int(v) for v in status.get('NSpid',str(pid)).split()],start_ticks=int(st[19]),cmd=(p/'cmdline').read_bytes().split(b'\0')[:-1])
 except (OSError,KeyError,ValueError):return None
save()
try:
 for _ in range(1440):
  d=json.loads((root/'manifest.json').read_text());a=json.loads((root/'host-ownership/manifest.json').read_text())
  if a.get('unknown_observed'):
   state['excluded_foreign_observed']=True
   for j in d.get('jobs',[]):
    if j.get('exit_code') is not None or not j.get('pid'):continue
    for p in pathlib.Path('/proc').iterdir():
     if not p.name.isdigit():continue
     v=proc(p.name)
     if not v or v['ns'][-1]!=j['pid']:continue
     par=proc(v['parent'])
     if not par or par['ns'][-1]!=d['pid']:continue
     argv=[v.decode(errors='replace') for v in v['cmd']]
     if not argv or pathlib.Path(argv[0]).name!='timeout':continue
     if argv[-len(j['command']):]!=j['command']:continue
     identity=proc(p.name)
     if not identity or identity['start_ticks']!=v['start_ticks']:continue
     os.kill(v['pid'],signal.SIGTERM)
     state['stops'].append(dict(time=time.time(),pid=v['pid'],namespace_pid=j['pid'],start_ticks=v['start_ticks'],parent=v['parent'],command=argv,reason='foreign GPU observed; formal timing excluded'))
     save()
   state.update(status='excluded',ended=time.time());save();break
  if d['status'] in ('complete','stopped'):
   state.update(status='complete',ended=time.time());save();break
  time.sleep(3)
 else:state.update(status='bounded-monitor-ended',ended=time.time());save()
except Exception as e:
 state.update(status='error',error=repr(e),ended=time.time());save();raise
