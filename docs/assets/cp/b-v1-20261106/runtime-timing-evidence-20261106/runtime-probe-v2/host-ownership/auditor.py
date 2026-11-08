"""Read-only host PID attribution for one bounded container experiment.

Run after the experiment manifest exists. Match entry identity AND the running
container controller in the host /proc ancestor/NSpid chain. Never signal PIDs.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def ancestry(pid):
    rows=[]
    for _ in range(16):
        try:text=Path(f'/proc/{pid}/status').read_text()
        except FileNotFoundError:break
        fields={line.split(':',1)[0]:line.split(':',1)[1].strip() for line in text.splitlines() if ':' in line}
        try:
            stat=Path(f'/proc/{pid}/stat').read_text()
            start_ticks=int(stat[stat.rfind(')')+2:].split()[19])
        except FileNotFoundError:start_ticks=None
        rows.append(dict(host_pid=pid,namespace_pids=[int(x) for x in fields.get('NSpid',str(pid)).split()],
            name=fields['Name'],uid=fields['Uid'],state=fields['State'],start_ticks=start_ticks))
        parent=int(fields['PPid'])
        if parent<=1 or parent==pid:break
        pid=parent
    return rows


def main():
    root=Path(sys.argv[1]).resolve()
    assert root.is_relative_to('/home/test/test01/tinacui/outputs')
    lock=(root/'host-audit.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert (root/'manifest.json').exists()
    out=root/'host-ownership';out.mkdir(exist_ok=False)
    script=Path(__file__).read_bytes();(out/'auditor.py').write_bytes(script)
    state=dict(status='running',pid=os.getpid(),started=time.time(),unknown_observed=False,samples=0,
        script_sha256=hashlib.sha256(script).hexdigest(),scope='host NVML+entry command and ancestor namespace PID attribution; read-only')
    def save():
        temporary=out/'manifest.tmp'
        temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(out/'manifest.json')
    state['boot_id']=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    save();until=time.monotonic()+7200;known={}
    while time.monotonic()<until:
        try:gpu=json.loads((root/'manifest.json').read_text())
        except json.JSONDecodeError:time.sleep(1);continue
        entries={arg for job in gpu.get('jobs',[]) for arg in job.get('command',[]) if arg.endswith('.py')}
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader'],text=True)
        owners=[]
        for row in apps.splitlines():
            if not row.strip():continue
            uuid,pid,memory=[x.strip() for x in row.split(',')]
            command=subprocess.run(['ps','-p',pid,'-o','args='],stdout=subprocess.PIPE,text=True).stdout.strip()
            chain=ancestry(int(pid))
            parent=next((r for r in chain if r['namespace_pids'][-1]==gpu['pid']),None)
            fingerprint=(chain[0]['namespace_pids'][-1],chain[0]['start_ticks'],parent['host_pid']) if chain and parent else None
            entry_match=bool(command and any(entry in command for entry in entries))
            prior=known.get(int(pid))
            seen=bool(chain and chain[0]['start_ticks'] is not None and prior and
                prior[:2]==(chain[0]['namespace_pids'][-1],chain[0]['start_ticks']))
            own=bool((parent and entry_match) or seen)
            if own and entry_match:known[int(pid)]=fingerprint
            if command and not own:state['unknown_observed']=True
            owners.append(dict(uuid=uuid,pid=int(pid),memory=memory,own=own,ancestry=chain,
                exact_entry_match=entry_match,previous_same_start_ticks_and_controller=seen,
                command=command if own else '[other/exited task argv omitted]'))
        with (out/'samples.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),gpu_status=gpu['status'],owners=owners))+'\n')
        state['samples']+=1;save()
        if gpu['status'] in ['complete','stopped','CPU-running','failed']:
            state.update(status='complete',ended=time.time());save();return
        time.sleep(5)
    state.update(status='timeout',ended=time.time());save()


if __name__=='__main__':main()
