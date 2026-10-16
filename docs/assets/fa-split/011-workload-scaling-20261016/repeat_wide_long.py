"""One adaptive reverse pair: first wide-long run had variable B durations."""
import copy
import json
from pathlib import Path
import subprocess
root=Path(__file__).resolve().parent
configs=json.loads((root/'configurations.json').read_text())
assert not any(c['name']=='wide-long-repeat' for c in configs)
cfg=copy.deepcopy(next(c for c in configs if c['name']=='wide-long'))
cfg['name']='wide-long-repeat';cfg['order']='after_dx fused'
cfg['reason']='First run: history4096 paired signs differ; history12288 split B range 5.098–6.655 ms.'
cfg['environment']['DSPP_COMBINED_MEASURE_ROOT']+='-repeat'
cfg['environment']['DSPP_COMBINED_MODE_ORDER']=cfg['order']
configs.append(cfg)
(root/'configurations.json').write_text(json.dumps(configs,indent=2)+'\n')
command=['docker','exec','dspp-workload-scaling','env',*[f'{k}={v}' for k,v in cfg['environment'].items()],
         'bash','examples/measure_dspp_combined_tasks.sh']
with (root/'wide-long-repeat-launch.log').open('w') as log:
    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
print('Reverse pair complete')
