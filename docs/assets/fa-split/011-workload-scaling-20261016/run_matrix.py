"""Sequential GPU runs; execute on host with the documented container alive."""
import json
from pathlib import Path
import subprocess

archive = Path(__file__).resolve().parent
container_archive = '/workspace/src/Megatron-LM-kwai/docs/assets/fa-split/' + archive.name
configs = [
    dict(name='small-short', hidden=512, ffn=1024, heads=4, groups=2, long=False, order='fused after_dx'),
    dict(name='wide-short', hidden=2048, ffn=4096, heads=16, groups=8, long=False, order='after_dx fused'),
    dict(name='small-long', hidden=512, ffn=1024, heads=4, groups=2, long=True, order='after_dx fused'),
    dict(name='wide-long', hidden=2048, ffn=4096, heads=16, groups=8, long=True, order='fused after_dx'),
]
for cfg in configs:
    name = cfg['name']
    data = ('/workspace/.cache/dspp-fa-split/workload-scaling-20261016/long-data' if cfg['long']
            else '/workspace/.cache/dspp-fa-split/clean-timing-measure-final-20261016/data')
    env = dict(DSPP_COMBINED_MEASURE_ROOT=container_archive+'/'+name,
               DSPP_COMBINED_DATA_PATH=data, DSPP_COMBINED_HIDDEN_SIZE=str(cfg['hidden']),
               DSPP_COMBINED_FFN_HIDDEN_SIZE=str(cfg['ffn']), DSPP_COMBINED_Q_HEADS=str(cfg['heads']),
               DSPP_COMBINED_QUERY_GROUPS=str(cfg['groups']), DSPP_COMBINED_SEQ_LENGTH='16384' if cfg['long'] else '8192',
               DSPP_COMBINED_CHUNK_LENGTH='2048', DSPP_COMBINED_MODE_ORDER=cfg['order'])
    cfg['environment'] = env
(archive/'configurations.json').write_text(json.dumps(configs, indent=2)+'\n')
for cfg in configs:
    name=cfg['name']
    print('Starting '+name, flush=True)
    command=['docker','exec','dspp-workload-scaling','env',*[f'{k}={v}' for k,v in cfg['environment'].items()],
             'bash','examples/measure_dspp_combined_tasks.sh']
    with (archive/(name+'-launch.log')).open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    result=json.loads((archive/name/'task-times.json').read_text())
    print(name, result['boundary_checks'], flush=True)
    for group in result['by_shape_stage']:
        if group['stage']==1 and group['model_chunk']==0 and group['layout_q_history'] in ([[2048,4096]],[[2048,12288]]):
            print(json.dumps(group),flush=True)
