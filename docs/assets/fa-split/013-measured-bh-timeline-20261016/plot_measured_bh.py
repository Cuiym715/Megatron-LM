"""Plot archived DSPP B/H CUDA-event samples; no invented F/W/communication events."""
from pathlib import Path
import hashlib
import json
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'012-width4096-theory-20261016/hidden4096-long'
font=Path('/plot-fonts/NotoSansCJK-Regular.ttc')
if not font.exists():font=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
font_manager.fontManager.addfont(str(font))
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(font)).get_name(),
    'font.size':11,'axes.titlesize':13,'axes.labelsize':11,'axes.unicode_minus':False,
    'svg.fonttype':'none','savefig.facecolor':'white'})
COLORS={'fused':'#31688E','after_dx':'#16877A','H':'#DB8524'}
RUNS={};sources={}
for mode in ('fused','after_dx'):
    RUNS[mode]={}
    for rank in range(3):
        p=SOURCE/mode/'timeline'/f'iteration_000003_rank_{rank}.json'
        d=json.loads(p.read_text());assert d['task_timing_only']
        assert set(e['kind'][0] for e in d['events'])<=set('BH')
        RUNS[mode][rank]=d
        sources[str(p.relative_to(ROOT.parent))]=hashlib.sha256(p.read_bytes()).hexdigest()

def key(e):return tuple(e[n] for n in ('stage','chunk','logical_microbatch','physical_microbatch'))
def selected(e):return e['chunk']==0 and [(i['token_length'],i['token_offset']) for i in e['items']]==[(2048,12288)]
def clean(ax):
    for s in ('top','right','left'):ax.spines[s].set_visible(False)
    ax.tick_params(axis='y',length=0);ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
def save(fig,name):
    for suffix in ('png','svg'):fig.savefig(ROOT/f'{name}.{suffix}',dpi=180,bbox_inches='tight')
    plt.close(fig)

fused={key(e):e for e in RUNS['fused'][1]['events'] if selected(e)}
split={key(e):e for e in RUNS['after_dx'][1]['events'] if selected(e) and e['kind'][0]=='B'}
history={key(e):e for e in RUNS['after_dx'][1]['events'] if selected(e) and e['kind'][0]=='H'}
assert fused.keys()==split.keys()==history.keys() and len(fused)==2
pairs=[]
fig,axes=plt.subplots(2,1,figsize=(13,7.6),sharex=True)
fig.subplots_adjust(top=.84,bottom=.13,left=.11,right=.97,hspace=.55)
fig.suptitle('DSPP 实测：完整 B 更早结束，历史 H 紧随其后',fontsize=19,fontweight='bold',y=.98)
fig.text(.5,.919,'hidden4096 · GQA32:16 · Q2048 / history12288 · stage1 / VPP chunk0 · 第3轮',ha='center',fontsize=11)
for ax,(k,b0) in zip(axes,sorted(fused.items())):
    b1=split[k];h=history[k];t0=b0['duration_ms'];t1=b1['duration_ms'];th=h['duration_ms']
    hs=h['start_ms']-b1['start_ms'];gap=h['start_ms']-b1['end_ms'];saving=t0-t1
    assert gap>=0 and abs((hs-t1)-gap)<.001
    for y,t,mode in ((1,t0,'fused'),(0,t1,'after_dx')):
        ax.barh(y,t,height=.46,color=COLORS[mode],edgecolor='white',linewidth=.8)
        ax.text(t/2,y,f'完整 B  {t:.3f} ms',color='white',ha='center',va='center',fontsize=12)
        ax.plot(t,y,marker='D',color='#202B36',markersize=5)
        ax.text(t,y+.31,'dx 就绪',ha='center',fontsize=10,color='#202B36')
    ax.barh(0,th,left=hs,height=.46,color=COLORS['H'],edgecolor='white',linewidth=.8,hatch='//')
    ax.text(hs+th/2,0,f'H  {th:.3f} ms',ha='center',va='center',color='white',fontsize=12)
    ax.annotate('',xy=(t1,1.80),xytext=(t0,1.80),arrowprops=dict(arrowstyle='<->',color='#16877A',lw=1.5))
    ax.text((t0+t1)/2,2.02,f'B 缩短 {saving:.3f} ms（{100*saving/t0:.1f}%）',ha='center',color='#146B61',fontsize=11)
    ax.text(.01,.96,f'对应任务 {k[2]+1}：logical={k[2]}，physical={k[3]}',transform=ax.transAxes,fontsize=11,va='top')
    ax.text(hs+th,-.43,f'B→H 间隔 {gap*1000:.2f} μs',ha='right',fontsize=9,color='#555')
    ax.set_yticks([0,1],['拆分后','原融合']);ax.set_ylim(-.60,2.55);ax.set_xlim(0,18.7);clean(ax)
    pairs.append(dict(task_key=k,fused_B_ms=t0,split_B_ms=t1,H_ms=th,
        B_saving_ms=saving,B_saving_percent=100*saving/t0,gap_B_H_ms=gap,
        source_fused_B=b0,source_split_B=b1,source_H=h))
axes[-1].set_xlabel('相对各自 B 起点的时间（ms）')
fig.text(.11,.035,'两次运行按各自 B 起点对齐；展示两个实际样本，非中位数拼图。dx 就绪不等于通信接收完成。\nB 包含完整 stage 输入反向，不含独立 W；此图不表示 iteration 加速。',fontsize=10,color='#505A65')
save(fig,'paired-long-history')

# Full-rank overview: retain each rank's own CUDA-event zero, WITHOUT inventing
# GPU-to-GPU synchronization from CPU launch timestamp alignment.
fig,axes=plt.subplots(3,1,figsize=(15,9.0),sharex=True)
fig.subplots_adjust(top=.83,bottom=.13,left=.10,right=.98,hspace=.34)
fig.suptitle('DSPP 三卡训练：实测 B/H 任务时序',fontsize=19,fontweight='bold',y=.98)
fig.text(.5,.924,'PP3 / VPP2 · hidden4096 · chunk2048 · 变长输入（最长14336 tokens）· 第3轮',ha='center',fontsize=11)
fig.legend(handles=[Patch(color=COLORS['fused'],label='融合 B'),Patch(color=COLORS['after_dx'],label='拆分 B'),
                    Patch(facecolor=COLORS['H'],hatch='//',label='历史 H'),Patch(facecolor='none',edgecolor='#8C3E8F',linewidth=2,label='长历史目标 B/H')],
           loc='upper center',bbox_to_anchor=(.5,.905),ncol=4,frameon=False)
max_end=max(e['end_ms'] for mode in RUNS.values() for p in mode.values() for e in p['events'])
for rank,ax in enumerate(axes):
    for mode,y in (('fused',1),('after_dx',0)):
        for e in RUNS[mode][rank]['events']:
            ish=e['kind'].startswith('H'); target=selected(e)
            ax.barh(y,e['end_ms']-e['start_ms'],left=e['start_ms'],height=.48,
                    color=COLORS['H' if ish else mode],edgecolor='#8C3E8F' if target else 'white',
                    linewidth=1.1 if target else .35,hatch='//' if ish else None)
    ax.set_yticks([0,1],['拆分 B/H','融合 B']);ax.set_ylim(-.52,1.65)
    ax.set_title(f'GPU / stage {rank}（含 VPP chunk0 与 chunk1）',loc='left',fontsize=12)
    ax.set_xlim(0,max_end+20);clean(ax)
axes[-1].set_xlabel('相对各 rank 本次采集 CUDA 锚点的时间（ms）')
fig.text(.10,.043,'仅绘制采集到的 B/H。空白包含未采集的 F/W、通信及等待，不能解释成 bubble。\n各 rank / 运行保留自身时间原点；未建立精确跨 GPU 公共时钟，因此不据此比较跨卡因果时延或 iteration time。',fontsize=10,color='#505A65')
save(fig,'three-stage-bh-overview')

# A local chronological window preserves measured spacing and labels VPP/task IDs.
fig,axes=plt.subplots(2,1,figsize=(13,5.0),sharex=False)
fig.subplots_adjust(top=.76,bottom=.19,left=.10,right=.98,hspace=.75)
fig.suptitle('stage1 局部原始时序：B 与 H 是独立、相邻的任务',fontsize=18,fontweight='bold',y=.98)
fig.text(.5,.87,'保留各运行原始起止位置；窗口覆盖第一个长历史 B0 周围的已采集事件',ha='center',fontsize=11)
windows={}
for ax,mode in zip(axes,('fused','after_dx')):
    es=RUNS[mode][1]['events']; target=next(e for e in es if selected(e) and e['kind']=='B0')
    lo=target['start_ms']-40;hi=target['start_ms']+65
    ev=[e for e in es if e['end_ms']>lo and e['start_ms']<hi];windows[mode]=ev
    for e in ev:
        ish=e['kind'][0]=='H'; istarget=key(e)==key(target)
        ax.barh(0,e['end_ms']-e['start_ms'],left=e['start_ms'],height=.5,color=COLORS['H' if ish else mode],
            edgecolor='#8C3E8F' if istarget else 'white',linewidth=1.3 if istarget else .6,hatch='//' if ish else None)
        mid=(e['start_ms']+e['end_ms'])/2
        ax.text(mid,0,e['kind'],ha='center',va='center',color='white',fontsize=9)
        ax.text(mid,.37,f"L{e['logical_microbatch']}/P{e['physical_microbatch']}",ha='center',fontsize=8)
    ax.set_xlim(lo,hi);ax.set_ylim(-.5,.9);ax.set_yticks([0],['融合' if mode=='fused' else '拆分'])
    ax.set_xlabel('相对该运行 stage1 的 CUDA 锚点（ms）');clean(ax)
fig.text(.10,.035,'B0/B1 与 H0/H1 的下标为 VPP chunk；L/P 为 logical/physical microbatch。\n空白不是已确认的空闲。两次运行不共享时钟，不能直接相减横坐标来计算收益。',fontsize=10,color='#505A65')
save(fig,'stage1-local-window')

metadata=dict(source_sha256=sources,scope='Actual B/H boundaries; no F/W or communication reconstruction.',
    paired_selection='All two stage1/VPP0/Q2048/history12288 samples, sorted by logical microbatch; each mode has its own B-start origin.',
    overview_alignment='Each rank keeps its own CUDA anchor, not CPU-timestamp-corrected; not a globally synchronized pipeline trace.',
    pairs=pairs,local_windows=windows,
    medians_ms={name:statistics.median(p[name] for p in pairs) for name in ('fused_B_ms','split_B_ms','H_ms')},
    counts={mode:{rank:dict(B=sum(e['kind'][0]=='B' for e in d['events']),H=sum(e['kind'][0]=='H' for e in d['events'])) for rank,d in ranks.items()} for mode,ranks in RUNS.items()})
(ROOT/'plotted-data.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(metadata['medians_ms']));print('Generated three PNG/SVG plots from six original JSON files.')
