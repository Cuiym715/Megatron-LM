#!/usr/bin/env python3
"""Draw stage-B actual GPU events and repeated API measurements."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.patches import Patch
import numpy as np

COLORS={'F_attn':'#2878B5','prepare':'#737B85','dQ':'#168C78','current':'#CC79A7',
        'history':'#D55E00','all_dKV':'#A87818','fused_B':'#8064A2'}
NAMES={'F_attn':'attention 前向','prepare':'预处理','dQ':'完整 dQ','current':'当前 dKV',
       'history':'历史 dKV','all_dKV':'全部 dKV','fused_B':'原融合反向'}
CASES=('fused','dq_all','regional','reverse')
LABELS=('原 FA','dQ + 全部 dKV','dQ + 当前 + 历史','历史先算（正确性对照）')


def save(fig,path):
    fig.savefig(path.with_suffix('.png'),dpi=180,bbox_inches='tight',facecolor='white')
    fig.savefig(path.with_suffix('.svg'),bbox_inches='tight',facecolor='white')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--font',type=Path,required=True);args=p.parse_args()
    fontManager.addfont(str(args.font));plt.rcParams.update({'font.family':FontProperties(fname=str(args.font)).get_name(),
                                                          'font.size':11,'axes.unicode_minus':False,'svg.fonttype':'path'})
    data=json.loads(args.evidence.read_text());root=args.evidence.parent
    fig,axes=plt.subplots(3,1,figsize=(14,10.4),sharex=True)
    fig.subplots_adjust(top=.85,bottom=.14,hspace=.42,left=.2,right=.97)
    fig.suptitle('阶段 B：第 3 个 chunk 的真实 GPU 执行',fontsize=21,y=.99)
    fig.text(.5,.945,'L40 · BF16 · 当前 Q=2048，历史 KV=4096，全部 KV=6144 · Hq=16，d=128',ha='center',fontsize=12)
    fig.legend(handles=[Patch(color=COLORS[k],label=NAMES[k]) for k in COLORS],
               loc='upper center',bbox_to_anchor=(.5,.926),ncol=4,frameon=False,fontsize=11)
    max_end=max(e['start_us']+e['duration_us'] for cases in data['timeline'].values() for rows in cases.values() for e in rows)/1000
    for ax,(variant,hk) in zip(axes,(('MHA',16),('GQA',4),('MQA',1))):
        for y,case in enumerate(CASES):
            for e in data['timeline'][variant][case]:
                start,duration=e['start_us']/1000,e['duration_us']/1000
                ax.broken_barh([(start,duration)],(y-.29,.58),color=COLORS[e['phase']],linewidth=0)
                if duration>.11:
                    label={'F_attn':'F','fused_B':'融合反向','all_dKV':'全部 dKV','current':'当前 dKV','history':'历史 dKV','dQ':'dQ'}.get(e['phase'],'')
                    ax.text(start+duration/2,y,label,ha='center',va='center',color='white',fontsize=9)
        ax.set_title(f'{variant}：{16//hk} 个 Q heads / KV head（Hkv={hk}）',loc='left',fontsize=12,pad=8)
        ax.set_yticks(range(4),LABELS);ax.set_ylim(3.6,-.6);ax.tick_params(axis='y',length=0,pad=9)
        ax.set_xlim(0,max_end*1.025);ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
        for s in ax.spines.values():s.set_visible(False)
    axes[-1].set_xlabel('GPU 时间（ms；每个样例从自身第一个 GPU kernel 起点对齐）')
    fig.text(.2,.04,'色块为真实 GPU kernels，包括对应阶段的清零、转换和归约；空白为采样时实际间隔。\n'
             '这是独立 attention 的单次 profiler 样本，不是完整 stage F/B/W，不把 dQ 或“当前 dKV 结束”标作 stage dx。\n'
             '历史在同一次调用中执行；没有跨任务延后。稳定性数据另由无 profiler 的重复计时给出。',fontsize=10,color='#555',linespacing=1.6)
    save(fig,root/'gpu_region_execution')

    fig,axes=plt.subplots(1,3,figsize=(14,4.8),sharey=True)
    fig.suptitle('重复计时：当前实现的 attention backward 调用总耗时',fontsize=18,y=1.02)
    fig.text(.5,.94,'CUDA events · 每项预热 10 次、计时 30 次 · 中位数，误差线为 P10–P90 · 不含前向',ha='center',fontsize=11)
    plot_colors={'fused':'#8064A2','dq_all':'#A87818','regional':'#168C78'}
    for ax,(variant,hk) in zip(axes,(('MHA',16),('GQA',4),('MQA',1))):
        for i,c in enumerate(('fused','dq_all','regional')):
            s=[data['benchmarks'][variant][str(n)][c]['total'] for n in (1,2,3)]
            mid=np.array([r['median_us']/1000 for r in s]);lo=np.array([r['p10_us']/1000 for r in s]);hi=np.array([r['p90_us']/1000 for r in s])
            ax.errorbar(np.arange(3)+(i-1)*.16,mid,yerr=[mid-lo,hi-mid],fmt='o-',capsize=4,color=plot_colors[c],
                        label={'fused':'原融合 FA','dq_all':'dQ + 全部 dKV','regional':'dQ + 当前 + 历史'}[c])
        ax.set_title(f'{variant}（Hkv={hk}）');ax.set_xticks(range(3),['第1段\nKV=2048','第2段\nKV=4096','第3段\nKV=6144'])
        ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('调用总耗时（ms）');axes[-1].legend(frameon=False,fontsize=9,loc='upper left')
    fig.subplots_adjust(top=.8,bottom=.25,wspace=.14)
    fig.text(.125,.05,'包含当前接口的预处理、清零、复制/GQA 归约与 GPU 提交间隔；总时间直接测量，不将分项中位数相加。\n'
             '当前原型尚未调优；这些是局部成本，不是训练吞吐或延后执行的加速结果。',fontsize=10,color='#555',linespacing=1.6)
    save(fig,root/'repeated_backward_timing')

if __name__=='__main__':main()
