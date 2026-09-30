"""Render aggregate iteration plots from saved evidence; no model calls."""
import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MaxNLocator

OUT=Path('reports/efficiency-rl-v1')
COLORS={'quality-only':'#4169a1','efficiency':'#dc752c'}
LABELS={'quality-only':'Quality-only control','efficiency':'Efficiency bonus'}
records={}
for arm in COLORS:
    root=OUT/'evidence/artifacts/experiments'/('efficiency-rl-v1-'+arm+'-seed42')
    events=[json.loads(s) for s in (root/'events.jsonl').read_text().splitlines()]
    batches=[e for e in events if e['event']=='training_batch']
    updates={e['attempted_batches'] for e in events if e['event']=='update' and e.get('acknowledged')}
    assert [b['attempted_batches'] for b in batches]==list(range(1,17))
    assert all(b['attempted_trajectories']==8 for b in batches)
    cumulative=[sum(j<=i for j in updates) for i in range(1,17)]
    assert all(b['optimizer_step']==sum(j<i for j in updates) for i,b in enumerate(batches,1))
    records[arm]={'batches':batches,'updates':sorted(updates),'cumulative':cumulative}
assert records['quality-only']['cumulative'][-1]==10
assert records['efficiency']['cumulative'][-1]==9
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
 'axes.spines.right':False,'axes.labelcolor':'#354052','text.color':'#172235',
 'xtick.color':'#556070','ytick.color':'#556070','axes.edgecolor':'#d6dde5',
 'grid.color':'#e5eaf0','figure.facecolor':'#ffffff','axes.facecolor':'#ffffff'})
fig,axes=plt.subplots(3,1,figsize=(12,10),sharex=True,gridspec_kw={'height_ratios':[1.15,1.05,.85]})
fig.subplots_adjust(left=.10,right=.89,top=.83,bottom=.18,hspace=.33)
fig.text(.10,.963,'What changed across training iterations?',fontsize=23,weight='bold')
fig.text(.10,.928,'16 attempted batches per arm · one seed · no additional training run',fontsize=12,color='#556070')
for arm,r in records.items():
    x=list(range(1,17));color=COLORS[arm]
    for ax,key in zip(axes[:2],['mean_output_tokens','accepted_rate']):
        y=[b[key] for b in r['batches']]
        ax.plot(x,y,color=color,alpha=.24,lw=1.2,marker='o',ms=3)
        rolling=[statistics.fmean(y[i-3:i+1]) for i in range(3,16)]
        ax.plot(x[3:],rolling,color=color,lw=2.8,label=LABELS[arm])
    axes[2].step([0]+x,[0]+r['cumulative'],where='post',color=color,lw=2.2)
    axes[2].scatter(r['updates'],[r['cumulative'][i-1] for i in r['updates']],s=28,color=color,zorder=4)
    axes[2].annotate(str(r['cumulative'][-1])+' updates',(16,r['cumulative'][-1]),xytext=(10,0),textcoords='offset points',color=color,va='center',weight='bold')
axes[0].set_title('Response length during training',loc='left',weight='bold',pad=10)
axes[0].set_ylabel('Generated tokens / attempt');axes[0].set_ylim(0,330)
axes[1].set_title('Accepted answers during training',loc='left',weight='bold',pad=10)
axes[1].set_ylabel('Acceptance rate');axes[1].set_ylim(-.02,.55);axes[1].yaxis.set_major_formatter(PercentFormatter(1))
axes[2].set_title('Actual parameter updates',loc='left',weight='bold',pad=10)
axes[2].set_ylabel('Cumulative updates');axes[2].set_ylim(-.3,11);axes[2].yaxis.set_major_locator(MaxNLocator(integer=True))
axes[2].set_xlabel('Attempted training batch');axes[2].set_xticks(range(1,17));axes[2].set_xlim(.7,16.5)
for ax in axes:ax.grid(axis='y');ax.set_axisbelow(True)
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper left',bbox_to_anchor=(.095,.90),frameon=False,ncol=2)
fig.text(.10,.087,'Faint dots / lines: raw batch means. Bold lines: trailing 4-batch means (32 attempts).',fontsize=10,color='#556070')
fig.text(.10,.063,'Staircase dots mark acknowledged updates; flat segments mean the update was skipped.',fontsize=10,color='#556070')
fig.text(.10,.032,'Task difficulty changes between batches. These training curves do not establish held-out improvement.',fontsize=10,weight='bold')
fig.savefig(OUT/'iteration-overview.png',dpi=180)
fig.savefig(OUT/'iteration-overview.svg')
plt.close(fig)
summary=json.loads((OUT/'summary.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(12,4.7))
fig.subplots_adjust(left=.08,right=.97,top=.69,bottom=.25,wspace=.30)
fig.text(.08,.93,'Held-out performance: only two measurements per arm',fontsize=20,weight='bold')
fig.text(.08,.86,'32 fixed selection questions · no intermediate checkpoint evaluations',fontsize=11,color='#556070')
for ax,key,title in zip(axes,['mean_output_tokens','accepted_rate'],['Generated tokens / task','Accepted answers']):
    for arm in COLORS:
        es=summary['arms'][arm]['evaluations'];color=COLORS[arm]
        for phase,marker in [('initial','o'),('final','D')]:
            e=es[phase];x=e['optimizer_step'];y=e[key]
            ax.scatter(x,y,color=color,marker=marker,s=60,zorder=3)
            label=f'{y:.1f}' if key=='mean_output_tokens' else f"{e['accepted_answers']}/32"
            ax.annotate(label,(x,y),xytext=(7,6 if arm=='quality-only' else -13),textcoords='offset points',fontsize=10,color=color)
    ax.set_title(title,loc='left',weight='bold');ax.set_xlim(-.8,12);ax.set_xticks([0,3,6,9,10]);ax.set_xlabel('Acknowledged optimizer updates');ax.grid(axis='y')
    if key=='accepted_rate':ax.set_ylim(0,.43);ax.yaxis.set_major_formatter(PercentFormatter(1))
    else:ax.set_ylim(225,260)
fig.legend(handles,labels,loc='upper left',bbox_to_anchor=(.075,.80),frameon=False,ncol=2)
fig.text(.08,.10,'Circles = initial policy; diamonds = final policy. No connecting curve: the intervening values were not measured.',fontsize=10,color='#556070')
fig.text(.08,.045,'Observed endpoints are noisy; neither arm met the required final grading coverage. No efficiency win established.',fontsize=10,weight='bold')
fig.savefig(OUT/'heldout-endpoints.png',dpi=180)
fig.savefig(OUT/'heldout-endpoints.svg')
plt.close(fig)
aggregate=[]
for i in range(16):
    row={'batch':i+1}
    for arm,r in records.items():
        b=r['batches'][i]
        row[arm]={'output_tokens':b['mean_output_tokens'],'accepted_rate':b['accepted_rate'],
                  'cumulative_updates':r['cumulative'][i],'updated':i+1 in r['updates']}
    aggregate.append(row)
(OUT/'iteration-aggregates.json').write_text(json.dumps(aggregate,indent=2))
print('Rendered iteration-overview and heldout-endpoints (PNG + SVG).')
