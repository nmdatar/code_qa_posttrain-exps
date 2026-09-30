"""Build a presentation chart and paired analysis from a frozen snapshot."""
import json, sys, statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out=Path(sys.argv[1])
s=json.loads((out/'snapshot.json').read_text())
labels=['Original prompt','Qwen397 subsections']
colors=['#3664AD','#DE7636']
evals={a:{e['optimizer_step']:e for e in s['arms'][a]['evaluations'] if e['optimizer_step']<=7} for a in labels}
common_steps=sorted(set(evals[labels[0]]) & set(evals[labels[1]]))
step=max(common_steps)
assert step>0, 'No shared post-training evaluation yet'
indexed={(a,k):{r['task_id']:r for r in evals[a][k]['results']} for a in labels for k in (0,step)}
common=set.intersection(*[{i for i,r in records.items() if r['status']=='resolved' and r['reward'] is not None} for records in indexed.values()])
assert common
paired={a:{str(k):statistics.mean(indexed[a,k][i]['reward'] for i in common) for k in (0,step)} for a in labels}
for values in paired.values(): values['gain']=values[str(step)]-values['0']
summary={'snapshot_utc':s['fetched_at_utc'],'status':'Preliminary; treatment still running' if s['arms'][labels[1]]['controller_exit'] is None else 'Completed snapshot',
 'shared_post_training_step':step,'matched_questions':len(common),'paired_scores':paired,
 'paired_difference_in_gains':paired[labels[1]]['gain']-paired[labels[0]]['gain'],
 'evaluations':[{ 'arm':a, **{k:e.get(k) for k in ('optimizer_step','mean_reward','resolved','expected','scoring_coverage','completion_rate','mean_output_tokens','mean_tool_calls')}} for a in labels for _,e in sorted(evals[a].items())],
 'limitations':['One seed and historical control','Same-family model judge; no independent human assessment','Treatment changes both inference prompt and training context','Step-6 treatment evaluation pending in this snapshot','Unresolved grades excluded; paired analysis uses the same questions at all four endpoints'],
 'sources':{a:'https://wandb.ai/nmdatar-harvard-university/repository-qa-training/runs/'+s['arms'][a]['config']['run_id'] for a in labels}}
(out/'presentation-data.json').write_text(json.dumps(summary,indent=2)+'\n')
paired_rows=[]
for i in sorted(common):
 row={'task_id':i}
 for a in labels:
  row[a]={'baseline':indexed[a,0][i]['reward'],'step_'+str(step):indexed[a,step][i]['reward']}
 paired_rows.append(row)
(out/'paired-task-scores.json').write_text(json.dumps(paired_rows,indent=2)+'\n')
(out/'prompt-examples.json').write_text(json.dumps(s['arms'][labels[1]]['prompts'],indent=2)+'\n')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False,
 'axes.labelcolor':'#334155','text.color':'#172235','xtick.color':'#475569','ytick.color':'#475569','axes.edgecolor':'#CBD5E1'})
fig,axes=plt.subplots(1,2,figsize=(13.33,7.5),gridspec_kw={'width_ratios':[1.15,1]})
fig.subplots_adjust(top=.75,bottom=.23,left=.07,right=.97,wspace=.30)
fig.text(.07,.94,'Higher baseline scores did not translate into larger early gains',fontsize=21,weight='bold')
fig.text(.07,.885,'Preliminary comparison · Qwen3.5-4B · bash-only GRPO · 32 validation questions · seed 42',fontsize=13,color='#475569')
ax=axes[0]
for a,color in zip(labels,colors):
 xs=common_steps; ys=[evals[a][k]['mean_reward'] for k in xs]
 ax.plot(xs,ys,color=color,marker='o',linewidth=3,markersize=8,label=a)
 for k,y in zip(xs,ys):
  e=evals[a][k]
  ax.annotate(f"{y:.3f}\n{e['resolved']}/32 graded",(k,y),textcoords='offset points',xytext=(10,(-32 if a==labels[0] else 7) if k==0 else (7 if a==labels[0] else -32)),fontsize=10,color=color)
ax.set(xlim=(-.25,step+1),ylim=(0,1),xticks=common_steps,xlabel='Optimizer updates',ylabel='Mean correctness reward')
ax.set_title('Reported evaluation scores',loc='left',pad=14,weight='bold')
ax.grid(axis='y',alpha=.2);ax.legend(loc='upper left',frameon=False,fontsize=11)
ax=axes[1]
gains=[paired[a]['gain'] for a in labels]
bars=ax.bar([0,1],gains,color=colors,width=.58)
for bar,gain in zip(bars,gains):ax.text(bar.get_x()+bar.get_width()/2,gain+.025,f'+{gain:.3f}',ha='center',fontsize=19,weight='bold')
ax.set(xticks=[0,1],xticklabels=['Original\nprompt','Qwen397\nsubsections'],ylim=(0,max(gains)+.17),ylabel=f'Reward gain: baseline to step {step}')
ax.set_title(f'Same {len(common)} questions at all four endpoints',loc='left',pad=14,weight='bold',fontsize=12)
ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.text(.07,.14,f"Matched gain difference (subsections − original): {summary['paired_difference_in_gains']:+.3f} reward points.",fontsize=13,weight='bold')
fig.text(.07,.09,'Step-6 treatment result pending. One seed; historical control; model-graded correctness.\nUnresolved grades are excluded, not scored as zero. Snapshot: '+s['fetched_at_utc']+' UTC.',fontsize=10,color='#64748B')
fig.savefig(out/'prompt-comparison.png',dpi=180,facecolor='white')
fig.savefig(out/'prompt-comparison.svg',facecolor='white')
print(json.dumps(summary,indent=2))
