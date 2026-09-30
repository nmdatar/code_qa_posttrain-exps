from pathlib import Path
import json, csv, hashlib, shutil
from collections import Counter
from reportlab.graphics.shapes import Drawing, String, Rect
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics import renderSVG
from reportlab.lib.colors import HexColor
R=Path(__file__).resolve().parents[2]; O=Path(__file__).resolve().parent
blue=HexColor('#3166b5'); teal=HexColor('#008576'); ink=HexColor('#172b43')
def read(p):return json.loads((R/p).read_text())
def title(d,x,y,s,size=15):d.add(String(x,y,s,fontName='Helvetica',fontSize=size,fillColor=ink))
def drawing(w,h,name,sub):
 d=Drawing(w,h);d.add(Rect(0,0,w,h,fillColor=HexColor('#ffffff'),strokeColor=None));title(d,28,h-32,name,21);title(d,28,h-54,sub,10);return d
def line(d,x,y,w,h,series,xmin,xmax,ymax,ystep):
 p=LinePlot();p.x=x;p.y=y;p.width=w;p.height=h;p.data=series
 p.xValueAxis.valueMin=xmin;p.xValueAxis.valueMax=xmax;p.xValueAxis.valueStep=5 if xmax==16 else 1
 p.yValueAxis.valueMin=0;p.yValueAxis.valueMax=ymax;p.yValueAxis.valueStep=ystep
 for i,c in enumerate([blue,teal]):p.lines[i].strokeColor=c;p.lines[i].strokeWidth=2
 p.xValueAxis.labels.fontSize=9;p.yValueAxis.labels.fontSize=9;d.add(p)
def bars(d,x,y,w,h,data,names,ymax,step):
 b=VerticalBarChart();b.x=x;b.y=y;b.width=w;b.height=h;b.data=data;b.categoryAxis.categoryNames=names
 b.valueAxis.valueMin=0;b.valueAxis.valueMax=ymax;b.valueAxis.valueStep=step
 b.bars[0].fillColor=blue;b.bars[1].fillColor=teal;b.barLabels.nudge=5;b.barLabelFormat='%g';b.barLabels.fontSize=10
 b.categoryAxis.labels.fontSize=10;d.add(b)
def legend(d,x,y,a,b):
 for xx,c,s in [(x,blue,a),(x+150,teal,b)]:d.add(Rect(xx,y-2,14,8,fillColor=c,strokeColor=None));title(d,xx+20,y-2,s,10)
comparison=read('reports/reinforce-v6/comparison.json')
curves={}
for key,folder,run in [('grpo','grpo-long-v6-results','grpo-long-v6-seed42-v1'),('reinforce','reinforce-v6-results','reinforce-v6-seed42-v1')]:
 p=R/f'artifacts/{folder}/artifacts/experiments/{run}/reward-curve.csv';shutil.copy2(p,O/f'{key}-reward-curve.csv');curves[key]=list(csv.DictReader(p.open()))
 assert sum(int(r['contributing_trajectories']) for r in curves[key])==comparison[key]['contributing_trajectories']
d=drawing(1100,430,'Why test REINFORCE after GRPO?','Recreated from archived v6 runs: same 32 training tasks, 128 attempts, 16 scheduled batches, seed 42.')
legend(d,760,398,'GRPO','REINFORCE')
title(d,48,335,'Trajectories contributing to learning',14)
series=[]
for k in curves:
 total=0;rows=[(0,0)]
 for r in curves[k]:total+=int(r['contributing_trajectories']);rows.append((int(r['attempted_batches']),total))
 series.append(rows)
line(d,58,140,260,165,series,0,16,128,32);title(d,85,110,'Scheduled training batch',10);title(d,48,85,'Final contributing trajectories: 60 vs 122',10)
title(d,395,335,'Mean reward in each training batch',14)
line(d,402,140,260,165,[[(int(r['attempted_batches']),float(r['mean_reward'])) for r in curves[k]] for k in curves],1,16,1,.25)
title(d,432,110,'Scheduled training batch',10);title(d,395,85,'Changing-task reward is not held-out quality.',10)
title(d,755,335,'Held-out demonstrated strict credit',14)
bars(d,775,140,255,165,[[100*comparison[k][s]['demonstrated_quality'] for s in ['initial','final']] for k in ['grpo','reinforce']],['Initial','Final'],40,10)
title(d,764,110,'Percent of the same 32 selection tasks',10);title(d,755,85,'GRPO: 8 to 6 passes; REINFORCE: 7 to 9.',10)
title(d,28,48,'Updates: 11/16 vs 16/16. GRPO zero-variance groups: 17/32. REINFORCE uses reward minus the prior running mean.',11)
title(d,28,27,'Difference in held-out changes: +12.5 percentage points; paired 95% interval [-12.5, +37.5]. Single seed; superiority not established.',10)
renderSVG.drawToFile(d,str(O/'grpo-vs-reinforce.svg'))
# Raw claim example, with source and full grader decisions retained.
base=Path('artifacts/atomic-claims-comparison-results/artifacts/experiments/atomic-claims-judge-comparison-reviewed-v1/comparison/Qwen3.5-397B-A17B')
raw=read(base/'original/private/saved-correct.assess.judge-raw.json')['request']
claim={'task_id':'import-c6fc34340a25ce91bf221303','answer':raw['untrusted']['answer'],'citations':raw['untrusted']['citations'],'original_rubric':raw['rubric'],'source_lines':[s for v in raw['untrusted']['evidence'].values() for s in v.get('numbered_source_lines',[]) if 2892<=int(s.split(':')[0])<=2916], 'original_grade':read(base/'original/private/saved-correct.judge.json'),'atomic_grade':read(base/'atomic/private/saved-correct.judge.json'), 'sources':{k:str(base/f'{k}/private/saved-correct.assess.judge-raw.json') for k in ['original','atomic']}}
(O/'claim-example.json').write_text(json.dumps(claim,indent=2)+'\n')
# Extract errors directly, preserving exact emitted text and the resulting observation.
root=R/'artifacts/grader-paraphrase-validation-results/artifacts/experiments/autoresearch-grpo-paginate-v2-seed42/trajectories'
examples=[];counts=Counter();episodes=0;exhausted=0
for p in sorted(root.glob('*.json')):
 a=json.loads(p.read_text())
 if a['split']!='train':continue
 episodes+=1;exhausted+=a['termination']=='budget_exhausted';text=''
 for e in a['events']:
  if e['kind']=='generation':text=e['text']
  if e['kind']=='observation' and 'error' in e.get('value',{}):
   v=e['value'];detail=v.get('detail','');counts[detail]+=1
   examples.append({'source':str(p.relative_to(R)),'episode_id':a['episode_id'],'task_id':a['task_id'],'termination':a['termination'],'generated_text':text,'observation':v})
assert episodes==80 and len(examples)==54
(O/'action-errors.json').write_text(json.dumps({'episodes':episodes,'action_error_count':len(examples),'counts':counts,'examples':examples},indent=2)+'\n')
# Fixed-context SFT probe counts recomputed from all 256 samples.
p=R/'artifacts/sft-tool-prefix-v1-probe-results/artifacts/experiments/sft-tool-prefix-v1-probe/samples.jsonl'
samples=[json.loads(s) for s in p.read_text().splitlines()];metrics={}
for k in ['before','after']:
 rows=[r for r in samples if r['policy']==k];metrics[k]={'n':len(rows),'valid':sum(r['valid'] for r in rows),'malformed':sum(not r['strict_json_object'] for r in rows)}
assert metrics['before']['valid']==109 and metrics['after']['valid']==121
(O/'sft-probe-counts.json').write_text(json.dumps(metrics,indent=2)+'\n')
d=drawing(900,370,'Tool-only SFT: formatting improved, task quality did not','Temperature-1 fixed-context probe and separate greedy end-to-end evaluation; do not pool their denominators.')
legend(d,590,302,'Before SFT','After SFT')
title(d,55,282,'Sampled actions: valid / 128',15)
bars(d,68,100,290,150,[[metrics[k]['valid']] for k in ['before','after']],['Fixed-context probe'],128,32)
title(d,495,282,'End-to-end: strict passes / 32',15)
bars(d,510,100,290,150,[[5],[3]],['Within-run comparison'],8,2)
title(d,55,65,'Malformed/non-object JSON: 8 to 2',11);title(d,495,65,'Completed answers: 32 to 28',11)
title(d,28,29,'Decision: reject this tool-only checkpoint; test complete investigations including supported final answers and stopping.',11)
renderSVG.drawToFile(d,str(O/'sft-format-vs-quality.svg'))
shutil.copy2(R/'reports/reinforce-v6/comparison.json',O/'grpo-reinforce-comparison.json')
print(json.dumps({'probe':metrics,'episodes':episodes,'errors':len(examples),'categories':dict(counts)},indent=2))
