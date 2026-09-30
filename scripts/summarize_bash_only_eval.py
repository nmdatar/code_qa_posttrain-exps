"""Matched task-level comparison, preserving unresolved grades as unknown."""
import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from statistics import mean, median
from training_pipeline.storage import read, atomic_json
from training_pipeline.trajectory_viewer import load_traces, render_report


def summarize(results, output):
 out=Path(output);out.mkdir(parents=True,exist_ok=True)
 cohort=read('configs/experiments/grpo-autoresearch/fixes-v1-cohorts.json')
 ids=cohort['selection']
 arms={};summary={};all_records=[]
 for arm in ('structured','bash'):
  root=Path(results)/'artifacts/experiments'/('bash-only-eval-v1-'+arm)
  evals=list((root/'evaluations').glob('*.json'))
  if len(evals)!=1:raise ValueError(f'{arm}: expected exactly one evaluation')
  evaluation=read(evals[0])
  records=load_traces(root)
  all_records.extend((t,{**c,'phase':arm}) for t,c in records)
  rows={}
  for trace,context in records:
   ident=trace['task_id']
   if ident in rows:raise ValueError('Duplicate task '+ident)
   v=trace.get('verification') or {}
   score=v.get('reward') if v.get('status')=='resolved' else None
   diag=v.get('diagnostics',{})
   feedback=diag.get('training_feedback') or {}
   coverage=(feedback.get('components') or {}).get('required_coverage')
   observations=[e.get('value') for e in trace.get('events',[]) if e.get('kind')=='observation']
   errors=[o for o in observations if isinstance(o,dict) and (o.get('error') or o.get('exit_code',0)!=0)]
   rows[ident]={'task_id':ident,'episode_id':trace['episode_id'],'score':score,'status':v.get('status'),
    'termination':trace['termination'],'audit_required_coverage':coverage,'usage':trace['usage'],'tool_errors':len(errors),'reasons':v.get('reasons',[])}
  if set(rows)!=set(ids):raise ValueError(f'{arm}: task identity mismatch')
  arms[arm]=rows
  scores=[r['score'] for r in rows.values() if r['score'] is not None]
  coverages=[r['audit_required_coverage'] for r in rows.values() if r['audit_required_coverage'] is not None]
  summary[arm]={'audit_coverage_resolved':len(coverages),'audit_required_coverage_mean':mean(coverages) if coverages else None,'tasks':len(ids),'resolved':len(scores),'unresolved':len(ids)-len(scores),
   'strict_passes':sum(x==1 for x in scores),'strict_pass_rate_all_tasks':sum(x==1 for x in scores)/len(ids),
   'strict_reward_resolved_mean':mean(scores) if scores else None,'demonstrated_reward_all_tasks':sum(scores)/len(ids),
   'terminations':dict(Counter(r['termination'] for r in rows.values())),
   'mean_tool_calls':mean(r['usage']['tool_calls'] for r in rows.values()),
   'tool_errors':sum(r['tool_errors'] for r in rows.values()),
   'input_tokens':sum(r['usage']['input_tokens'] for r in rows.values()),
   'output_tokens':sum(r['usage']['output_tokens'] for r in rows.values()),
   'median_episode_seconds':median(r['usage']['latency_seconds'] for r in rows.values()),
   'rollout_and_grading_seconds':evaluation['wall_seconds'],
   'reserved_rollout_and_grading_usd':evaluation['reserved_cost_usd']}
 paired=[]
 for ident in ids:
  a,b=arms['structured'][ident],arms['bash'][ident]
  paired.append({'task_id':ident,'structured':a['score'],'bash':b['score'],
   'structured_audit_coverage':a['audit_required_coverage'],'bash_audit_coverage':b['audit_required_coverage'],
   'delta':b['score']-a['score'] if a['score'] is not None and b['score'] is not None else None,
   'structured_termination':a['termination'],'bash_termination':b['termination']})
 deltas=[r['delta'] for r in paired if r['delta'] is not None]
 rng=random.Random(42);boot=sorted(mean(rng.choices(deltas,k=len(deltas))) for _ in range(10000)) if deltas else []
 result={'arms':summary,'paired':{'resolved':len(deltas),'unknown':len(ids)-len(deltas),
  'improved':sum(d>0 for d in deltas),'tied':sum(d==0 for d in deltas),'worsened':sum(d<0 for d in deltas),
  'mean_delta_resolved':mean(deltas) if deltas else None,'task_bootstrap_95_interval':[boot[250],boot[9749]] if boot else None},
  'cohort_manifest_hash':cohort['manifest_hash'],'temperature':0,'model':'Qwen/Qwen3.5-4B',
  'training_updates':0,'confirmation_used':False,
  'limitations':'Single greedy attempt per task. Same-family uncalibrated judge. Unresolved grades are unknown, not failures; all-task demonstrated metrics count observed credit only. Bootstrap excludes judge and generation variability. Bash compact citations use pinned catalog binding, structured compact citations require observed files.'}
 atomic_json(out/'summary.json',result);atomic_json(out/'task-details.json',arms)
 with (out/'paired-results.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
 (out/'trajectories.html').write_text(render_report(all_records,'Bash-only versus structured tools · 32 validation tasks'))
 print(json.dumps(result,indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--output',default='reports/bash-only-eval-v1')
 args=p.parse_args();summarize(args.results,args.output)
