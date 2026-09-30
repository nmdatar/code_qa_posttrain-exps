import json,statistics,collections,re
from pathlib import Path
root=Path('artifacts/experiment-01-v4-results')
rows=[];unresolved=[]
def percentile(xs,q):
 xs=sorted(xs)
 if not xs:return None
 pos=(len(xs)-1)*q;i=int(pos);return xs[i]+(xs[min(i+1,len(xs)-1)]-xs[i])*(pos-i)
for p in sorted(root.glob('artifacts/experiments/*')):
 evaluations=list(p.glob('evaluations/*.json'));b=p/'benchmark.json'
 if not evaluations and not b.exists():continue
 report=json.loads((evaluations[0] if evaluations else b).read_text())
 traces=[json.loads(t.read_text()) for t in p.glob('trajectories/*.json')]
 complete=[t for t in traces if t['termination']=='completed'];resolved=[t for t in traces if (t.get('verification') or {}).get('status')=='resolved']
 scores=[t['verification']['reward'] for t in resolved];latencies=[t['usage']['latency_seconds'] for t in traces if 'latency_seconds' in t.get('usage',{})]
 wall=report['wall_seconds'];cost=report['reserved_cost_usd']
 grades=[json.loads(t.read_text()) for t in p.glob('private/*.judge-raw.json')]
 timings={k:sum(t.get('usage',{}).get(k,0) for t in traces) for k in ['provision_seconds','generation_seconds','action_seconds','verification_seconds','cleanup_seconds']}
 tracking=[json.loads(t.read_text()) for t in p.glob('tracking*.json')]
 row={'run_id':p.name,'expected':report.get('expected',report.get('expected_episodes')),'attempted':len(traces),'resolved':len(resolved),'completed':len(complete),'scoring_coverage':report['scoring_coverage'],'completion_rate':report['completion_rate'],'demonstrated_quality':sum(scores)/len(traces) if traces else None,'resolved_mean':statistics.mean(scores) if scores else None,'score_distribution':dict(collections.Counter(scores)),'wall_seconds':wall,'completed_per_minute':len(complete)*60/wall,'resolved_per_minute':len(resolved)*60/wall,'output_tokens':sum(t['usage'].get('output_tokens',0) for t in traces),'tool_calls':sum(t['usage'].get('tool_calls',0) for t in traces),'latency_p50':percentile(latencies,.5),'latency_p95':percentile(latencies,.95),'timing_sum_seconds':timings,'judge_queue_seconds':sum(g.get('timing',{}).get('queue_seconds',0) for g in grades),'judge_sampling_seconds':sum(g.get('timing',{}).get('sampling_seconds',0) for g in grades),'reserved_episode_cost_usd':cost,'reserved_cost_per_scored':cost/len(resolved) if resolved else None,'actual_billing_usd':None,'terminations':dict(collections.Counter(t['termination'] for t in traces)),'invalid_action_events':sum(e.get('kind')=='observation' and str(e.get('value',{}).get('error','')).startswith('Invalid action') for t in traces for e in t['events']),'tracking_ids':[t['id'] for t in tracking],'evaluation_file':str(evaluations[0] if evaluations else b)}
 row['tokens_per_second']=row['output_tokens']/wall;rows.append(row)
 for t in traces:
  v=t.get('verification') or {}
  if v.get('status')=='unresolved':
   error=p/'private'/(t['episode_id']+'.judge-error.json')
   unresolved.append({'run_id':p.name,'task_id':t['task_id'],'episode_id':t['episode_id'],'reasons':v.get('reasons'),'error':json.loads(error.read_text()) if error.exists() else None,'ambiguities':v.get('diagnostics',{}).get('semantic',{}).get('disagreements',[])})
Path('reports/experiment-01-v4/metrics.json').write_text(json.dumps(rows,indent=2)+'\n')
Path('reports/experiment-01-v4/unresolved.json').write_text(json.dumps(unresolved,indent=2)+'\n')
print(json.dumps([{k:r[k] for k in ['run_id','attempted','resolved','completed','demonstrated_quality','wall_seconds','reserved_episode_cost_usd']} for r in rows],indent=2))
