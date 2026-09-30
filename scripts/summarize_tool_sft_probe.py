import collections,json,random,statistics
from pathlib import Path
from training_pipeline.storage import atomic_json
root=Path('artifacts/sft-tool-prefix-v1-probe-results/artifacts/experiments/sft-tool-prefix-v1-probe')
r=json.loads((root/'report.json').read_text());f=json.loads((root/'fixture.json').read_text());assert r['status']=='complete'
contexts={c['id']:c for c in f['contexts']};initial={t:min([c for c in f['contexts'] if c['task_id']==t],key=lambda c:int(c['id'].rsplit('-',1)[1]))['id'] for t in {c['task_id'] for c in f['contexts']}}
result={'samples':r['samples'],'reserved_usd':r['reserved_usd'],'metrics':{},'delta_after_minus_before':{}}
for policy in ['before','after']:
 rows=[x for x in r['results'] if x['policy']==policy];last=[x for x in rows if x['context_id']!=initial[x['task_id']]]
 result['metrics'][policy]={'samples':len(rows),'valid_actions':sum(x['valid'] for x in rows),'valid_rate':statistics.mean(x['valid'] for x in rows),'strict_json_objects':sum(x['strict_json_object'] for x in rows),'last_context_answers':sum(x['action_kind']=='answer' for x in last),'last_context_samples':len(last),'errors':dict(collections.Counter(x['error'] for x in rows if not x['valid'])),'output_tokens':sum(x['output_tokens'] for x in rows)}
means={}
for key in contexts:
 vals={p:[x['valid'] for x in r['results'] if x['context_id']==key and x['policy']==p] for p in ['before','after']}
 assert all(len(v)==2 for v in vals.values());means[key]=statistics.mean(vals['after'])-statistics.mean(vals['before'])
families=collections.defaultdict(set)
for x in r['results']:families[x['family_id']].add(x['context_id'])
result['delta_after_minus_before']['valid_rate']=statistics.mean(means.values())
for cluster in [False,True]:
 rng=random.Random(42);units=sorted(families) if cluster else sorted(contexts);draws=[]
 for _ in range(10000):
  chosen=rng.choices(units,k=len(units));ids=[i for k in chosen for i in sorted(families[k])] if cluster else chosen;draws.append(statistics.mean(means[i] for i in ids))
 draws.sort();result['repository_clustered_95pct' if cluster else 'paired_context_95pct']=[draws[249],draws[9749]]
result['scope']='Fixed-context sampled action validity only, not rollout success or correctness; secondary diagnostic.'
atomic_json('reports/sft-tool-warmup-v1/probe-results.json',result);print(json.dumps(result,indent=2))
