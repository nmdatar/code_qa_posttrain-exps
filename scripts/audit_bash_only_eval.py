"""Verify the completed comparison's task matching, tool use, and archived files."""
import json
import hashlib
from collections import Counter
from pathlib import Path
from training_pipeline.storage import read, atomic_json

root=Path('artifacts/bash-only-eval-v1-results')
report=Path('reports/bash-only-eval-v1')
archive=read(report/'archive-checksums.json')
for record in archive:
 raw=(root/record['path']).read_bytes()
 assert len(raw)==record['bytes'] and hashlib.sha256(raw).hexdigest()==record['sha256']
ids=set(read('configs/experiments/grpo-autoresearch/fixes-v1-cohorts.json')['selection'])
configs={};audit={};traces={}
for arm in ['structured','bash']:
 folder=root/'artifacts/experiments'/('bash-only-eval-v1-'+arm)
 configs[arm]=read(folder/'config.json')
 rows=[read(p) for p in (folder/'trajectories').glob('*.json')]
 assert len(rows)==32 and {t['task_id'] for t in rows}==ids
 assert all(t['split']=='development' for t in rows)
 traces[arm]={t['task_id']:t for t in rows}
 table=read(folder/'answers.json')
 actual={dict(zip(table['columns'],r))['episode_id'] for r in table['data']}
 assert actual=={t['episode_id'] for t in rows}
 tools=Counter();errors=Counter();first=Counter();zero=[];truncations=0
 for t in rows:
  if t['usage']['tool_calls']==0:zero.append(t['task_id'])
  first_call=True
  for e in t['events']:
   v=e.get('value',{})
   if e['kind']=='parsed_action' and isinstance(v,dict) and 'tool' in v:
    tools[v['tool']]+=1
    if first_call:
     first[v.get('arguments',{}).get('command',v['tool'])]+=1;first_call=False
   if e['kind']=='observation' and isinstance(v,dict):
    if v.get('error'):errors['invalid_action']+=1
    if v.get('exit_code',0)!=0:errors['nonzero_command_exit']+=1
    truncations+=bool(v.get('stdout_truncated'))
 if arm=='bash':assert set(tools)=={'bash'}
 else:assert set(tools)<={'search_code','read_file','list_files'}
 budget=read(root/'artifacts/project-budget/bash-only-eval-v1'/(arm+'.json'))
 audit[arm]={'tool_counts':dict(tools),'zero_tool_tasks':zero,'first_calls':dict(first),'errors':dict(errors),
  'stdout_truncations':truncations,'episodes_with_max_five_calls':sum(t['usage']['tool_calls']==5 for t in rows),
  'reserved_usd_including_controller':budget['reserved_usd']}
for k in ['model','judge','limits','seed','concurrency','training_reward']:
 assert configs['structured'][k]==configs['bash'][k],k
for ident in ids:
 a,b=traces['structured'][ident],traces['bash'][ident]
 assert a['policy_id']==b['policy_id']
 public=[]
 for t in (a,b):
  initial=next(e for e in t['events'] if e['kind']=='initial')
  assert initial['limits']==next(e for e in a['events'] if e['kind']=='initial')['limits']
  p=json.loads(initial['messages'][1]['content']);p.pop('permitted_tools');public.append(p)
 assert public[0]==public[1],ident
value={'status':'verified','files_checked':len(archive),'trace_count':64,'matched_tasks':32,
 'arms':audit,'reserved_total_usd':sum(x['reserved_usd_including_controller'] for x in audit.values())}
atomic_json(report/'verification.json',value)
print(json.dumps(value,indent=2))
