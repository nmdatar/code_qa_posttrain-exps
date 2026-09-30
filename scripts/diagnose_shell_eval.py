"""Analyze command use and failures in the frozen shell-only-v1 campaign."""
import json,csv
from pathlib import Path
from collections import Counter
root=Path('artifacts/shell-only-v1-results');out=Path('reports/shell-only-v1')
configs=Path('configs/experiments/shell-only-v1');runs={};paired={}
for arm in ['control','shell']:
 for repeat in [1,2]:
  name=f'{arm}-r{repeat}';cfg=json.loads((configs/(name+'.json')).read_text());folder=root/cfg['output']
  report=json.loads(next((folder/'evaluations').glob('*.json')).read_text())
  counts=Counter();errors=Counter();reasons=Counter();commands=Counter()
  for row in report['results']:
   trace=json.loads((folder/'trajectories'/(row['episode_id']+'.json')).read_text());action={};read_count=0;successful=0;attempts=0
   for e in trace['events']:
    if e['kind']=='generation':
     try: action=json.loads(e['text'])
     except (ValueError,KeyError): action={}
     if not isinstance(action,dict): action={}
    if e['kind']=='parsed_action':
     action=e['value']
     if isinstance(action,dict) and 'tool' in action:
      attempts+=1
      if action['tool']=='shell':commands[action['arguments'].get('command','')]+=1
    if e['kind']=='observation' and isinstance(e['value'],dict):
     obs=e['value']
     if str(obs.get('error','')).startswith('Invalid action'):
      counts['invalid_tool_actions' if 'tool' in action else ('invalid_answer_actions' if 'answer' in action else 'invalid_unparsed_actions')]+=1
      errors[obs.get('detail','')]+=1
     if 'tool' in action and obs.get('exit_code')==0:
      successful+=1
      try: data=json.loads(obs['stdout'])
      except (ValueError,KeyError):data={}
      if action.get('tool')=='read_file' or ('path' in data and 'file_sha256' in data):read_count+=1
   counts.update(episodes=1,tool_attempts=attempts,successful_tool_calls=successful,successful_source_reads=read_count,
                 episodes_without_tool_attempts=int(attempts==0),episodes_without_successful_tool_calls=int(successful==0),
                 episodes_without_source_reads=int(read_count==0),episodes_without_citations=int(not (trace.get('submission') or {}).get('citations')),
                 strict_passes=int(row['reward']==1),unresolved=int(row['status']=='unresolved'))
   reasons.update(trace['verification'].get('reasons',[]))
   paired.setdefault(row['task_id'],{'task_id':row['task_id'],'family_id':row['family_id']}).update({name+'_reward':row['reward'],name+'_status':row['status']})
  ledger=json.loads((root/cfg['spend']['ledger']).read_text())
  runs[name]={'counts':dict(counts),'errors':dict(errors),'verification_reasons':dict(reasons),'commands':dict(commands),'ledger_reserved_usd':ledger['reserved_usd'],'ledger_cap_usd':ledger['cap'],'actual_billing_usd':ledger.get('actual_billing_usd')}
result={'runs':runs,'total_reserved_usd':sum(r['ledger_reserved_usd'] for r in runs.values()),'actual_billing_usd':None,'note':'Recorded conservative reservations including controller allocation, not provider invoice; no retraining.'}
(out/'diagnostics.json').write_text(json.dumps(result,indent=2)+'\n')
with (out/'paired-tasks.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=next(iter(paired.values())).keys());w.writeheader();w.writerows(paired.values())
print(json.dumps(result,indent=2))
