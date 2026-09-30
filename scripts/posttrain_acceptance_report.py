"""Summarize recorded evidence; never infer an update from a rollout alone."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from posttrain.storage import read,atomic
from posttrain.tracking import EventLog
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
d=Path(a.run).resolve();events=EventLog(d).read();c=read(d/'config.json')
updates=[e for e in events if e['kind']=='update'];evaluations=[read(e['artifact']) for e in events if e['kind']=='evaluation']
ledger=read(c['budget_ledger']);charged=sum(x['charged_usd'] for x in ledger['calls'].values())
result={'run':str(d),'model':c['model'],'simulated':c['backend']=='fake','acknowledged_updates':len(updates),'updates':updates,
 'checkpoint':read(d/'latest-checkpoint.json') if (d/'latest-checkpoint.json').exists() else None,
 'evaluation_steps':[e['optimizer_step'] for e in evaluations],
 'evaluations':[ {k:v for k,v in e.items() if k!='rows'} for e in evaluations],
 'data':{'source_reviewed_training_tasks':100,'ready_training_environments':10,'development_cohort':20,'original_development_tasks':992,'human_calibration_passed':False},
 'budget':{'conservatively_charged_or_reserved_usd':charged,'provider_actual_billing_known':False,'dispatch_remaining_usd':max(0,18-charged),'cap_usd':20},
 'limitations':['Diagnostic integration only; not evidence of improved model quality.','Human reward calibration remains pending.','Two development tasks used in paid diagnostic; full20-task configuration is separate.']}
reload_path=Path('reports/posttrain/qwen4b-checkpoint-reload.json')
if reload_path.is_file():
    reload_result=read(reload_path)
    if Path(reload_result['checkpoint']).is_relative_to(d):result['checkpoint_reload']=reload_result
atomic('reports/posttrain/repository-rl-acceptance.json',result)
lines=['# Repository RL acceptance evidence','',f"Model: `{c['model']}`. Run: `{d.name}`.",'',f"Acknowledged optimizer updates: **{len(updates)}**. Development evaluation steps: {result['evaluation_steps']}.",'',f"Conservatively charged/reserved across campaign: **${charged:.3f}**. Actual provider billing is not yet known.",'','## Evidence','',f"- [Durable events]({d}/events.jsonl)",f"- [Local dashboard]({d}/report.html)"]
if result['checkpoint']:lines.append(f"- [Checkpoint manifest]({result['checkpoint']['manifest']})")
lines+=['','## Limits','']+['- '+x for x in result['limitations']]
Path('requirements/POSTTRAIN_RL_ACCEPTANCE.md').write_text('\n'.join(lines)+'\n')
print({'acknowledged_updates':len(updates),'evaluation_steps':result['evaluation_steps'],'charged_or_reserved':charged})
