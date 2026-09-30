"""Summarize preserved partial teacher collection without inferring model gains."""
import json,collections
from pathlib import Path
root=Path('artifacts/expanded-studies-phase1-billing-stop-results');run=root/'artifacts/experiments/expanded-teacher-collection-v1-seed42';rows=[];invalid=[]
for p in (run/'trajectories').glob('*.json'):
 try:rows.append(json.loads(p.read_text()))
 except (ValueError,UnicodeError):invalid.append(str(p))
verified=[r for r in rows if isinstance(r.get('verification'),dict)];resolved=[r for r in verified if r['verification']['status']=='resolved'];reasons=collections.Counter(str(x) for r in verified if r['verification']['status']!='resolved' for x in r['verification'].get('reasons',[]))
ledgers={p.stem:json.loads(p.read_text()) for p in (root/'artifacts/project-budget/expanded-studies-v1').glob('*.json')}
summary={'status':'stopped_provider_billing_402','planned_teacher_trajectories':1200,'trajectory_records':len(rows),'verification_records':len(verified),'resolved':len(resolved),'strict_passes':sum((r['verification'].get('diagnostics') or {}).get('strict_score')==1 for r in verified),'mean_factual_reward_resolved':sum(r['verification'].get('reward') or 0 for r in resolved)/len(resolved) if resolved else None,'unresolved_reasons':dict(reasons),'invalid_trajectory_json':invalid,'reserved_usd':{k:x['reserved_usd'] for k,x in ledgers.items()},'ledger_snapshot_total_reserved_usd':sum(x['reserved_usd'] for x in ledgers.values()),'phase1_reserved_usd':354.0393030000065+18.233856,'ledger_snapshot_note':'Student ledger was downloaded after the separately authorized first REINFORCE launch added another controller reservation; preserve that newer authoritative snapshot, but do not attribute it to stopped phase1.','provider_invoiced_usd':None,'reinforce_updates':0,'sft_updates':0,'confirmation_touched':False,'new_student_checkpoint':None,'automation_at_billing_stop':'PAUSED'}
Path('reports/expanded-studies/BILLING-STOP.json').write_text(json.dumps(summary,indent=2))
text=f'''# Expanded study stopped at provider billing block

Tinker returned HTTP 402 (access blocked due to billing status). In accordance with the user's instruction to continue until credits run out, the controller was synced and terminated; monitoring was paused at the stop. No credits were purchased by the agent. The user subsequently reported adding credits and authorized continuation under the unchanged cap. Provider billing status is known; an exact remaining credit balance is not available.

The partial teacher run preserved {summary['trajectory_records']} trajectory records of 1,200 planned, with {summary['verification_records']} verification records and {summary['resolved']} resolved grades. There are {summary['strict_passes']} strict passes before source/token/lineage SFT admission. The required 200 distinct fully admitted demonstrations have not been established. Missing grades and infrastructure failures are not model errors.

Stopped phase1 reservations: ${summary['phase1_reserved_usd']:.6f}, not invoices. The archived ledger snapshot totals ${summary['ledger_snapshot_total_reserved_usd']:.6f} because the student ledger was downloaded after the separately authorized REINFORCE first launch reserved its controller. Per-ledger totals and unresolved reasons are in BILLING-STOP.json. The $5,000 additional cap was not exhausted; the provider billing block occurred first.

No REINFORCE or SFT optimizer update occurred in this campaign, no new student checkpoint was trained, and confirmation remains untouched. The queued direct REINFORCE run did not start in phase1; a separate first-launch campaign was prepared after the user replenished credits. These partial teacher results do not establish model improvement or complete any of the three experiments.

Artifacts: `{root.resolve()}`. File hashes and JSON parse diagnostics: billing-stop-archive.json. The stopped frozen run must not be blindly resumed or submitted again. Any future recovery needs credit availability, archived-state reconciliation and an explicitly designed continuation without duplicate training.

[Teacher run on W&B](https://wandb.ai/nmdatar-harvard-university/repository-qa-training/runs/expanded-teacher-collection-v1-seed42)
'''
Path('reports/expanded-studies/BILLING-STOP.md').write_text(text);print(json.dumps(summary,indent=2))
