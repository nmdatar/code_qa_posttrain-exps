import json,hashlib,collections,csv
from pathlib import Path
base=Path.cwd();out=base/'reports/reward-shaping-v2'
roots=[base/'artifacts/reward-shaping-v1-results',base/'artifacts/reward-shaping-v2-results']
summary=[];answers=[];files={}
for root in roots:
 for path in root.rglob('*'):
  if path.is_file():files[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
 for p in sorted((root/'artifacts/experiments').iterdir()):
  if not p.is_dir():continue
  evals=list(p.glob('evaluations/*.json'));b=p/'benchmark.json'
  if not evals and not b.exists():continue
  report=json.loads((evals[0] if evals else b).read_text());config=json.loads((p/'config.json').read_text())
  traces=[json.loads(f.read_text()) for f in p.glob('trajectories/*.json')]
  expected=report.get('expected',report.get('expected_episodes'));assert len(traces)==expected
  assert len(json.loads((p/'answers.json').read_text())['data'])==expected
  assert all(t['verification'] for t in traces)
  events=[json.loads(l) for l in (p/'events.jsonl').read_text().splitlines()]
  assert not any(e.get('event')=='update' for e in events)
  tracking=json.loads(next(p.glob('tracking*.json')).read_text())
  sem=[t for t in traces if t['verification'].get('diagnostics',{}).get('semantic')]
  row={'run_id':p.name,'grader':config['environment']['grading_version'],'reward_version':config['training_reward']['version'],
   'attempted':len(traces),'resolved':sum(t['verification']['status']=='resolved' for t in traces),
   'completed':sum(t['termination']=='completed' for t in traces),'semantic_audits':len(sem),
   'resolved_fractional_training_rewards':sum(t['verification']['status']=='resolved' and 0<(t['verification'].get('diagnostics',{}).get('training_reward') or 0)<1 for t in traces),
   'strict_zero_positive_training':sum(t['verification']['status']=='resolved' and t['verification'].get('diagnostics',{}).get('strict_score')==0 and (t['verification'].get('diagnostics',{}).get('training_reward') or 0)>0 for t in traces),
   'scoring_coverage':report['scoring_coverage'],'completion_rate':report['completion_rate'],
   'strict_demonstrated_quality':sum((t['verification'].get('diagnostics',{}).get('strict_score') or 0) for t in traces)/len(traces),
   'wall_seconds':report['wall_seconds'],'reserved_episode_usd':report['reserved_cost_usd'],
   'reward_distribution':dict(collections.Counter(str(t['verification']['reward']) for t in traces)),
   'unresolved_reasons':dict(collections.Counter(reason for t in traces if t['verification']['status']=='unresolved' for reason in t['verification']['reasons'])),
   'reward_signal':report.get('reward_signal'),'mechanical_repairs':len(list(p.glob('private/*.repair-1.judge-raw.json'))),
   'tracking_failures':sum(e.get('event')=='tracking_failure' for e in events),
   'wandb':'https://wandb.ai/nmdatar-harvard-university/repository-qa-training/runs/'+tracking['id'],
   'report_path':str(evals[0] if evals else b),'log_path':str(next(root.glob('campaigns/*/'+p.name+'.log'))),
   'answers_path':str(p/'answers.json')}
  summary.append(row)
  for t in traces:
   v=t['verification'];d=v.get('diagnostics',{})
   answers.append({'run_id':p.name,'task_id':t['task_id'],'episode_id':t['episode_id'],'termination':t['termination'],
    'status':v['status'],'strict_score':d.get('strict_score'),'training_reward':d.get('training_reward'),
    'applied_reward':v['reward'],'reasons':'; '.join(v['reasons'])})
final=next(r for r in summary if r['run_id']=='reward-shaping-v2-grader-v5-four-attempt-diagnostic')
control=next(r for r in summary if r['run_id']=='reward-shaping-v2-grader-v5-selection-validation')
prior=json.loads((base/'reports/reward-shaping-v1/ledger-before.json').read_text())['reserved_usd']
current=json.loads((out/'ledger-final.json').read_text())['reserved_usd']
gates={'training_scoring_coverage':final['scoring_coverage']>=.95,'training_answer_completion':final['completion_rate']>=.90,
 'training_group_variation':final['reward_signal']['training_contributing_groups']>0,
 'selection_scoring_coverage':control['scoring_coverage']>=.95,'selection_answer_completion':control['completion_rate']>=.90}
result={'final':final,'control':control,'runs':summary,'gates':gates,'small_diagnostic_ready':all(gates.values()),
 'optimizer_updates':0,'new_reserved_usd':current-prior,'cumulative_reserved_usd':current,'actual_billing_usd':None,
 'episodes':sum(r['attempted'] for r in summary)}
(out/'results.json').write_text(json.dumps(result,indent=2)+'\n');(out/'checksums.json').write_text(json.dumps(files,indent=2)+'\n')
with (out/'per-attempt.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(answers[0]));w.writeheader();w.writerows(answers)
lines=['# Reward shaping and grader reliability — implementation and validation','',
 '**Implementation complete; no GRPO updates were run.** The final train reward is `supported-coverage-v2` and the final grader is `all-claims-v5`. Strict evaluation retains the original scoring criterion. All historical controls and unsuccessful diagnostics are retained.','',
 '## What changed','',
 '- Training credit comes only from supported required facts, with explicit penalties for contradictions, unsupported assertions, and citation defects. Scores are clipped to [0,1]. Extra correct prose earns no positive credit and cannot dilute penalties.',
 '- Development evaluations and checkpoint selection use strict scores. Trajectories and answer tables retain strict score, training reward, and component breakdown separately.',
 '- Overlapping evidence is compacted without dropping source lines or keys. Reference context and output budgets increased. At most one mechanical repair per judge stage is priced and logged; genuine uncertainty and valid unfavorable grades are not retried.',
 '- Grader v5 checks extraction ranges against catalog line counts/hashes before source reads. The exact v4 failure now reaches bounded repair instead of immediately losing a whole group.',
 '- Unresolved results remain null and exclude the whole training group. The launcher rejects the retired v1 reward config.','',
 'See [reward contract and implementation details]('+str(base/'docs/REWARD_SHAPING.md')+').','',
 '## Frozen checks','',
 '474 tests passed before the source-range follow-up; the final test log records the final suite. Frozen source-grounded synthetic examples rank correct=1.0, partial=0.5, incorrect=0, citation-defective=0.9, unsupported-extra=0.8, and unsupported-extra-plus-true-filler=0.8. False execution claims receive 0; unresolved judgments stay null. These checks validate reward arithmetic, not human calibration or live judge correctness.','',
 'An offline projection of the historical selection audits gave positive shaped credit to 11 of 20 strictly-zero answers with complete semantic audits. Four other zero answers had no semantic audit; unresolved answers remained null. This illustrates available feedback but does not establish on-policy group variation. No confirmation answers were used to tune the reward.','',
 '## All live runs','',
 '| Run | Grader / reward | Scored | Completed | Strict quality | Contributing groups (strict → training) |','|---|---|---:|---:|---:|---:|']
for r in summary:
 signal=r['reward_signal'];gain=f"{signal['strict_contributing_groups']} → {signal['training_contributing_groups']} ({signal['eligible_groups']} eligible)" if signal else 'selection evaluation'
 lines.append(f"| {r['run_id']} | {r['grader']} / {r['reward_version']} | {r['resolved']}/{r['attempted']} | {r['completed']}/{r['attempted']} | {r['strict_demonstrated_quality']:.2%} | {gain} |")
lines+=['','The v1 prototype allowed negative scores. Its apparent 1→4 group improvement partly rewarded missing or uncited answers over false answers. It was rejected, and all its records were kept. The first nonnegative v2 check had no contributing groups and one out-of-bounds source request. That mechanical failure motivated grader v5; it was not hidden or resampled under the same identity.','',
 'Coverage counts both semantic audits and deterministic zero outcomes. Final training semantic audits: '+str(final['semantic_audits'])+'/'+str(final['attempted'])+'; final selection semantic audits: '+str(control['semantic_audits'])+'/'+str(control['attempted'])+'. Thus 100% scoring coverage would not imply every answer was semantically graded.','',
 '## Final decision','',f"Small-diagnostic readiness: **{'passed' if all(gates.values()) else 'not passed'}**.",'']
for key,value in gates.items():lines.append(f"- {key}: {'pass' if value else 'fail'}")
lines+=['',f"The final training run has {final['reward_signal']['training_contributing_groups']} contributing groups, {final['reward_signal']['excluded_groups']} excluded groups, and {final['strict_zero_positive_training']} answers with strict zero but positive training credit. {'This is a small signal check, not a training result.' if all(gates.values()) else 'Do not launch GRPO on the assumption that shaping alone has solved the signal problem.'}",
 '', 'A shaped reward cannot create factual credit when attempts lack supported required facts. Valid cited-answer completion and sufficiently granular trustworthy references remain prerequisites. Single-claim references can still limit factual partial credit. The original full Experiment 1 throughput comparison and full fresh-control gates remain outside this bounded diagnostic.','',
 '## Cost and provenance','',f"{result['episodes']} new episodes across all retained diagnostic versions; zero optimizer updates. New conservative reservations: **${current-prior:.4f}**. Cumulative shared baseline ledger: **${current:.4f}/$300**. These are reservations, not actual invoices; the project ceiling remains $1,000. Every attempt, including failed grading and mechanical repairs, remains in the ledger.",'',
 '[Final test log]('+str(out/'tests-final.log')+') · [Results JSON]('+str(out/'results.json')+') · [Per-attempt CSV]('+str(out/'per-attempt.csv')+') · [Frozen fixtures]('+str(out/'frozen-fixtures.json')+') · [Artifact checksums]('+str(out/'checksums.json')+')','',
 '## Log links','']
for r in summary:lines.append(f"- **{r['run_id']}**: [W&B]({r['wandb']}) · [controller log]({r['log_path']}) · [metrics]({r['report_path']}) · [answers]({r['answers_path']})")
(out/'README.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({k:result[k] for k in ['episodes','gates','new_reserved_usd','cumulative_reserved_usd']},indent=2))
