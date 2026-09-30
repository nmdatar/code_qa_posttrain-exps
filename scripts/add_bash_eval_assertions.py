"""Add explicitly requested grader assertions to the local completed-run viewer only."""
import hashlib
import json
from pathlib import Path
from training_pipeline.storage import read
from training_pipeline.trajectory_viewer import load_traces, render_report

bundle=Path('artifacts/bash-only-eval-v1-funded-bundle')
manifest=read(bundle/'bundle.json')
records=[]
for arm,index in [('structured',0),('bash',1)]:
 rel=f'inputs/{index}/release/private/grading.jsonl'
 raw=(bundle/rel).read_bytes()
 assert hashlib.sha256(raw).hexdigest()==manifest['files'][rel]
 references={r['task_id']:r for r in map(json.loads,raw.decode().splitlines())}
 root=Path('artifacts/bash-only-eval-v1-results/artifacts/experiments')/('bash-only-eval-v1-'+arm)
 for trace,context in load_traces(root):
  verification=trace.get('verification') or {}
  diag=verification.get('diagnostics',{})
  semantic=diag.get('semantic') or {}
  findings={c['id']:c for c in diag.get('claims',[])}
  required=[]
  for i,claim in enumerate(references[trace['task_id']]['reviewed_claims'],1):
   ident='c'+str(i);text=claim['claim']
   if claim['verdict']=='qualified':text+='\nQualification: '+claim['reason']
   finding=findings.get(ident)
   required.append({'id':ident,'assertion':text,'weight':1.0,
    'reference_source':[claim['evidence']]+claim.get('supporting_evidence',[]),
    'recorded_assessment':finding if finding is not None else {'status':'not assessed / unavailable','episode_reasons':verification.get('reasons',[])}})
  # Cross-check the displayed assertions against any actual saved judge request.
  for path in (root/'private').glob(trace['episode_id']+'.assess*.judge-raw.json'):
   request=read(path).get('request',{});rubric=request.get('rubric',{})
   actual=rubric.get('claims',[]) if isinstance(rubric,dict) else []
   if actual:
    assert [(r['id'],r['assertion']) for r in required]==[(r['id'],r['text']) for r in actual]
  feedback=diag.get('training_feedback')
  scorecard={'strict_evaluation_score':verification.get('reward'),
   'saved_training_reward_diagnostic':diag.get('training_reward'),
   'diagnostic_status':('assessed from strict audit' if feedback and feedback.get('eligible') else
     'evaluation gate returned zero; factual coverage was not assessed' if not feedback and diag.get('training_reward')==0 else 'unavailable / unresolved'),
   'applied_score':'strict evaluation; no training updates in this run',
   'independent_training_coverage_pass':'not run on these evaluation tasks',
   'formula':'Weighted mean: complete supported required fact = 1, partial = 0.5, absent or unsupported = 0. This positive-coverage reward does not deduct citation/additional-claim penalties.',
   'saved_feedback':feedback}
  review={'scorecard':scorecard,'required_assertions':required,'answer_checks':{k:semantic.get(k) for k in
   ['extracted_claims','additional_claims','citation_links','uncited_claim_ids','critical_error','false_execution_claim','needs_review','disagreements']}}
  records.append((trace,{**context,'phase':arm,'grading_review':review}))
assert len(records)==64
out=Path('reports/bash-only-eval-v1/trajectories.html')
out.write_text(render_report(records,'Bash-only versus structured tools · 32 validation tasks'))
print(f'Updated {out}: {len(records)} episodes with frozen required assertions; missing assessments remain explicit.')
