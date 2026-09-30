"""Freeze source-grounded synthetic ranking cases and replay selection diagnostics.

Synthetic annotations test the reward contract, not live judge calibration.
Confirmation answers are deliberately excluded from design/validation.
"""
import copy
import json
import tempfile
from pathlib import Path
from qa_eval.demo import fixture
from training_pipeline.storage import atomic_json, digest
from training_pipeline.shaped_reward import shape
from training_pipeline.strict_grading import strict_score

out=Path('reports/reward-shaping-v3');out.mkdir(exist_ok=True,parents=True)
with tempfile.TemporaryDirectory() as tmp:
    task,answer,_,_,semantic=fixture(tmp)
    task['human_reviewed']=False;task['gold_status']='draft'
    task['claims'][0]['weight']=1
    task['claims'].append({**copy.deepcopy(task['claims'][0]),'id':'c2','text':'cancel does not call a job.'})
    semantic['required_claims'].append({**copy.deepcopy(semantic['required_claims'][0]),'id':'c2'})
    semantic['additional_claims'].append({**copy.deepcopy(semantic['additional_claims'][0]),'id':'a2'})
    semantic['extracted_claims'].append({'id':'a2','text':'cancel does not call a job.','source':'text'})
    semantic['citation_links'].append({**semantic['citation_links'][0],'claim_id':'a2'})
    answer['text']='cancel sets cancelled to True and does not call a job.'
    cases=[]
    def add(name,a,s):
        strict,tier,_,_=strict_score(task,a,s)
        cases.append({'name':name,'answer':copy.deepcopy(a),'semantic':copy.deepcopy(s),
                      'strict_score':strict,'training':shape(s,task,strict,version='supported-coverage-v3'),'tier':tier})
    add('correct',answer,semantic)
    a=copy.deepcopy(answer);a['text']='cancel sets cancelled to True.'
    s=copy.deepcopy(semantic);s['required_claims'][1]['coverage']='absent';s['additional_claims']=s['additional_claims'][:1];s['extracted_claims']=s['extracted_claims'][:1];s['citation_links']=s['citation_links'][:1]
    add('partial',a,s)
    a=copy.deepcopy(answer);a['text']='cancel sets cancelled to False.'
    s=copy.deepcopy(semantic)
    s['required_claims'][0].update(verdict='contradicted',material_error=True)
    s['required_claims'][1]['coverage']='absent'
    s['additional_claims']=s['additional_claims'][:1]
    s['additional_claims'][0].update(verdict='contradicted',material_error=True)
    s['extracted_claims']=[{'id':'a1','text':a['text'],'source':'text'}]
    s['citation_links']=s['citation_links'][:1]
    s['citation_links'][0]['supported']=False
    s['uncited_claim_ids']=['a1']
    add('incorrect',a,s)
    a=copy.deepcopy(answer);s=copy.deepcopy(semantic);s['citation_links'][0]['supported']=False;s['uncited_claim_ids']=['a1'];add('citation_defect',a,s)
    a=copy.deepcopy(answer);a['text']+=' It also emails the administrator.'
    s=copy.deepcopy(semantic);s['additional_claims'].append({**s['additional_claims'][0],'id':'a3','verdict':'insufficient'});s['extracted_claims'].append({'id':'a3','text':'It also emails the administrator.','source':'text'});s['uncited_claim_ids']=['a3'];add('unsupported_extra',a,s)
    a['text']+=' The function is named cancel.';s['additional_claims'].append({**s['additional_claims'][0],'id':'a4'});s['extracted_claims'].append({'id':'a4','text':'The function is named cancel.','source':'text'});s['citation_links'].append({**s['citation_links'][0],'claim_id':'a4'});add('unsupported_plus_true_filler',a,s)
    a=copy.deepcopy(answer);a['text']+=' I ran the program.';s=copy.deepcopy(semantic);s['false_execution_claim']=True;add('false_execution',a,s)
    s=copy.deepcopy(semantic);s['needs_review']=True;add('unresolved',answer,s)
    fixtures={'kind':'synthetic_source_grounded_reward_contract','human_calibrated':False,
              'source':(Path(tmp)/'executor.py').read_text(),'task':task,'cases':cases}
    fixtures['fixture_hash']=digest(fixtures);atomic_json(out/'frozen-fixtures.json',fixtures)
    values={c['name']:c['training']['reward'] for c in cases}
    assert values['correct']>values['partial']>values['incorrect']
    assert values['correct']>values['unsupported_extra']==values['unsupported_plus_true_filler']
    assert 0<values['citation_defect']<values['correct']
    assert values['false_execution']==0 and values['unresolved'] is None
    print(values)
# Replay retained selection assessments only; never relabel old stored rewards.
root=Path('artifacts/experiment-01-v4-results/artifacts/experiments/01-baseline-qwen4b-qwen397-seed42-all-claims-v3-modal-protocol-v4')
rows=[]
for path in sorted(root.glob('trajectories/*.json')):
    t=json.loads(path.read_text());v=t['verification'];s=v.get('diagnostics',{}).get('semantic')
    if s:
        # Collection references in this cohort use unit-weight claims. Verify the
        # original assessment's rubric separately before reporting this projection.
        rubric_path=root/'private'/(t['episode_id']+'.assess.judge-raw.json')
        rubric=json.loads(rubric_path.read_text())['request']['rubric']
        feedback=shape(s,rubric,v['reward'],version='supported-coverage-v3')
        rows.append({'task_id':t['task_id'],'episode_id':t['episode_id'],'strict_score':v['reward'],**feedback})
atomic_json(out/'selection-reward-projection.json',{'note':'Offline projection on historical selection judgments, not a new baseline or human calibration. No confirmation data used.','rows':rows})
print('Projected selection assessments:',len(rows))
