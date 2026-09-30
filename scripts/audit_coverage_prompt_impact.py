"""Identify affected frozen cases after deterministic source lookup changes."""
import json
from pathlib import Path
from training_pipeline.coverage_judge import coverage_request
from training_pipeline.claim_grading import request_payload
from training_pipeline.storage import digest
root=Path('reports/grader-paraphrase-validation');fixture=json.loads((root/'fixture-v2.json').read_text());raw=Path('artifacts/grader-paraphrase-validation-results/artifacts/experiments/qwen-paraphrase-validation-v4/private');impact=[]
for case in fixture['cases']:
    old=json.loads((raw/(case['name']+'.coverage-0.json')).read_text())['payload']
    p=raw/(case['name']+'.assess.judge-raw.json')
    audit=json.loads(p.read_text())['request']['untrusted']['evidence'] if p.exists() else {}
    request=coverage_request(case['request'],audit);new=request_payload(request)
    new['candidate_answer']={'text':request['answer']['text']};new.pop('candidate_citation_evidence',None)
    impact.append({'case':case['name'],'changed':new!=old,'before':digest(old),'after':digest(new)})
(root/'prompt-impact.json').write_text(json.dumps(impact,indent=2)+'\n')
print('Changed:',[x['case'] for x in impact if x['changed']])
