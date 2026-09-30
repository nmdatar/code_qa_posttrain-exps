"""Independent factual-coverage pass: strict audit defects cannot change reward."""
import copy
import json
import ast
import re
from dataclasses import asdict
from .claim_grading import PROMPT, request_payload, aggregate
from .storage import atomic_json, digest
from .admission import blob, sha

VERSION = 'independent-factual-coverage-v3'


def mentioned_function_evidence(request):
    """Find explicitly mentioned functions in pinned, already-known files.

    An uncited paraphrase must not lose source solely because the strict extractor
    did not request its helper. This reads source, never executes candidate text.
    """
    row=request['source_row'];inventory=row['image_result']['snapshot_files']
    known={e['path'] for e in request['rubric']['evidence']} | set(request.get('observed_files',{}))
    text=request['answer']['text']
    # Avoid interpreting ordinary prose words such as "name", "values", or
    # "export" as requests for every same-named method in a large module.
    names={n for n in re.findall(r'\b[A-Za-z_]\w*\b',text) if '_' in n}
    names.update(re.findall(r'`([A-Za-z_]\w*)`',text))
    refs=[]
    for path in sorted(known):
        if not path.endswith('.py') or path not in inventory:continue
        content=blob(row['snapshot_root'],row['public']['repository']['commit'],path)
        if sha(content)!=inventory[path]:raise ValueError('Mentioned-function source hash mismatch')
        try:tree=ast.parse(content.decode())
        except (SyntaxError,UnicodeDecodeError):continue
        for node in ast.walk(tree):
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in names:
                refs.append({'path':path,'start_line':node.lineno,'end_line':node.end_lineno,'file_sha256':inventory[path]})
    return refs
POLICY = PROMPT + '''
The required claim text is the ENTIRE criterion, not the broader question. The question
only resolves scope. Never add a requirement for explanations, mechanisms, names, or citations
that the claim does not request. Correct source-supported paraphrases receive full credit.
Ordinary words such as context do not name a class unless the source establishes that.
Describing an implementing helper can cover an operation attributed to its caller when the
answer connects them. Inheritance supplies inherited methods and grammar; a subclass need
not repeat their definitions. Separate runtime processing order from source declaration order
or output-container ordering. Use normal language semantics, not speculative reinterpretations.
Assess only what the candidate actually says. A source fact missing from the candidate earns
no credit. A correct fact still earns credit when another independent fact is wrong or absent.
Do not assess citations or extra assertions in this pass; the separate strict audit handles them.
Give a brief reason identifying the specific supported or missing fact before your verdict.
If source cannot verify the reference claim itself, use unresolved, not a fabricated zero.
'''


def coverage_request(request, audit_evidence=None):
    req = copy.deepcopy(request)
    rubric = req['rubric']; row = req['source_row']
    # Reuse verified source gathered by the strict evidence reader, without its
    # verdicts or citation judgments. Source discovered from a candidate's valid
    # citation is still useful evidence even though citations earn no reward.
    seen={(e['path'],e['start_line'],e['end_line']) for e in rubric['evidence']}
    additional=list(request.get('answer_evidence',[]))+list((audit_evidence or {}).values())
    available=list(rubric['evidence'])+additional
    for ref in mentioned_function_evidence(request):
        if not any(isinstance(e,dict) and e.get('path')==ref['path'] and
                   e.get('start_line',float('inf'))<=ref['start_line'] and
                   e.get('end_line',0)>=ref['end_line'] for e in available):
            additional.append(ref)
    for ref in additional:
        if not isinstance(ref,dict) or not {'path','start_line','end_line'} <= ref.keys():continue
        key=(ref['path'],ref['start_line'],ref['end_line'])
        if key in seen:continue
        expected=row['image_result']['snapshot_files'].get(ref['path'])
        if expected is None or ref.get('file_sha256',expected)!=expected:continue
        content=blob(row['snapshot_root'],row['public']['repository']['commit'],ref['path'])
        if sha(content)!=expected:raise ValueError('Additional coverage source hash mismatch')
        lines=content.decode().splitlines();first,last=ref['start_line'],ref['end_line']
        if type(first) is not int or type(last) is not int or not 1<=first<=last<=len(lines):continue
        rubric['evidence'].append({'id':'verified-'+str(len(rubric['evidence'])+1),'path':ref['path'],
            'file_sha256':expected,'start_line':first,'end_line':last,'text':'\n'.join(lines[first-1:last])})
        seen.add(key)
    # Include bounded surrounding source for terse references, with pinned hashes.
    for original in list(rubric['evidence']):
        content = blob(row['snapshot_root'],row['public']['repository']['commit'],original['path'])
        if sha(content) != original['file_sha256']:
            raise ValueError('Coverage source hash mismatch')
        lines = content.decode().splitlines()
        first=max(1,original['start_line']-40);last=min(len(lines),original['end_line']+40)
        rubric['evidence'].append({**original,'id':'context-'+original['id'], 'start_line':first,'end_line':last,
                                  'text':'\n'.join(lines[first-1:last])})
    rubric['rubric_hash'] = digest({k:rubric[k] for k in ('version','claims','evidence')})
    return req


def assess_coverage(factory, request, audit_evidence=None):
    req = coverage_request(request, audit_evidence)
    payload = request_payload(req)
    # Deliberately exclude all citation and strict-assessment information.
    payload['candidate_answer'] = {'text':req['answer']['text']}
    payload.pop('candidate_citation_evidence',None)
    for attempt in range(factory.config['judge'].get('repair_attempts',0)+1):
        sample = factory.judge.sample([{'role':'system','content':POLICY}, {'role':'user','content':json.dumps(payload)}],
                                      factory.config['judge']['max_tokens'],factory.config['judge'].get('temperature',0))
        record={'generation':asdict(sample),'payload':copy.deepcopy(payload),'policy':POLICY,
                'judge_identity':factory.judge.identity,'attempt':attempt}
        path=factory.root/'private'/(request['episode_id']+f'.coverage-{attempt}.json')
        atomic_json(path,record)
        try:
            if sample.stop_reason == 'length':raise ValueError('Truncated factual coverage judgment')
            result=aggregate(sample.text,req)
            atomic_json(factory.root/'private'/(request['episode_id']+'.coverage.json'),result)
            return result
        except (ValueError,KeyError,TypeError) as exc:
            record['validation_error']=str(exc);atomic_json(path,record)
            if attempt == factory.config['judge'].get('repair_attempts',0):raise
            payload['format_repair']={'error':str(exc),'instruction':'Return the complete required JSON schema; preserve honest uncertainty and every claim.'}


def merge_coverage(strict, coverage):
    """Preserve strict diagnostics; select eligibility from the independent reward pass."""
    result = copy.deepcopy(strict)
    result['strict_status'] = strict['status']
    result['strict_reason'] = strict.get('reason')
    result['reason'] = coverage['reason']
    result['coverage_assessment'] = coverage
    result['status'] = coverage['status']
    result['training_feedback'] = {'version':'positive-coverage-v4','reward':coverage['score'],
        'eligible':coverage['status']=='resolved','components':{**strict.get('training_feedback',{}).get('components',{}), 'required_coverage':coverage['score']},
        'assessment':VERSION}
    return result
