"""Private strict scoring adapter; simulated scores never enter live training."""
import json
import os
import re
from pathlib import Path
from qa_eval.security import bindings, seal, digest
from qa_eval.grading import evaluate
from qa_eval.judging import adapter_call
from qa_eval.source import GitSource


def simulated_verify(trajectory):
    if trajectory.get('termination') == 'infrastructure_error': return {'status':'unresolved','reward':None,'simulated':True}
    good = 'Supported example.' in trajectory.get('final_answer','')
    return {'status':'resolved','reward':1.0 if good else 0.0,
            'tier':'accepted' if good else 'failed','coverage':float(good),'simulated':True}


def verify(trajectory, task, experiment, root, budget, upper_usd, diagnostic=False, catalog_mode="reference_and_observed", automated_review=None):
    if trajectory.get('termination') == 'infrastructure_error':
        return {'status':'unresolved','reward':None,'reason':'rollout infrastructure failure'}
    source = GitSource(root, task['repository']['commit'])
    answer = trajectory['final_answer']
    citations=[]
    # Extract only explicit paths and line ranges present in the submitted answer.
    for path,a,b in re.findall(r'([A-Za-z0-9_./-]+):(?:L)?(\d+)(?:-(?:L)?(\d+))?',answer):
        try:
            citations.append({'id':'c'+str(len(citations)), 'path':path,'start_line':int(a),
                              'end_line':int(b or a),'file_sha256':source.sha(path)})
        except ValueError:
            # Preserve invalid citations as defects rather than hiding them.
            citations.append({'id':'c'+str(len(citations)), 'path':path,'start_line':int(a),
                              'end_line':int(b or a),'file_sha256':'0'*64})
    if catalog_mode=='reference_and_observed':
        paths={ref['path'] for claim in task['claims'] for ref in claim['evidence']}
        paths.update(cite['path'] for cite in citations)
        for action in trajectory['actions']:
            try:
                command=json.loads(action['text'])
                if command.get('tool')=='read_file':paths.add(command['arguments']['path'])
            except (ValueError,KeyError,TypeError):pass
        source.catalog_paths=frozenset(paths)
    submission={'schema_version' :'1.0','task_id':task['id'],'text':answer,'citations':citations,'diagram':None}
    tools=[]
    for action in trajectory['actions']:
        try:
            parsed=json.loads(action['text'])
            if 'tool' in parsed: tools.append(parsed['tool'])
        except (ValueError,TypeError): pass
    termination=trajectory['termination']
    termination=('completed' if termination=='completed' else 'budget_exhausted' if 'budget' in termination else 'agent_error')
    payload={'schema_version':'1.0', **bindings(task,submission),'experiment_id':experiment['id'],
             'episode_id':trajectory['episode_id'],'trajectory_ref':trajectory['artifact_path'],
             'latency_seconds':trajectory['elapsed_seconds'],'time_to_first_token_seconds':None,
             'input_tokens':trajectory['input_tokens'],'output_tokens':trajectory['output_tokens'],
             'tool_seconds':trajectory.get('tool_seconds',0.0),'tool_calls':tools,'retries':0,'cost':None,
             'termination_reason':termination,'integrity_violation':False,'execution_records':[], 'probes':[]}
    # Cost is not consumed by the current reward formula. Its unknown status is
    # retained separately; source-only tasks have no claimed runtime probes.
    key=os.urandom(32)
    def call(command, request, timeout):
        from .tinker_judge import judge_call
        operation = judge_call if command and command[0]=='tinker' else adapter_call
        return budget.execute('judge',upper_usd,lambda:operation(command,request,timeout),
                              {'task_id':task['id'],'stage':request['stage']})
    role='training' if task['split']=='train' else 'evaluation'
    result,_svg=evaluate(task,submission,seal('EpisodeMetrics',payload,key),experiment,source,key,
                    role=role,call=call,diagnostic_machine_review=diagnostic and automated_review is None,
                    automated_review=automated_review)
    return {'status':'unresolved' if result['tier']=='unresolved' else 'resolved',
            'reward':result['reward'],'tier':result['tier'],'coverage':result.get('coverage'),
            'report':result,'diagnostic':diagnostic,
            'review_policy':'automated' if automated_review is not None else 'human',
            'human_reviewed':task['human_reviewed'],'cost_usd':None,'catalog_scope':catalog_mode}
