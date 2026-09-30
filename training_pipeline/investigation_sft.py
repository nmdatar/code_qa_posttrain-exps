"""Admit complete, strictly passing archived investigations for experimental SFT.

These are verified policy-generated demonstrations, not human or expert gold.
Admission binds the exact current rubric, judge version, source and token trace.
"""
import json
from types import SimpleNamespace
from qa_eval.schema import validate, SUBMISSION
from .admission import blob, sha
from .sft_collection import candidate as tool_candidate

VERSION = 'verified-investigation-v1'


def candidate(trace, row, reward_version):
    verification=trace.get('verification') or {}
    diagnostics=verification.get('diagnostics',{})
    if (trace.get('termination') != 'completed' or verification.get('status') != 'resolved'
            or verification.get('version') != reward_version
            or diagnostics.get('strict_status','resolved') != 'resolved'
            or diagnostics.get('strict_score') != 1.
            or diagnostics.get('rubric_hash') != row['rubric']['rubric_hash']):
        raise ValueError('Investigation requires a matched current strict-passing assessment')
    example=tool_candidate(trace,row)
    messages=trace['events'][0]['messages']
    generations=[e for e in trace['events'] if e['kind']=='generation']
    assistant=[i for i,m in enumerate(messages) if m['role']=='assistant']
    if (len(assistant) != len(generations) or len(example['assistant_turns']) != len(generations)-1
            or not generations or generations[-1]['stop_reason'] != 'stop'):
        raise ValueError('Only complete investigations with clean verified tool actions are admitted')
    i=assistant[-1];g=generations[-1]
    if messages[i]['content'] != g['text']:
        raise ValueError('Answer generation does not match transcript')
    from .collection import CollectionEpisode
    episode=object.__new__(CollectionEpisode)
    episode.row=row
    episode.factory=SimpleNamespace(config={'environment':{'tool_action_policy':'action-alias-v1'}})
    episode.observed_files={}
    for j in example['assistant_turns']:
        action=json.loads(messages[j]['content'])
        if action['tool']=='read_file':
            path=action['arguments']['path'];episode.observed_files[path]=row['image_result']['snapshot_files'][path]
    action=episode.parse_action(g['text'])
    if set(action) != {'answer'} or action['answer'] != trace['submission']:
        raise ValueError('Final answer binding mismatch')
    answer=action['answer'];validate(answer,SUBMISSION)
    if not answer['text'].strip() or not answer['citations'] or answer['task_id'] != row['id']:
        raise ValueError('Missing answer/citations or incorrect task identity')
    for ref in answer['citations']:
        if ref['path'] not in episode.observed_files:
            raise ValueError('Answer cites unread source')
        raw=blob(row['snapshot_root'],row['public']['repository']['commit'],ref['path'])
        if sha(raw) != ref['file_sha256'] or not 1<=ref['start_line']<=ref['end_line']<=len(raw.splitlines()) or ref['end_line']-ref['start_line']>=120:
            raise ValueError('Answer evidence changed or invalid')
    example.update(id=row['id']+'-investigation-sft',admission=VERSION,
                   target_scope='verified_tools_and_strict_passing_answer',messages=messages[:i+1],
                   assistant_turns=[*example['assistant_turns'],i],
                   proofs=[*example['proofs'],{'message_index':i,'generation':g}],
                   grading_version=reward_version,rubric_hash=row['rubric']['rubric_hash'])
    return example
