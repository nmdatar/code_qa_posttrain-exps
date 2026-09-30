"""Experimental, tool-only SFT from source-checked archived policy trajectories.

No answer correctness or human-gold admission is implied by this route.
"""
import json
from pathlib import Path
from agent_harness.repository_tools import command
from .admission import blob, sha
from .contracts import ConfigurationError
from .storage import digest

VERSION = 'verified-tool-prefix-v1'


def candidate(trace, row):
    from .collection import PROTOCOL
    if trace.get('split') != 'train' or row['split'] != 'train' or trace['task_id'] != row['id']:
        raise ValueError('Training task binding mismatch')
    initial = trace['events'][0]
    messages = initial['messages']
    protocols = [PROTOCOL, PROTOCOL.replace('Read at most 120 lines per call.', 'read_file returns at most 120 lines per call; larger requests return a first page with next_start_line. Follow that pointer only if needed.')]
    if messages[0] not in [{'role':'system','content':p} for p in protocols]:
        raise ValueError('Historical protocol is not compatible')
    public = json.loads(messages[1]['content'])
    if {k:v for k,v in public.items() if k != 'budgets'} != {k:v for k,v in row['public'].items() if k != 'budgets'}:
        raise ValueError('Public task changed')
    generations = [e for e in trace['events'] if e['kind'] == 'generation']
    observations = [e['value'] for e in trace['events'] if e['kind'] == 'observation']
    indices = [i for i,m in enumerate(messages) if m['role'] == 'assistant']
    selected, proofs = [], []
    # Only a clean prefix is admitted. A failure/truncation ends it; later recovery
    # and final answers are never silently relabelled as supervised successes.
    for i,g,o in zip(indices,generations,observations):
        if messages[i]['content'] != g['text'] or g['stop_reason'] != 'stop': break
        try:
            a = json.loads(g['text'])
            if set(a) != {'tool','arguments'} or a['tool'] not in row['public']['permitted_tools']: break
            command(a['tool'],a['arguments'],row['image_result']['snapshot_files'],paginate_reads=True)
            if o.get('exit_code') != 0 or o.get('stdout_truncated') or o.get('stderr_truncated'): break
            if i+1 >= len(messages) or '[output truncated]' in messages[i+1]['content']: break
            if json.loads(messages[i+1]['content'].removeprefix('Tool observation: ')) != o: break
            out = json.loads(o['stdout'])
            if a['tool'] == 'read_file':
                raw = blob(row['snapshot_root'],row['public']['repository']['commit'],a['arguments']['path'])
                if sha(raw) != out['file_sha256'] or sha(raw) != row['image_result']['snapshot_files'][out['path']]: break
                lo,hi = out['start_line'],out['end_line']
                if out['path'] != a['arguments']['path'] or lo != a['arguments'].get('start_line',1) or not 1<=lo<=hi or hi-lo>=120: break
                lines=raw.decode().splitlines()
                if out['lines'] != [str(j)+': '+lines[j-1] for j in range(lo,hi+1)]: break
                useful = True
            elif a['tool'] == 'search_code':
                useful = bool(out['matches'])
                for match in out['matches']:
                    raw = blob(row['snapshot_root'],row['public']['repository']['commit'],match['path'])
                    if sha(raw) != row['image_result']['snapshot_files'][match['path']]: raise ValueError('Search source hash')
                    line = raw.decode().splitlines()[match['line']-1]
                    if match['text'] != line[:500] or a['arguments']['query'] not in line: raise ValueError('Search observation mismatch')
            else:
                useful = bool(out['files']) and all(p in row['image_result']['snapshot_files'] for p in out['files'])
            if useful:
                selected.append(i); proofs.append({'message_index':i,'generation':g})
        except (ValueError,KeyError,TypeError,IndexError): break
    if not selected: raise ValueError('No useful verified prefix')
    return {'id':row['id']+'-tool-sft','task_id':row['id'],'split':'train','family_id':row['family_id'],
            'lineage_id':row['lineage_id'],'status':'accepted','admission':VERSION,'human_reviewed':False,
            'target_scope':'successful_tools_only_no_answers','messages':messages[:selected[-1]+1],
            'assistant_turns':selected,'source_episode_id':trace['episode_id'],'source_policy_id':trace['policy_id'],
            'proofs':proofs}


def load_supervised(spec, data):
    from .config import verified_file
    root = Path(spec['manifest']).resolve().parent
    manifest = json.loads(verified_file(spec['manifest'],spec['sha256']).read_text())
    from .investigation_sft import VERSION as investigation_version, candidate as investigation_candidate
    if manifest['version'] not in (VERSION, investigation_version) or manifest['data_identity'] != data['identity'] or manifest.get('human_reviewed') is not False:
        raise ConfigurationError('SFT admission/version mismatch')
    loss_scope = spec.get('loss_scope', 'all-admitted-turns')
    if loss_scope not in {'all-admitted-turns', 'final-answer-only'}:
        raise ConfigurationError('Unsupported supervised loss scope')
    if loss_scope == 'final-answer-only' and manifest['version'] != investigation_version:
        raise ConfigurationError('Answer-only loss requires admitted complete investigations')
    tasks = {r['id']:r for r in data['tasks']}
    examples, seen = [], set()
    for record in manifest['records']:
        rel = Path(record['source'])
        if rel.is_absolute() or '..' in rel.parts: raise ConfigurationError('Invalid SFT source path')
        path = (root/rel).resolve()
        if not path.is_relative_to(root): raise ConfigurationError('SFT path escapes release')
        trace = json.loads(verified_file(path, record['sha256']).read_text())
        row = tasks.get(trace['task_id'])
        if row is None or row['lineage_id'] in seen: raise ConfigurationError('SFT split/lineage violation')
        seen.add(row['lineage_id'])
        example = (investigation_candidate(trace,row,manifest['grading_version'])
                   if manifest['version'] == investigation_version else candidate(trace,row))
        if digest(example) != record['example_hash']: raise ConfigurationError('SFT proof changed')
        if loss_scope == 'final-answer-only':
            # Verify the complete original proof first. Keep the teacher context
            # and native alignment; change only which assistant outputs get loss.
            example['assistant_turns'] = [example['assistant_turns'][-1]]
            example['target_scope'] = 'final_answer_only_with_investigation_context'
        examples.append(example)
    if not examples: raise ConfigurationError('Empty supervised release')
    return examples


def validate_rendering(renderer, examples):
    for example in examples:
        renderer.supervised(example)
        for proof in example['proofs']:
            i = proof['message_index']; g = proof['generation']
            if renderer.prompt(example['messages'][:i]) != g['prompt']:
                raise ConfigurationError('Archived generation prefix does not match renderer')
            if renderer.tokenizer.decode(g['tokens'], skip_special_tokens=True) != g['text']:
                raise ConfigurationError('Archived target token mismatch')
