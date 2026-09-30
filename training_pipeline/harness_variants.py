"""Opt-in solver harness variants, isolated from grader evidence and gold."""
import copy
import json
from pathlib import Path
from .contracts import ConfigurationError
from .storage import digest, atomic_json


def validate(spec):
    required = {'symbols','retrieval','history','source_manifest','source_manifest_sha256'}
    if not isinstance(spec,dict) or not required <= set(spec) or set(spec) - required - {'planning', 'interface'}:
        raise ConfigurationError('Invalid experimental harness specification')
    if spec.get('interface', 'structured-v1') not in {'structured-v1', 'source-shell-v1'}:
        raise ConfigurationError('Unsupported tool interface')
    if spec.get('interface') == 'source-shell-v1' and (spec['symbols'] or spec['retrieval'] != 'none' or spec['history'] != 'raw'):
        raise ConfigurationError('Shell-only interface cannot expose additional structured tools')
    if spec.get('planning', 'none') not in {'none', 'question-checklist-v1'}:
        raise ConfigurationError('Unsupported planning variant')
    if type(spec['symbols']) is not bool or spec['retrieval'] not in {'none','lexical-v1','hybrid-subword-v1'} or spec['history'] not in {'raw','evidence-ledger-v1'}:
        raise ConfigurationError('Unsupported experimental harness variant')


def source_manifest(data):
    repos = {}
    for row in data['tasks'] + data['development']:
        repo = row['public']['repository']
        key = repo['family_id']+'@'+repo['commit']
        value = {'repository':repo,'inventory_sha256':digest(row['image_result']['snapshot_files'])}
        if key in repos and repos[key] != value:
            raise ConfigurationError('Inconsistent source inventory')
        repos[key] = value
    return {'version':'source-tools-v1','embedding':'char-trigram-hash-256-v1',
            'chunk_lines':12,'hybrid_weights':{'lexical':.5,'embedding':.5},
            'context_tokens':512,'source_only':True,'repositories':repos}


def validate_sources(spec, data):
    from .config import verified_file
    actual = json.loads(verified_file(spec['source_manifest'],spec['source_manifest_sha256']).read_text())
    if actual != source_manifest(data):
        raise ConfigurationError('Harness source/index manifest changed')


def tools(spec):
    names=[]
    if spec.get('symbols'): names += ['find_definition','find_references']
    if spec.get('retrieval','none') != 'none': names += ['get_context']
    if spec.get('history','raw') != 'raw': names += ['read_observation']
    return names


def instructions(spec):
    text = ''
    if spec.get('planning') == 'question-checklist-v1':
        text += ('\nBefore your first search, organize the user question into a short checklist of subquestions. '
                 'Derive it only from the user question; do not invent required facts. '
                 'Use the checklist to choose searches and gather source evidence for each requested part. '
                 'Before answering, check which parts have evidence and state any unresolved gaps. '
                 'Keep this planning internal; continue returning only the existing tool or answer JSON schema. '
                 'All planning and investigation must fit the existing response and tool budgets.')
    if spec.get('symbols'):
        text += '\nAdditional tools: {"tool":"find_definition","arguments":{"symbol":"identifier"}} and {"tool":"find_references","arguments":{"symbol":"identifier"}}. These inspect Python ASTs. References are identifier occurrences, not resolved cross-file bindings; use read_file for context.'
    if spec.get('retrieval','none') != 'none':
        text += '\nRetrieve source snippets with {"tool":"get_context","arguments":{"query":"short query","budget":512}}. Source snippets include citation paths, lines and hashes. Maximum response is 512 solver-tokenizer tokens.'
    if spec.get('history','raw') != 'raw':
        text += '\nOlder tool observations are replaced with evidence handles. Recover their archived JSON using {"tool":"read_observation","arguments":{"id":"o1","offset":0}}. next_offset pages through the result. Handles are private to this episode.'
    return text


def limit_context(observation, tokenizer, budget, byte_limit):
    if tokenizer is None:
        raise ConfigurationError('Retrieval needs the solver tokenizer for its token limit')
    value = copy.deepcopy(observation)
    content = json.loads(value['stdout'])
    snippets = content['snippets']
    while True:
        value['stdout'] = json.dumps(content)
        serialized = json.dumps(value,ensure_ascii=True)
        if len(tokenizer.encode(serialized,add_special_tokens=False)) <= budget and len(serialized.encode()) <= byte_limit:
            return value
        if not snippets:
            raise ValueError('Requested retrieval budget is too small for response metadata')
        snippets[-1]['lines'].pop();snippets[-1]['end_line'] -= 1
        if not snippets[-1]['lines']: snippets.pop()


class ObservationHistory:
    def __init__(self, root, episode_id):
        self.root=Path(root)/'observations'/episode_id
        self.records={};self.messages=[]

    def remember(self, observation, message_index):
        ident='o'+str(len(self.records)+1)
        self.records[ident]=copy.deepcopy(observation)
        atomic_json(self.root/(ident+'.json'),self.records[ident])
        self.messages.append((message_index,ident))
        return ident

    def compact(self, messages):
        for index,ident in self.messages[:-1]:
            record=self.records[ident]
            refs=[]
            try:
                content=json.loads(record.get('stdout','{}'))
                candidates = content.get('snippets',content.get('matches',[]))
                if 'path' in content: candidates=[content]
                refs=[{k:r[k] for k in ('path','start_line','end_line','file_sha256') if k in r} for r in candidates]
            except (ValueError,TypeError,AttributeError):
                pass
            messages[index]={'role':'user','content':'Tool observation archived: '+json.dumps({'id':ident,'source_handles':refs,'retrieve':'read_observation'})}

    def read(self,args):
        if (not isinstance(args,dict) or set(args)-{'id','offset'} or not isinstance(args.get('id'),str)
                or args['id'] not in self.records or type(args.get('offset',0)) is not int or args.get('offset',0)<0):
            raise ValueError('Unknown episode observation or invalid offset')
        # Read back the actual archive, rejecting tampering or partial writes.
        record=json.loads((self.root/(args['id']+'.json')).read_text())
        if record != self.records[args['id']]:
            raise ValueError('Observation archive changed')
        text=json.dumps(record);start=args.get('offset',0);end=min(len(text),start+1200)
        return {'id':args['id'],'text':text[start:end],'next_offset':end if end<len(text) else None}
