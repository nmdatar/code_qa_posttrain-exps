"""Import attributed benchmark Q&A into explicit source-reading task bundles.

References remain upstream drafts. Citation integrity is not semantic approval.
This importer does not generate solver trajectories or executable behavior probes.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from .build import SYSTEM_PROMPT, write_json, write_jsonl, read_jsonl
from .contracts import canonical_hash, validate_public_task
from .environment import _export_snapshot
from .source_evidence import inspect_reference
from qa_eval.deterministic import snapshot

BUDGETS = {'latency_seconds':1200,'compute_units':100000,'max_tool_calls':40,
           'max_output_tokens':6000,'max_submission_bytes':64000}
RECIPE = {'base_image':'python:3.12-slim','capability':'source_reading','install_commands':[],
          'readiness_command':['python','-I','-c','from pathlib import Path; import json; p=Path("/workspace"); assert p.is_dir() and any(p.iterdir()); print(json.dumps({"source_reading":True}))']}
EXECUTION_REQUIRED = re.compile(r'\b(?:please\s+(?:run|execute)|run\s+(?:the\s+)?(?:tests?|benchmark)|execute\s+(?:the\s+)?(?:tests?|commands?)|measure\s+(?:latency|performance)|empirically\s+(?:verify|test)|reproduce\s+this\s+(?:bug|failure))\b',re.I)


def normalized_question(text):
    return ' '.join(unicodedata.normalize('NFKC',text).casefold().split())


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def source_rows(root):
    base=root/'artifacts/task-generation-1000'
    pins=load(base/'benchmark-research/sweqa-full-pins.json')
    rows=[]
    for pin in pins:
        repo=pin['repository'];path=base/'benchmark-research/sweqa'/(repo.split('/')[-1]+'.jsonl')
        for i,row in enumerate(read_jsonl(path)):
            rows.append({'repo':repo,'commit':pin['commit'],'question':row['question'],'answer':row['answer'],
                'category':'repository_qa','provenance':{'source':'SWE-QA','dataset_revision':'f048f07da234b39e32937f115d027c5b71acd54e',
                'dataset_url':'https://huggingface.co/datasets/swe-qa/SWE-QA-Benchmark','source_file':str(path.relative_to(root)),
                'source_file_sha256':file_sha(path),'source_row_index':i,'upstream_split':'benchmark','license':'Apache-2.0',
                'license_path':'artifacts/task-generation-1000/benchmark-research/sweqa-LICENSE','original_short_commit':pin['source_short_commit']}})
    path=base/'training-research/probench-test.jsonl'
    for i,row in enumerate(read_jsonl(path)):
        rows.append({'repo':row['repo'],'commit':row['commit_id'],'question':row['question'],'answer':row['answer'],
            'category':str(row['cluster']),'provenance':{'source':'SWE-QA-Pro','dataset_revision':'596892dac60b6f500f01a7dc2becb9f66593b7b7',
            'dataset_url':'https://huggingface.co/datasets/TIGER-Lab/SWE-QA-Pro-Bench','source_file':str(path.relative_to(root)),
            'source_file_sha256':file_sha(path),'source_row_index':i,'upstream_split':'test','license':'MIT',
            'license_path':'artifacts/task-generation-1000/training-research/repository-license.txt','qa_type':row['qa_type']}})
    return rows


def prepare_collection(root, output, resume=False):
    root,output=Path(root),Path(output)
    if output.exists() and any(output.iterdir()) and not resume:raise ValueError('Use a new collection directory; do not overwrite')
    plan=load(root/'reports/task-generation-1000/collection-plan.json')
    assigned={(r['repo'],r['commit']):r for r in plan['repositories']}
    acquisition={}
    for p in (root/'reports/task-generation-1000/acquisition').glob('*.json'):
        if p.name=='summary.json':continue
        r=load(p);acquisition[(r['repo'],r['commit'])]=r
    groups=defaultdict(list);rejected=[];seen={}
    for row in source_rows(root):
        identifier='import-'+canonical_hash([row['provenance']['source'],row['repo'],row['commit'],row['question']])[:24]
        row['id']=identifier
        reason=None;key=(row['repo'],row['commit']);norm=normalized_question(row['question'])
        if key not in assigned:reason='repository_not_preassigned'
        elif not row['question'].strip() or not row['answer'].strip():reason='empty_question_or_reference'
        elif norm in seen:reason='duplicate_question:'+seen[norm]
        elif EXECUTION_REQUIRED.search(row['question']):reason='explicit_execution_requirement_needs_executable_environment'
        elif acquisition.get(key,{}).get('status')!='ready':reason='snapshot_not_ready'
        if reason:
            rejected.append({'id':identifier,'repo':row['repo'],'commit':row['commit'],'reason':reason,'provenance':row['provenance']});continue
        seen[norm]=identifier;groups[key].append(row)
    output.mkdir(parents=True,exist_ok=True);inventory=[]
    for key,rows in sorted(groups.items()):
        family=assigned[key];acquired=acquisition[key];checkout=Path(acquired['snapshot_path'])
        bundle=output/(key[0].replace('/','--')+'-'+key[1][:12])
        if (bundle/'manifest.json').exists():
            manifest=load(bundle/'manifest.json')
            for relative,digest in manifest['artifacts'].items():
                if file_sha(bundle/relative)!=digest:raise ValueError('Existing bundle changed')
            prior=read_jsonl(bundle/'private/references.jsonl')
            if {(r['id'],r['question'],r['reference_answer']) for r in prior}!={(r['id'],r['question'],r['answer']) for r in rows}:raise ValueError('Source rows changed; use new version')
            env=load(bundle/'public/environment.json');quality=read_jsonl(bundle/'private/quality.jsonl')
            inventory.append({'bundle':str(bundle.relative_to(root)),'repository':env['repository'],'environment_id':env['id'],'tasks':len(prior),
                              'citation_status_counts':dict(Counter(q['citation_integrity']['status'] for q in quality))})
            continue
        mode=acquired.get('source_inventory_mode');blob_reader=None
        if mode=='git_objects_primary':
            from .git_source_environment import inventory as git_inventory, read_git_file
            metadata=git_inventory(checkout,key[1]);hashes=metadata['snapshot_files'];links={}
            blob_reader=lambda path:read_git_file(checkout,key[1],path)
        else:
            snapshot(checkout,key[1]);hashes,links=_export_snapshot(checkout,key[1])
        repository={'url':'https://github.com/'+key[0],'commit':key[1],'family_id':family['family_id']}
        identity={'repository':repository,'recipe':RECIPE}
        if mode:identity['source_inventory_mode']=mode
        env_id='env-'+canonical_hash(identity)[:20]
        env={'id':env_id,'repository':repository,'snapshot_path':str(checkout),'recipe':RECIPE,'status':'prepared_not_built'}
        if mode:
            env.update(source_inventory_mode=mode,source_scope='Primary repository regular Git blobs only; symbolic links are target metadata and submodules are unavailable pinned metadata. No repository execution.',symbolic_links=metadata['symbolic_links'],submodules=metadata['submodules'])
        public=[];refs=[];quality=[];assertions=[]
        for row in rows:
            task={'schema_version':'1.0','id':row['id'],'system_prompt':SYSTEM_PROMPT+' This task uses source-reading tools only; do not claim to execute repository code.',
                  'user_prompt':row['question'],'repository':repository,'environment_id':env_id,'split':family['split'],
                  'permitted_tools':['list_files','search_code','read_file'],'budgets':BUDGETS}
            validate_public_task(task);public.append(task)
            evidence=inspect_reference(row['answer'],checkout,hashes,blob_reader=blob_reader)
            quality.append({'task_id':row['id'],'question_sha256':hashlib.sha256(row['question'].encode()).hexdigest(),
                'reference_sha256':hashlib.sha256(row['answer'].encode()).hexdigest(),'citation_integrity':evidence,
                'semantic_review':'not_locally_reviewed','reference_origin':'upstream_benchmark','human_reviewed':False,
                'training_eligible':False,'environment_capability':'source_reading','execution_requirement_screen':'no_explicit_execution_request_detected'})
            refs.append({'schema_version':'source-reference-1.0','id':row['id'],'question':row['question'],
                'reference_answer':row['answer'],'repository':repository,'environment_id':env_id,'split':family['split'],
                'lineage_id':row['id'],'category':row['category'],'provenance':row['provenance'],
                'grading':{'kind':'reference_comparison','criteria':['Substantive correctness against the private reference and pinned source',
                    'Coverage of the question without unsupported additions','Citations support substantive source claims',
                    'No claim of execution in a source-reading task'],'semantic_judge_required':True,
                    'upstream_reference_is_not_automatically_correct':True},'gold_status':'draft','human_reviewed':False})
            assertions.append({'task_id':row['id'],'kind':'source_integrity_not_answer_correctness','repository_commit':key[1],
                'question_sha256':quality[-1]['question_sha256'],'reference_sha256':quality[-1]['reference_sha256'],
                'evidence':evidence['evidence'],'file_references':evidence['file_references'],'unresolved_citations':evidence['unresolved'],
                'semantic_assertions_status':'upstream_reference_comparison_requires_review','executable_behavior_probes':[]})
        write_jsonl(bundle/'public/tasks.jsonl',public);write_json(bundle/'public/environment.json',env)
        write_jsonl(bundle/'private/references.jsonl',refs);write_jsonl(bundle/'private/quality.jsonl',quality)
        write_jsonl(bundle/'private/assertions.jsonl',assertions)
        write_json(bundle/'private/source-spec.json',{'repository':repository,'kind':'attributed_benchmark_import','task_count':len(public),'family_assignment':family,'environment':RECIPE})
        manifest={'schema_version':'source-bundle-1.0','repository':repository,'environment_id':env_id,'split':family['split'],
            'task_count':len(public),'status':'prepared_source_reading','training_eligible':False,'human_reviewed':False,
            'source_inventory_sha256':canonical_hash(hashes),'artifacts':{str(p.relative_to(bundle)):file_sha(p) for p in sorted(bundle.rglob('*')) if p.is_file()}}
        write_json(bundle/'manifest.json',manifest)
        inventory.append({'bundle':str(bundle.relative_to(root)),'repository':repository,'environment_id':env_id,'tasks':len(public),
                          'citation_status_counts':dict(Counter(q['citation_integrity']['status'] for q in quality))})
    report={'input_records':len(source_rows(root)),'prepared_tasks':sum(x['tasks'] for x in inventory),'environments':inventory,
            'rejections':rejected,'status':'prepared_not_runtime_verified','semantic_acceptance_claimed':False}
    write_json(root/'reports/task-generation-1000/preparation.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--resume',action='store_true');p.add_argument('--output',type=Path,default=Path('data/generated/collection-1000-v1'));a=p.parse_args()
    r=prepare_collection(a.root.resolve(),(a.root/a.output).resolve(),resume=a.resume);print(json.dumps({'input':r['input_records'],'prepared':r['prepared_tasks'],'environments':len(r['environments']),'rejected':len(r['rejections'])}))


if __name__=='__main__':main()
