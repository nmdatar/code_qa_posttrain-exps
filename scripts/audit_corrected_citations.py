"""Check recognized citations in corrected answers; retain explicit parser scope."""
import hashlib
from pathlib import Path
from dataset_builder.collection_review import review_records
from dataset_builder.build import read_jsonl,write_json
from dataset_builder.collection import load
from dataset_builder.source_evidence import inspect_reference
from dataset_builder.git_source_environment import read_git_file

root=Path.cwd();reviews=review_records(root);results=[]
context_path=root/'reports/task-generation-1000/corrected-citation-context-review.json'
contexts={r['task_id']:r for r in load(context_path)['records']} if context_path.exists() else {}
for b in sorted((root/'data/generated/collection-1000-v1').iterdir()):
 env=load(b/'public/environment.json');built=load(b/'environment-build/result.json');checkout=Path(env['snapshot_path']);commit=env['repository']['commit']
 for ref in read_jsonl(b/'private/references.jsonl'):
  review=reviews.get(ref['id'],{});answer=review.get('corrected_reference_answer')
  if not answer:continue
  digest=hashlib.sha256(answer.encode()).hexdigest()
  report=inspect_reference(answer,checkout,built['snapshot_files'],blob_reader=lambda p:read_git_file(checkout,commit,p))
  item={'task_id':ref['id'],'reference_sha256':digest,'review_report':review['review_report'],'citation_integrity':report,'status':'recognized_citations_verified'}
  if report['unresolved']:
   context=contexts.get(ref['id'])
   if not context or context['reference_sha256']!=digest:item['status']='needs_context_review'
   else:
    for span in context['verified_evidence']:
     data=read_git_file(checkout,commit,span['path'])
     if hashlib.sha256(data).hexdigest()!=span['file_sha256'] or built['snapshot_files'].get(span['path'])!=span['file_sha256']:raise ValueError('Context evidence hash mismatch')
     if not 1<=span['start_line']<=span['end_line']<=len(data.decode().splitlines()):raise ValueError('Context evidence line mismatch')
    item['status']='recognized_citations_verified_with_context_review';item['context_review']=context
  results.append(item)
write_json(root/'reports/task-generation-1000/corrected-answer-citations.json',{'records':len(results),'scope':'Recognized explicit citation forms plus manually reviewed context resolutions. Not a guarantee of exhaustive extraction or semantic truth; independent semantic reviews are separate.','results':results})
print({'checked':len(results),'needs_context_review':sum(r['status']=='needs_context_review' for r in results)})
