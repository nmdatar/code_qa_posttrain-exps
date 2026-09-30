"""Record current public repository activity separately from benchmark pins."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import datetime as dt,json,subprocess
from dataset_builder.build import write_json
root=Path.cwd();plan=json.loads((root/'reports/task-generation-1000/collection-plan.json').read_text());repos=sorted({r['repo'] for r in plan['repositories']})
now=dt.datetime.now(dt.timezone.utc)
def fetch(url):
 return json.loads(subprocess.check_output(['curl','--location','--retry','2','--retry-all-errors','--retry-delay','1','--fail','--silent','--show-error','--max-time','30','-H','Accept: application/vnd.github+json',url],text=True,stderr=subprocess.PIPE))

def worker(repo):
 url='https://api.github.com/repos/'+repo;r={'repository':repo,'checked_at':now.isoformat(),'metadata_url':url}
 try:
  data=fetch(url);r.update(status='metadata_verified',canonical_repository=data['full_name'],archived=data['archived'],disabled=data['disabled'],pushed_at=data['pushed_at'],default_branch=data['default_branch'])
  r['commit_sample_url']=url+'/commits?per_page=10'
  try:
   commits=fetch(r['commit_sample_url']);r['recent_commit_sample']=[{'sha':x['sha'],'committer_date':x['commit']['committer']['date']} for x in commits]
   dates=[dt.datetime.fromisoformat(x['committer_date'].replace('Z','+00:00')) for x in r['recent_commit_sample']]
   r['sample_commits_within_30_days']=sum(dt.timedelta(0)<=now-d<=dt.timedelta(days=30) for d in dates)
   r['activity_assessment']='recent_frequent_sample' if not r['archived'] and r['sample_commits_within_30_days']>=5 else 'not_recent_frequent_in_sample'
  except Exception as exc:r['commit_sample_error']=str(exc);r['activity_assessment']='frequency_unverified'
 except Exception as exc:r.update(status='unverified',error=str(exc),activity_assessment='unverified')
 return r
previous_path=root/'reports/task-generation-1000/repository-activity.json'
previous=json.loads(previous_path.read_text())['repositories'] if previous_path.exists() else []
known={r['repository']:r for r in previous if r.get('status')=='metadata_verified' and 'commit_sample_error' not in r}
with ThreadPoolExecutor(max_workers=2) as pool:
 refreshed=list(pool.map(worker,[r for r in repos if r not in known]))
known.update({r['repository']:r for r in refreshed});records=[known[r] for r in repos]
write_json(root/'reports/task-generation-1000/repository-activity.json',{'checked_at':now.isoformat(),'scope':'Current public default-branch metadata and latest ten commits; historical task pins remain unchanged. Five sampled commits in thirty days is a selection indicator, not a complete activity history. Unknowns remain explicit.','repositories':records})
from collections import Counter
print(dict(Counter(r['activity_assessment'] for r in records)))
