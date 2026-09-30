"""Report lexical near-duplicates without silently deleting distinct tasks."""
import json,re
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from dataset_builder.build import read_jsonl,write_json
rows=[]
for p in Path('data/generated/collection-1000-v1').glob('*/public/tasks.jsonl'):rows+=read_jsonl(p)
postings=defaultdict(list);grams={}
for i,r in enumerate(rows):
 words=re.findall(r'\w+',r['user_prompt'].casefold());g=set(zip(words,words[1:],words[2:]));grams[i]=g
 for x in g:postings[x].append(i)
pairs=set()
for ids in postings.values():
 if len(ids)<60:pairs.update(combinations(ids,2))
flagged=[]
for i,j in sorted(pairs):
 a,b=grams[i],grams[j];score=len(a&b)/max(1,len(a|b))
 if score>=.65:flagged.append({'task_ids':[rows[i]['id'],rows[j]['id']],'repositories':[rows[i]['repository'],rows[j]['repository']],'questions':[rows[i]['user_prompt'],rows[j]['user_prompt']],'trigram_jaccard':round(score,4),'status':'needs_comparison'})
write_json(Path('reports/task-generation-1000/near-duplicate-audit.json'),{'records':len(rows),'method':'Lowercased word-trigram Jaccard >=0.65; lexical screening only, not a guarantee of semantic uniqueness. Common grams appearing in >=60 questions are excluded from candidate generation.','flagged_pairs':flagged})
print(json.dumps({'records':len(rows),'flagged_pairs':len(flagged)}))
