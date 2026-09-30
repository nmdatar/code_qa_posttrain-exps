import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from posttrain.preparation import prepare_candidates, build_prepared
from posttrain.storage import digest

class PreparationTests(unittest.TestCase):
    def test_prepare_reviewed_source_snapshot_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);repo=root/'repo';repo.mkdir()
            def git(*args):return subprocess.run(['git','-C',str(repo),*args],capture_output=True,check=True).stdout.decode().strip()
            git('init','-q');(repo/'example.py').write_text('VALUE = 3\n')
            git('add','example.py');git('-c','user.name=Test','-c','user.email=test@example.invalid','commit','-qm','fixture')
            sha=git('rev-parse','HEAD')
            task={'schema_version':'1.0','id':'sample','lineage_id':'sample','repository':{'url':'https://github.com/test/example','commit':sha,'family_id':'test/example'},
                  'split':'train','question':'What value is assigned?','category':'source_behavior','answerability':'answerable','claims':[{'id':'c1','text':'VALUE is3','weight':3,'separable_subparts':[],
                  'evidence':[{'path':'example.py','start_line':1,'end_line':1,'file_sha256':hashlib.sha256((repo/'example.py').read_bytes()).hexdigest()}]}],
                  'critical_errors':[],'diagram':{'required':False,'criteria':[],'allowed_abstractions':[]},'probes':[],'human_reviewed':False,'gold_status':'draft',
                  'permitted_tools':['list_files','search_code','read_file'],'budgets':{'compute_units':50000,'latency_seconds':300,'max_output_tokens':3000,'max_submission_bytes':32000,'max_tool_calls':20}}
            candidates=root/'candidates.jsonl';candidates.write_text(json.dumps({'task':task,'reference_answer':'private VALUE3'})+'\n')
            reviews=root/'reviews.json';reviews.write_text(json.dumps({'reviews':[{'task_id':'sample','status':'supported','task_sha256':digest(task)}]}))
            with patch('posttrain.preparation.fetch_snapshot',return_value=repo):
                result=prepare_candidates(candidates,reviews,root/'repos',root/'output')
                self.assertEqual(result['prepared_tasks'],1)
                bundle=Path(result['environments'][0]['bundle'])
                public=json.loads((bundle/'public/tasks.jsonl').read_text())
                self.assertNotIn('claims',public);self.assertNotIn('reference_answer',public)
                self.assertFalse(json.loads((bundle/'manifest.json').read_text())['training_eligible'])
                self.assertEqual(prepare_candidates(candidates,reviews,root/'repos',root/'output')['prepared_tasks'],1)
                candidates.write_text(json.dumps({'task':dict(task,question='Changed'),'reference_answer':'private'})+'\n')
                self.assertEqual(prepare_candidates(candidates,reviews,root/'repos',root/'output')['prepared_tasks'],0)

    def test_build_requires_known_estimate(self):
        with self.assertRaises(ValueError):build_prepared('missing',None,{})
        with self.assertRaises(ValueError):build_prepared('missing',None,{'price_source':'url','upper_usd':2})

if __name__=='__main__':unittest.main()
