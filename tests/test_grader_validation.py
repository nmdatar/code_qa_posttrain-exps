import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from training_pipeline.storage import atomic_json, digest, read
from training_pipeline.grader_validation import run

class ValidationTests(unittest.TestCase):
    def test_invalid_and_unresolved_grades_are_not_zero_rewards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            fixture={'purpose':'test','cases':[{'name':n,'request':{'episode_id':n}} for n in ['good','invalid','uncertain']]}
            fixture['fixture_hash']=digest(fixture);atomic_json(root/'fixture.json',fixture)
            class Ledger:
                def __init__(self,*args):
                    self.path=root/'ledger.json';atomic_json(self.path,{'reserved_usd':0})
            class Factory:
                reward_version='frozen-test'
                def __init__(self,*args):pass
                def close(self):pass
                def grade(self,request):
                    if request['episode_id']=='invalid':raise ValueError('invalid schema')
                    r=None if request['episode_id']=='uncertain' else .5
                    return {'status':'unresolved' if r is None else 'resolved','training_feedback':{'reward':r}}
            config={'environment':{'grading_version':'all-claims-v6'},'output':str(root/'output'),'spend':{'ledger':'unused','cap_usd':5,'prices':{}},'model':{'checkpoint_ttl_seconds':1},'judge':{'context_tokens':100,'max_tokens':100,'prices':{'prefill':1,'sample':1}}}
            with patch('training_pipeline.budget.SpendLedger',Ledger),patch('training_pipeline.collection.CollectionFactory',Factory):
                run(config,{'fixture':str(root/'fixture.json'),'reservation_limit_usd':1})
            report=read(root/'output/report.json');cases={c['name']:c for c in report['cases']}
            self.assertEqual(report['status'],'complete')
            self.assertEqual(cases['good']['reward'],.5)
            self.assertIsNone(cases['invalid']['reward']);self.assertIsNone(cases['uncertain']['reward'])
            self.assertEqual(cases['invalid']['error_type'],'ValueError')
