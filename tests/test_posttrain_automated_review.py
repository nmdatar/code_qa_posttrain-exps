"""Automated admission is source-review-bound and never fabricates human labels."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from qa_eval.review import automated_rubric_supported
from posttrain.storage import digest
from posttrain.config import load_config
from posttrain.runner import freeze_data, readiness
from posttrain.data import inventory_release

ROOT = Path(__file__).resolve().parents[1]

def review_for(task):
    return dict(task_id=task['id'], status='supported', author='author', reviewer='verifier', rubric_sha256=digest(task))

class AutomatedReviewTests(unittest.TestCase):
    def test_valid_independent_review_and_stale_rejections(self):
        task={'id':'t','gold_status':'draft','human_reviewed':False}
        review=review_for(task)
        self.assertTrue(automated_rubric_supported(task,review))
        for field,value in [('status','needs_review'),('reviewer',' AUTHOR '),('task_id','other'),('rubric_sha256','0'*64)]:
            altered={**review,field:value}
            self.assertFalse(automated_rubric_supported(task,altered))
        self.assertFalse(automated_rubric_supported({**task,'gold_status':'ambiguous'},review))
        self.assertFalse(automated_rubric_supported(task,None))

    def test_real_releases_ready_without_changing_human_flags(self):
        for name,count in [('repo-qa-training-posttrain-v1',100),('repo-qa-development-posttrain-v1',20)]:
            root=ROOT/'data/releases'/name
            if not root.exists(): self.skipTest('Release unavailable')
            report=inventory_release(root)
            self.assertEqual(report['ready'],count)
            self.assertEqual(report['automated_admitted'],count)
            self.assertEqual(report['human_reviewed'],0)
            self.assertEqual(inventory_release(root,review_policy='human')['ready'],0)

    def test_long_live_readiness_uses_bound_reviews_not_human_calibration(self):
        path=ROOT/'examples/posttrain/repository-diagnostic.json'
        if not path.exists(): self.skipTest('Configured release unavailable')
        c=load_config(path);c['diagnostic']=False
        c['stages'][0]['max_updates']=10
        if any(not Path(c[k]).exists() for k in ('training_release', 'evaluation_release')): self.skipTest('Configured release unavailable')
        c=freeze_data(c)
        with patch('posttrain.runner.importlib.util.find_spec',return_value=object()),patch('posttrain.services.tinker_auth_available',return_value=True):
            ready=readiness(c)
            self.assertEqual(ready['blockers'],[])
            self.assertFalse(ready['human_calibration_required'])
            broken=copy.deepcopy(c)
            broken['rubric_reviews'].clear()
            self.assertTrue(any('automated rubric review' in x for x in readiness(broken)['blockers']))
            strict=copy.deepcopy(c);strict['review_policy']='human'
            self.assertTrue(any('Human calibration' in x for x in readiness(strict)['blockers']))

    def test_normal_scoring_accepts_machine_review_without_human_promotion(self):
        from qa_eval.demo import fixture, KEY
        from qa_eval.dataset import freeze
        from qa_eval.security import bindings, digest as qdigest, seal
        from qa_eval.grading import evaluate
        from qa_eval.source import GitSource
        with tempfile.TemporaryDirectory() as root:
            task,answer,experiment,metrics,semantic=fixture(Path(root))
            task.update(human_reviewed=False,gold_status='draft')
            experiment=freeze(experiment,[task],{'synthetic':True})
            metrics.update(bindings(task,answer))
            semantic.update(bindings(task,answer));semantic['experiment_hash']=qdigest(experiment)
            args=(task,answer,seal('EpisodeMetrics',metrics,KEY),experiment,GitSource(root,task['repository']['commit']),KEY)
            kwargs=dict(semantic_envelope=seal('SemanticAssessment',semantic,KEY))
            self.assertEqual(evaluate(*args,**kwargs)[0]['tier'],'unresolved')
            result=evaluate(*args,**kwargs,automated_review=review_for(task))[0]
            self.assertEqual(result['tier'],'accepted')
            self.assertTrue(any(x['name']=='automated_rubric_review' for x in result['checks']))
            self.assertFalse(task['human_reviewed'])
            invalid={**review_for(task),'reviewer':'author'}
            self.assertEqual(evaluate(*args,**kwargs,automated_review=invalid,diagnostic_machine_review=True)[0]['tier'],'unresolved')
