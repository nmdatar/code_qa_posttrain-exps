import copy
import json
import tempfile
import unittest
from pathlib import Path
from tests.test_training_pipeline import trajectory, FakeBackend, PRICES
from training_pipeline.contracts import VerificationResult
from training_pipeline.strategies import reinforce_batch, grpo_batch
from training_pipeline.orchestrator import Pipeline
from training_pipeline.smoke import smoke_config
from training_pipeline.storage import load_checkpoint
from training_pipeline.config import validate_config as validate
from training_pipeline.launch import estimate


class RunningReinforceTests(unittest.TestCase):
    def test_equal_successes_reinforce_and_later_failures_discourage(self):
        group=[trajectory(i,1) for i in range(4)]
        self.assertEqual(grpo_batch([group])[0],[])
        rows,stats,state=reinforce_batch([group])
        self.assertEqual(stats['advantages'],[1]*4)
        self.assertAlmostEqual(sum(sum(r.weights) for r in rows),1)
        rows,stats,new=reinforce_batch([[trajectory(i,0) for i in range(4)]],state)
        self.assertEqual(stats['advantages'],[-1]*4)
        self.assertAlmostEqual(sum(sum(r.weights) for r in rows),-1)
        self.assertEqual(state,{'reward_sum':4.,'count':4})
        self.assertEqual(new,{'reward_sum':4.,'count':8})

    def test_baseline_is_prior_only_and_unresolved_groups_are_excluded(self):
        bad=[trajectory(4,1),trajectory(5,0)]
        bad[1].verification=VerificationResult('unresolved',None,'v1')
        rows,stats,state=reinforce_batch([[trajectory(0,0),trajectory(1,1)],bad],{'reward_sum':1.,'count':4})
        self.assertEqual(stats['advantages'],[-.25,.75])
        self.assertEqual(state,{'reward_sum':2.,'count':6})
        self.assertEqual(stats['excluded_groups'],1)
        for row in rows:self.assertEqual(row.weights[0],0)  # prompt receives no gradient
        with self.assertRaises(ValueError):reinforce_batch([[trajectory(0,1),trajectory(0,1)]])
        with self.assertRaises(ValueError):reinforce_batch([[trajectory(0,1),trajectory(1,1)]],{'count':0,'reward_sum':1})

    def test_resume_preserves_baseline_and_sampler_boundary(self):
        with tempfile.TemporaryDirectory() as d:
            c=smoke_config(d,PRICES,Path(d)/'ledger.json');c['limits']['context_tokens']=4096
            stage=c['stages'][1];stage.update(kind='reinforce',baseline='prior-mean-v1',max_batches=3,max_updates=3)
            c['stages']=[stage];c['evaluation']['every']=0
            backend=FakeBackend();p=Pipeline(c,backend=backend);path=p.run(stop_after_updates=1)
            m=load_checkpoint(path);state=m['state']['reinforce_baselines']['0'];self.assertEqual(state['count'],8)
            restored=FakeBackend(backend.registry);q=Pipeline(c,backend=restored);last=q.run(path,'resume');end=load_checkpoint(last)
            self.assertEqual(end['state']['reinforce_baselines']['0']['count'],24)
            events=[json.loads(l) for l in (Path(c['output'])/'events.jsonl').read_text().splitlines()]
            batches=[e for e in events if e['event']=='training_batch']
            self.assertEqual(batches[1]['advantage_baseline'],state['reward_sum']/state['count'])
            self.assertEqual(end['state']['optimizer_step'],3)
            self.assertEqual(len({e['behavior_policy'] for e in events if e['event']=='update'}),3)

    def test_v7_launch_bounds_reinforce_like_grpo_with_train_only_third_judge(self):
        c=json.loads(Path('configs/experiments/grpo-long-v6/run.json').read_text());base=estimate(c,c['spend']['prices'])
        c['stages'][0].update(kind='reinforce',baseline='prior-mean-v1');validate(c)
        result=estimate(c,c['spend']['prices']);self.assertEqual(result['components_usd'],base['components_usd'])
        self.assertEqual(result['episodes_including_retries'],192)
        j=c['judge'];unit=((j['context_tokens']-j['max_tokens'])*j['prices']['prefill']+j['max_tokens']*j['prices']['sample'])/1e6
        self.assertAlmostEqual(result['components_usd']['reference_grading'],(128*3+64*2)*2*unit)
        c['execution']['operation']='benchmark'
        benchmark=estimate(c,c['spend']['prices'],evaluation_only=True)
        self.assertAlmostEqual(benchmark['components_usd']['reference_grading'],benchmark['episodes_including_retries']*3*2*unit)

if __name__=='__main__':unittest.main()
