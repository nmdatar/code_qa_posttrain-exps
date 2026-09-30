import copy
import json
import unittest
from training_pipeline.efficiency_reward import token_penalty
from tests.test_correctness_only import CorrectnessTests
from training_pipeline.contracts import Trajectory

class TokenPenaltyTests(unittest.TestCase):
    def test_correctness_dominates_and_missing_is_not_zero(self):
        self.assertIsNone(token_penalty(None, 100, 6000))
        self.assertEqual(token_penalty(0, 0, 6000), 0)
        self.assertEqual(token_penalty(1, 6000, 6000), .9)
        self.assertEqual(token_penalty(1, 12000, 6000), .9)
        self.assertGreater(token_penalty(1, 6000, 6000), token_penalty(.8, 0, 6000))
        with self.assertRaises(ValueError): token_penalty(1, None, 6000)

class TokenPenaltyIntegrationTests(CorrectnessTests):
    def test_penalty_only_changes_training_reward(self):
        self.config['environment'].update(scoring_policy='correctness-only-v1', grading_version='all-claims-v7')
        self.config['training_reward']={'version':'positive-coverage-v4','efficiency_penalty':'output-token-fraction-v1'}
        self.config['limits']['max_output_tokens']=6000
        result={'status':'resolved','score':0.,'strict_status':'resolved','reason':'coverage',
                'training_feedback':{'reward':.75,'eligible':True},'claims':[],'claim_count':1,'rubric_hash':'hash'}
        self.factory.grade=lambda r:copy.deepcopy(result)
        for split, expected in [('train',.7125),('development',.75)]:
            self.row['split']=split
            episode=self.episode('penalty-'+split)
            episode.recorder.record['output_tokens']=3000
            action=episode.parse_action(json.dumps({'answer':'A correct fact','citations':[]}))
            t=Trajectory('r',0,'task','h','g','ep','p','e','x',split,submission=action['answer'],termination='completed')
            grade=episode.verify(t)
            self.assertAlmostEqual(grade.reward,expected)
            self.assertEqual(grade.diagnostics['correctness_score'],.75)
            episode.close()
