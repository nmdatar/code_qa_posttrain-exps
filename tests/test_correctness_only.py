import copy
import json
from unittest.mock import patch
from tests.test_collection_training import CollectionTests
from training_pipeline.contracts import Trajectory
from training_pipeline.correctness_metrics import citation_diagnostics

class CorrectnessTests(CollectionTests):
    def test_uncited_eval_and_invalid_citations_keep_correctness_reward(self):
        self.config['environment'].update(scoring_policy='correctness-only-v1', grading_version='all-claims-v7')
        self.config['training_reward']={'version':'positive-coverage-v4'}
        self.row['split']='development'
        result={'status':'resolved','score':0.,'strict_status':'resolved','reason':'coverage',
                'training_feedback':{'reward':.75,'eligible':True},'claims':[],'claim_count':1,'rubric_hash':'hash'}
        captured=[]
        self.factory.grade=lambda r:(captured.append(r) or copy.deepcopy(result))
        for i,citations in enumerate([[],[{'path':'unknown.py','start_line':1,'end_line':2}]]):
            episode=self.episode('correct-'+str(i))
            action=episode.parse_action(json.dumps({'answer':'A correct fact','citations':citations}))
            t=Trajectory('r',0,'task','h','g','ep','p','e','x','development',submission=action['answer'],termination='completed')
            grade=episode.verify(t)
            self.assertEqual(grade.reward,.75)
            self.assertEqual(grade.diagnostics['citation_score'],0)
            self.assertEqual(grade.diagnostics['reward_applied'],'correctness')
            self.assertEqual(captured[-1]['answer']['text'],'A correct fact')
            self.assertEqual(captured[-1]['answer']['citations'],[])
            episode.close()

    def test_citation_score_measures_claim_support_not_reward(self):
        value=citation_diagnostics({'citations':[{'id':'c1'}]},
          {'assessment_complete':True,'extracted_claims':[{'id':'a'},{'id':'b'}],
           'citation_links':[{'claim_id':'a','supported':True}]},[])
        self.assertEqual(value['citation_score'],.5)
        self.assertIsNone(citation_diagnostics({'citations':[{}]},None,[])['citation_score'])
