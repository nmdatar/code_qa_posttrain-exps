import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts.prepare_complete_sft_study import audit


class CompleteStudyAuditTests(unittest.TestCase):
    def scan(self, traces):
        with tempfile.TemporaryDirectory() as directory:
            paths=[]
            for i, trace in enumerate(traces):
                p=Path(directory)/f'{i}.json'; p.write_text(json.dumps(trace)); paths.append(p)
            example={'lineage_id':'x','proofs':[{'generation':{'tokens':[1,2,3]}}]}
            with patch('scripts.prepare_complete_sft_study.candidate',return_value=example) as admit, patch('scripts.prepare_complete_sft_study.validate_rendering'):
                result=audit(paths,{'x':{}},'current',object())
                return result,admit.call_count

    def trace(self, **changes):
        t={'split':'train','task_id':'x','verification':{'version':'current','diagnostics':{'strict_score':1}}}
        t.update(changes)
        return t

    def test_perfect_factual_reward_cannot_admit_failed_strict_answer(self):
        t=self.trace(verification={'version':'current','reward':1,'diagnostics':{'training_reward':1,'strict_score':0}})
        (choices,counts,_),calls=self.scan([t])
        self.assertEqual(calls,0);self.assertFalse(choices)
        self.assertEqual(counts['perfect_training_reward_not_admission'],1)

    def test_evaluation_and_other_grading_versions_never_reach_admission(self):
        (choices,counts,_),calls=self.scan([self.trace(split='development'), self.trace(verification={'version':'old','diagnostics':{'strict_score':1}})])
        self.assertEqual(calls,0);self.assertFalse(choices)
        self.assertEqual(counts['other_grading_identity'],1)

    def test_duplicate_archives_and_lineage_repeats_do_not_inflate_dataset(self):
        t=self.trace();u=self.trace(episode_id='second')
        (choices,counts,_),calls=self.scan([t,t,u])
        self.assertEqual(calls,2);self.assertEqual(len(choices),1)
        self.assertEqual(counts['duplicate_archive_files'],1)
        self.assertEqual(counts['complete_verified_candidates'],2)


if __name__ == '__main__':unittest.main()
