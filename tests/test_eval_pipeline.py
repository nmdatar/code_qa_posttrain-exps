import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from eval_pipeline.run import main, summarize, validate_scores, DIMS


class EvaluationTests(unittest.TestCase):
    def test_bad_judge_scores_are_rejected(self):
        for value in ({}, dict.fromkeys(DIMS, 11), dict.fromkeys(DIMS, True)):
            with self.assertRaises(ValueError):
                validate_scores(value)

    def test_missing_predictions_stay_in_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks = [dict(id=str(i), repo='test/repo', commit_id='a'*40, question='Where?') for i in range(3)]
            refs = [dict(id=str(i), reference_answer='reference') for i in range(3)]
            for name, rows in [('tasks', tasks), ('references', refs), ('predictions', [dict(id='0', answer='answer')])]:
                (root/(name+'.jsonl')).write_text(''.join(json.dumps(x)+'\n' for x in rows))
            (root/'manifest.json').write_text(json.dumps(dict(revision='pinned', source_sha256='hash',task_ids=['0','1','2'])))
            with patch('eval_pipeline.run.chat', return_value=(json.dumps(dict(scores=dict.fromkeys(DIMS, 8),reason='ok')), {})):
                exit_code = main(['--data', str(root), '--mode', 'import', '--model', 'test-model', '--judge-model', 'test-judge', '--predictions',str(root/'predictions.jsonl'), '--output',str(root/'out')])
            summary = json.loads((root/'out/summary.json').read_text())
            self.assertEqual(exit_code, 1)
            self.assertEqual(summary['expected'], 3)
            self.assertEqual(summary['scored'], 1)
            self.assertEqual(summary['failed'], 2)
            self.assertEqual(summary['mean_score'], 40)
            self.assertEqual(summary['completion_rate'], 1/3)
            self.assertIsNone(summary['total_tokens'])
            self.assertTrue((root/'out/report.html').exists())

    def test_duplicate_predictions_rejected(self):
        from eval_pipeline.run import read_rows
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'rows.jsonl'
            path.write_text('{"id":"x"}\n{"id":"x"}\n')
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                read_rows(path)

    def test_no_scores_is_unknown_not_zero(self):
        s = summarize([dict(status='agent_error', answer='', scores=None,
                            latency_seconds=1, input_tokens=None, output_tokens=None)])
        self.assertIsNone(s['mean_score'])
        self.assertEqual(s['scoring_rate'], 0)
