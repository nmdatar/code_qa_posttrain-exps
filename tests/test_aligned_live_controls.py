import copy
import unittest
from training_pipeline.aligned_controls import VARIANTS, evaluate_gate, judge_bound


class AlignedLiveControlTests(unittest.TestCase):
    def fixture_records(self):
        fixture={'sources':[{'task_id':str(i)} for i in range(4)]}
        rows=[]
        for source in fixture['sources']:
            for variant in VARIANTS:
                for replicate in (1,2):
                    reward={'correct':1.,'partial':.5,'wrong':0.,'mixed_contradiction':.25,
                            'unsupported_extra':.8,'uncited':.9,'invalid_citation':.85}[variant]
                    components={'uncited':int(variant=='uncited'),'contradicted':int(variant in ('wrong','mixed_contradiction')),
                                'unsupported':int(variant=='unsupported_extra'),
                                'bad_citation':int(variant=='invalid_citation')}
                    rows.append({'task_id':source['task_id'],'variant':variant,'replicate':replicate,
                                 'eligible':True,'reward':reward,'coverage':.5 if variant=='partial' else 1.,
                                 'components':components})
        return fixture,rows

    def test_expected_controls_pass(self):
        fixture,rows=self.fixture_records()
        self.assertTrue(evaluate_gate(rows,fixture)['passed'])

    def test_unsupported_mislabelled_as_contradiction_blocks_training(self):
        fixture,rows=self.fixture_records()
        for row in rows:
            if row['variant']=='unsupported_extra':
                row['components'].update(unsupported=0,contradicted=1)
        result=evaluate_gate(rows,fixture)
        self.assertFalse(result['passed'])
        self.assertTrue(any('unsupported_not_distinguished' in failure for failure in result['failures']))

    def test_correct_false_penalty_or_low_coverage_blocks_training(self):
        fixture,rows=self.fixture_records()
        correct=next(r for r in rows if r['variant']=='correct');correct['components']['contradicted']=1
        self.assertFalse(evaluate_gate(rows,fixture)['passed'])
        fixture,rows=self.fixture_records()
        for row in rows[:3]:row['eligible']=False
        self.assertFalse(evaluate_gate(rows,fixture)['passed'])

    def test_missing_variant_does_not_pass_via_averages(self):
        fixture,rows=self.fixture_records()
        for row in rows:
            if row['task_id']=='0' and row['variant']=='wrong':row['eligible']=False
        result=evaluate_gate(rows,fixture)
        self.assertEqual(result['eligible'],54)
        self.assertFalse(result['passed'])

    def test_all_three_stages_and_repair_calls_budgeted(self):
        config={'judge':{'context_tokens':65536,'max_tokens':8192,'repair_attempts':1,
                         'prices':{'prefill':3.,'sample':7.5}}}
        self.assertAlmostEqual(judge_bound(config,56),78.446592)
        config['judge']['repair_attempts']=0
        self.assertAlmostEqual(judge_bound(config,56),39.223296)

    def test_nonfinite_or_missing_numbers_never_pass(self):
        for field in ('reward', 'coverage', 'contradicted', 'unsupported', 'uncited', 'bad_citation'):
            for value in (float('nan'), float('inf'), None):
                fixture, rows = self.fixture_records()
                row = next(r for r in rows if r['variant']=='uncited')
                target = row if field in ('reward','coverage') else row['components']
                target[field] = value
                self.assertFalse(evaluate_gate(rows, fixture)['passed'], (field,value))

    def test_correct_citation_penalties_fail(self):
        for field in ('uncited','bad_citation'):
            fixture, rows = self.fixture_records()
            next(r for r in rows if r['variant']=='correct')['components'][field] = 1
            self.assertFalse(evaluate_gate(rows, fixture)['passed'])
