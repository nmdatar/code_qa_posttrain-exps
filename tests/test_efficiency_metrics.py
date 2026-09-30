import unittest
from types import SimpleNamespace as NS
from training_pipeline.efficiency_metrics import summarize


class EfficiencyMetricsTests(unittest.TestCase):
    def trajectory(self, tier, output=50, status='resolved'):
        return NS(usage={'input_tokens':100, 'output_tokens':output, 'tool_calls':1,
                         'tool_seconds':2, 'latency_seconds':10},
                  verification=NS(status=status, diagnostics={'tier':tier}))

    def test_all_attempt_costs_including_failures_and_unresolved(self):
        rows = [self.trajectory('accepted'), self.trajectory('failed', 100),
                self.trajectory('accepted', 150, 'unresolved')]
        result = summarize(rows)
        self.assertEqual(result['accepted_answers'], 1)
        self.assertAlmostEqual(result['accepted_rate'], 1/3)
        self.assertEqual(result['mean_output_tokens'], 100)
        self.assertEqual(result['output_tokens_per_accepted_answer'], 300)
        self.assertEqual(result['compute_units_per_accepted_answer'], 1500)

    def test_missing_measurements_and_no_acceptances_are_not_zero_cost(self):
        row = self.trajectory('partial')
        del row.usage['tool_seconds']
        result = summarize([row])
        self.assertIsNone(result['output_tokens_per_accepted_answer'])
        self.assertIsNone(result['mean_tool_seconds'])
        self.assertNotIn('mean_compute_units', result)
        self.assertEqual(result['tool_seconds_measurement_coverage'], 0)

    def test_bonus_separate_from_strict_acceptance(self):
        row = self.trajectory('accepted')
        row.verification.diagnostics['training_feedback'] = {
            'eligible':True, 'tier':'accepted', 'reward':.97}
        self.assertAlmostEqual(summarize([row])['mean_accepted_efficiency_bonus'], .07)


if __name__ == '__main__':
    unittest.main()
