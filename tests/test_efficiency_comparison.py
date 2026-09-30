import unittest
from scripts.analyze_efficiency_rl import compare, efficiency_signal


def report(accepted, output=100):
    return {'results':[{'task_id':str(i),'family_id':'repo'+str(i//2),
        'status':'resolved','tier':'accepted' if i in accepted else 'failed',
        'usage':{'input_tokens':100,'output_tokens':output,'tool_seconds':1}}
        for i in range(8)]}


class ComparisonTests(unittest.TestCase):
    def test_bonus_must_survive_group_normalization(self):
        result=efficiency_signal([{'rewards':[0.,0.,0.,.97,.92,.94,.97,.99]}])
        self.assertEqual(result['eligible_groups'],2)
        self.assertEqual(result['groups_with_changed_normalized_advantages'],1)
        self.assertEqual(result['all_accepted_ties_broken'],1)

    def test_paired_cost_reduction_without_quality_change(self):
        result=compare(report(set(range(8))),report(set(range(8)),50))
        self.assertEqual(result['repository_clusters'],4)
        self.assertEqual(result['observed']['acceptance_delta'],0)
        self.assertEqual(result['observed']['relative_compute_per_acceptance'],-.25)
        self.assertEqual(result['cluster_bootstrap_95']['relative_compute_per_acceptance'],[-.25,-.25])

    def test_zero_acceptance_bootstraps_are_not_silently_removed(self):
        result=compare(report({0}),report({0},50))
        self.assertGreater(result['undefined_bootstrap_draws']['relative_compute_per_acceptance'],0)
        self.assertIsNone(result['cluster_bootstrap_95']['relative_compute_per_acceptance'])

    def test_unresolved_is_not_a_success(self):
        a=report(set(range(8)));b=report(set(range(8)))
        b['results'][0]['status']='unresolved'
        self.assertEqual(compare(a,b)['observed']['acceptance_delta'],-.125)

    def test_mismatched_tasks_rejected(self):
        a=report({0});b=report({0});b['results'].pop()
        with self.assertRaises(AssertionError):compare(a,b)


if __name__ == '__main__':
    unittest.main()
