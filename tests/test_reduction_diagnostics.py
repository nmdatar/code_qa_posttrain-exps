"""Loss-reduction diagnostics preserve strict fail-closed numerical checks."""
import json
import math
import unittest
from types import SimpleNamespace as NS
from training_pipeline.tinker_backend import check_reduction


def output(values, measured):
    return NS(loss_fn_outputs=[{'logprobs': NS(data=values)}], metrics={'loss:sum': measured})


def row(weights, old=None):
    return NS(weights=weights, logprobs=old or [0.] * len(weights))


class ReductionDiagnosticsTests(unittest.TestCase):
    def reject(self, rows, result):
        with self.assertRaises(ValueError) as caught:
            check_reduction(rows, 'importance_sampling', result)
        detail = caught.exception.reduction_diagnostics
        json.dumps(detail, allow_nan=False)
        return detail

    def test_cancelled_objective_is_diagnosed_not_silently_accepted(self):
        detail = self.reject([row([.25, -.24999])], output([0., 0.], .00002))
        self.assertEqual(detail['reason'], 'sum_mismatch')
        self.assertGreater(detail['cancellation_ratio'], 49000)
        self.assertAlmostEqual(detail['sum_absolute_terms'], .49999)
        self.assertAlmostEqual(detail['accepted_tolerance'], 1e-5)
        self.assertEqual((detail['positive_terms'], detail['negative_terms']), (1, 1))

    def test_wrong_scaling_still_rejected(self):
        detail = self.reject([row([.25, .25])], output([0., 0.], -.25))
        self.assertEqual(detail['expected'], -.5)
        self.assertEqual(detail['measured'], -.25)
        self.assertEqual(detail['cancellation_ratio'], 1.)

    def test_shape_failures_are_distinct(self):
        mismatch = NS(loss_fn_outputs=[], metrics={})
        self.assertEqual(self.reject([row([1.])], mismatch)['reason'], 'output_count_mismatch')
        self.assertEqual(self.reject([row([1., 2.])], output([0.], 0))['reason'], 'token_count_mismatch')

    def test_nonfinite_learner_and_metric_are_json_safe(self):
        detail = self.reject([row([1.])], output([float('nan')], 0))
        self.assertEqual(detail['reason'], 'nonfinite_learner_logprob')
        detail = self.reject([row([1.])], output([0.], float('nan')))
        self.assertEqual(detail['reason'], 'sum_mismatch')
        self.assertFalse(detail['measured_finite'])
        self.assertIsNone(detail['measured'])

    def test_original_acceptance_and_masking_preserved(self):
        self.assertEqual(check_reduction([row([0., 1.])], 'importance_sampling',
                                        output([float('nan'), 0.], -1.)), -1.)

    def test_stable_and_float32_input_values_are_diagnostics_only(self):
        detail = self.reject([row([1e16, 1., -1e16])], output([0., 0., 0.], -1.))
        self.assertEqual(detail['expected'], 0.)
        self.assertEqual(detail['stable_expected'], -1.)
        self.assertIn('float32_input_expected', detail)
