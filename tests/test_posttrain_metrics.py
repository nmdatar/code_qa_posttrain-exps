import unittest
from posttrain.metrics import evaluation_metrics

class MetricsTests(unittest.TestCase):
    def test_unresolved_stays_in_denominator_and_unknown_cost(self):
        score={'status':'resolved','coverage':1,'report':{'material_error':False,'semantic':{'citation_links':[{'claim_id':'c','citation_id':'x','supported':True}]*2},'metrics':{'latency_seconds':2,'termination_reason':'completed','cost':None}}}
        m=evaluation_metrics([{'verification':score},{'verification':{'status':'unresolved'}}])
        self.assertEqual(m['required_claim_coverage'],.5)
        self.assertEqual(m['scoring_coverage'],.5)
        self.assertEqual(m['completion_rate'],.5)
        self.assertEqual(m['unresolved_tasks'],1)
        self.assertIsNone(m['cost_usd'])
        self.assertEqual(m['citation_support_rate'],1)
    def test_empty_is_unknown(self):
        self.assertIsNone(evaluation_metrics([])['required_claim_coverage'])
