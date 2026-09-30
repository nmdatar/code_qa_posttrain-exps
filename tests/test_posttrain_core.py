import concurrent.futures
import json
from pathlib import Path
import tempfile
import unittest
from posttrain.storage import BudgetLedger,BudgetExceeded,read,atomic
from posttrain.config import load_config,fingerprint
from posttrain.strategies import prepare_group,validate_actions,validate_sft
from posttrain.checkpoints import save_checkpoint,load_checkpoint

class CoreTests(unittest.TestCase):
    def test_concurrent_reservations(self):
        with tempfile.TemporaryDirectory() as d:
            ledger=BudgetLedger(Path(d)/'budget.json',20,2)
            def reserve(_):
                try: return ledger.reserve('test',3)
                except BudgetExceeded: return None
            with concurrent.futures.ThreadPoolExecutor(10) as pool: ids=list(pool.map(reserve,range(20)))
            self.assertEqual(sum(x is not None for x in ids),6)
            self.assertEqual(ledger.summary()['available_for_dispatch_usd'],0)

    def test_failed_request_keeps_reservation(self):
        with tempfile.TemporaryDirectory() as d:
            ledger=BudgetLedger(Path(d)/'budget.json')
            with self.assertRaises(RuntimeError): ledger.execute('test',2,lambda: (_ for _ in ()).throw(RuntimeError()))
            self.assertEqual(ledger.summary()['charged_or_reserved_usd'],2)
            self.assertEqual(next(iter(ledger.summary()['calls'].values()))['status'],'unknown_outcome')

    def test_configuration_freeze(self):
        a=load_config({'run_id':'test'})
        b=load_config({'run_id':'other','tracking_mode':'offline'})
        self.assertEqual(fingerprint(a),fingerprint(b))
        b['group_size']=8
        self.assertNotEqual(fingerprint(a),fingerprint(b))
        with self.assertRaises(ValueError): load_config({'run_id':'x','budget_cap_usd':21})
        with self.assertRaises(ValueError): load_config({'run_id':'x','stages':[{'algorithm':'grpo','max_updates':4,'learning_rate':1e-5}]})

    def test_group_integrity_and_variance(self):
        ts=[{'task_id':'a','policy_id':'p'} for _ in range(2)]
        rs=[{'status':'resolved','reward':0},{'status':'resolved','reward':1}]
        self.assertEqual(prepare_group(ts,rs)['advantages'],[-1,1])
        self.assertEqual(prepare_group(ts,[rs[0],rs[0]])['advantages'],[0,0])
        self.assertEqual(prepare_group(ts,[rs[0],{'status':'unresolved'}])['status'],'retry_or_quarantine')
        ts[1]['policy_id']='q'
        with self.assertRaises(ValueError): prepare_group(ts,rs)

    def test_token_alignment(self):
        t={'policy_id':'p','actions':[{'prompt_tokens':[1],'token_ids':[2,3],'logprobs':[-1,-2]}]}
        self.assertTrue(validate_actions(t))
        t['actions'][0]['logprobs'].pop()
        with self.assertRaises(ValueError): validate_actions(t)
        with self.assertRaises(ValueError): validate_sft({'split':'development'})

    def test_partial_checkpoint_not_published(self):
        class Backend:
            def save(self,name): raise RuntimeError('provider save failed')
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError): save_checkpoint(d,Backend(),{'optimizer_step':0},load_config({'run_id':'x'}))
            self.assertFalse((Path(d)/'latest-checkpoint.json').exists())

    def test_checkpoint_tamper(self):
        class Backend:
            def save(self,name):
                atomic(name+'.json',{'optimizer':3})
                return {'training_state':name+'.json','sampling_state':name+'.json'}
        with tempfile.TemporaryDirectory() as d:
            p=save_checkpoint(d,Backend(),{'optimizer_step':3},load_config({'run_id':'x'}))
            self.assertEqual(load_checkpoint(p)['state']['optimizer_step'],3)
            m=read(p);m['state']['optimizer_step']=4;atomic(p,m)
            with self.assertRaises(ValueError): load_checkpoint(p)
