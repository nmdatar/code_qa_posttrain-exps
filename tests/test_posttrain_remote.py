import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock,patch
from posttrain.remote import validate_remote_plan,launch_remote,RemoteConfig,_coordinator_command
from posttrain.storage import BudgetLedger,BudgetExceeded

class RemoteTests(unittest.TestCase):
    def config(self):
        return {'run_id':'remote-test','artifacts_root':'/runs/experiments','budget_ledger':'/runs/budgets/remote-test.json','budget_cap_usd':3,'budget_reserve_usd':.25}
    def test_plan_is_offline_and_bounds_child(self):
        with patch('posttrain.remote.build_modal_app') as build:
            p=validate_remote_plan(self.config(),'/private-grading/config.json',4,1)
            build.assert_not_called()
        self.assertEqual(p['status'],'offline_plan_not_deployed')
        self.assertEqual(p['child_cap_usd'],3)
        self.assertFalse(p['provider_enforced_billing_cap'])
        with self.assertRaises(ValueError):validate_remote_plan(self.config(),'/private-grading/config.json',3,1)
    def test_path_and_mount_restrictions(self):
        for path in ('config.json','/runs/config.json','/private-grading/../etc/passwd'):
            with self.assertRaises(ValueError):validate_remote_plan(self.config(),path,4,1)
        with self.assertRaises(ValueError):RemoteConfig(original_workspace='/etc').validate()
        with self.assertRaises(ValueError):RemoteConfig(runs_volume='x',private_volume='x').validate()
        with self.assertRaises(ValueError):_coordinator_command('/private-grading/config.json','/tmp/checkpoint.json')
        self.assertIn('--config',_coordinator_command('/private-grading/config.json','/runs/checkpoint.json'))
    def test_parent_reservation_before_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger=BudgetLedger(Path(tmp)/'ledger.json')
            app=MagicMock();coordinator=MagicMock();coordinator.spawn.return_value.object_id='fc-example'
            def factory(_):
                self.assertEqual(ledger.summary()['charged_or_reserved_usd'],4)
                return app,coordinator
            with patch('posttrain.remote.build_modal_app',side_effect=factory):
                result=launch_remote('/private-grading/config.json',ledger,4,run_config=self.config(),coordinator_upper_usd=1)
            self.assertEqual(result['function_call_id'],'fc-example')
            self.assertEqual(ledger.summary()['charged_or_reserved_usd'],4)
            allocation=coordinator.spawn.call_args.args[1]
            self.assertEqual(allocation['envelope_usd'],4)
            self.assertNotIn('calls',allocation)
    def test_failed_launch_retains_parent_charge(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger=BudgetLedger(Path(tmp)/'ledger.json')
            with patch('posttrain.remote.build_modal_app',side_effect=RuntimeError('failure')):
                with self.assertRaises(RuntimeError):launch_remote('/private-grading/config.json',ledger,4,run_config=self.config(),coordinator_upper_usd=1)
            self.assertEqual(ledger.summary()['charged_or_reserved_usd'],4)
            self.assertEqual(next(iter(ledger.summary()['calls'].values()))['status'],'unknown_outcome')
    def test_oversubscription_never_dispatches(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger=BudgetLedger(Path(tmp)/'ledger.json');ledger.reserve('previous',15)
            with patch('posttrain.remote.build_modal_app') as build:
                with self.assertRaises(BudgetExceeded):launch_remote('/private-grading/config.json',ledger,4,run_config=self.config(),coordinator_upper_usd=1)
                build.assert_not_called()
            self.assertEqual(ledger.summary()['charged_or_reserved_usd'],15)

if __name__=='__main__':unittest.main()
