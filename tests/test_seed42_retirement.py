import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('retire_seed42', Path(__file__).resolve().parents[1]/'scripts/retire_direct_seed42_allocation.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Seed42RetirementTests(unittest.TestCase):
    def test_retirement_preserves_complete_history_and_original(self):
        state = {'cap': 1800, 'reserved_usd': module.EXPECTED_RESERVED,
                 'reservations': [{'kind': 'train', 'upper_estimate_usd': 1}],
                 'prices': {'train': .737}, 'actual_billing_usd': None, 'ttl_seconds': 172800}
        before = copy.deepcopy(state)
        retired = module.retirement_state(state)
        self.assertEqual(state, before)
        self.assertEqual(retired['cap'], state['reserved_usd'])
        for key in state:
            if key != 'cap':
                self.assertEqual(retired[key], state[key])
        self.assertEqual(retired['allocation_history'][-1]['previous_cap'], 1800)

    def test_changed_cap_or_reservations_fail_closed(self):
        for state in [{'cap': 1801, 'reserved_usd': module.EXPECTED_RESERVED},
                      {'cap': 1800, 'reserved_usd': module.EXPECTED_RESERVED+1}]:
            with self.subTest(state=state), self.assertRaises(ValueError):
                module.retirement_state(state)
