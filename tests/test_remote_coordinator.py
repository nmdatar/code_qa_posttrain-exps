import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent_harness.coordinator import coordinate


def manifest(n=6, **changes):
    value = {'run_id': 'run-1', 'max_rollouts': 3, 'max_graders': 1,
             'jobs': [{'episode_id': f'e{i}', 'task_id': 't1', 'group_id': f'g{i // 2}'} for i in range(n)]}
    value.update(changes)
    return value


class Crash(BaseException):
    pass


class Executor:
    def __init__(self, path, *, grade_delay=0):
        self.path, self.grade_delay = path, grade_delay
        self.calls, self.submissions = {}, []
        self.fail_submit, self.fail_poll, self.mismatch, self.unresolved = set(), set(), set(), set()
        self.crash_submit = False

    def submit(self, kind, job):
        # Intent must be durably written before an external side effect.
        state = json.loads(self.path.read_text())['episodes'][job['episode_id']]['state']
        if state != 'submitting_' + kind:
            raise AssertionError('missing pre-submit journal')
        self.submissions.append((kind, job['episode_id']))
        if self.crash_submit:
            raise Crash()
        if (kind, job['episode_id']) in self.fail_submit:
            raise RuntimeError('private exception text')
        call_id = str(len(self.submissions))
        self.calls[call_id] = [kind, job, 0]
        return call_id

    def poll(self, call_id):
        kind, job, polls = self.calls[call_id]
        self.calls[call_id][2] += 1
        if (kind, job['episode_id']) in self.fail_poll:
            raise RuntimeError('private exception text')
        if kind == 'grade' and polls < self.grade_delay:
            return None
        result = {k: job[k] for k in ('run_id', 'episode_id', 'task_id', 'group_id')}
        result['status'] = 'completed' if kind == 'rollout' else 'resolved'
        if kind == 'grade':
            result['reward'] = .5
        if (kind, job['episode_id']) in self.mismatch:
            result['task_id'] = 'wrong'
        if (kind, job['episode_id']) in self.unresolved:
            result['status'] = 'unresolved'
        return result


class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'journal.json'

    def run_cohort(self, value=None, executor=None, **kwargs):
        return coordinate(value or manifest(), executor or Executor(self.path), self.path,
                          sleep=lambda seconds: None, **kwargs)

    def test_complete_cohort_and_bounded_backpressure(self):
        observations = []
        executor = Executor(self.path, grade_delay=4)
        def persisted():
            entries = json.loads(self.path.read_text())['episodes'].values()
            counts = {state: sum(x['state'] == state for x in entries)
                      for state in ('active_rollout', 'active_grade', 'ready')}
            observations.append(counts)
            self.assertLessEqual(counts['active_rollout'], 3)
            self.assertLessEqual(counts['active_grade'], 1)
            self.assertLessEqual(counts['ready'], 1)
        result = self.run_cohort(executor=executor, persist=persisted)
        self.assertEqual(result['status'], 'finished')
        self.assertEqual(result['completed_episodes'], 6)
        self.assertEqual(result['unresolved_episodes'], 0)
        self.assertEqual(len(executor.submissions), 12)
        self.assertEqual(result['groups']['g0'], {'status': 'ready', 'episode_ids': ['e0', 'e1'], 'rewards': [.5, .5]})
        self.assertTrue(any(x['ready'] == 1 and x['active_grade'] == 1 for x in observations))
        self.assertTrue(all(x['history'] for x in result['episodes']))

    def test_completed_resume_has_no_resubmissions(self):
        executor = Executor(self.path)
        first = self.run_cohort(executor=executor)
        count = len(executor.submissions)
        second = self.run_cohort(executor=executor)
        self.assertEqual(first, second)
        self.assertEqual(len(executor.submissions), count)

    def test_active_handles_resume_without_resubmission(self):
        executor = Executor(self.path)
        with self.assertRaises(Crash):
            coordinate(manifest(), executor, self.path, sleep=lambda _: (_ for _ in ()).throw(Crash()))
        first = list(executor.submissions)
        self.assertEqual(first, [('rollout', 'e0'), ('rollout', 'e1'), ('rollout', 'e2')])
        result = self.run_cohort(executor=executor)
        self.assertEqual(result['completed_episodes'], 6)
        for call in first:
            self.assertEqual(executor.submissions.count(call), 1)

    def test_ambiguous_submission_quarantined_on_resume(self):
        executor = Executor(self.path)
        executor.crash_submit = True
        with self.assertRaises(Crash):
            self.run_cohort(manifest(2), executor)
        executor.crash_submit = False
        result = self.run_cohort(manifest(2), executor)
        self.assertEqual(executor.submissions.count(('rollout', 'e0')), 1)
        self.assertEqual(result['unresolved_episodes'], 1)
        self.assertEqual(result['episodes'][0]['reason'], 'ambiguous_submission_requires_manual_recovery')
        self.assertEqual(result['groups']['g0']['status'], 'excluded')
        self.assertIsNone(result['groups']['g0']['rewards'])

    def test_each_failure_remains_in_complete_cohort(self):
        executor = Executor(self.path)
        executor.fail_submit.add(('rollout', 'e0'))
        executor.fail_poll.add(('rollout', 'e1'))
        executor.mismatch.add(('grade', 'e2'))
        executor.unresolved.add(('grade', 'e3'))
        result = self.run_cohort(executor=executor)
        self.assertEqual(result['expected_episodes'], 6)
        self.assertEqual(result['completed_episodes'], 2)
        self.assertEqual(result['unresolved_episodes'], 4)
        self.assertEqual(len(result['episodes']), 6)
        self.assertNotIn('private exception text', self.path.read_text())
        self.assertEqual(result['groups']['g0']['status'], 'excluded')
        self.assertEqual(result['groups']['g1']['status'], 'excluded')
        self.assertEqual(result['groups']['g2']['status'], 'ready')

    def test_identity_mismatch_does_not_launch_grader(self):
        executor = Executor(self.path)
        executor.mismatch.add(('rollout', 'e0'))
        result = self.run_cohort(manifest(1), executor)
        self.assertEqual(result['unresolved_episodes'], 1)
        self.assertEqual(executor.submissions, [('rollout', 'e0')])

    def test_deadline_preserves_active_ids_and_pending_cohort(self):
        executor = Executor(self.path)
        clock = [1000.0]
        def tick(seconds):
            clock[0] += 2
        with patch('agent_harness.coordinator.time.time', side_effect=lambda: clock[0]):
            result = coordinate(manifest(coordinator_seconds=1), executor, self.path, sleep=tick)
        self.assertEqual(result['unresolved_episodes'], 6)
        self.assertTrue(all(x['reason'] == 'coordinator_deadline_exceeded' for x in result['episodes']))
        self.assertEqual(result['episodes'][0]['call_id'], '1')
        self.assertEqual(len(executor.submissions), 3)

    def test_deadline_cancels_active_rollouts_and_graders_only(self):
        executor = Executor(self.path, grade_delay=100)
        cancelled = []
        executor.cancel = cancelled.append
        clock = [1000.0]
        sleeps = [0]
        def tick(seconds):
            sleeps[0] += 1
            if sleeps[0] == 2:
                clock[0] += 2
        with patch('agent_harness.coordinator.time.time', side_effect=lambda: clock[0]):
            result = coordinate(manifest(coordinator_seconds=1), executor, self.path, sleep=tick)
        self.assertTrue(any(executor.calls[c][0] == 'rollout' for c in cancelled))
        self.assertTrue(any(executor.calls[c][0] == 'grade' for c in cancelled))
        self.assertEqual(len(cancelled), len(set(cancelled)))
        for entry in result['episodes']:
            if entry.get('call_id') in cancelled:
                self.assertEqual(entry['cancellation_status'], 'requested')
            else:
                self.assertNotIn('cancellation_status', entry)
        self.assertEqual(result['unresolved_episodes'], 6)
        self.assertTrue(all(group['status'] == 'excluded' for group in result['groups'].values()))

    def test_cancellation_failure_preserves_unresolved_cohort_without_secret_output(self):
        executor = Executor(self.path)
        cancelled = []
        def cancel(call_id):
            cancelled.append(call_id)
            raise RuntimeError('private credential exception text')
        executor.cancel = cancel
        clock = [1000.0]
        def tick(seconds):
            clock[0] += 2
        with patch('agent_harness.coordinator.time.time', side_effect=lambda: clock[0]):
            result = coordinate(manifest(coordinator_seconds=1), executor, self.path, sleep=tick)
        self.assertEqual(len(cancelled), 3)
        self.assertEqual(result['unresolved_episodes'], 6)
        self.assertEqual(result['episodes'][0]['cancellation_status'], 'failed')
        self.assertEqual(result['episodes'][0]['cancellation_error_type'], 'RuntimeError')
        self.assertNotIn('private credential exception text', self.path.read_text())
        # Completed coordinator runs do not repeatedly cancel already terminal calls.
        self.run_cohort(manifest(coordinator_seconds=1), executor)
        self.assertEqual(len(cancelled), 3)

    def test_manifest_mutation_rejected(self):
        self.run_cohort(manifest(1))
        changed = manifest(1, max_rollouts=2)
        with self.assertRaisesRegex(ValueError, 'manifest'):
            self.run_cohort(changed)

    def test_validation(self):
        cases = [manifest(max_rollouts=0), manifest(max_graders=True), manifest(max_graders=1.5),
                 manifest(coordinator_seconds=float('inf')), manifest(run_id='../escape'), manifest(0)]
        duplicate = manifest(2)
        duplicate['jobs'][1]['episode_id'] = 'e0'
        cases.append(duplicate)
        mismatch = manifest(1)
        mismatch['jobs'][0]['run_id'] = 'another'
        cases.append(mismatch)
        mixed = manifest(2)
        mixed['jobs'][1]['task_id'] = 'another'
        cases.append(mixed)
        for value in cases:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.run_cohort(value)
        self.assertFalse(self.path.exists())

    def test_persistence_failure_prevents_external_submit(self):
        executor = Executor(self.path)
        def persist():
            journal = json.loads(self.path.read_text())
            if any(x['state'].startswith('submitting_') for x in journal['episodes'].values()):
                raise OSError('volume unavailable')
        with self.assertRaises(OSError):
            self.run_cohort(manifest(1), executor, persist=persist)
        self.assertEqual(executor.submissions, [])
        result = self.run_cohort(manifest(1), executor)
        self.assertEqual(result['unresolved_episodes'], 1)
        self.assertEqual(executor.submissions, [])

    def test_invalid_grade_reward_excludes_group(self):
        executor = Executor(self.path)
        original = executor.poll
        def poll(call_id):
            result = original(call_id)
            if result['status'] == 'resolved':
                result['reward'] = True
            return result
        executor.poll = poll
        result = self.run_cohort(manifest(2), executor)
        self.assertEqual(result['groups']['g0']['status'], 'excluded')
        self.assertEqual(result['unresolved_episodes'], 2)

    def test_missing_result_identities_are_unresolved(self):
        executor = Executor(self.path)
        executor.poll = lambda _: {'episode_id': 'e0', 'status': 'completed'}
        result = self.run_cohort(manifest(1), executor)
        self.assertEqual(result['episodes'][0]['reason'], 'remote_result_identity_mismatch')

    def test_duplicate_call_id_is_not_polled_for_another_stage(self):
        executor = Executor(self.path)
        original = executor.submit
        def submit(kind, job):
            call_id = original(kind, job)
            return '1' if kind == 'grade' else call_id
        executor.submit = submit
        result = self.run_cohort(manifest(1), executor)
        self.assertEqual(result['episodes'][0]['reason'], 'duplicate_call_id_requires_manual_recovery')
        self.assertEqual(result['unresolved_episodes'], 1)

    def test_corrupt_completed_journal_rejected(self):
        self.run_cohort(manifest(1))
        journal = json.loads(self.path.read_text())
        journal['episodes']['e0']['reward'] = True
        self.path.write_text(json.dumps(journal))
        with self.assertRaisesRegex(ValueError, 'completed journal'):
            self.run_cohort(manifest(1))
