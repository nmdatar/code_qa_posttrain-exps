import json
import time
from dataclasses import replace
import unittest
import tempfile
from pathlib import Path
from agent_harness import (AgentRunner, ArtifactStore, FinalAnswer, ResearchRequest,
    RunLimits, ToolCall, ToolCallBatch, ToolObservation, ToolRegistry, ToolSpec)
from agent_harness.models import ScriptedModel, ChatCompletionsModel


class ReadTool:
    spec = ToolSpec('read', '1', 'read', ('read',),
        {'type':'object', 'properties': {'value': {'type':'integer'}, 'delay': {'type':'number'}},
         'required':['value'], 'additionalProperties':False}, parallel_safe=True)

    def execute(self, args, context):
        time.sleep(args.get('delay', 0))
        context.artifacts.put(context.episode_id, {'same': 'artifact'})
        return ToolObservation('ok', args['value'])



class TimedReadTool(ReadTool):
    def execute(self, args, context):
        started = time.monotonic()
        time.sleep(.1)
        return ToolObservation('ok', [started, time.monotonic()])

def run(tmp_path, calls, *, cap=2, parallel=2, tool=None, wall=5):
    model = ScriptedModel([ToolCallBatch(tuple(calls)), FinalAnswer('done')])
    return AgentRunner(model, ToolRegistry([tool or ReadTool()]), ArtifactStore(tmp_path)).run(
        ResearchRequest('task', 'Question?', limits=RunLimits(max_tool_calls=cap,
          max_parallel_tool_calls=parallel, wall_time_seconds=wall)), episode_id='test')


def test_results_ordered_and_artifacts_committed_by_parent(tmp_path):
    result = run(tmp_path, [ToolCall('read', {'value':1,'delay':.05},'a'),
                            ToolCall('read', {'value':2},'b')])
    assert result.termination_reason == 'completed'
    assert result.metrics['tool_calls'] == 2
    events = [json.loads(l) for l in Path(result.trajectory_path).read_text().splitlines()]
    observations = [e for e in events if e['kind']=='tool_observation']
    assert [e['observation']['content'] for e in observations] == [1,2]
    assert [e['call_id'] for e in observations] == ['a','b']
    assert len(list((tmp_path/'test'/'artifacts').glob('*.json'))) == 3
    assert [e['sequence'] for e in events] == list(range(len(events)))


def test_rejected_batches_execute_nothing(tmp_path, cap, parallel, reason):
    result = run(tmp_path,[ToolCall('read',{'value':1}),ToolCall('read',{'value':2})],cap=cap,parallel=parallel)
    assert result.termination_reason == reason
    assert result.metrics['tool_calls'] == 0
    assert not list((tmp_path/'test'/'artifacts').glob('*.json'))


def test_unsafe_and_duplicate_batches_execute_nothing(tmp_path):
    tool = ReadTool()
    tool.spec = replace(tool.spec, parallel_safe=False)
    result = run(tmp_path/'unsafe',[ToolCall('read',{'value':1}),ToolCall('read',{'value':2})],tool=tool)
    assert result.termination_reason == 'agent_error'
    result = run(tmp_path/'duplicate',[ToolCall('read',{'value':1},'same'),ToolCall('read',{'value':2},'same')])
    assert result.termination_reason == 'agent_error'
    assert result.metrics['tool_calls'] == 0


def test_worker_deadline_is_bounded_and_reaped(tmp_path):
    import multiprocessing
    before = {p.pid for p in multiprocessing.active_children()}
    started=time.monotonic()
    result=run(tmp_path,[ToolCall('read',{'value':1,'delay':2}),ToolCall('read',{'value':2,'delay':2})],wall=.1)
    assert result.termination_reason == 'budget_exhausted'
    assert time.monotonic()-started < 1.5
    assert {p.pid for p in multiprocessing.active_children()} == before


def test_native_parser_is_opt_in_and_checks_all_calls():
    message={'tool_calls':[{'id':str(i),'type':'function','function':{'name':'read','arguments':'{"value":1}'}} for i in range(2)]}
    model=ChatCompletionsModel(model='test',base_url='http://localhost',max_parallel_tool_calls=2)
    assert isinstance(model._action(message),ToolCallBatch)
    with unittest.TestCase().assertRaises(ValueError):
        ChatCompletionsModel(model='test',base_url='http://localhost')._action(message)
    message['tool_calls'][1]['function']['arguments']='[]'
    with unittest.TestCase().assertRaises(ValueError): model._action(message)


def test_compaction_preserves_complete_batch(tmp_path):
    store=ArtifactStore(tmp_path);store.create_episode('test')
    runner=AgentRunner(ScriptedModel([]),ToolRegistry(),store)
    messages=[{'role':'system','content':'s'},{'role':'user','content':'q'},
      {'role':'assistant','tool_calls':[{'id':'a'},{'id':'b'}]},
      {'role':'tool','tool_call_id':'a','content':'x'*1000},
      {'role':'tool','tool_call_id':'b','content':'x'*1000},
      {'role':'assistant','tool_calls':[{'id':'c'}]},
      {'role':'tool','tool_call_id':'c','content':'latest'}]
    assert runner._compact(messages,0,600,'test',lambda *a,**kw:None)
    assert [m.get('tool_call_id') for m in messages if m['role']=='tool']==['c']


class ParallelAgentToolsTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name)

    def test_order_and_artifacts(self):
        test_results_ordered_and_artifacts_committed_by_parent(self.path)

    def test_rejected_budget(self):
        test_rejected_batches_execute_nothing(self.path, 1, 2, 'budget_exhausted')

    def test_disabled(self):
        test_rejected_batches_execute_nothing(self.path, 2, 1, 'agent_error')

    def test_unsafe_duplicate(self):
        test_unsafe_and_duplicate_batches_execute_nothing(self.path)

    def test_deadline(self):
        test_worker_deadline_is_bounded_and_reaped(self.path)

    def test_parser(self):
        test_native_parser_is_opt_in_and_checks_all_calls()

    def test_compaction(self):
        test_compaction_preserves_complete_batch(self.path)


    def test_workers_actually_overlap(self):
        result = run(self.path, [ToolCall('read', {'value':1}), ToolCall('read', {'value':2})],
                     tool=TimedReadTool())
        self.assertEqual(result.termination_reason, 'completed')
        events = [json.loads(line) for line in Path(result.trajectory_path).read_text().splitlines()]
        spans = [e['observation']['content'] for e in events if e['kind']=='tool_observation']
        self.assertLess(max(span[0] for span in spans), min(span[1] for span in spans))

    def test_limit_validation(self):
        for limit in (0, 9, True, 1.5):
            with self.assertRaises(ValueError):
                AgentRunner._validate_request(ResearchRequest('t', 'q', limits=RunLimits(max_parallel_tool_calls=limit)))
        AgentRunner._validate_request(ResearchRequest('t', 'q', limits=RunLimits(max_parallel_tool_calls=8)))


    def test_invalid_member_starts_no_worker(self):
        from unittest.mock import patch
        with patch('agent_harness.runner.dispatch_parallel') as dispatch:
            result = run(self.path, [ToolCall('read', {'value':1}), ToolCall('read', {'value':'bad'})])
        dispatch.assert_not_called()
        self.assertEqual(result.termination_reason, 'agent_error')
        self.assertEqual(result.metrics['tool_calls'], 0)
        self.assertEqual(list((self.path/'test'/'artifacts').glob('*.json')), [])

    def test_dispatch_entrypoint_prevalidates_all_members(self):
        from unittest.mock import patch
        from agent_harness.parallel_tools import dispatch_parallel
        from agent_harness.contracts import ToolContext
        from agent_harness.registry import ToolInputError
        store = ArtifactStore(self.path); store.create_episode('test')
        with patch('agent_harness.parallel_tools.multiprocessing.get_context') as workers:
            with self.assertRaises(ToolInputError):
                dispatch_parallel(ToolRegistry([ReadTool()]),
                    [ToolCall('read', {'value':1}), ToolCall('read', {'value':'bad'})],
                    ToolContext({}, store, 'test'),
                    ResearchRequest('t', 'q', limits=RunLimits(max_parallel_tool_calls=2)), 5)
        workers.assert_not_called()

    def test_parallel_wall_metric(self):
        result = run(self.path, [ToolCall('read', {'value':1}), ToolCall('read', {'value':2})],
                     tool=TimedReadTool())
        self.assertGreater(result.metrics['tool_wall_seconds'], 0)
        self.assertLess(result.metrics['tool_wall_seconds'], result.metrics['tool_seconds'])

    def test_cli_flag_enables_model_and_request(self):
        import contextlib
        import io
        from unittest.mock import patch
        from agent_harness.cli import main
        from qa_eval.demo import fixture
        task, answer, *_ = fixture(self.path/'repo')
        task_path = self.path/'task.json'; task_path.write_text(json.dumps(task))
        output = io.StringIO()
        with patch('agent_harness.cli.ChatCompletionsModel',
                   return_value=ScriptedModel([FinalAnswer(answer)])) as model:
            with contextlib.redirect_stdout(output):
                code = main(['run', '--repo', str(self.path/'repo'), '--commit', task['repository']['commit'],
                    '--output', str(self.path/'runs'), '--task', str(task_path), '--model','offline',
                    '--base-url','https://example.invalid/v1','--max-parallel-tool-calls','4'])
        self.assertEqual(code, 0)
        self.assertEqual(model.call_args.kwargs['max_parallel_tool_calls'], 4)
        trajectory = Path(json.loads(output.getvalue())['trajectory'])
        first = json.loads(trajectory.read_text().splitlines()[0])
        self.assertEqual(first['limits']['max_parallel_tool_calls'], 4)

    def test_remote_model_config_accepts_opt_in(self):
        from agent_harness.remote_contracts import validate_model
        validate_model({'kind':'http','model':'offline','base_url':'https://example.invalid',
                        'max_parallel_tool_calls':4})
        validate_model({'kind':'tinker','base_model':'offline','renderer_name':'offline',
                        'max_parallel_tool_calls':4})
        with self.assertRaises(ValueError):
            validate_model({'kind':'http','model':'offline','base_url':'https://example.invalid',
                            'max_parallel_tool_calls':9})
