import tempfile
import unittest
from pathlib import Path
from eval_pipeline.baseline import grade_result, solve, summary, parse_action

class BaselineTests(unittest.TestCase):
    def test_terminal_brace_repair_never_repairs_truncation(self):
        action, repaired = parse_action('{"tool":"python_probe","code":"print(1)"', 'stop')
        self.assertTrue(repaired)
        self.assertEqual(action['args']['code'], 'print(1)')
        with self.assertRaises(ValueError):
            parse_action('{"tool":"python_probe","code":"print(1)"', 'length')
        with self.assertRaises(ValueError):
            parse_action('{"tool":"python_probe","code":"print(1)', 'stop')

    def test_partial_and_duplicate_judgments(self):
        ref={'claims':[{'id':'a','weight':3},{'id':'b','weight':1}]}
        result={'claims':[{'id':'a','verdict':'supported','reason':'ok'},{'id':'b','verdict':'partial','reason':'part'}], 'execution_supported':False,'reason':'review'}
        self.assertEqual(grade_result(result,ref)['weighted_claim_coverage'],.875)
        result['claims'][1]['id']='a'
        with self.assertRaises(ValueError): grade_result(result,ref)

    def test_failure_denominators(self):
        result=summary([{'status':'agent_error','answer':''}],12)
        self.assertEqual(result['grading_coverage'],0)
        self.assertEqual(result['not_attempted'],11)
        self.assertFalse(result['pipeline_complete'])

    def test_private_data_not_in_solver_and_budget_enforced(self):
        class Model:
            spec={'max_tokens_per_call':2}
            def sample(self,messages,max_tokens,timeout):
                assert 'PRIVATE_REFERENCE' not in str(messages)
                return {'text':'invalid','input_tokens':10,'output_tokens':max_tokens}
        task={'id':'test','system_prompt':'public','user_prompt':'question','budgets':{'max_output_tokens':3,'max_tool_calls':1,'latency_seconds':10,'max_submission_bytes':100}}
        config={'rollout':{'max_output_tokens_total':3,'max_tool_calls':1,'latency_seconds':10,'max_submission_bytes':100}}
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'Output token budget'):
                solve(task,{'private':'PRIVATE_REFERENCE'},Model(),config,Path(d)/'attempt',lambda *x: None)
            self.assertTrue((Path(d)/'attempt/usage.json').exists())

    def test_tool_roundtrip(self):
        class Model:
            spec={'max_tokens_per_call':20}
            calls=0
            def sample(self,messages,max_tokens,timeout):
                self.calls+=1
                text='{"tool":"list_files","args":{}}' if self.calls==1 else '{"answer":"Found file.py"}'
                return {'text':text,'input_tokens':10,'output_tokens':10}
        task={'id':'test','system_prompt':'public','user_prompt':'question','budgets':{'max_output_tokens':40,'max_tool_calls':1,'latency_seconds':10,'max_submission_bytes':100}}
        config={'rollout':{'max_output_tokens_total':40,'max_tool_calls':1,'latency_seconds':10,'max_submission_bytes':100}}
        with tempfile.TemporaryDirectory() as d:
            answer,events,stats=solve(task,{},Model(),config,Path(d)/'attempt',lambda *x: {'exit_code':0,'stdout':'file.py','timed_out':False,'truncated':False})
            self.assertEqual(stats['successful_tool_calls'],1)
            self.assertEqual(answer,'Found file.py')

    def test_invalid_tool_is_observation_and_final_answer_reserved(self):
        class Model:
            spec={'max_tokens_per_call':20}
            calls=0
            def sample(self,messages,max_tokens,timeout):
                self.calls+=1
                if self.calls==1:
                    return {'text':'{"tool":"read_file","args":{"path":"missing"}}','input_tokens':10,'output_tokens':10}
                self.seen=messages
                return {'text':'Unable to verify the behavior: the requested file was not found.','input_tokens':20,'output_tokens':10}
        task={'id':'test','system_prompt':'public','user_prompt':'question','budgets':{'max_output_tokens':40,'max_tool_calls':1,'latency_seconds':10,'max_submission_bytes':100}}
        config={'rollout':{'max_output_tokens_total':40,'max_tool_calls':1,'latency_seconds':10,'max_submission_bytes':100,'final_answer_reserved_tokens':20,'recover_invalid_tool_arguments':True}}
        def execute(*args): raise ValueError('Path not a tracked snapshot file')
        model=Model()
        with tempfile.TemporaryDirectory() as d:
            answer,events,stats=solve(task,{},model,config,Path(d)/'attempt',execute)
            self.assertTrue(stats['forced_final_answer'])
            self.assertEqual(stats['successful_tool_calls'],0)
            self.assertIn('Path not a tracked',str(model.seen))
            self.assertIn('Unable to verify',answer)

if __name__=='__main__': unittest.main()
