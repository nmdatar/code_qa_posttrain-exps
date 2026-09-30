import unittest
from training_pipeline.trajectory_viewer import render_report, summarize, episode_html, tool_call_rows

class ViewerTests(unittest.TestCase):
    def trace(self):
        return dict(episode_id='ep-1', task_id='task', termination='budget_exhausted',
                    usage={'tool_calls':1}, verification={'status':'resolved','reward':0,'reasons':['bad citation'], 'diagnostics':{'private':'SECRET'}},
                    events=[{'kind':'generation','text':'</pre><script>alert(1)</script>'},
                            {'kind':'parsed_action','value':{'tool':'read_file','arguments':{'path':'a.py'}}},
                            {'kind':'observation','value':{'error':'Invalid action'}}])
    def test_untrusted_text_is_escaped_and_private_diagnostics_omitted(self):
        value=render_report([(self.trace(),{})])
        self.assertNotIn('<script>alert(1)</script>',value)
        self.assertIn('&lt;script&gt;',value)
        self.assertNotIn('SECRET',value)
        self.assertIn('Tool call · read_file',value)
        self.assertIn('budget_exhausted errors failed',value)
    def test_malformed_historical_action_is_visible(self):
        trace=self.trace();trace['events']=[{'kind':'parsed_action','value':'bad action'}]
        self.assertIn('Invalid parsed action',episode_html(trace))

    def test_native_tool_table_separates_arguments_response_and_errors(self):
        columns, rows=tool_call_rows([(self.trace(),{})])
        row=dict(zip(columns,rows[0]))
        self.assertEqual(row['tool'],'read_file')
        self.assertIn('a.py',row['arguments'])
        self.assertIn('Invalid action',row['error'])
        self.assertEqual(row['step'],1)

    def test_nested_tool_name_remains_a_string_in_wandb(self):
        trace=self.trace();trace['events'][1]['value']={'tool':{'tool':'search_code'}}
        columns,rows=tool_call_rows([(trace,{})])
        self.assertIsInstance(dict(zip(columns,rows[0]))['tool'],str)

    def test_unresolved_is_not_zero_and_context_preserved(self):
        trace=self.trace();trace['verification']=None
        meta=summarize(trace,{'phase':'evaluation','optimizer_step':13})
        self.assertIsNone(meta['score'])
        self.assertEqual(meta['errors'],1)
        self.assertEqual(meta['phase'],'evaluation')
        self.assertEqual(meta['optimizer_step'],13)
    def test_initial_message_alias_does_not_duplicate_conversation(self):
        trace=self.trace();trace['events']=[{'kind':'initial','messages':[{'role':'user','content':'question'},{'role':'assistant','content':'later response'}]}]
        self.assertNotIn('later response',episode_html(trace))

if __name__=='__main__':unittest.main()
