import unittest
from types import SimpleNamespace
from unittest.mock import patch
from posttrain import tinker_judge

class JudgeAdapterTests(unittest.TestCase):
    def invoke(self,content,stop='stop',role='training',prompt_length=3):
        command=['tinker','train-model','train-renderer','eval-model','eval-renderer']
        request={'routing_role':role,'policy':'P','instructions':'I','output_schema':{'type':'object','properties':{'ok':{'type':'boolean'}},'required':['ok'],'additionalProperties':False},'stage':'assess','untrusted':{'answer':'DATA'}}
        self.sample_calls=[]
        def sample(**kwargs):
            self.sample_calls.append(kwargs)
            return SimpleNamespace(result=lambda timeout:SimpleNamespace(sequences=[SimpleNamespace(tokens=[1],stop_reason=stop)]))
        renderer=SimpleNamespace(build_generation_prompt=lambda messages:SimpleNamespace(to_ints=lambda:[1]*prompt_length),get_stop_sequences=lambda:[2],parse_response=lambda tokens:({'content':[{'type':'text','text':content}]},True))
        clients={(command[1],command[2]):(None,SimpleNamespace(sample=sample),renderer),(command[3],command[4]):(None,SimpleNamespace(sample=sample),renderer)}
        fake_tinker=SimpleNamespace(types=SimpleNamespace(SamplingParams=lambda **kwargs:kwargs))
        with patch.dict(tinker_judge._clients,clients,clear=True),patch.dict('sys.modules',{'tinker':fake_tinker}):
            return tinker_judge.judge_call(command,request,12)

    def test_structured_content_and_schema(self):
        self.assertEqual(self.invoke('```json\n{"ok":true}\n```'),{'ok':True})
        self.assertEqual(self.sample_calls[0]['sampling_params']['max_tokens'],4096)
    def test_invalid_json_rejected(self):
        for content in ['{"ok":true,"ok":false}','{"ok":NaN}','{"ok":"yes"}','{"ok":true,"reward":1}']:
            with self.subTest(content=content),self.assertRaises(ValueError):self.invoke(content)
    def test_truncated_output_rejected(self):
        with self.assertRaises(ValueError):self.invoke('{"ok":true}',stop='length')
    def test_unknown_role_rejected(self):
        with self.assertRaises(ValueError):self.invoke('{"ok":true}',role='heldout-typo')
    def test_context_bound_prevents_dispatch(self):
        with self.assertRaises(ValueError):self.invoke('{"ok":true}',prompt_length=24001)
        self.assertEqual(self.sample_calls,[])
