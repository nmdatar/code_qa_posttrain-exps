import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from training_pipeline.sft_collection import candidate, load_supervised, VERSION
from training_pipeline.collection import PROTOCOL
from training_pipeline.storage import digest
from training_pipeline.admission import sha
from training_pipeline.rendering import ChatRenderer, sft_batch
from training_pipeline.launch import estimate

class Tokenizer:
    def get_chat_template(self): return 'test'
    def apply_chat_template(self,messages,tokenize=True,add_generation_prompt=False,**kw):
        s=''.join('<'+m['role']+'>'+m['content']+'!' for m in messages)
        return [ord(c) for c in s+('<assistant>' if add_generation_prompt else '')]

class ToolSFTTests(unittest.TestCase):
    def fixture(self):
        raw=b'hello\nworld\n';public={'id':'a','split':'train','repository':{'commit':'abc'},'permitted_tools':['read_file']}
        row={'id':'a','split':'train','family_id':'repo','lineage_id':'a','public':public,'snapshot_root':'unused','image_result':{'snapshot_files':{'a.py':sha(raw)}}}
        text=json.dumps({'tool':'read_file','arguments':{'path':'a.py','start_line':1,'end_line':2}})
        obs={'exit_code':0,'stdout':json.dumps({'path':'a.py','file_sha256':sha(raw),'start_line':1,'end_line':2,'lines':['1: hello','2: world']}),'stdout_truncated':False,'stderr_truncated':False}
        messages=[{'role':'system','content':PROTOCOL},{'role':'user','content':json.dumps(public)},{'role':'assistant','content':text},{'role':'user','content':'Tool observation: '+json.dumps(obs)},{'role':'assistant','content':'{"answer":"UNVERIFIED"}'}]
        gen={'kind':'generation','text':text,'stop_reason':'stop','prompt':[],'tokens':[]}
        trace={'task_id':'a','split':'train','episode_id':'ep','policy_id':'base','events':[{'kind':'initial','messages':messages},gen,{'kind':'observation','value':obs}]}
        return raw,row,trace
    def test_only_successful_action_is_supervised(self):
        raw,row,t=self.fixture()
        with patch('training_pipeline.sft_collection.blob',return_value=raw): e=candidate(t,row)
        self.assertEqual(e['assistant_turns'],[2]);self.assertEqual(len(e['messages']),3)
        rows=sft_batch(ChatRenderer(Tokenizer(),10000),[e]);r=rows[0]
        self.assertAlmostEqual(sum(r.weights),1)
        prefix=len(ChatRenderer(Tokenizer(),10000).prompt(e['messages'][:2]))-1
        self.assertTrue(all(w==0 for w in r.weights[:prefix]));self.assertTrue(all(w>0 for w in r.weights[prefix:]))
        self.assertNotIn('UNVERIFIED',''.join(chr(t) for t in r.input_tokens))
    def test_rejects_eval_and_source_tampering(self):
        raw,row,t=self.fixture()
        with patch('training_pipeline.sft_collection.blob',return_value=b'wrong'):
            with self.assertRaises(ValueError):candidate(t,row)
        t['split']='development'
        with self.assertRaises(ValueError):candidate(t,row)
    def test_rejects_truncated_and_failed_observations(self):
        for field,value in [('exit_code',1),('stdout_truncated',True)]:
            raw,row,t=self.fixture();t['events'][2]['value'][field]=value
            with patch('training_pipeline.sft_collection.blob',return_value=raw):
                with self.assertRaises(ValueError):candidate(t,row)
    def test_manifest_cannot_relabel_dev_as_train_or_duplicate_lineage(self):
        raw,row,t=self.fixture()
        with tempfile.TemporaryDirectory() as d,patch('training_pipeline.sft_collection.blob',return_value=raw):
            root=Path(d);p=root/'source.json';p.write_text(json.dumps(t));e=candidate(t,row)
            record={'source':'source.json','sha256':sha(p.read_bytes()),'example_hash':digest(e)}
            m={'version':VERSION,'data_identity':'x','human_reviewed':False,'records':[record]}
            def load(tasks):
                mp=root/'manifest.json';mp.write_text(json.dumps(m))
                return load_supervised({'manifest':str(mp),'sha256':sha(mp.read_bytes())},{'identity':'x','tasks':tasks})
            self.assertEqual(len(load([row])),1)
            with self.assertRaises(ValueError):load([])
            m['records'].append(record)
            with self.assertRaises(ValueError):load([row])
    def test_sft_estimate_excludes_rl_rollouts_but_includes_training(self):
        c=json.loads(Path('configs/experiments/grpo-v6-parallel/baseline-v1.json').read_text())
        c['execution']['operation']='run';c['execution'].pop('cohort',None)
        c['stages']=[{'kind':'sft','batch_size':8,'max_batches':4,'max_updates':4,'learning_rate':1e-4}]
        c['evaluation']['every']=5
        r=estimate(c,c['spend']['prices'])
        self.assertEqual(r['episodes_including_retries'],64)
        self.assertGreater(r['components_usd']['training'],0)

class SampledProbeTests(unittest.TestCase):
    def test_alias_and_malformed_actions_remain_distinct(self):
        from training_pipeline.tool_sft_probe import check
        row={'id':'x','public':{'permitted_tools':['read_file','search_code']},'image_result':{'snapshot_files':{'a.py':'hash'}}}
        config={'environment':{'tool_action_policy':'action-alias-v1'}}
        valid='{"tool":"read_file","arguments":{"path":"a.py","start_line":1,"end_line":3}}'
        self.assertTrue(check(valid,row,{},config)['valid'])
        self.assertTrue(check(valid.replace('"tool"','"action"'),row,{},config)['valid'])
        self.assertFalse(check(valid.replace('"a.py"','"missing.py"'),row,{},config)['valid'])
        self.assertFalse(check(valid+' {}',row,{},config)['valid'])
        self.assertFalse(check('{"arguments":{}}',row,{},config)['valid'])
        self.assertFalse(check('{"answer":"made up","citations":[{"path":"a.py","start_line":1,"end_line":2}]}',row,{},config)['valid'])
