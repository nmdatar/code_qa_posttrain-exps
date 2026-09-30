import copy
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from agent_harness.shell_tools import command, parse
from agent_harness.process import CommandResult
from training_pipeline.collection import CollectionEpisode
from training_pipeline.harness_variants import validate


class ShellTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.raw = "".join(f"line {i} symbol\n" for i in range(1, 251))
        (self.root / "a file.py").write_text(self.raw)
        self.files = {"a file.py": hashlib.sha256(self.raw.encode()).hexdigest()}
        self.permissions = ["list_files", "search_code", "read_file"]

    def run_command(self, text):
        argv, _, _ = command({"command": text}, self.files, permitted_tools=self.permissions)
        argv[2] = argv[2].replace("pathlib.Path('/workspace')", "pathlib.Path(" + repr(str(self.root)) + ")")
        return subprocess.run(argv, capture_output=True, text=True)

    def test_allowlist_real_sandbox_reader_and_pagination(self):
        for text in ["ls", "rg --files -g '*.py'", "ls '*.py' --offset 0"]:
            self.assertEqual(json.loads(self.run_command(text).stdout)['files'], ['a file.py'])
        found = json.loads(self.run_command("rg -n -F -g '*.py' symbol").stdout)
        self.assertEqual(len(found['matches']), 100)
        for text in ["cat 'a file.py'", "head -n 120 'a file.py'", "sed -n '1,250p' 'a file.py'"]:
            data = json.loads(self.run_command(text).stdout)
            self.assertEqual(len(data['lines']), 120)
            self.assertEqual(data['file_sha256'], self.files['a file.py'])
        data = json.loads(self.run_command("sed -n '121,250p' 'a file.py'").stdout)
        self.assertEqual(data['next_start_line'], 241)
        self.assertEqual(json.loads(self.run_command("sed -n '250p' 'a file.py'").stdout)['lines'], ['250: line 250 symbol'])
        self.assertEqual(parse({'command': "rg -n -F -- '-symbol'"})[1]['query'], '-symbol')

    def test_injection_and_unknown_flags_rejected_before_execution(self):
        for text in ['bash -c ls', '/bin/cat a', 'python3 -c pass', 'cat a; ls', 'cat a | head',
                     'ls && cat a', 'ls > a', 'ls < a', 'cat $(id)', 'cat `id`', 'ls\nls',
                     'A=1 ls', 'env ls', 'sed -i p a', "sed -n '1e id' a", 'rg --pre python x',
                     'rg -n -F --files-with-matches x', 'head -n 0 a', 'sed -n 0,2p a',
                     'ls --offset -1', 'ls --offset 0 --offset 1', 'rg -n -F x y', 'cat a b']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                command({'command': text}, self.files, permitted_tools=self.permissions)
        for args in [{}, {'command': []}, {'command': 'ls', 'cwd': '/'}, {'command': 'x'*8001}]:
            with self.assertRaises(ValueError): parse(args)

    def test_catalog_permissions_and_symlink_confinement(self):
        for path in ['/etc/passwd', '../secret', 'missing.py']:
            with self.assertRaises(ValueError):
                command({'command': 'cat '+path}, self.files, permitted_tools=self.permissions)
        with self.assertRaises(ValueError):
            command({'command': 'ls'}, self.files, permitted_tools=['read_file'])
        (self.root/'a file.py').unlink()
        (self.root/'a file.py').symlink_to('/etc/passwd')
        self.assertNotEqual(self.run_command("cat 'a file.py'").returncode, 0)

    def test_collection_exposes_only_shell_and_binds_read_evidence(self):
        row = {'id':'task', 'public': {'id':'task', 'permitted_tools':self.permissions,
               'budgets':{'max_tool_calls':5,'max_submission_bytes':64000}}, 'image_result':{'snapshot_files':self.files}}
        original = copy.deepcopy(row)
        config = json.loads(Path('configs/experiments/shell-only-v1/shell-r1.json').read_text())
        factory = SimpleNamespace(root=self.root, config=config)
        def execute(argv):
            argv[2] = argv[2].replace("pathlib.Path('/workspace')", "pathlib.Path("+repr(str(self.root))+")")
            out = subprocess.run(argv, capture_output=True, text=True)
            return CommandResult(out.stdout, out.stderr, out.returncode, .01, len(out.stdout), len(out.stderr), False, False)
        sandbox = SimpleNamespace(execute=execute)
        episode = CollectionEpisode(row, sandbox, factory, 'ep', self.root/'ep.json')
        self.assertEqual(row, original)
        self.assertEqual(json.loads(episode.messages[1]['content'])['permitted_tools'], ['shell'])
        self.assertNotIn('"tool":"read_file"', episode.messages[0]['content'])
        with self.assertRaises(ValueError): episode.step({'tool':'read_file','arguments':{'path':'a file.py'}})
        episode.step({'tool':'shell','arguments':{'command':"rg -n -F symbol"}})
        self.assertEqual(episode.observed_files, {})
        episode.step({'tool':'shell','arguments':{'command':"sed -n '1,2p' 'a file.py'"}})
        action = episode.parse_action(json.dumps({'answer':'Has symbol.', 'citations':[{'path':'a file.py','start_line':1,'end_line':2}]}))
        self.assertEqual(action['answer']['citations'][0]['file_sha256'], self.files['a file.py'])
        self.assertEqual(episode.recorder.record['tool_calls'], ['shell','shell'])
        self.assertTrue(episode.step(action)[0])
        for truncated, changed in [(True,False),(False,True)]:
            episode.observed_files.clear()
            stdout = json.dumps({'path':'a file.py','file_sha256':'bad' if changed else self.files['a file.py']})
            sandbox.execute = lambda argv:CommandResult(stdout,'',0,.01,len(stdout),0,truncated,False)
            if changed:
                with self.assertRaises(ValueError): episode.step({'tool':'shell','arguments':{'command':"cat 'a file.py'"}})
            else: episode.step({'tool':'shell','arguments':{'command':"cat 'a file.py'"}})
            self.assertEqual(episode.observed_files,{})

    def test_config_rejects_mixed_or_unknown_interfaces(self):
        config = json.loads(Path('configs/experiments/shell-only-v1/shell-r1.json').read_text())
        spec = config['harness']; validate(spec)
        for key,value in [('interface','bash-unrestricted'),('symbols',True),('retrieval','lexical-v1'),('history','evidence-ledger-v1')]:
            bad = {**spec,key:value}
            with self.assertRaises(ValueError): validate(bad)
        control = json.loads(Path('configs/experiments/shell-only-v1/control-r1.json').read_text())
        for key in ['model','environment','limits','evaluation','judge','concurrency','seed']:
            self.assertEqual(control[key],config[key],key)


    def test_matched_summary_and_mismatched_condition_rejected(self):
        from scripts.summarize_shell_eval import summarize
        configs = self.root/'configs'; configs.mkdir()
        for arm in ['control','shell']:
            for repeat in [1,2]:
                name = f'{arm}-r{repeat}'
                spec = json.loads(Path('configs/experiments/shell-only-v1/'+name+'.json').read_text())
                (configs/(name+'.json')).write_text(json.dumps(spec))
                folder = self.root/spec['output']; (folder/'evaluations').mkdir(parents=True); (folder/'trajectories').mkdir()
                rows=[]
                for task in ['one','two']:
                    usage={'latency_seconds':1.,'input_tokens':10,'output_tokens':5,'tool_calls':1}
                    rows.append({'task_id':task,'episode_id':task,'family_id':task,'status':'resolved','reward':.75 if arm=='shell' else .25,'termination':'completed'})
                    (folder/'trajectories'/(task+'.json')).write_text(json.dumps({'usage':usage,'events':[]}))
                report={'data_identity':'same','reward_version':'same','cohort':'same','policy_id':'base:qwen','optimizer_step':0,'environment':arm,'expected':2,'resolved':2,'demonstrated_quality':rows[0]['reward'],'mean_reward':rows[0]['reward'],'scoring_coverage':1.,'completion_rate':1.,'reserved_cost_usd':1.,'results':rows}
                (folder/'evaluations/report.json').write_text(json.dumps(report))
        result=summarize(self.root,configs)
        self.assertEqual(result['mean_quality_difference'],.5)
        self.assertEqual(result['environment_identities']['shell-r1'],'shell')
        path=configs/'shell-r2.json';spec=json.loads(path.read_text());spec['limits']['max_tool_calls']=99;path.write_text(json.dumps(spec))
        with self.assertRaisesRegex(ValueError,'Unmatched condition'): summarize(self.root,configs)


if __name__ == '__main__': unittest.main()
