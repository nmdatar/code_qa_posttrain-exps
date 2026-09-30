import copy
import json
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from dataclasses import dataclass

from agent_harness.bash_tool import command
from training_pipeline.collection import CollectionEpisode
from training_pipeline.config import validate_config


class BashOnlyTests(unittest.TestCase):
    def test_arbitrary_shell_and_pipelines(self):
        argv = command({'command': "for x in alpha beta; do printf '%s\\n' \"$x\"; done | sed -n '2p'"})
        self.assertEqual(subprocess.check_output(argv, text=True).strip(), 'beta')
        self.assertEqual(command({'command':'python3 -c "print(1)"; cat /etc/os-release'})[2], 'python3 -c "print(1)"; cat /etc/os-release')
        for args in ({}, {'command': 1}, {'command': '\0'}, {'command': 'ls', 'extra': 1}):
            with self.assertRaises(ValueError): command(args)

    def test_bash_episode_and_pinned_citations(self):
        @dataclass
        class Result:
            stdout: str = 'a.py:1:hello'
            exit_code: int = 0
        with TemporaryDirectory() as root:
            config=json.loads(Path('configs/experiments/strong-model-validation-v1/student.json').read_text())
            config['environment']['solver_tools']='bash-only-v1'
            validate_config(config)
            public={'id':'t','permitted_tools':['list_files','read_file','search_code'], 'budgets':{}}
            row={'id':'t','public':public,'image_result':{'snapshot_files':{'a.py':'a'*64}}}
            calls=[]
            sb=SimpleNamespace(execute=lambda argv:(calls.append(argv) or Result()))
            factory=SimpleNamespace(config=config, root=Path(root))
            with unittest.mock.patch('training_pipeline.collection.EpisodeRecorder') as recorder:
                recorder.return_value.tool.side_effect=lambda name, fn:fn()
                e=CollectionEpisode(row,sb,factory,'e',Path(root)/'e.json')
                self.assertEqual(e.row['public']['permitted_tools'], ['bash'])
                self.assertEqual(public['permitted_tools'], ['list_files','read_file','search_code'])
                self.assertNotIn('"tool":"read_file"',e.messages[0]['content'])
                e.step({'tool':'bash','arguments':{'command':'cat a.py | head'}})
                self.assertEqual(calls, [['bash','-lc','cat a.py | head']])
                with self.assertRaises(ValueError):e.step({'tool':'read_file','arguments':{'path':'a.py'}})
                answer=e.parse_action(json.dumps({'answer':'hello','citations':[{'path':'a.py','start_line':1,'end_line':1}]}))
                self.assertEqual(answer['answer']['citations'][0]['file_sha256'],'a'*64)
                self.assertEqual(e.observed_files,{})
                with self.assertRaises(ValueError):e.parse_action(json.dumps({'answer':'hello','citations':[{'path':'outside','start_line':1,'end_line':1}]}))

if __name__=='__main__':unittest.main()
