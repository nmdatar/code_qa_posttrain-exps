import json,subprocess,tempfile,unittest
from pathlib import Path
from agent_harness.repository_tools import command
class ReadPaginationTests(unittest.TestCase):
 def test_bounded_pagination_and_legacy_failure(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);(root/'a.py').write_text('\n'.join(str(i) for i in range(1,251)))
   def run(start,end,paginate):
    argv=command('read_file',{'path':'a.py','start_line':start,'end_line':end},['a.py'],paginate_reads=paginate)
    argv[2]=argv[2].replace("pathlib.Path('/repo')",'pathlib.Path('+repr(tmp)+')')
    return subprocess.run(argv,capture_output=True,text=True)
   self.assertNotEqual(run(1,250,False).returncode,0)
   first=json.loads(run(1,250,True).stdout);second=json.loads(run(first['next_start_line'],250,True).stdout);third=json.loads(run(second['next_start_line'],250,True).stdout)
   self.assertEqual([len(x['lines']) for x in [first,second,third]],[120,120,10])
   self.assertIsNone(third['next_start_line']);self.assertEqual(first['file_sha256'],third['file_sha256'])
   self.assertNotEqual(run(251,260,True).returncode,0)
   self.assertNotEqual(run(10,1,True).returncode,0)
 def test_unknown_paths_still_rejected(self):
  with self.assertRaises(ValueError):command('read_file',{'path':'../a.py'},['a.py'],paginate_reads=True)
