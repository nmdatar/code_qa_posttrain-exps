import json
from pathlib import Path
import unittest
from unittest.mock import patch
from dataset_builder.collection import prepare_collection, normalized_question
from dataset_builder.build import write_json, read_jsonl
from tests import test_dataset_build as fixtures


class CollectionImportTests(unittest.TestCase):
    def setUp(self):
        fixture=fixtures.DatasetBuildTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        self.root=fixture.base;self.repo=fixture.repo;self.commit=fixture.commit
        row={'repo':'example/repository','commit':self.commit,'family_id':'example/repository','split':'development'}
        write_json(self.root/'reports/task-generation-1000/collection-plan.json',{'repositories':[row]})
        write_json(self.root/'reports/task-generation-1000/acquisition/repo.json',{**row,'status':'ready','snapshot_path':str(self.repo)})
        self.row={'repo':'example/repository','commit':self.commit,'question':'Where is the timeout defined?',
                  'answer':'PRIVATE_REFERENCE: client.py lines 1-2 returns 30.','category':'localization',
                  'provenance':{'source':'fixture','dataset_revision':'abc','source_row_index':0}}

    def test_source_reading_keeps_private_material_out_and_no_fake_probes(self):
        with patch('dataset_builder.collection.source_rows',return_value=[self.row]):
            result=prepare_collection(self.root,self.root/'out')
        self.assertEqual(result['prepared_tasks'],1)
        bundle=self.root/result['environments'][0]['bundle'];task=read_jsonl(bundle/'public/tasks.jsonl')[0]
        self.assertNotIn('python_probe',task['permitted_tools'])
        self.assertNotIn('PRIVATE_REFERENCE',(bundle/'public/tasks.jsonl').read_text())
        reference=read_jsonl(bundle/'private/references.jsonl')[0]
        self.assertEqual(reference['reference_answer'],self.row['answer'])
        self.assertFalse(reference['human_reviewed'])
        checks=read_jsonl(bundle/'private/assertions.jsonl')[0]
        self.assertEqual(checks['executable_behavior_probes'],[])
        self.assertEqual(len(checks['evidence']),1)

    def test_dedup_and_execution_requirements_not_silently_downgraded(self):
        duplicate={**self.row,'question':'  WHERE is the timeout defined?  '}
        execute={**self.row,'question':'Please run the tests and report failures.'}
        with patch('dataset_builder.collection.source_rows',return_value=[self.row,duplicate,execute]):
            result=prepare_collection(self.root,self.root/'out')
        self.assertEqual(result['prepared_tasks'],1)
        self.assertEqual(len(result['rejections']),2)
        self.assertTrue(any('execution_requirement' in r['reason'] for r in result['rejections']))

    def test_unknown_pin_quarantined_and_nonempty_output_rejected(self):
        with patch('dataset_builder.collection.source_rows',return_value=[{**self.row,'commit':'f'*40}]):
            result=prepare_collection(self.root,self.root/'out')
        self.assertEqual(result['prepared_tasks'],0)
        self.assertEqual(result['rejections'][0]['reason'],'repository_not_preassigned')
        (self.root/'out/saved').write_text('preserve')
        with self.assertRaises(ValueError):prepare_collection(self.root,self.root/'out')

    def test_unicode_normalization(self):
        self.assertEqual(normalized_question(' Ａ  Test\n'),normalized_question('a test'))


if __name__=='__main__':unittest.main()
