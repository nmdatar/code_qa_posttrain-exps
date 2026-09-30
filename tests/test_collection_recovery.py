import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from training_pipeline.benchmark import task_manifest,selected_tasks
from training_pipeline.collection_recovery import create_manifest,validate_manifest,verified_archive_inventory,disposition
from training_pipeline.contracts import ConfigurationError
from training_pipeline.storage import digest


class CollectionRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.data={'identity':'dataset','tasks':[{'id':'a','split':'train','family_id':'f'},{'id':'b','split':'train','family_id':'g'}]}
        self.original=task_manifest(self.data,2)
        self.config={'model':{'base_model':'teacher'},'environment':{'kind':'collection','release':'/local'},'limits':{'context_tokens':8192},'judge':{'version':'v7'},'run_id':'original','group_retries':0,'execution':{'operation':'benchmark'},'benchmark':{'attempts':2}}

    def trace(self,episode,task='a',termination='completed',status='resolved',reward=0,submission=None):
        return {'trace':{'episode_id':episode,'task_id':task,'run_id':'original','policy_id':'base:teacher','split':'train','termination':termination,'submission':submission,'verification':{'status':status,'reward':reward}},'sha256':episode,'path':episode+'.json'}

    def plan(self,traces):
        m=create_manifest(self.original,self.config,traces,{'archived_trajectory_count':len(traces)})
        c=copy.deepcopy(self.config);c['benchmark']={'attempts':1,'manifest_hash':m['manifest_hash']}
        return m,c

    def rehash(self,m,c):
        m['manifest_hash']=digest({k:v for k,v in m.items() if k!='manifest_hash'});c['benchmark']['manifest_hash']=m['manifest_hash']

    def test_preserves_semantic_failure_and_unresolved_completed_answer(self):
        m,c=self.plan([self.trace('one',reward=0),self.trace('two',status='unresolved',reward=None)])
        self.assertEqual(len(m['retained']),2)
        self.assertEqual([t['id'] for t in validate_manifest(m,c,self.data)],['b','b'])
        self.assertTrue(all(s['reason']=='missing' for s in m['slots']))

    def test_infrastructure_without_answer_replaced_but_answer_regraded(self):
        m,c=self.plan([self.trace('one',termination='infrastructure_error',status='unresolved',submission={'text':'final'}),self.trace('two',termination='infrastructure_error',status='unresolved')])
        self.assertEqual(len(m['retained']),1);self.assertEqual(len(validate_manifest(m,c,self.data)),3)
        self.assertEqual(m['slots'][0]['replaces_episode_id'],'two')
        t=self.trace('raw',termination='infrastructure_error')['trace'];t['generations']=[{'text':'{"answer":"existing"}'}]
        self.assertEqual(disposition(t),'preserve-answer')

    def test_cannot_resample_kept_outcome_even_with_recomputed_manifest_hash(self):
        m,c=self.plan([self.trace('one')]);kept=m['retained'].pop()
        m['slots'].append({'task_id':'a','attempt_slot':0,'reason':'infrastructure_error','replaces_episode_id':'one'});m['task_ids']=[s['task_id'] for s in m['slots']];self.rehash(m,c)
        with self.assertRaisesRegex(ConfigurationError,'preserved outcome'):validate_manifest(m,c,self.data)

    def test_wrong_hash_science_retries_or_training_operation_rejected(self):
        for mutation in ('hash','limits','retry','operation','attempts'):
            m,c=self.plan([])
            if mutation=='hash':m['source']={'tampered':True}
            elif mutation=='limits':c['limits']['context_tokens']=999
            elif mutation=='retry':c['group_retries']=1
            elif mutation=='operation':c['execution']['operation']='run'
            else:c['benchmark']['attempts']=2
            with self.assertRaises(ConfigurationError):validate_manifest(m,c,self.data)

    def test_partition_and_extra_source_attempts_rejected(self):
        with self.assertRaises(ConfigurationError):self.plan([self.trace('1'),self.trace('2'),self.trace('3')])
        m,c=self.plan([]);m['slots'].append(copy.deepcopy(m['slots'][0]));m['task_ids']=[s['task_id'] for s in m['slots']];self.rehash(m,c)
        with self.assertRaises(ConfigurationError):validate_manifest(m,c,self.data)

    def test_archive_requires_all_trajectory_events_and_untampered_files(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);(r/'trajectories').mkdir();cp=r/'config.json';ep=r/'events.jsonl';tp=r/'trajectories'/'one.json'
            cp.write_text(json.dumps(self.config));ep.write_text(json.dumps({'event':'trajectory','episode_id':'one'})+'\n');tp.write_text(json.dumps(self.trace('one')['trace']))
            checks=r/'checksums.json';checks.write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (cp,ep,tp)}))
            self.assertEqual(len(verified_archive_inventory(r,checks)[1]),1)
            tp.unlink()
            with self.assertRaisesRegex(ConfigurationError,'Incomplete archived'):verified_archive_inventory(r,checks)
            tp.write_text('{}')
            with self.assertRaisesRegex(ConfigurationError,'changed'):verified_archive_inventory(r,checks)

    def test_union_replaces_only_failed_original_and_has_exact_slot_coverage(self):
        from training_pipeline.collection_recovery import merge_recovery_inventory
        old=[self.trace('keep',reward=0),self.trace('failed',termination='infrastructure_error',status='unresolved')]
        m,c=self.plan(old);c['run_id']='recovery'
        fresh=[]
        for i,slot in enumerate(m['slots']):
            row=self.trace('new'+str(i),task=slot['task_id']);row['trace']['run_id']='recovery'
            row['trace']['group_id']='recovery-'+m['manifest_hash'][:16]+'-'+slot['task_id']+'-'+str(slot['attempt_slot'])
            fresh.append(row)
        merged=merge_recovery_inventory(m,old,fresh,c)
        self.assertEqual(len(merged),4)
        self.assertIn('keep',[r['trace']['episode_id'] for r in merged])
        self.assertNotIn('failed',[r['trace']['episode_id'] for r in merged])
        with self.assertRaisesRegex(ConfigurationError,'missing intended slots'):merge_recovery_inventory(m,old,fresh[:-1],c)
        duplicate=copy.deepcopy(fresh);duplicate[1]['trace']['group_id']=duplicate[0]['trace']['group_id']
        with self.assertRaisesRegex(ConfigurationError,'uniquely bind'):merge_recovery_inventory(m,old,duplicate,c)
        changed=copy.deepcopy(old);changed[0]['sha256']='different'
        with self.assertRaisesRegex(ConfigurationError,'hash changed'):merge_recovery_inventory(m,changed,fresh,c)

    def test_runtime_persists_distinct_stable_slot_ids_without_retrying(self):
        from types import SimpleNamespace
        from training_pipeline.collection_recovery import run_recovery_slots
        m,c=self.plan([self.trace('one')]);tasks=validate_manifest(m,c,self.data);calls=[]
        def rollout(task,group_id,temperature):
            calls.append((task['id'],group_id,temperature));return SimpleNamespace(policy_id='base:teacher')
        p=SimpleNamespace(tracker=SimpleNamespace(context={}),state={'optimizer_step':0},backend=SimpleNamespace(policy_id='base:teacher'),config={'concurrency':{'rollouts':1}},rollout=rollout)
        result=run_recovery_slots(p,tasks,m)
        self.assertEqual(len(result),3);self.assertEqual(len({g for _,g,_ in calls}),3)
        self.assertTrue(all(g.endswith('-'+str(slot['attempt_slot'])) for (_,g,_),slot in zip(calls,m['slots'])))
        self.assertTrue(all(temp==1 for _,_,temp in calls))

    def test_existing_benchmark_route_selects_recovery_slot_multiplicity(self):
        m,c=self.plan([self.trace('one')])
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'manifest.json';p.write_text(json.dumps(m));c['benchmark']['task_manifest']=str(p)
            self.assertEqual([t['id'] for t in selected_tasks(c,self.data)],['a','b','b'])


if __name__=='__main__':unittest.main()
