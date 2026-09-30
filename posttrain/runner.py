"""Synchronous experiment coordinator with committed-step recovery."""
import concurrent.futures
import copy
import importlib.util
import json
import os
from pathlib import Path
import random
import threading
import time
import uuid
from .storage import atomic, read, digest, lock, BudgetLedger
from .config import load_config, fingerprint
from .checkpoints import save_checkpoint, load_checkpoint
from .tracking import EventLog, sync_tracking
from .backends import FakeBackend, TinkerBackend
from .environments import FakeEnvironment, ModalEnvironment
from .rollout import run_episode
from .strategies import prepare_group, validate_actions, validate_sft
from .verifier import simulated_verify, verify


def make_backend(c, ledger):
    if c['backend']=='fake': return FakeBackend(model=c['model'])
    return TinkerBackend(model=c['model'],budget=ledger,renderer_name=c['renderer_name'],
                         rank=c['lora_rank'],bounds=c['paid_call_bounds'].get('tinker',{}))


def freeze_data(c):
    """Resolve releases into exact public rows and private rubrics before model calls."""
    from .data import validate_artifacts, read_jsonl, adapt_task
    c=copy.deepcopy(c)
    c['rubric_reviews'] = {}  # Rebuilt from hashed release artifacts, never trusted from config.
    if c['backend']=='tinker' and (not c['training_release'] or not c['evaluation_release']):
        raise ValueError('Live training requires immutable training and evaluation releases')
    for field,split,out in [('training_release','train','train_tasks'),('evaluation_release','development','eval_tasks')]:
        if not c[field]: continue
        root=Path(c[field]);validate_artifacts(root)
        reviews={}
        if c['backend']=='tinker':
            artifacts=read(root/'manifest.json').get('artifacts',{})
            names=set(artifacts) if isinstance(artifacts,dict) else {a['path'] for a in artifacts}
            required={'public/tasks.jsonl','private/grading.jsonl','private/rubric_reviews.jsonl'}
            if not required.issubset(names):
                raise ValueError('Live release lacks manifest-bound tasks, grading, or independent rubric reviews')
            for review in read_jsonl(root/'private/rubric_reviews.jsonl'):
                if review['task_id'] in reviews: raise ValueError('Duplicate rubric review')
                reviews[review['task_id']]=review
        public=read_jsonl(root/'public/tasks.jsonl')
        from .data import _index
        private=_index(read_jsonl(root/'private/grading.jsonl'),'task_id')
        selected={x['id'] for x in c[out]} if c[out] else None
        tasks=[]
        for t in public:
            if t['split'] not in (split,'dev' if split=='development' else split):continue
            if selected is not None and t['id'] not in selected:continue
            strict=adapt_task(t,private.get(t['id']),c['strict_tasks'].get(t['id']))
            if c['backend']=='tinker':
                review=reviews.get(t['id'],{})
                from qa_eval.review import automated_rubric_supported
                if not automated_rubric_supported(strict, review):
                    raise ValueError('Missing independent review bound to strict rubric: '+t['id'])
                c['rubric_reviews'][t['id']] = review
            tasks.append(t);c['strict_tasks'][t['id']]=strict
        if selected and selected != {t['id'] for t in tasks}: raise ValueError('Selected task IDs not present in pinned split')
        if selected:
            order={t['id']:i for i,t in enumerate(c[out])}
            tasks.sort(key=lambda t:order[t['id']])
        c[out]=tasks
    if c['backend']=='tinker' and any(stage['algorithm']=='sft' for stage in c['stages']):
        root=Path(c['training_release'])
        artifacts=read(root/'manifest.json').get('artifacts',{})
        names=set(artifacts) if isinstance(artifacts,dict) else {a['path'] for a in artifacts}
        if not {'training/sft.jsonl','private/sft_reviews.jsonl'}.issubset(names):
            raise ValueError('Live SFT requires manifest-bound examples and independent reviews')
        examples=_index(read_jsonl(root/'training/sft.jsonl'),'id')
        reviews=_index(read_jsonl(root/'private/sft_reviews.jsonl'),'task_id')
        for example in c['sft_examples']:
            validate_sft(example)
            if example['id'] not in examples or digest(example)!=digest(examples[example['id']]):
                raise ValueError('SFT example differs from immutable training release')
            review=reviews.get(example['id'],{})
            if (example['provenance'].get('collection_mode')!='blind_solver'
                    or review.get('status')!='supported' or not review.get('author') or not review.get('reviewer')
                    or review['author']==review['reviewer'] or review.get('example_sha256')!=digest(example)):
                raise ValueError('SFT requires independently reviewed blind solver investigations')
    return c


def readiness(c):
    blockers=[]
    if any(s['algorithm']=='grpo' for s in c['stages']) and not c['train_tasks']: blockers.append('No training tasks')
    if not c['eval_tasks']: blockers.append('No development evaluation tasks')
    if any(s['algorithm']=='sft' for s in c['stages']) and not c['sft_examples']:blockers.append('No admitted SFT examples')
    families={}
    for key,expected in [('train_tasks','train'),('eval_tasks','development')]:
        ids=set()
        for t in c[key]:
            if t['id'] in ids:blockers.append('Duplicate task ID');continue
            ids.add(t['id'])
            split={'dev':'development'}.get(t.get('split'),t.get('split'))
            if split!=expected:blockers.append('Task split mismatch: '+t['id'])
            family=t['repository']['family_id'].casefold()
            if family in families and families[family]!=expected:blockers.append('Family split leakage: '+family)
            families[family]=expected
    evaluation_ids={t['id'] for t in c['eval_tasks']}
    training_ids={t['id'] for t in c['train_tasks']}
    for example in c['sft_examples']:
        try: validate_sft(example)
        except ValueError as exc: blockers.append(str(exc)); continue
        if example.get('id') in evaluation_ids:
            blockers.append('SFT task overlaps evaluation: '+example['id'])
        family=example.get('repository',{}).get('family_id','').casefold()
        if family and families.get(family)=='development':
            blockers.append('SFT family split leakage: '+family)
        if c['backend']=='tinker' and example.get('id') not in training_ids:
            blockers.append('SFT example lacks admitted training task: '+str(example.get('id')))
    if c['backend']=='tinker':
        if not c['renderer_name']:blockers.append('Explicit compatible renderer required')
        if not c['pricing_snapshot']:blockers.append('Frozen model/pricing/capability evidence required')
        for mod in ('tinker','tinker_cookbook','modal'):
            if importlib.util.find_spec(mod) is None:blockers.append('Dependency missing: '+mod)
        from .services import tinker_auth_available
        if not tinker_auth_available():blockers.append('Tinker authentication unavailable in environment or official SDK store')
        judge_command=(c['verifier_config'] or {}).get('judge_command',[])
        if judge_command and judge_command[0]=='tinker':
            if len(judge_command)!=5 or judge_command[1]==judge_command[3]:blockers.append('Distinct Tinker judge models/renderers required')
        else:
            for prefix in ('QA_TRAIN_JUDGE_','QA_EVAL_JUDGE_'):
                for suffix in ('MODEL','BASE_URL'):
                    if not os.environ.get(prefix+suffix):blockers.append(prefix+suffix+' missing')
        if not c['verifier_config']:blockers.append('Strict verifier configuration missing')
        for t in c['train_tasks']+c['eval_tasks']:
            if t['id'] not in c['strict_tasks']:blockers.append('Strict rubric missing: '+t['id'])
            if t['id'] not in c['environment_bundles']:blockers.append('Environment bundle missing: '+t['id'])
            if t['id'] not in c['repository_roots']:blockers.append('Pinned source root missing: '+t['id'])
            strict=c['strict_tasks'].get(t['id'],{})
            if strict and strict.get('probes'):blockers.append('Live private probe attestation adapter required: '+t['id'])
            if c.get('review_policy', 'automated') == 'automated':
                from qa_eval.review import automated_rubric_supported
                if not automated_rubric_supported(strict, c.get('rubric_reviews', {}).get(t['id'])):
                    blockers.append('Independent automated rubric review missing or stale: '+t['id'])
            elif not c['diagnostic'] and (not strict.get('human_reviewed') or strict.get('gold_status')!='accepted'):
                blockers.append('Human gold admission pending: '+t['id'])
        bounds=c['paid_call_bounds']
        if any(k not in bounds for k in ('judge','modal','tinker')):blockers.append('All paid operations need conservative cost bounds')
    if not c['diagnostic'] and c.get('review_policy', 'automated') == 'human':
        from qa_eval.reporting import calibration
        try:
            evidence=read(c['calibration_report']);actual=calibration(evidence['rows'])
            if actual['status']!='passed':blockers.append('Human calibration gate not passed')
        except (TypeError,ValueError,KeyError,OSError):blockers.append('Human calibration evidence missing or invalid')
    return {'status':'blocked' if blockers else 'ready','blockers':sorted(set(blockers)),
            'simulated':c['backend']=='fake','diagnostic':c['diagnostic'],
            'review_policy':c.get('review_policy', 'automated'),
            'human_calibration_required':c.get('review_policy', 'automated') == 'human' and not c['diagnostic']}


class Coordinator:
    def __init__(self,c,run_dir,backend,ledger):
        self.c,self.directory,self.backend,self.ledger=c,Path(run_dir),backend,ledger
        self.events=EventLog(run_dir);self.judges=threading.Semaphore(c['judge_concurrency'])
        self.rng=random.Random(c['seed'])

    def episode(self,task,episode_id,seed):
        out=self.directory/'episodes'/episode_id
        if self.c['backend']=='fake':env=FakeEnvironment(allowed_tools=task['permitted_tools'])
        else:env=ModalEnvironment(self.c['environment_bundles'][task['id']],task['id'],out/'tools',
                                 self.ledger,self.c['paid_call_bounds']['modal'])
        if self.c['backend']!='fake':
            bound=env.record['task']
            if any(bound.get(k)!=task.get(k) for k in ('id','repository','environment_id','permitted_tools')):
                raise ValueError('Environment bundle does not match admitted task binding')
        result=run_episode(self.backend,env,task,self.c,seed=seed,output_dir=out)
        result.update(episode_id=episode_id,artifact_path=str(out/'episode.json'),split=task['split'],
                      experiment_hash=fingerprint(self.c),reward_version=self.c['reward_version'])
        atomic(out/'episode.json',result)
        with self.judges:
            if self.c['backend']=='fake':score=simulated_verify(result)
            else:
                try:score=verify(result,self.c['strict_tasks'][task['id']],self.experiment(),
                                 self.c['repository_roots'][task['id']],self.ledger,
                                 self.c['paid_call_bounds']['judge'],self.c['diagnostic'],self.c['judge_catalog_mode'],
                                 automated_review=(self.c.get('rubric_reviews', {}).get(task['id'])
                                                   if self.c.get('review_policy', 'automated') == 'automated' else None))
                except Exception as exc:score={'status':'unresolved','reward':None,'reason':type(exc).__name__}
        atomic(out/'verification.json',score)
        self.events.emit('episode',task_id=task['id'],episode_id=episode_id,reward=score['reward'],
                         termination=result['termination'],artifact=str(out/'episode.json'),simulated=self.c['backend']=='fake')
        return result,score

    def experiment(self):
        from qa_eval.security import digest as qdigest
        ex=copy.deepcopy(self.c['verifier_config'])
        ex.update(id=self.c['run_id'],frozen=True,task_manifest=[{'id':t['id'],'task_hash':qdigest(t)} for t in self.c['strict_tasks'].values()])
        return ex

    def evaluation(self,state,checkpoint):
        # Called only between optimizer updates; sampler identity cannot change.
        rows=[]
        with concurrent.futures.ThreadPoolExecutor(self.c['max_concurrency']) as pool:
            futures=[pool.submit(self.episode,t,'eval-'+str(state['optimizer_step'])+'-'+uuid.uuid4().hex[:12],self.c['seed']+i)
                     for i,t in enumerate(self.c['eval_tasks'])]
            for t,f in zip(self.c['eval_tasks'],futures):
                try:
                    trajectory,score=f.result();rows.append({'task_id':t['id'],'verification':score,'trajectory':trajectory['artifact_path']})
                except Exception as exc:rows.append({'task_id':t['id'],'verification':{'status':'unresolved','reward':None,'reason':type(exc).__name__}})
        total=len(rows);resolved=[r['verification'] for r in rows if r['verification']['status']=='resolved']
        report={'checkpoint':checkpoint,'policy_id':self.backend.policy_id,'optimizer_step':state['optimizer_step'],
                'expected':total,'resolved':len(resolved),'coverage':len(resolved)/total if total else None,
                'accepted_rate':sum(s.get('tier')=='accepted' for s in resolved)/total if total else None,
                'mean_reward_resolved':sum(s['reward'] for s in resolved)/len(resolved) if resolved else None,
                'status':'complete' if len(resolved)==total else 'incomplete','rows':rows,'simulated':self.c['backend']=='fake'}
        from .metrics import evaluation_metrics
        report['metrics']=evaluation_metrics(rows)
        path=self.directory/'evaluations'/('step-%06d-%s.json'%(state['optimizer_step'],uuid.uuid4().hex[:8]))
        atomic(path,report)
        self.events.emit('evaluation',optimizer_step=state['optimizer_step'],coverage=report['coverage'],
                         accepted_rate=report['accepted_rate'],metrics=report['metrics'],artifact=str(path),checkpoint=checkpoint)
        return report

    def checkpoint(self,state):
        state=copy.deepcopy(state);state['rng_state']=self.rng.getstate();state['policy_id']=self.backend.policy_id
        state['configuration_hash']=fingerprint(self.c)
        path=save_checkpoint(self.directory,self.backend,state,self.c)
        self.events.emit('checkpoint',optimizer_step=state['optimizer_step'],artifact=path)
        return path

    def train(self,state):
        if hasattr(self.backend,'connect'):self.backend.connect()
        if state.get('rng_state'):
            def tuples(x):return tuple(tuples(v) for v in x) if isinstance(x,list) else x
            self.rng.setstate(tuples(state['rng_state']))
        checkpoint=self.checkpoint(state)
        if state['optimizer_step']==0:self.evaluation(state,checkpoint)
        while state['stage_index']<len(self.c['stages']):
            stage=self.c['stages'][state['stage_index']]
            if state['stage_updates']>=stage['max_updates']:
                state['stage_index']+=1;state['stage_updates']=0
                if state['stage_index']<len(self.c['stages']):
                    # Stage transition deliberately starts a fresh optimizer.
                    m=load_checkpoint(checkpoint);self.backend.load(m['references'],optimizer=False)
                    checkpoint=self.checkpoint(state)
                continue
            if state['attempted_batches']>=self.c['max_attempted_batches']:
                state['status']='attempt_limit';break
            state['attempted_batches']+=1
            trajectories=[];advantages=[];scores=[]
            if stage['algorithm']=='sft':
                example=self.c['sft_examples'][state['cursor']%len(self.c['sft_examples'])];validate_sft(example)
                trajectories=[self.backend.render_sft(example)];state['cursor']+=1
                if any(len(a['prompt_tokens'])+len(a['token_ids'])>self.c['max_context_tokens'] for a in trajectories[0]['actions']):
                    raise ValueError('SFT example exceeds context limit')
            else:
                # Current backend requires one task per optimizer group.
                task=self.c['train_tasks'][state['cursor']%len(self.c['train_tasks'])];state['cursor']+=1
                for retry in range(self.c['full_group_retries']+1):
                    seed=self.rng.randrange(2**30)
                    with concurrent.futures.ThreadPoolExecutor(self.c['max_concurrency']) as pool:
                        fs=[pool.submit(self.episode,task,'train-'+uuid.uuid4().hex,seed+i) for i in range(self.c['group_size'])]
                        pairs=[]
                        for f in fs:
                            try:pairs.append(f.result())
                            except Exception as exc:
                                self.events.emit('episode_failure',error_type=type(exc).__name__)
                    if len(pairs)!=self.c['group_size']:group={'status':'retry_or_quarantine'}
                    else:
                        trajectories=[x[0] for x in pairs];scores=[x[1] for x in pairs]
                        for trajectory in trajectories:
                            if trajectory['split']!='train' or trajectory['policy_id']!=self.backend.policy_id:raise ValueError('Rollout identity mismatch')
                        group=prepare_group(trajectories,scores)
                    if group['status']=='ready':break
                    self.events.emit('group_unresolved',retry=retry,task_id=task['id'])
                if group['status']!='ready':continue
                advantages=group['advantages']
                if group['zero_variance']:
                    self.events.emit('zero_variance',optimizer_step=state['optimizer_step'],task_id=task['id']);continue
                for t in trajectories:validate_actions(t)
            intent={'status':'update_pending','attempted_batch':state['attempted_batches'],'last_checkpoint':checkpoint,
                    'policy_id':self.backend.policy_id,'optimizer_step_before':state['optimizer_step']}
            atomic(self.directory/'update-journal.json',intent)
            try:result=self.backend.update(trajectories,advantages if stage['algorithm']=='grpo' else None,
                                            stage['learning_rate'],stage['algorithm'])
            except BaseException:
                self.events.emit('update_unknown',last_checkpoint=checkpoint)
                raise
            if not result.get('updated'):raise ValueError('Expected contributing update')
            state['optimizer_step']+=1;state['stage_updates']+=1
            atomic(self.directory/'update-journal.json',{**intent,'status':'acknowledged','optimizer_step':state['optimizer_step']})
            self.events.emit('update',optimizer_step=state['optimizer_step'],attempted_batches=state['attempted_batches'],
                             algorithm=stage['algorithm'],loss=result['loss'],learning_rate=stage['learning_rate'],
                             mean_reward=sum(s['reward'] for s in scores)/len(scores) if scores else None,
                             policy_id=self.backend.policy_id,simulated=self.c['backend']=='fake')
            if state['optimizer_step']%self.c['checkpoint_every']==0 or state['optimizer_step']%self.c['eval_every']==0 or state['stage_updates']==stage['max_updates']:
                checkpoint=self.checkpoint(state)
            if state['optimizer_step']%self.c['eval_every']==0:self.evaluation(state,checkpoint)
            atomic(self.directory/'state.json',state)
            sync_tracking(self.directory,mode=self.c['tracking_mode'],project=self.c['tracking_project'])
        if state.get('status')!='attempt_limit':state['status']='completed'
        checkpoint=self.checkpoint(state);atomic(self.directory/'state.json',state)
        self.events.emit('run_finished',optimizer_step=state['optimizer_step'],status=state['status'],checkpoint=checkpoint)
        return {'run_dir':str(self.directory),'state':state,'checkpoint':checkpoint,'simulated':self.c['backend']=='fake'}


def run(config,checkpoint=None,resume=False):
    c=freeze_data(load_config(config))
    check=readiness(c)
    if check['blockers']:raise ValueError('Readiness blocked: '+'; '.join(check['blockers']))
    if c['batch_groups']!=1:raise ValueError('Initial recipe supports one task group per update')
    directory=Path(c['artifacts_root'])/c['run_id']
    directory.mkdir(parents=True,exist_ok=True)
    with lock(directory/'.writer.lock',blocking=False):
        if (directory/'config.json').exists() and not resume:
            raise ValueError('Run already exists; use resume or a new run ID')
        ledger=BudgetLedger(c['budget_ledger'],c['budget_cap_usd'],c['budget_reserve_usd'])
        backend=make_backend(c,ledger)
        state={'optimizer_step':0,'attempted_batches':0,'stage_index':0,'stage_updates':0,'cursor':0,'status':'running'}
        if checkpoint:
            m=load_checkpoint(checkpoint)
            for key in ('backend','model','lora_rank','renderer_name'):
                if c[key]!=m['config'][key]:raise ValueError('Checkpoint identity mismatch: '+key)
            if Path(c['budget_ledger']).resolve()!=Path(m['config']['budget_ledger']).resolve():
                raise ValueError('Checkpoint descendants must retain the shared spending ledger')
            if resume and fingerprint(c)!=fingerprint(m['config']):raise ValueError('Resume configuration mismatch; fork instead')
            backend.load(m['references'],optimizer=resume)
            if resume:state=copy.deepcopy(m['state']);state['status']='running'
            state['parent_checkpoint']=str(checkpoint)
        atomic(directory/'config.json',c)
        coordinator=Coordinator(c,directory,backend,ledger)
        coordinator.events.emit('run_start',run_id=c['run_id'],simulated=c['backend']=='fake',resume=resume)
        try:return coordinator.train(state)
        except BaseException as exc:
            coordinator.events.emit('run_interrupted',error_type=type(exc).__name__)
            atomic(directory/'state.json',{**state,'status':'interrupted'})
            raise
