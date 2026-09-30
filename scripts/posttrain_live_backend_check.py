"""Small paid backend check on explicit synthetic supervision, never dataset evidence."""
from pathlib import Path
import json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from posttrain.backends import TinkerBackend
from posttrain.storage import BudgetLedger,atomic
from posttrain.checkpoints import save_checkpoint,load_checkpoint
from posttrain.config import load_config
from posttrain.tracking import EventLog,sync_tracking

out=Path('artifacts/posttrain/live-backend-check');out.mkdir(parents=True,exist_ok=True)
if (out/'started.json').exists():raise SystemExit('This paid check already started; inspect artifacts before retrying')
atomic(out/'started.json',{'started_at':time.time(),'scope':'synthetic SFT connectivity only'})
ledger=BudgetLedger('artifacts/posttrain/spending.json')
c=load_config({'run_id':'live-backend-check','backend':'tinker','model':'openai/gpt-oss-20b','renderer_name':'gpt_oss_no_sysprompt','tracking_mode':'online'})
atomic(out/'config.json',c);events=EventLog(out)
b=TinkerBackend(c['model'],ledger,c['renderer_name'],rank=16,bounds={'connect':.01,'save':.01,'sample':.02,'update':.05})
try:
 b.connect();events.emit('backend_connected',model=c['model'])
 sample=b.sample([{'role':'user','content':'Reply with exactly the digit 4.'}],max_tokens=128,temperature=0,seed=42)
 atomic(out/'sample.json',sample)
 example={'id':'synthetic-arithmetic','split':'train','status':'accepted','provenance':{'synthetic':True,'purpose':'backend API contract'},'messages':[{'role':'user','content':'What is 2 + 2? Reply with the digit.'},{'role':'assistant','content':'4'}]}
 trajectory=b.render_sft(example)
 result=b.update([trajectory],learning_rate=1e-5,algorithm='sft')
 events.emit('update',optimizer_step=1,loss=result['loss'],scope='synthetic SFT; not repository training',simulated=False)
 checkpoint=save_checkpoint(out,b,{'optimizer_step':1,'stage_index':0,'stage_updates':1,'cursor':1,'attempted_batches':1,'status':'completed'},c)
 manifest=load_checkpoint(checkpoint)
 evaluator=TinkerBackend(c['model'],ledger,c['renderer_name'],bounds={'connect':.01,'save':.01,'sample':.02,'update':.05})
 evaluator.load(manifest['references'],purpose='evaluate')
 after=evaluator.sample([{'role':'user','content':'Reply with exactly the digit 4.'}],max_tokens=128,temperature=0,seed=42)
 atomic(out/'sample-after.json',after)
 resumed=TinkerBackend(c['model'],ledger,c['renderer_name'],bounds={'connect':.01,'save':.01,'sample':.02,'update':.05})
 resumed.load(manifest['references'],optimizer=True)
 forked=TinkerBackend(c['model'],ledger,c['renderer_name'],bounds={'connect':.01,'save':.01,'sample':.02,'update':.05})
 forked.load(manifest['references'],optimizer=False)
 report={'status':'passed','scope':'real Tinker sampling, synthetic SFT update, save/evaluate/optimizer-resume/fresh-fork API connectivity',
         'repository_rl_verified':False,'model':c['model'],'loss':result['loss'],'checkpoint':checkpoint,'sampled_tokens':len(sample['token_ids']),
         'budget':ledger.summary(),'tracking':sync_tracking(out,mode='online',project='repo-qa-posttrain')}
 atomic('reports/posttrain/live-backend-check.json',report);print(json.dumps({k:v for k,v in report.items() if k!='budget'},indent=2))
except BaseException as exc:
 atomic('reports/posttrain/live-backend-check.json',{'status':'failed','error_type':type(exc).__name__,'stage_events':events.read(),'scope':'synthetic backend check'})
 raise
