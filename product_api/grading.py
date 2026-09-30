"""Separate grading process: frozen judge, blind candidates, validated claim coverage."""
import json
import os
import signal
import threading
import sys
import time
from pathlib import Path
from product_api.catalog import read
from product_api.worker import atomic_write
from training_pipeline.claim_grading import PROMPT, request_payload, aggregate
from training_pipeline.judge import TinkerJudge

JUDGE_MODEL = 'Qwen/Qwen3.5-397B-A17B'
POLICY = PROMPT + '\nThe required assertion is the criterion. Accept correct source-supported paraphrases. Do not demand extra facts or a particular writing style. This score measures reference-assertion coverage, not an exhaustive audit of all additional claims.'


def judge_config(models):
    model = next((m for m in models if m.get('base_model') == JUDGE_MODEL and m['kind'] == 'base' and m['ready']), None)
    if not model: raise ValueError('The fixed 397B rubric judge is unavailable; refresh models before starting a scored comparison')
    price = model.get('pricing')
    return {'base_model':JUDGE_MODEL, 'renderer':'hf-chat-no-thinking-v1', 'context_tokens':32768,
        'max_tokens':8192, 'temperature':0, 'provider_timeout_seconds':180,
        'prices':{'prefill':price['input_per_million'],'sample':price['output_per_million']} if price else {'prefill':0,'sample':0},
        'pricing_available':bool(price)}


def grade_answer(judge, benchmark, answer):
    request = {'question':benchmark['question'],'rubric':benchmark['rubric'],'answer':{'text':answer['text']}}
    payload = request_payload(request)
    generation = judge.sample([{'role':'system','content':POLICY},{'role':'user','content':json.dumps(payload)}],8192,0)
    if generation.stop_reason == 'length': raise ValueError('Judge output truncated')
    result = aggregate(generation.text,request)
    result['input_tokens'], result['output_tokens'] = len(generation.prompt), len(generation.tokens)
    return result


def main(root, comparison_id):
    parent = os.getppid()
    def watch_parent():
        while True:
            time.sleep(1)
            if os.getppid() != parent:
                os.kill(os.getpid(), signal.SIGTERM)
                return
    threading.Thread(target=watch_parent, daemon=True).start()
    root = Path(root)
    record = read(root/'comparisons'/(comparison_id+'.json'))
    path = root/'comparisons'/(comparison_id+'.grade.json')
    result = {'status':'running','judge_model':record['judge']['base_model'], 'rubric_hash':record['benchmark']['rubric']['rubric_hash'],
              'scores':{}, 'started_at':time.time(), 'estimated_cost_usd':None}
    if path.exists() and read(path).get('status') == 'cancelled': return
    atomic_write(path,result)
    judge = None
    try:
        for run_id in record['run_ids']:
            answer = read(root/run_id/'result.json').get('answer')
            if not answer:
                result['scores'][run_id] = {'status':'not_scored','score':None,'reason':'No final answer was submitted.'}
            else:
                try:
                    if judge is None: judge = TinkerJudge(record['judge'])
                    grade = grade_answer(judge,record['benchmark'],answer)
                    prices = record['judge']['prices']
                    grade['estimated_cost_usd'] = (grade['input_tokens']*prices['prefill'] + grade['output_tokens']*prices['sample'])/1e6 if record['judge']['pricing_available'] else None
                    result['scores'][run_id] = grade
                except Exception:
                    result['scores'][run_id] = {'status':'failed','score':None,'reason':'Grading could not be completed or validated. No numeric score was assigned.'}
            if path.exists() and read(path).get('status') == 'cancelled': return
            atomic_write(path,result)
        result['status'] = 'completed'
        result['elapsed_seconds'] = time.time()-result['started_at']
        costs = [g.get('estimated_cost_usd') for g in result['scores'].values()]
        if costs and all(v is not None for v in costs): result['estimated_cost_usd'] = sum(costs)
        if judge: result['judge_identity'] = judge.identity
        atomic_write(path,result)
    finally:
        if judge:
            try: judge.close()
            except Exception: pass

if __name__ == '__main__': main(sys.argv[1],sys.argv[2])
