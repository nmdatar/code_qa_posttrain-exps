"""Private judge requests through stored Tinker auth, without training allocation.

The caller must wrap each call in the shared BudgetLedger. Models and renderers
are explicit command arguments, not inferred from submitted answers.
"""
import json
import threading

_clients={}
_lock=threading.Lock()


def judge_call(command,request,timeout):
    if len(command)!=5 or command[0]!='tinker':
        raise ValueError('Tinker judge command requires training model/renderer and evaluation model/renderer')
    role=request['routing_role']
    if role not in {'training','evaluation'}:
        raise ValueError('Unknown judge routing role')
    model,renderer_name=command[1:3] if role=='training' else command[3:5]
    with _lock:
        if (model,renderer_name) not in _clients:
            import tinker
            from tinker_cookbook.renderers import get_renderer
            service=tinker.ServiceClient(timeout=timeout,max_retries=0)
            sampler=service.create_sampling_client(base_model=model)
            renderer=get_renderer(renderer_name,sampler.get_tokenizer(),model_name=model)
            _clients[model,renderer_name]=(service,sampler,renderer)
        _,sampler,renderer=_clients[model,renderer_name]
    from tinker import types
    body={k:v for k,v in request.items() if k not in {'policy','instructions','output_schema','routing_role'}}
    messages=[{'role':'system','content':request['policy']+'\n'+request['instructions']+'\nReturn JSON only. Schema:\n'+json.dumps(request['output_schema'])},
              {'role':'user','content':json.dumps(body)}]
    prompt=renderer.build_generation_prompt(messages)
    if len(prompt.to_ints())>24000:raise ValueError('Judge prompt exceeds bounded24000-token allowance; curate evidence catalog')
    result=sampler.sample(prompt=prompt,num_samples=1,sampling_params=types.SamplingParams(
        max_tokens=4096,temperature=0.0,stop=renderer.get_stop_sequences())).result(timeout=timeout)
    if len(result.sequences)!=1:raise ValueError('Judge returned unexpected sequence count')
    seq=result.sequences[0]
    if seq.stop_reason=='length':raise ValueError('Judge output exhausted token budget')
    parsed=renderer.parse_response(seq.tokens)
    message=parsed[0] if isinstance(parsed,tuple) else parsed
    content=message.get('content','')
    if isinstance(content,list):content=''.join(p.get('text','') for p in content if p.get('type')=='text')
    if not isinstance(content,str):raise ValueError('Judge produced unsupported content')
    content=content.strip()
    if content.startswith('```json') and content.endswith('```'):content=content[7:-3].strip()
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ValueError('Duplicate judge JSON field')
            result[key]=value
        return result
    result=json.loads(content,object_pairs_hook=unique,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite judge JSON')))
    from qa_eval.schema import validate
    validate(result,request['output_schema'])
    return result
