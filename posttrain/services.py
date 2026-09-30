"""Read-only service discovery. Never returns credential values or identifiers."""
from datetime import datetime, timezone
from importlib.metadata import version
import hashlib
import json
import netrc
import os
import ssl
import subprocess
import urllib.error
from pathlib import Path
import urllib.request
from urllib.parse import urlparse

PRICING_URL='https://tinker-docs.thinkingmachines.ai/tinker/models.json'


def tinker_auth_status():
    if os.environ.get('TINKER_API_KEY'):
        return {'available':True,'source':'environment','verified':False}
    try:
        from tinker.lib.credentials import JsonCredentialStore, default_credentials_path
        record=JsonCredentialStore(default_credentials_path()).get_default_key()
        return {'available':bool(record and record.key),'source':'official_sdk_store','verified':False}
    except ImportError:
        return {'available':False,'source':'sdk_not_installed','verified':False}
    except Exception as exc:
        return {'available':False,'source':'official_sdk_store','verified':False,'error_type':type(exc).__name__}


def tinker_auth_available():
    return tinker_auth_status()['available']


def wandb_auth_status():
    if os.environ.get('WANDB_API_KEY'):
        return {'available':True,'source':'environment','verified':False}
    try:
        host=urlparse(os.environ.get('WANDB_BASE_URL','https://api.wandb.ai')).hostname
        authentication=netrc.netrc().authenticators(host)
        return {'available':bool(authentication and authentication[2]),'source':'netrc','verified':False}
    except (OSError,netrc.NetrcParseError):
        return {'available':False,'source':'netrc','verified':False}


def probe_tinker_capabilities():
    """Metadata only: does not create a training or sampling model client."""
    import tinker
    with tinker.ServiceClient(timeout=30,max_retries=0) as client:
        response=client.get_server_capabilities()
        models=[{'model':m.model_name,'max_context_tokens':m.max_context_length,
                 'trainable':m.trainable,'sampleable':m.sampleable} for m in response.supported_models]
    return {'status':'verified','checked_at':datetime.now(timezone.utc).isoformat(),
            'sdk_version':version('tinker'),'models':models,'paid_model_calls':0}


def fetch_pricing():
    try:
        import certifi
        context=ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context=ssl.create_default_context()
    try:
        with urllib.request.urlopen(PRICING_URL,timeout=30,context=context) as response:
            raw=response.read()
    except urllib.error.URLError:
        # Native curl uses platform trust and HTTP negotiation. TLS stays verified.
        raw=subprocess.run(['curl','--fail','--silent','--show-error','--max-time','30',PRICING_URL],
                           check=True,capture_output=True,timeout=35).stdout
    return {'source':PRICING_URL,'retrieved_at':datetime.now(timezone.utc).isoformat(),
            'sha256':hashlib.sha256(raw).hexdigest(),'models':json.loads(raw)}


def rank_training_candidates(capabilities,pricing,min_context=16384):
    """Rank a frozen reference workload, not a claim of measured speed/quality.

    Workload: one million input, output and training tokens each. Missing
    capability flags stay unknown rather than silently being considered ready.
    """
    from tinker_cookbook.model_info import get_recommended_renderer_names
    prices={p['tinker_id']:p for p in pricing['models']}
    candidates=[]
    for entry in capabilities['models']:
        if entry['trainable'] is not True or entry['sampleable'] is not True:
            continue
        if not entry['max_context_tokens'] or entry['max_context_tokens']<min_context:
            continue
        price=prices.get(entry['model'])
        if not price: continue
        try:
            rates={k:float(str(price[k]).replace('$','').replace(',','')) for k in ('prefill','sample','train')}
            renderers=get_recommended_renderer_names(entry['model'])
        except (ValueError,KeyError):continue
        if not renderers:continue
        candidates.append({**entry,'pricing':rates,'pricing_units':'USD per million tokens',
                           'renderers':renderers,'reference_workload_usd':sum(rates.values()),
                           'model_type':price.get('type'),'live_tokenization_verified':False,
                           'tool_rollout_verified':False,'sampled_logprobs_verified':False})
    return sorted(candidates,key=lambda r:(r['reference_workload_usd'],r['model']))
