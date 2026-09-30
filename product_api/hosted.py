"""Tinker hosted-model discovery and versioned, optional token price estimates."""
import asyncio
import json
import math
import os
import httpx
from product_api.limits import CONTEXT_TOKENS
from datetime import datetime, timezone

METADATA_URL = 'https://tinker-docs.thinkingmachines.ai/tinker/models.json'
BASE = 'Qwen/Qwen3.5-4B'


def metadata():
    try:
        response = httpx.get(METADATA_URL, timeout=8, follow_redirects=True)
        response.raise_for_status()
        rows = response.json()
        return {r['tinker_id']: r for r in rows}
    except Exception:
        return {}


def valid_price(value):
    return isinstance(value, dict) and all(type(value.get(k)) in (int, float) and math.isfinite(value[k]) and value[k] >= 0
        for k in ('input_per_million', 'output_per_million'))


def pricing_overrides():
    path = os.environ.get('PRODUCT_MODEL_PRICING_FILE')
    if not path:
        return {}
    try:
        with open(path) as stream:
            rows = json.load(stream)
        if not isinstance(rows, dict) or not all(valid_price(v) for v in rows.values()):
            raise ValueError('Invalid rates')
        return rows
    except (OSError, ValueError):
        raise ValueError('PRODUCT_MODEL_PRICING_FILE must contain model IDs mapped to nonnegative input_per_million and output_per_million rates') from None


def discover(service, base_id):
    # The SDK can pause and retry billing errors indefinitely even with a
    # per-request timeout. Bound the whole lookup so checkpoint discovery runs.
    async def capabilities():
        return await asyncio.wait_for(service.get_server_capabilities_async(), timeout=15)
    caps = asyncio.run(capabilities()).supported_models
    info = metadata() if caps else {}
    rows = []
    for cap in caps:
        name = cap.model_name
        details = info.get(name, {})
        compatible = name.startswith(('Qwen/Qwen3.5-', 'Qwen/Qwen3.6-', 'Qwen/Qwen3.8-')) and not name.split(':')[0].endswith('-Base')
        ready = bool(cap.sampleable and cap.max_context_length and compatible)
        price = None
        try:
            price = {'input_per_million': float(str(details['prefill']).removeprefix('$')),
                'output_per_million': float(str(details['sample']).removeprefix('$')),
                'source': METADATA_URL, 'checked_at': datetime.now(timezone.utc).isoformat()}
            if not valid_price(price): price = None
        except (KeyError, ValueError, TypeError):
            pass
        rows.append(dict(id='base' if name == base_id else 'hosted:' + name,
            name=details.get('name', name.split('/')[-1]), kind='base', base_model=name,
            model_path=None, renderer='hf-chat-no-thinking-v1', context_tokens=min(CONTEXT_TOKENS, cap.max_context_length or CONTEXT_TOKENS),
            max_context_tokens=cap.max_context_length, ready=ready,
            reason=None if ready else ('Unsupported model renderer' if not compatible else 'Sampling unavailable'),
            parameters=details.get('params'), active_parameters=details.get('active_params'), pricing=price,
            run='Tinker hosted', step=None))
    return sorted(rows, key=lambda r: (r['id'] != 'base', not r['ready'], r.get('parameters') or 0, r['name']))


def apply_pricing(models):
    overrides = pricing_overrides()
    hosted = {m['base_model']: m.get('pricing') for m in models if m['kind'] == 'base'}
    for model in models:
        model['pricing'] = overrides.get(model['id'], overrides.get(model['base_model'], hosted.get(model['base_model'])))
    return models
