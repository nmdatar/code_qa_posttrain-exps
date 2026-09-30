"""Source-only experimental tools. This module also runs inside the sandbox.

Hybrid retrieval uses deterministic subword hashing, not a learned semantic model.
Python references are AST identifier occurrences, not resolved cross-file bindings.
"""
import ast
import fnmatch
import hashlib
import heapq
import json
import math
from pathlib import Path
import re

VERSION = 'source-tools-v1'
EMBEDDING = 'char-trigram-hash-256-v1'
TOOLS = {'find_definition', 'find_references', 'get_context'}


def _source(root, name, inventory):
    path = root / name
    if (name not in inventory or path.is_symlink() or not path.resolve().is_relative_to(root)
            or any(p.is_symlink() for p in path.parents if p != root and p.is_relative_to(root))):
        raise ValueError('Not a regular pinned source file')
    if path.stat().st_size > 2_000_000:
        return None
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != inventory[name]:
        raise ValueError('Pinned source hash mismatch')
    try:
        return raw.decode('utf-8')
    except UnicodeError:
        return None


def _embedding(text):
    text = ' '.join(re.findall(r'\w+', text.lower()))[:4096]
    out = {}
    for i in range(max(0, len(text) - 2)):
        h = hashlib.sha256(text[i:i+3].encode()).digest()
        key = int.from_bytes(h[:2], 'big') % 256
        out[key] = out.get(key, 0) + (1 if h[2] & 1 else -1)
    norm = math.sqrt(sum(v*v for v in out.values()))
    return {k:v/norm for k,v in out.items()} if norm else {}


def run(root, inventory, name, args, retrieval='lexical-v1'):
    root = Path(root).resolve()
    matches = []
    if name in {'find_definition', 'find_references'}:
        symbol = args['symbol']
        for path in sorted(inventory):
            if not path.endswith('.py') or not fnmatch.fnmatch(path, args.get('glob', '*')):
                continue
            source = _source(root, path, inventory)
            if source is None:
                continue
            try:
                tree = ast.parse(source)
            except (SyntaxError, ValueError, RecursionError):
                continue
            lines = source.splitlines()
            found = set()
            for node in ast.walk(tree):
                definition = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == symbol
                reference = ((isinstance(node, ast.Name) and node.id == symbol) or
                             (isinstance(node, ast.Attribute) and node.attr == symbol))
                if (definition if name == 'find_definition' else reference):
                    found.add(node.lineno)
            for line in sorted(found):
                matches.append({'path':path, 'start_line':line, 'end_line':line,
                                'file_sha256':inventory[path], 'text':lines[line-1][:300]})
        total = len(matches)
        value = {'matches':matches, 'scope':'Python AST names; references are occurrences, not resolved bindings'}
        while len(json.dumps(value).encode()) > 1800 and value['matches']:
            value['matches'].pop()
        value['truncated'] = len(value['matches']) < total
        return value
    if name != 'get_context':
        raise ValueError('Unknown research tool')
    if retrieval not in {'lexical-v1', 'hybrid-subword-v1'}:
        raise ValueError('Unknown retrieval variant')
    query = args['query']; terms = set(re.findall(r'\w+', query.lower()))
    vector = _embedding(query); heap = []; scanned = 0
    for path in sorted(inventory):
        if not fnmatch.fnmatch(path, args.get('glob', '*')):
            continue
        source = _source(root, path, inventory)
        if source is None:
            continue
        lines = source.splitlines()
        for offset in range(0, len(lines), 12):
            chunk = lines[offset:offset+12]; text = '\n'.join(chunk)
            scanned += 1
            words = set(re.findall(r'\w+', path.lower()+' '+text.lower()))
            lexical = len(terms & words) / max(1, len(terms))
            score = lexical
            if retrieval == 'hybrid-subword-v1':
                embedded = _embedding(path+' '+text)
                cosine = max(0., sum(v*embedded.get(k,0) for k,v in vector.items()))
                score = .5*lexical + .5*cosine
            if score <= 0:
                continue
            item = (score, path, offset, text)
            if len(heap) < 8:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
    snippets = []
    value = {'snippets':snippets, 'retrieval':retrieval, 'chunks_scanned':scanned,
             'embedding':EMBEDDING if retrieval == 'hybrid-subword-v1' else None}
    for _, path, offset, text in sorted(heap, key=lambda x:(-x[0],x[1],x[2])):
        lines = text.splitlines()
        ref = {'path':path,'start_line':offset+1,'end_line':offset+len(lines),
               'file_sha256':inventory[path], 'lines':[f'{offset+i+1}: {line}' for i,line in enumerate(lines)]}
        snippets.append(ref)
        while len(json.dumps(value).encode()) > 1800 and ref['lines']:
            ref['lines'].pop(); ref['end_line'] -= 1
        if not ref['lines']:
            snippets.pop()
        if len(json.dumps(value).encode()) > 1500:
            break
    return value


def command(name, args, inventory, workspace_path, retrieval='lexical-v1'):
    specs = {'find_definition':({'symbol':str},{'glob':str}),
             'find_references':({'symbol':str},{'glob':str}),
             'get_context':({'query':str},{'glob':str,'budget':int})}
    if name not in specs or not isinstance(args, dict):
        raise ValueError('Unknown research tool')
    required, optional = specs[name]
    if set(required)-args.keys() or args.keys()-required.keys()-optional.keys():
        raise ValueError('Invalid research tool arguments')
    for k,v in args.items():
        if type(v) is not {**required,**optional}[k] or isinstance(v,str) and (not v or len(v.encode()) > 8000 or '\0' in v):
            raise ValueError('Invalid research tool argument')
    if 'symbol' in args and not args['symbol'].isidentifier():
        raise ValueError('Use a single Python identifier')
    if 'budget' in args and not 1 <= args['budget'] <= 512:
        raise ValueError('Context budget must be between 1 and 512 tokens')
    if workspace_path not in {'/repo','/workspace'}:
        raise ValueError('Invalid workspace')
    catalog = json.dumps(inventory,sort_keys=True)
    if len(catalog.encode()) > 4_000_000:
        raise ValueError('Research source catalog too large')
    return ['python3','-c',Path(__file__).read_text(),workspace_path,name,json.dumps(args),retrieval,
            *[catalog[i:i+32000] for i in range(0,len(catalog),32000)]]


if __name__ == '__main__':
    import sys
    print(json.dumps(run(sys.argv[1],json.loads(''.join(sys.argv[5:])),sys.argv[2],json.loads(sys.argv[3]),sys.argv[4])))
