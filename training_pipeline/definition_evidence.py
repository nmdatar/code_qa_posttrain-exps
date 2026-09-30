"""Pinned Python definition headers for scoped judge evidence, never score repair."""
import ast
import re

from .admission import blob, sha

VERSION = 'definition-context-v1'
INSTRUCTION = ('Verify function/class attribution from supplied definition headers and source. '
               'Do not infer a different function name from a body fragment, library familiarity, '
               'or an omitted header. If attribution cannot be established, record insufficient '
               'evidence rather than inventing a contradictory identifier. Source context does '
               'not expand the candidate citation ranges or make a citation supported automatically.')


def definition_context(request):
    row = request['source_row']
    inventory = row['image_result']['snapshot_files']
    refs = request['rubric']['evidence'] + request.get('answer_evidence', [])
    known = {ref['path'] for ref in refs} | set(request.get('observed_files', {}))
    text = request['answer']['text'] + ' ' + ' '.join(c['text'] for c in request['rubric']['claims'])
    names = set(re.findall(r'\b[A-Za-z_]\w*\b', text))
    output = []
    for path in sorted(known):
        if not path.endswith('.py') or path not in inventory:
            continue
        content = blob(row['snapshot_root'], row['public']['repository']['commit'], path)
        if sha(content) != inventory[path]:
            raise ValueError('Definition source hash mismatch')
        try:
            tree = ast.parse(content.decode())
        except (SyntaxError, UnicodeDecodeError):
            continue
        spans = [(r['start_line'], r['end_line']) for r in refs if r['path'] == path]
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if node.name not in names and not any(node.lineno <= a <= b <= node.end_lineno for a, b in spans):
                continue
            # Include the signature and docstring, not an unbounded function body.
            first = node.body[0]
            end = max(node.lineno, first.lineno - 1)
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                end = first.end_lineno
            output.append({'path': path, 'start_line': node.lineno,
                           'end_line': min(end, node.lineno + 59), 'file_sha256': inventory[path]})
    return sorted(output, key=lambda r: (r['path'], r['start_line']))
