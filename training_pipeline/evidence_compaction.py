"""Losslessly represent overlapping pinned line ranges once per source file."""
import copy
from qa_eval.security import digest


def compact(evidence):
    result = copy.deepcopy(evidence)
    files = {}
    for key, value in evidence.items():
        if not isinstance(value, dict) or not {'path', 'start_line', 'end_line', 'text'} <= value.keys():
            continue
        lines = value['text'].splitlines()
        if len(lines) != value['end_line']-value['start_line']+1:
            raise ValueError('Evidence line count mismatch')
        rows = files.setdefault(value['path'], {'lines': {}, 'keys': []})
        rows['keys'].append(key)
        for number, text in zip(range(value['start_line'], value['end_line']+1), lines):
            if number in rows['lines'] and rows['lines'][number] != text:
                raise ValueError('Overlapping evidence disagrees')
            rows['lines'][number] = text
    for path, rows in files.items():
        if len(rows['keys']) < 2:
            continue
        source_key = 'source_' + digest({'path': path, 'lines': rows['lines']})[:24]
        if source_key in result:
            raise ValueError('Evidence source key collision')
        # Numbered lines preserve gaps explicitly; no omitted line is implied.
        result[source_key] = {'path': path, 'numbered_source_lines':
                             [text for _, text in sorted(rows['lines'].items())]}
        for key in rows['keys']:
            value = result[key]
            del value['text']
            value['source_key'] = source_key
    return result
