"""Conservative citation integrity checks; never a semantic support judgment."""
import hashlib
from pathlib import Path, PurePosixPath
import re
from urllib.parse import unquote

_SCOPE = ("Explicit source filenames with recognized extensions; adjacent path:line-range, "
          "path (lines N-M), lines N-M in path, code-fence N:M:path headings, and GitHub blob #LN-LM anchors. "
          "File-only references are separate. Unattached prose line numbers and semantic "
          "support are not assessed; this parser does not guarantee complete citation extraction.")
_EXT = r"(?:py|pyi|js|jsx|ts|tsx|c|h|cpp|hpp|cc|rs|go|java|rb|php|cs|scala|swift|kt|m|sh|bash|sql|md|rst|txt|toml|yaml|yml|json|html|css|vue|svelte|ini|cfg)"
_PATH = re.compile(r"(?<![\w./\\-])(?P<path>(?:[A-Za-z]:)?[A-Za-z0-9_./\\-]+\." + _EXT + r")(?![\w/-]|\.[A-Za-z0-9_])", re.I)
_URL = re.compile(r"https?://github\.com/[^\s/]+/[^\s/]+/blob/[^\s/]+/(?P<path>[^\s`<>?#)]+)(?:#L(?P<start>\d+)(?:-L?(?P<end>\d+))?)?")
_RANGE = r"(?:\s*(?:[-–—]|to)\s*(\d+))?"
_AFTER = re.compile(r"^[`\"']?\s*(?:(?::\s*(?:lines?\s+)?)(\d+)" + _RANGE + r"|(?:[,:(]\s*)?(?:(?:at|on)\s+)?lines?\s+(\d+)" + _RANGE + r")", re.I)
_BEFORE = re.compile(r"\blines?\s+(\d+)" + _RANGE + r"\s+(?:in|of|from)\s*[`\"']?$", re.I)
_FENCE = re.compile(r"```(\d+):(\d+):$")



def inspect_reference(answer: str, root: Path, snapshot_files: dict, *, blob_reader=None) -> dict:
    """Resolve explicit citations against a pinned inventory and verify actual bytes.

    `evidence` contains only explicit line spans. `file_references` never invents
    a line range. `unresolved` retains parser-recognized invalid references.
    `status` describes span/file integrity only, not reference-answer correctness.
    """
    if not isinstance(answer, str):
        raise TypeError('answer must be a string')
    root = Path(root).resolve()
    candidates, covered = [], []
    for match in _URL.finditer(answer):
        start = int(match['start']) if match['start'] else None
        candidates.append((unquote(match['path']), start,
                           int(match['end'] or match['start']) if start is not None else None,
                           match.group(0)))
        covered.append(match.span())
    for match in _PATH.finditer(answer):
        if any(a <= match.start() < b for a, b in covered):
            continue
        path = match['path']
        after = _AFTER.match(answer[match.end():match.end() + 90])
        prefix = answer[max(0, match.start() - 90):match.start()]
        before = _BEFORE.search(prefix)
        fence = _FENCE.search(prefix)
        start = end = None
        if after:
            groups = after.groups()
            first, last = (groups[0], groups[1]) if groups[0] is not None else (groups[2], groups[3])
            start, end = int(first), int(last or first)
        elif before:
            start, end = int(before[1]), int(before[2] or before[1])
        elif fence:
            start, end = int(fence[1]), int(fence[2])
        candidates.append((path, start, end, match.group(0)))
    evidence, files, unresolved, seen = [], [], [], set()
    cache = {}
    for raw, start, end, original in candidates:
        key = (raw, start, end)
        if key in seen:
            continue
        seen.add(key)
        diagnostic = {'reference': original, 'path': raw, 'start_line': start, 'end_line': end}
        normalized = raw
        while normalized.startswith('./'):
            normalized = normalized[2:]
        path = PurePosixPath(normalized)
        if (path.is_absolute() or '\\' in raw or ':' in raw or '\x00' in raw
                or any(part in ('', '.', '..') for part in normalized.split('/'))):
            unresolved.append({**diagnostic, 'reason': 'unsafe_path'}); continue
        resolved = normalized
        if resolved not in snapshot_files:
            matches = [p for p in snapshot_files if PurePosixPath(p).name == normalized] if '/' not in normalized else []
            if len(matches) == 1:
                resolved = matches[0]
            else:
                unresolved.append({**diagnostic, 'reason': 'ambiguous_basename' if len(matches) > 1 else 'not_tracked',
                                   'candidates': sorted(matches)}); continue
        if resolved not in cache:
            try:
                relative = PurePosixPath(resolved)
                if relative.is_absolute() or any(p in ('', '.', '..') for p in resolved.split('/')) or '\\' in resolved:
                    raise ValueError('unsafe_inventory_path')
                if blob_reader is not None:
                    data = blob_reader(resolved)
                else:
                    local = root
                    for component in relative.parts:
                        local = local / component
                        if local.is_symlink():
                            raise ValueError('symlink_not_allowed')
                    if not local.resolve().is_relative_to(root):
                        raise ValueError('outside_snapshot')
                    if not local.is_file():
                        raise ValueError('missing_file')
                    data = local.read_bytes()
                sha = hashlib.sha256(data).hexdigest()
                if sha != snapshot_files[resolved]:
                    raise ValueError('file_hash_mismatch')
                if b'\x00' in data:
                    raise ValueError('binary_file')
                try:
                    lines = data.decode('utf-8').splitlines()
                except UnicodeDecodeError:
                    raise ValueError('non_utf8_file') from None
                cache[resolved] = {'file_sha256': sha, 'line_count': len(lines)}
            except (ValueError, OSError) as exc:
                cache[resolved] = {'error': str(exc)}
        inspected = cache[resolved]
        if 'error' in inspected:
            unresolved.append({**diagnostic, 'resolved_path': resolved, 'reason': inspected['error']}); continue
        if start is None:
            item = {'kind': 'file_reference', 'path': resolved, **inspected}
            if item not in files:
                files.append(item)
        elif not 1 <= start <= end <= inspected['line_count']:
            unresolved.append({**diagnostic, 'resolved_path': resolved, 'reason': 'line_range_out_of_bounds',
                               'line_count': inspected['line_count']})
        else:
            item = {'path': resolved, 'start_line': start, 'end_line': end,
                    'file_sha256': inspected['file_sha256']}
            if item not in evidence:
                evidence.append(item)
    status = 'partial' if (evidence or files) and unresolved else ('source_spans_verified' if evidence and not unresolved else 'no_resolvable_citations')
    return {'evidence': evidence, 'file_references': files, 'unresolved': unresolved,
            'parser_scope': _SCOPE, 'status': status, 'semantic_validation': 'not_performed'}
