"""Resolve explicit prose-reference citations against the pinned file inventory."""
import re
from .admission import blob, sha

CITATION = re.compile(r'([\w./-]+\.py):(\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*)')


def cited_ranges(text):
    for path, spans in CITATION.findall(text):
        for span in spans.split(','):
            endpoints = span.split('-')
            yield path, int(endpoints[0]), int(endpoints[-1])


def enrich(reference, row):
    refs = list(reference['verified_evidence'])
    seen = {(r['path'],r['start_line'],r['end_line']) for r in refs}
    rejected = []
    inventory = row['image_result']['snapshot_files']
    for path, start, end in cited_ranges(reference['reference_answer']):
        if path not in inventory:
            rejected.append({'path':path,'reason':'not_in_pinned_inventory'});continue
        content = blob(row['snapshot_root'],row['public']['repository']['commit'],path)
        if sha(content) != inventory[path]:
            raise ValueError('Pinned reference source changed: '+path)
        if not 1 <= start <= end <= len(content.splitlines()):
            rejected.append({'path':path,'start':start,'end':end,'reason':'invalid_line_range'});continue
        for first in range(start,end+1,600):
            last=min(first+599,end);key=(path,first,last)
            if key not in seen:
                refs.append({'path':path,'start_line':first,'end_line':last,'file_sha256':inventory[path]});seen.add(key)
    return {**reference,'verified_evidence':refs,'evidence_revision':'reference-citations-v1'},rejected
