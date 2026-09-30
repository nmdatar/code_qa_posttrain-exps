"""Independent automated rubric admission, without asserting human approval."""
import hashlib
import json


def automated_rubric_supported(task, review):
    if not isinstance(review, dict) or not isinstance(task, dict):
        return False
    author, reviewer = review.get('author'), review.get('reviewer')
    if (not isinstance(author, str) or not author.strip() or
            not isinstance(reviewer, str) or not reviewer.strip() or
            author.strip().casefold() == reviewer.strip().casefold()):
        return False
    rubric_hash = hashlib.sha256(json.dumps(task, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return (review.get('task_id') == task.get('id') and
            review.get('status') == 'supported' and
            review.get('rubric_sha256') == rubric_hash and
            task.get('gold_status') in {'draft', 'accepted'})
