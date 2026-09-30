"""Summarize frozen matched judge outputs without treating missing grades as zero."""
import csv
import json
from collections import Counter
from pathlib import Path

root = Path('reports/atomic-claims-judge-comparison')
report = json.loads((root / 'comparison-progress.json').read_text())
if report['status'] != 'complete' or len(report['comparisons']) != 28:
    raise ValueError('The paired comparison is not complete')
rows = []
claim_rows = []
for case in report['comparisons']:
    grade = case.get('grade', {})
    row = {'model': case['model'], 'rubric': case['rubric'], 'case': case['case'],
           'status': 'error' if 'error' in case else grade['status'],
           'reward': case['reward'], 'strict_score': grade.get('strict_score'),
           'error': case.get('error', ''), 'reason': grade.get('reason', '')}
    rows.append(row)
    for claim in grade.get('claims', []):
        claim_rows.append({**{k: row[k] for k in ('model', 'rubric', 'case')},
                           **{k: claim[k] for k in ('id', 'verdict', 'coverage', 'material_error', 'reason')}})
for filename, data in [('comparison-results.csv', rows), ('claim-decisions.csv', claim_rows)]:
    with (root / filename).open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
conditions = []
for model in sorted({row['model'] for row in rows}):
    for rubric in ('original', 'atomic'):
        selected = [r for r in rows if r['model'] == model and r['rubric'] == rubric]
        conditions.append({'model': model, 'rubric': rubric, 'attempts': len(selected),
                           'status_counts': dict(Counter(r['status'] for r in selected)),
                           'numeric_rewards': sum(r['reward'] is not None for r in selected)})
summary = {'status': report['status'], 'attempts': len(rows), 'optimizer_updates': report['optimizer_updates'],
           'conditions': conditions, 'model_reservations_usd': report['reserved_usd'],
           'elapsed_seconds': report['elapsed_seconds'],
           'note': 'No pooled accuracy or mean that imputes failed grades as zero; seven correlated cases across three tasks.'}
(root / 'comparison-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
