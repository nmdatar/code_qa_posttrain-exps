"""Read the original and subsection arms and compare baseline-adjusted reward."""
from pathlib import Path
import json
import compare_bash_efficiency_grpo as comparison
from training_pipeline.storage import atomic_json

comparison.OUT = Path('reports/bash-prompt-decomposition-grpo-7-v1')
comparison.ARMS = {'original':'bash-correctness-grpo-15-v1',
                   'subsections':'bash-prompt-decomposition-grpo-7-v1'}

if __name__ == '__main__':
    summary = comparison.fetch()
    rows = {arm:{e['optimizer_step']:e for e in value['evaluations']}
            for arm,value in summary.items()}
    gains = []
    for step in (3,6):
        values = {}
        for arm, evaluations in rows.items():
            if 0 in evaluations and step in evaluations:
                base = evaluations[0].get('mean_correctness_score')
                score = evaluations[step].get('mean_correctness_score')
                if base is not None and score is not None:
                    values[arm] = {'baseline':base,'score':score,'gain':score-base,
                        'scoring_coverage':evaluations[step].get('scoring_coverage')}
        gains.append({'step':step, 'arms':values,
            'subsections_minus_original_gain':values['subsections']['gain']-values['original']['gain']
                if len(values)==2 else None})
    result = {'comparisons':gains, 'interpretation':'Exploratory single seed. Compare scoring coverage; aggregate resolved-case gains can include different cases. Confirm with paired task-level scores before claiming an improvement.'}
    atomic_json(comparison.OUT/'reward-gains.json', result)
    print(json.dumps(result, indent=2))
