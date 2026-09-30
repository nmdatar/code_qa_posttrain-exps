# Available 600-task assessment

Expanded the initial 100-task pilot using all 1,200 compatible historical teacher attempts over 600 training tasks, spanning 31 repository families. No new calls were needed for this offline expansion.

582 tasks have two resolved factual-coverage scores; 18 remain unranked because one or both grading results are unresolved. Existing model failures retain their scores. Ranking uses mean coverage over two attempts, descending, with deterministic tie handling.

- Recommended inclusive 0.2–0.8 score band: **202 tasks across 29 families**.
- Requested middle-percentile band: **348 tasks across 31 families**. Boundary ties mean some scores remain at 0 or 1.

Use `ranked-tasks.csv` for the ranking, `score-20-80-tasks.jsonl` for the recommended public-task export, and the corresponding manifest for IDs/provenance. `trajectories.html` links all 1,200 traces by repository. Existing private reference/rubric records remain in the unchanged v6 source release.

This is an interim result. The separately approved 516-rollout expansion will assess the remaining 258 training tasks. Neither this ranking nor the new campaign uses development/confirmation tasks. Two teacher attempts are an exploratory solvability signal, not proof the student can learn each task.
