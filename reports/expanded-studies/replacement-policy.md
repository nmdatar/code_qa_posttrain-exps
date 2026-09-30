# Infrastructure replacement policy — frozen before replacement launch

An incomplete run that terminates because of an unacknowledged backend update may receive at most one fresh-base replacement under a new run ID, within the existing project allocation. This rule applies equally to all arms/seeds and is based on infrastructure failure, never observed quality. Preserve failed-run trajectories, checkpoints, acknowledgements and spending as reported infrastructure attempts; do not treat them as completed replicates or choose whichever run scores better.

For direct seed42 v1, retain the immutable checkpoint after acknowledged update1 and leave the uncertain trainer abandoned. Replacement v2 starts from the unchanged base model with seed42, task permutation, reward, hyperparameters, episode limits and32scheduled batches. It does not restore the uncertain state, reuse its gradients, resubmit the old optimizer request or replay the old generated trajectories. Generations are newly sampled. No resumption or ordinary fork of v1 is permitted.

The only source change for replacement is sanitized failure-phase observability: distinguish forward/backward, local reduction validation and optimizer dispatch. Numerical tolerances, objective and retry behavior remain unchanged. If another ambiguous failure occurs, stop further RL launches and investigate using the new diagnostics; do not launch repeated paid replacements.

Source/data/grader comparisons and remaining-budget preflight are required before launch. Teacher recovery is currently using the shared controller; replacement is prepared only and must wait until it is free. The three-arm comparison remains incomplete, and confirmation must remain untouched until all required models and analysis are frozen.
