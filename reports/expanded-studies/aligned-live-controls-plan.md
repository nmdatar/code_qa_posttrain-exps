# Frozen live aligned-reward gate

Prepared locally; no submission or paid grading was performed during preparation. Launch from the repository root with:

```sh
PYTHONPATH=. .venv-eval/bin/python scripts/launch_aligned_live_controls.py
```

This custom worker grades 56 fixed submissions: four source-verified admitted training tasks, seven variants per task, two independent grading replicates. The tasks cover scikit-learn SVC probability methods, requests Link parsing, Sphinx LaTeX theme configuration, and Matplotlib OffsetBox containment. The fixture contains exact pinned source excerpts, complete-file hashes, repository commits, rubric hashes, and submissions. No selection or confirmation answer was used. Full dataset validation checked membership against 858 training tasks and 117 development tasks.

The seven variants are correct, partial, wrong, correct plus source contradiction, correct plus unsupported benchmark claim, uncited, and invalid citation. Grades use the production full semantic verifier, independent required-fact coverage, and frozen aligned-coverage-v1 coefficients, with the existing v6 rubrics/v7 grader and unchanged judge model/prices. The fixture envelope explicitly forbids training admission. The worker creates no policy client, policy generation, gradient, optimizer call, or training trajectory.

The frozen gate requires at least 54/56 eligible grades and at least one eligible replicate for every task/variant. Every eligible reward, coverage and penalty component must be present and finite. Per task, correct mean reward must be at least 0.9, partial between 0.1 and 0.8, wrong at most 0.05, and the order must be strictly correct > partial > wrong. Correct controls require at least 0.95 coverage and zero contradiction, unsupported, uncited and bad-citation components; partial coverage must fall between 0.25 and 0.75. Source contradiction must be detected and reduce reward by at least 0.5; the unsupported claim must be identified as unsupported rather than contradiction and reduce reward by at least 0.1. Uncited and invalid-citation controls must reduce reward by at least 0.05, and every eligible invalid-citation control must receive a bad-citation component. Thresholds and coefficients remain frozen after live results. Failure blocks aligned training; success permits a separate reviewed launch but does not launch it automatically.

This is a directional prerequisite, not an estimate establishing grader accuracy or model improvement. Four tasks and two replicates cannot establish generalization across repositories or all reward failure modes. Unresolved or failed controls must be reported, not silently retried or replaced to pass.

The private allocation is $100 from the previously unallocated $506.079295360982. Judge reservations have a conservative upper bound of $78.446592 for 56 cases × three judge stages × two attempts at the frozen context/output limits and prices. The two-hour controller upper bound is $1.519488, yielding $79.96608 total. The complete allocation remains $6000, retaining both seed43 and seed44 allocations of $1800 and $406.0792953609819 for future work. These are reservations, not invoices.

The launcher checks all frozen file hashes and the immutable source input bundle, refuses an existing named controller or private volume, and writes a pending receipt before submission. The worker additionally refuses an existing ledger/output and checks the whole gate fits before grading. Inspect any pending receipt rather than resubmitting. Results will be in the private volume at `/artifacts/controls/aligned-live-controls-v1/results.json`, with campaign status at `/control-campaign/status.json` and ledger at `/artifacts/project-budget/aligned-live-controls-v1-isolated.json`.

Frozen fixture: `3013c87268b0b155b29fe147e34ba31c888b92f3259768afc6954f0b1fc0b648`.
Frozen control bundle: `2568f97f39ec5e31085ba0da3c2160ceb1c1431ac4ea705caa61cef905931890`.
Private volume: `qa-state-evups7zz5rpdccc3udndyilaz2y4cqy2ytvhaxfkmhhpsbmtdcia`.

Validation: 43 tests passed across `tests.test_aligned_live_controls` and `tests.test_reward_alignment`, including nonfinite/missing numeric rejection, correct-answer false penalties, missing variants, unsupported-versus-contradiction distinction, and complete repair-call budgeting. Worker/launcher Python syntax checked; all 56 submissions pass the submission schema and source/membership binding checks.
