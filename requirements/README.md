# Training and evaluation requirements

Design baseline v0.1 · 2026-09-28 · **Original design baseline; machinery implementation now available.**

This package specifies a reusable Tinker-backed SFT/GRPO framework with repository Q&A as its first environment. It adds to the existing dataset roadmap without changing its contents. The original design deliverable included no training. The subsequent implementation and live toy validation are documented in [the machinery report](../reports/training-pipeline-smoke.md).

## Review in this order

| View | Editable Excalidraw scene | Local editable backup |
|---|---|---|
| 01 — System overview | [Open overview](https://app.excalidraw.com/s/8Ufs2ZMhWhu/3eHAQBP8Lwr) | [system-overview.excalidraw](diagrams/system-overview.excalidraw) |
| 02 — GRPO loop | [Open GRPO loop](https://app.excalidraw.com/s/8Ufs2ZMhWhu/3G3d4F8XY6y) | [grpo-loop.excalidraw](diagrams/grpo-loop.excalidraw) |
| 03 — Checkpoint lifecycle | [Open checkpoint lifecycle](https://app.excalidraw.com/s/8Ufs2ZMhWhu/6gLedvh7Ud8) | [checkpoint-lifecycle.excalidraw](diagrams/checkpoint-lifecycle.excalidraw) |

Scenes were created in the connected private collection. Access follows the existing Excalidraw account permissions; these are editable scene links, not newly published public shares. Import the `.excalidraw` files into Excalidraw to work from the local copies. Scene IDs and export versions are recorded in [the diagram manifest](diagrams/manifest.json).

Blue shapes denote framework responsibilities, green denotes data/artifacts, and pink denotes external services. Solid arrows show initial-scope relationships; the dashed extension arrow denotes future strategy plugins. Diagrams describe logical responsibilities rather than deployment services. SFT, GRPO and eval all emit tracking events even where the overview omits repeated arrows. Sampling and update calls both pass through the Tinker backend. The shared rollout/environment mechanics serve both GRPO and evaluation.

## Specification

- [Architecture](TRAINING_ARCHITECTURE.md): ownership, end-to-end flows, compatibility, recovery and proposed defaults.
- [Interface contracts](TRAINING_INTERFACES.md): model, data, strategy, environment, verifier, checkpoint, evaluation and tracking boundaries.
- [Acceptance scenarios](TRAINING_ACCEPTANCE.md): observable conditions for the eventual implementation, separate from checks performed on this design package.
- [Decision log](TRAINING_DECISIONS.md): user-confirmed scope, proposed technical defaults and questions for the next review.
- [Dataset roadmap](DATASET_ROADMAP.md): existing upstream dataset plan and neutral release records.

## Iteration process

Review the overview first: component ownership, optional SFT, direct GRPO, and extension boundaries. Next inspect the GRPO loop: policy versions, tool isolation, rewards and token attribution. Finally review checkpoint publication and the three consumer operations.

Record changes in the decision log, revise the relevant specification and scene together, inspect the scene screenshot, and re-export its local copy and manifest. Manual changes to a remote scene do not automatically update local files. Do not start trainer implementation merely because the design artifacts exist; complete the requested design review first.

Validation for this package checks diagram readability, valid Excalidraw JSON, bound-arrow references, required views, and local document links. The training acceptance scenarios are requirements, not tests that have already passed.
