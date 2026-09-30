# Completed seed 43/44 allocation retirement — prepared, not executed

No remote calls, retirement-script execution, remote mutation, frozen budget edits or checkpoint export were performed for this preparation. The script was statically parsed with `ast.parse`; its local archived ledger amounts and hashes were checked independently. Runtime controller status and live-ledger equality remain checks for the parent to perform after review.

| Private ledger | Archived cap | Proposed retired cap | Allocation released | Reservations preserved |
|---|---:|---:|---:|---:|
| Direct seed 43 | $1,800 | $364.8522665304345 | $1,435.1477334695655 | 10,316 |
| Direct seed 44 | $1,800 | $343.0981421804351 | $1,456.9018578195648 | 9,981 |
| Total | $3,600 | $707.9504087108696 | **$2,892.04959128913** | 20,297 |

These are allocations and reservations, not refunds or provider invoices. The $6,000 project ceiling and separate $100 control-evaluation allocation remain untouched. Releasing an unused ceiling does not authorize or launch another experiment.

## Prepared implementation

`scripts/retire_completed_direct_allocations.py` accepts one seed per invocation. It follows the successful seed-42 procedure, with stricter byte equality and explicit exclusivity guards:

1. Require a completed archive audit with zero JSON errors and all hashes reverified; verify the ledger and terminal event-log hashes against that audit.
2. Require the exact archived prior cap of $1,800 and exact authoritative reservations above. The only ledger changes are `cap` and one appended `allocation_history` entry; all reservation records, existing history, prices, TTL and other fields are preserved.
3. Discover all local submission receipts pointing to the private volume, requiring the known original/continuation receipts. Poll every known controller and fail if any is active or cannot be inspected. Seed 43 requires original `sb-ykJIAn3ZLVMq24UbMn2ywa` and continuation `sb-wgP6F2auiBuHkd9SVCLTpD`; seed 44 requires `sb-Pd8ZzwoRRi4ms3zMuQ5uYE`.
4. Require the live ledger bytes to exactly equal the verified archived bytes. A JSON-equivalent reserialization also fails; reconcile it explicitly rather than silently overwrite.
5. Default invocation reports a read-only dry-run audit. Applying additionally requires `--exclusive-volume-control`, no existing local retirement audit, a second terminal-controller check and unchanged live ledger bytes.
6. Save original ledger bytes locally with exclusive file creation, replace the private ledger and add a remote retirement audit, then verify exact ledger and audit readback. Only after successful readback write the local completed audit and after-ledger snapshot. An interrupted/failed apply is not replayed automatically; the saved before file intentionally blocks a blind repeat.

Volume upload is not a transactional compare-and-swap. Polling known receipts cannot detect an unregistered writer. The parent must coordinate exclusive ownership of each private volume and inspect any failed or changed precondition. The separate controls may continue only if they do not write these private ledgers/volumes.

Prepared commands, **not executed**:

```sh
.venv-eval/bin/python scripts/retire_completed_direct_allocations.py --seed 43
.venv-eval/bin/python scripts/retire_completed_direct_allocations.py --seed 44
```

After parent review, terminal checks and exclusive coordination, apply each separately:

```sh
.venv-eval/bin/python scripts/retire_completed_direct_allocations.py --seed 43 --apply --exclusive-volume-control
.venv-eval/bin/python scripts/retire_completed_direct_allocations.py --seed 44 --apply --exclusive-volume-control
```

Expected local audit paths are `reports/expanded-studies/seed43-allocation-retirement.json` and `seed44-allocation-retirement.json`, with before/after ledger snapshots alongside them. Exact archive paths and SHA-256 values are recorded in `seed43-seed44-retirement-plan.json`.

## Fixed-final checkpoint export plan — no trainer or optimizer

An existing sampler checkpoint can be downloaded through the installed Tinker REST interface without creating a trainer, loading weights, sampling, updating an optimizer or saving another checkpoint. The local SDK exposes `ServiceClient.create_rest_client()` followed by `get_checkpoint_archive_url_from_tinker_path(existing_sampler_path)`. The latter calls the archive-URL endpoint for an existing model/checkpoint and returns a signed URL. It may assemble an export server-side; it is not a pure static HTTP file lookup. Existence and expiration still require a live read request, which has not been made here.

Use each **fixed-final** sampler path from its verified manifest, never `best.json`:

- Seed 42: `tinker://33ea3593-23f7-5639-b2d5-ae69e5bb0249:train:1/sampler_weights/ckpt-fc0b245037464f3a937d57c737deb7bc`
- Seed 43: `tinker://795b8b20-8f3d-5a81-b3ac-c5ab312307a2:train:0/sampler_weights/ckpt-526b475d98334d8585e99d96125eaa38` — resolve the exact path from the archived manifest at execution; the manifest is authoritative over prose.
- Seed 44: `tinker://3eb58ae5-f25f-5fc6-bc8f-21514da03a24:train:1/sampler_weights/ckpt-3eb1a5bfe97745b9ba9b3a691b242e1b` — likewise use the manifest verbatim.

Before execution, read the final checkpoint's manifest and evaluation, assert matching checkpoint IDs and step 32, and use `artifacts.sampler` exactly. Stream the signed URL to a new `.pending` file with an 8-GiB bound, require nonzero bytes, flush/fsync, calculate SHA-256, then rename atomically. Inspect tar members without extracting: reject absolute/traversal paths and link members, and require adapter configuration, adapter weights and a completion marker. Record final checkpoint ID, manifest hash, sampler URI, bytes, hash and validation results in a separate export receipt. Do not log the signed URL. Do not claim a Tinker reload or optimizer restoration based only on archive structure.

Do **not** call `TinkerBackend.archive()` or `training_pipeline.archive_validation.worker`: the former extends TTL, reserves checkpoint spend, creates a fresh trainer and loads state; the latter also creates a trainer. Those are outside this download-only plan. Do not extend retention implicitly. If the fixed-final checkpoint has expired, report it and stop; do not recreate it by replaying training. The repository records that training-state/optimizer archive export is unsupported by the live service; plan only sampler-weight export, with no promise of optimizer preservation or Tinker archive reimport.

Seed 44's existing verified local sampler export is step 16's best eligible checkpoint, not its fixed final. A successful download of the exact step-32 sampler would close that distinct gap without changing results, optimization or the private budget ledger.
