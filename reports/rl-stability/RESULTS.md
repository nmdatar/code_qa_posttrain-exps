# Autoresearch stopped: parallel command transport limit

The control completed eight batches and five acknowledged optimizer updates. The controller then stopped because21/64training trajectories had infrastructure errors. Four candidate arms never started. No incumbent was selected or optimizer call replayed; confirmation stayed untouched. Exit0 means the controller handled the stop, not that the experiment succeeded.

## Reproduced cause

The new parallel dispatcher passes repository catalogs in argv through Modal exec. The SDK rejects total argument lengths above65,536characters. All31failed commands exceeded the limit (77,917–282,355characters) and reproduced the saved InvalidError using the SDK's local validator, without dispatching anything. They affected21training trajectories; a batch can have multiple failed commands. This is a defect in the new parallel transport, not evidence against an LR or RL strategy. The serial persistent-stream path uses a different transport. Earlier mocked concurrency tests missed realistic large catalogs.

Before relaunch: move catalogs out of argv into stdin or sandbox-local files, add a realistic >64KiB transport regression and validate the real remote path. The monitor made no frozen-code change and launched no additional paid work.

## Results

| Control metric | Initial | Final |
|---|---:|---:|
| Strict full passes /32 |5|5|
| Demonstrated strict reward |15.625%|15.625%|
| Resolved grades |30/32|29/32|
| Completed answers |31/32|29/32|

Neither evaluation met95%coverage. Unknown grades remain unknown. No measured improvement or candidate comparison is available. Mean training reward was0.219512 over41resolved trajectories;23were unresolved, including21infrastructure failures. Seventeen trajectories contributed to updates;23format failures were recorded. These changing-task training rewards do not establish generalization.

Final checkpoint: ckpt-9a6082178e4e470db202f8cc986ad253, optimizer step5, attempted batches8. Nine checkpoint manifests were archived, including skipped-batch boundaries. Routine remote retention is48hours from save; neither evaluation qualified for extended best retention. Manifests are not permanent weight/optimizer exports.

Final ledger reservations: **$47.3230127616/$1000**, not provider invoices. All861files were downloaded, JSON-checked where applicable, checksummed and reread to verify SHA-256. Archive: artifacts/rl-stability-v1-results. Evidence: archive.json, archive-checksums.json, results.json. No extra model calls, grading, training or credit purchases occurred during this heartbeat.

[Control W&B](https://wandb.ai/nmdatar-harvard-university/repository-qa-training/runs/stability-auto-v1-control-seed42)

Automation monitor-parallel-rl-autoresearch is **PAUSED**. No relaunch is authorized by these monitoring instructions.
