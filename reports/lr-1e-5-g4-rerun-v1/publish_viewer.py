"""Publish the readable trajectories after approval; no training/evaluation is run."""
from pathlib import Path
import wandb
from training_pipeline.trajectory_viewer import write_report, log_report
root=Path('artifacts/lr-1e-5-g4-rerun-v1-results/artifacts/experiments/lr-1e-5-group-4-v6-rerun-v1-seed42')
path,records=write_report(root, 'reports/lr-1e-5-g4-rerun-v1/trajectory-viewer.html')
run=wandb.init(entity='nmdatar-harvard-university',project='repository-qa-training',id=root.name,resume='must',dir='reports/lr-1e-5-g4-rerun-v1',settings=wandb.Settings(disable_git=True))
log_report(run,wandb,path,records)
artifact=wandb.Artifact('trajectory-viewer-'+root.name,type='trajectory-viewer')
artifact.add_file(str(path),name='trajectory-viewer.html')
run.log_artifact(artifact)
run.finish()
print('Uploaded',len(records),'episodes')
