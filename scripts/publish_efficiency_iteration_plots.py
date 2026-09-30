"""Append only aggregate figures to the completed comparison; no training."""
import json
from pathlib import Path
import wandb
OUT=Path('reports/efficiency-rl-v1')
with wandb.init(entity='nmdatar-harvard-university',project='repository-qa-training',
                id='efficiency-rl-v1-comparison',resume='must',dir=str(OUT),
                settings=wandb.Settings(disable_git=True)) as run:
    run.log({'iteration_view/01_training_progress':wandb.Image(str(OUT/'iteration-overview.png'),
                 caption='16 batches per arm. Raw batch means plus trailing 4-batch means; cumulative acknowledged updates. Task mix changes.'),
             'iteration_view/02_heldout_endpoints':wandb.Image(str(OUT/'heldout-endpoints.png'),
                 caption='Only initial and final held-out evaluations exist. Intermediate checkpoint quality was not measured.')})
    run.summary['iteration_view_description']='One seed, two arms, 16 attempted batches each; 10 control updates and 9 efficiency updates. No new training or evaluation. Bold lines are trailing 4-batch means; raw data remain visible.'
    artifact=wandb.Artifact('efficiency-rl-v1-iteration-figures',type='aggregate-figures')
    for name in ['iteration-overview.png','iteration-overview.svg','heldout-endpoints.png','heldout-endpoints.svg','iteration-aggregates.json']:
        artifact.add_file(str(OUT/name),name=name)
    run.log_artifact(artifact).wait()
    receipt={'url':run.url,'artifact':artifact.qualified_name,'new_training':False}
(OUT/'iteration-plots-upload.json').write_text(json.dumps(receipt,indent=2))
api=wandb.Api(timeout=30)
r=api.run('nmdatar-harvard-university/repository-qa-training/efficiency-rl-v1-comparison')
s=dict(r.summary)
assert r.state=='finished'
assert 'iteration_view/01_training_progress' in s
assert 'iteration_view/02_heldout_endpoints' in s
assert s['decision']=='no_demonstrated_efficiency_win'
print(json.dumps({'url':r.url,'verified':True,'state':r.state}))
