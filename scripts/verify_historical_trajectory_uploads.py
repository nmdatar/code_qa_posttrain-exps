import json
from pathlib import Path
import wandb
root=Path('reports/trajectory-history');receipts=json.loads((root/'wandb-upload.json').read_text())
api=wandb.Api(timeout=60);results=json.loads((root/'wandb-verification.json').read_text()) if (root/'wandb-verification.json').exists() else []
verified={r['run'] for r in results}
for ident,receipt in receipts.items():
 if ident in verified:continue
 run=api.run('nmdatar-harvard-university/repository-qa-training/'+ident)
 names={a.qualified_name for a in run.logged_artifacts() if a.type=='trajectory-inspection'}
 art=api.artifact(receipt['artifact'])
 entries=set(art.manifest.entries)
 assert receipt['artifact'] in names,ident
 assert {'tool_calls.table.json','trajectories.table.json'} <= entries,ident
 summary_link_present=run.summary.get('trajectory_inspection_url')==receipt['url']
 if receipt['original_state'] != 'running':assert summary_link_present,ident
 results.append({'run':ident,'attached':True,'native_tables':True,'state':run.state,'original_state':receipt['original_state'],'summary_link_present':summary_link_present})
(root/'wandb-verification.json').write_text(json.dumps(results,indent=2))
print('Verified',len(results),'run attachments and native table pairs')
