"""Execute in the trusted controller, using its existing W&B credentials."""
import json
from pathlib import Path
import wandb_workspaces.reports.v2 as wr

RUN = 'structured-correctness-grpo-15-v1'
ROOT = Path('/state/artifacts/experiments') / RUN
entity = json.loads((ROOT/'tracking-url.json').read_text())['url'].split('/')[3]
project = 'repository-qa-training'
runset = wr.Runset(entity=entity, project=project, name=RUN,
                  filters=f"Config('run_id') == '{RUN}'")

def plot(title, x, *ys):
    return wr.LinePlot(title=title, x=x, y=list(ys))

sections = [
 ('Correctness and citation diagnostics', [
  plot('Training correctness reward', 'attempted_batches', 'training/mean_correctness_score', 'training/reward_ema'),
  plot('Fixed-cohort evaluation correctness reward', 'optimizer_step', 'evaluation/mean_correctness_score'),
  plot('Training citation score (diagnostic only)', 'attempted_batches', 'training/mean_citation_score'),
  plot('Evaluation citation score (diagnostic only)', 'optimizer_step', 'evaluation/mean_citation_score'),
 ]),
 ('Tool use and rollout latency', [
  plot('Training tool execution seconds per attempt', 'attempted_batches', 'training/mean_tool_seconds'),
  plot('Evaluation tool execution seconds per attempt', 'optimizer_step', 'evaluation/mean_tool_seconds'),
  plot('Training tool calls per attempt', 'attempted_batches', 'training/mean_tool_calls'),
  plot('Evaluation tool calls per attempt', 'optimizer_step', 'evaluation/mean_tool_calls'),
  plot('Training rollout latency seconds', 'attempted_batches', 'training/mean_latency_seconds'),
  plot('Evaluation rollout latency seconds', 'optimizer_step', 'evaluation/mean_latency_seconds'),
 ]),
 ('Iteration and evaluation wall time', [
  plot('Training batch wall seconds (excludes evaluation)', 'attempted_batches', 'training/batch_seconds'),
  plot('Batch timing breakdown in seconds', 'attempted_batches', 'training/collection_and_grading_seconds', 'training/update_seconds', 'training/checkpoint_seconds'),
  plot('Evaluation wall seconds', 'optimizer_step', 'evaluation/wall_seconds'),
 ]),
 ('Scoring health and sampling cost', [
  plot('Training scoring coverage', 'attempted_batches', 'training/scoring_coverage'),
  plot('Evaluation scoring coverage', 'optimizer_step', 'evaluation/scoring_coverage'),
  plot('Excluded and zero-variance groups', 'attempted_batches', 'training/excluded_groups', 'training/zero_variance_groups'),
  plot('Contributing training trajectories', 'attempted_batches', 'training/contributing_trajectories'),
  plot('Training tokens per attempt', 'attempted_batches', 'training/mean_input_tokens', 'training/mean_output_tokens'),
  plot('Evaluation tokens per attempt', 'optimizer_step', 'evaluation/mean_input_tokens', 'evaluation/mean_output_tokens'),
  plot('Tool timing measurement coverage', 'attempted_batches', 'training/tool_seconds_measurement_coverage'),
 ])]
blocks = [wr.P('Live charts for 15 GRPO iterations, batch 8, group 8, Qwen3.5-4B rank-8 LoRA, LR 1e-5, seed 42. Original list_files/search_code/read_file harness. Correctness-only reward; citations are diagnostic. Fresh base model; same 32 training tasks cycle in shuffled epochs. Disjoint fixed 32-case validation at optimizer steps 0, 3, 6, 9, 12, 15. Training uses temperature 1, evaluation temperature 0, so the two reward curves are not directly comparable. Empty charts await measurements. Tool time is total execution time per attempt, not time per individual call. Batch wall time excludes periodic evaluation. This report does not change project access permissions.')]
for title, panels in sections:
    blocks += [wr.H1(title), wr.PanelGrid(runsets=[runset], panels=panels)]
report = wr.Report(entity=entity, project=project,
    title='Three-tool GRPO 15 iterations - correctness and timing', blocks=blocks).save()
result = {'report_url': report.url, 'run_url': json.loads((ROOT/'tracking-url.json').read_text())['url'],
          'panels': sum(len(p) for _,p in sections)}
(ROOT/'plots-report.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result))
