import type { AssertionGrade, Comparison } from './types';
export function scoreLabel(grade?: AssertionGrade, status?: string) {
  if (grade?.status === 'resolved' && grade.score != null) return `${(grade.score * 100).toFixed(1)} / 100`;
  if (grade?.status === 'unresolved') return 'Unresolved';
  if (grade?.status === 'not_scored') return 'No answer';
  if (grade?.status === 'failed') return 'Unavailable';
  return status === 'running' ? 'Grading…' : status === 'pending' ? 'Awaiting answers' : 'Not scored';
}
export default function ComparisonScore({ pair, openEvidence }: { pair: Comparison; openEvidence: (path: string, start: number, end: number) => void }) {
  const benchmark = pair.benchmark;
  if (!benchmark) return <div className="score-unavailable">Not scored · {pair.preview ? 'Scripted preview' : 'Free-form question without a saved answer rubric'}. Start a known-answer comparison for assertion scores.</div>;
  const grading = pair.grading;
  return <section className="assertion-score" aria-label="Assertion rubric scores">
    <div className="score-heading"><div><span className="eyebrow">KNOWN-ANSWER BENCHMARK</span><h2>Assertion coverage</h2></div><span>{benchmark.claim_count} assertions · {benchmark.reference_status}{!benchmark.human_reviewed ? ' · not human-reviewed' : ''}</span></div>
    <p>Both answers are graded against the same frozen assertions and pinned source. Supported = 1, partial = ½, missing or contradicted = 0. Weighted coverage is shown out of 100; it is not a full audit of every extra claim.</p>
    <div className="score-totals">{pair.runs.map(run => <div key={run.id}><span>{run.model.name}</span><strong>{scoreLabel(grading?.scores[run.id], grading?.status)}</strong><small>{grading?.scores[run.id]?.reason}</small></div>)}</div>
    {grading?.error && <p role="alert">{grading.error}</p>}
    <div className="score-table-scroll"><table><thead><tr><th>Expected assertion</th>{pair.runs.map(r => <th key={r.id}>{r.model.name}</th>)}</tr></thead><tbody>
      {benchmark.rubric?.claims.map(claim => <tr key={claim.id}><th><strong>{claim.text}</strong><small>Weight {claim.weight}</small>{claim.evidence_ids.map(id => { const e = benchmark.rubric?.evidence.find(e=>e.id===id); return e ? <button className="score-source" key={id} onClick={()=>openEvidence(e.path,e.start_line,e.end_line)}>{e.path}:{e.start_line}–{e.end_line}</button> : null; })}</th>
        {pair.runs.map(run => { const grade = grading?.scores[run.id]; const result = grade?.claims?.find(c=>c.id===claim.id); return <td key={run.id}>{result ? <><span className={`verdict verdict-${result.verdict}`}>{result.verdict}{result.credit != null ? ` · ${result.credit * claim.weight}/${claim.weight}` : ''}</span><p>{result.reason}</p>{result.answer_quotes?.map((quote,i)=><blockquote key={i}>{quote}</blockquote>)}</> : <span>{grade?.reason || scoreLabel(grade,grading?.status)}</span>}</td>; })}
      </tr>)}
    </tbody></table></div>
    <details className="reference-answer"><summary>Reference answer and rubric provenance</summary><p className="reference-text">{benchmark.reference_answer}</p><small>Task: {benchmark.id} · Rubric: {benchmark.rubric?.rubric_hash.slice(0,12)}. Reference answers are withheld from the answering models.</small></details>
    <p className="judge-caption">Judge: {grading?.judge_model || 'Qwen/Qwen3.5-397B-A17B'} · temperature 0 · candidate identities withheld · same settings for both answers.{grading?.estimated_cost_usd != null ? ` Estimated grading token cost: $${grading.estimated_cost_usd.toFixed(5)} (separate from run costs).` : ''}</p>
  </section>;
}
