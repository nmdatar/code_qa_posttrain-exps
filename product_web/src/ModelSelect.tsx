import { useId, useState } from 'react';
import type { Model } from './types';

export default function ModelSelect({ label, value, models, disabled, onChange }: {
  label: string; value: string; models: Model[]; disabled: boolean;
  onChange: (value: string) => void;
}) {
  const [query, setQuery] = useState('');
  const hintId = useId();
  const words = query.trim().toLowerCase().split(/\s+/).filter(Boolean);
  const filtered = models.filter(model => {
    const text = [model.name, model.base_model, model.run, model.model_path, model.id,
      model.step == null ? '' : `step ${model.step}`].join(' ').toLowerCase();
    return words.every(word => text.includes(word));
  });
  const selected = models.find(model => model.id === value);
  const visibleValue = filtered.some(model => model.id === value) ? value : '';
  return <div className="searchable-model">
    <input type="search" aria-label={`Search ${label}`} placeholder="Search name, checkpoint ID, or run…"
      value={query} disabled={disabled} onChange={event => setQuery(event.target.value)}
      aria-describedby={hintId} />
    <select aria-label={label} value={visibleValue} disabled={disabled || !filtered.length}
      onChange={event => { onChange(event.target.value); setQuery(''); }}>
      <option value="" disabled>{filtered.length ? 'Choose a model' : 'No matching models'}</option>
      {(['base', 'checkpoint', 'preview'] as const).map(kind => {
        const items = filtered.filter(model => model.kind === kind);
        if (!items.length) return null;
        return <optgroup key={kind} label={kind === 'base' ? 'Tinker hosted models' : kind === 'preview' ? 'Scripted demo' : 'Project checkpoints · newest first'}>
          {items.map(model => <option key={model.id} value={model.id} disabled={!model.ready}>
            {model.name}{model.kind === 'checkpoint' && model.run && !model.name.includes(model.run) ? ` · ${model.run}${model.step == null ? '' : ` · step ${model.step}`}` : ''}{model.ready ? '' : ' · unavailable'}
          </option>)}
        </optgroup>;
      })}
    </select>
    <small id={hintId} role="status">{query.trim() ? `${filtered.length} match${filtered.length === 1 ? '' : 'es'}${selected && !visibleValue ? ` · Selected: ${selected.name}` : ''}` : 'Search by model name, checkpoint ID, or run'}</small>
  </div>;
}
