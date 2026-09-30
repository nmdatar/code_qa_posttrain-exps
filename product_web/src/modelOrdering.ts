import type { Model } from './types';

// Keep hosted models in provider order; checkpoint creation time controls recency.
export function newestCheckpointsFirst(models: Model[]): Model[] {
  const timestamp = (model: Model) => {
    const value = Date.parse(model.created_at || '');
    return Number.isFinite(value) ? value : -Infinity;
  };
  return [
    ...models.filter(model => model.kind !== 'checkpoint'),
    ...models.filter(model => model.kind === 'checkpoint')
      .sort((a, b) => timestamp(b) - timestamp(a)),
  ];
}
