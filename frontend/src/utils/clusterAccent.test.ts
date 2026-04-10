import { describe, expect, it } from 'vitest';

import { getClusterAccent } from './clusterAccent';

describe('getClusterAccent', () => {
  it('uses the provided cluster color when available', () => {
    const accent = getClusterAccent({ id: 3, color: '#ff6600' });

    expect(accent.base).toBe('#ff6600');
    expect(accent.dot).toBe('#ff6600');
  });

  it('creates deterministic non-default fallback colors for clusters without a color', () => {
    const first = getClusterAccent({ id: 1, color: null });
    const second = getClusterAccent({ id: 2, color: null });

    expect(first.base).not.toBe('#6B7280');
    expect(second.base).not.toBe('#6B7280');
    expect(first.base).not.toBe(second.base);
    expect(getClusterAccent({ id: 1, color: null }).base).toBe(first.base);
  });
});
