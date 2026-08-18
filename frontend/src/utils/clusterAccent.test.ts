import { describe, expect, it } from 'vitest';

import { contrastOnLightSurface, getClusterAccent } from './clusterAccent';

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

describe('cluster accent text contrast', () => {
  // Chips render on the cream `--bg-main` surface. The previous implementation
  // used the raw hue at 88% alpha, which put bright accents at ~2.4:1 and made
  // every chip on the Data page fail WCAG AA.
  const HUES = ['#10b981', '#3b82f6', '#ef4444', '#eab308', '#22d3ee', '#a3e635'];

  it.each(HUES)('meets the AA 4.5:1 floor for %s', (hue) => {
    const { text } = getClusterAccent({ id: 1, color: hue });
    expect(contrastOnLightSurface(text)).toBeGreaterThanOrEqual(4.5);
  });

  it('keeps an already-dark hue untouched', () => {
    const { text } = getClusterAccent({ id: 1, color: '#0F766E' });
    expect(text.toLowerCase()).toBe('#0f766e');
  });

  it('leaves every fallback palette colour readable', () => {
    for (let id = 0; id < 8; id += 1) {
      const { text } = getClusterAccent({ id, color: null });
      expect(contrastOnLightSurface(text)).toBeGreaterThanOrEqual(4.5);
    }
  });

  it('still exposes the undarkened hue for dots and borders', () => {
    const accent = getClusterAccent({ id: 1, color: '#10b981' });
    expect(accent.dot).toBe('#10b981');
    expect(accent.base).toBe('#10b981');
  });
});
