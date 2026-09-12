import { describe, expect, it, vi } from 'vitest';

import { runThemeViewTransition } from './themeTransition';

describe('runThemeViewTransition', () => {
  it('applies the update immediately when motion is reduced', () => {
    const update = vi.fn();
    const matchMedia = vi.fn().mockReturnValue({ matches: true });
    vi.stubGlobal('matchMedia', matchMedia);

    runThemeViewTransition({ clientX: 10, clientY: 10 }, update);

    expect(update).toHaveBeenCalledOnce();
    vi.unstubAllGlobals();
  });

  it('applies the update immediately when no origin is given', () => {
    const update = vi.fn();
    runThemeViewTransition(undefined, update);
    expect(update).toHaveBeenCalledOnce();
  });
});
