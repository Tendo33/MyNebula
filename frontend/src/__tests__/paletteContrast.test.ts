import { describe, expect, it } from 'vitest';

import { readFileSync } from 'node:fs';
import { join } from 'node:path';

/**
 * Contrast contract for the colour tokens.
 *
 * The palette carries three text tiers over three very light surfaces, and the
 * surfaces sit within a narrow luminance band. That leaves little headroom, so
 * each tier has an explicit role and an explicit floor rather than a vague
 * "lighter than the last one":
 *
 *   text-main   primary text                       >= 4.5 (WCAG 1.4.3 AA)
 *   text-muted  secondary text that carries info   >= 4.5
 *   text-dim    disabled controls and decorative
 *               icons only                         >= 3.0 (WCAG 1.4.11)
 *
 * `text-dim` was previously #8A92A0, which is 2.52:1 on the hover surface — it
 * failed even the non-text floor, while being used for repo counts, cluster
 * counts and slider values. Those moved to `text-muted`; the token now only
 * covers the WCAG-exempt disabled case and decoration.
 */

/**
 * Read the tokens out of `index.css`.
 *
 * That file is the source of truth now: `tailwind.config.js` only holds
 * `var(--color-…)` references, so parsing it would yield no hex at all. The
 * "actually parsed" test below exists because a silently-empty parse would make
 * every assertion here pass vacuously.
 */
const readThemeTokens = (): Record<'light' | 'dark', Record<string, string>> => {
  const source = readFileSync(join(process.cwd(), 'src/index.css'), 'utf8');
  // Both `:root` and `html.dark` appear more than once in the file, so collect
  // from every matching block rather than the first one.
  const grab = (selector: string) => {
    const out: Record<string, string> = {};
    let cursor = 0;
    for (;;) {
      const start = source.indexOf(selector, cursor);
      if (start === -1) break;
      const end = source.indexOf('}', start);
      for (const [, name, value] of source
        .slice(start, end)
        .matchAll(/--color-([\w-]+):\s*(#[0-9a-fA-F]{6})/g)) {
        out[name] = value;
      }
      cursor = end + 1;
    }
    return out;
  };
  return { light: grab(':root {'), dark: grab('html.dark {') };
};

const themes = readThemeTokens();

const relativeLuminance = (hex: string): number => {
  const normalized = hex.replace('#', '');
  const channels = [0, 2, 4].map((i) => Number.parseInt(normalized.slice(i, i + 2), 16) / 255);
  const linear = channels.map((c) => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4));
  return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2];
};

export const contrastRatio = (a: string, b: string): number => {
  const l1 = relativeLuminance(a);
  const l2 = relativeLuminance(b);
  return (Math.max(l1, l2) + 0.05) / (Math.min(l1, l2) + 0.05);
};

const SURFACE_KEYS = ['bg-main', 'bg-sidebar', 'bg-hover'] as const;
const TEXT_TIERS = [
  ['text-main', 4.5],
  ['text-muted', 4.5],
  ['text-dim', 3.0],
] as const;

for (const theme of ['light', 'dark'] as const) {
  describe(`${theme} palette contrast contract`, () => {
    const tokens = themes[theme];

    it('actually parsed the tokens', () => {
      for (const key of [...SURFACE_KEYS, 'text-main', 'text-muted', 'text-dim', 'action-primary', 'on-action']) {
        expect(tokens[key], `missing --color-${key} in ${theme}`).toMatch(/^#[0-9a-fA-F]{6}$/);
      }
    });

    for (const [tier, floor] of TEXT_TIERS) {
      for (const surface of SURFACE_KEYS) {
        it(`${tier} on ${surface} clears ${floor}:1`, () => {
          expect(contrastRatio(tokens[tier], tokens[surface])).toBeGreaterThanOrEqual(floor);
        });
      }
    }

    it('keeps the tiers visually ordered', () => {
      const onMain = (hex: string) => contrastRatio(hex, tokens['bg-main']);
      expect(onMain(tokens['text-main'])).toBeGreaterThan(onMain(tokens['text-muted']));
      expect(onMain(tokens['text-muted'])).toBeGreaterThan(onMain(tokens['text-dim']));
    });

    it('keeps text on the action colour readable', () => {
      // The skip link and primary buttons put `on-action` on `action-primary`.
      // White works in light but drops to 2.56:1 on the lightened dark accent.
      expect(contrastRatio(tokens['on-action'], tokens['action-primary'])).toBeGreaterThanOrEqual(4.5);
    });
  });
}
