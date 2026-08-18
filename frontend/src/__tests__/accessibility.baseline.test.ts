import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

/**
 * Static accessibility baseline.
 *
 * These are structural guards, not a substitute for an audit: automated rules
 * catch roughly a third of real accessibility problems. Deliberately out of
 * scope here are colour-contrast checking, screen-reader transcripts, and the
 * `react-force-graph` canvas, which has no accessible node tree at all.
 */

const SRC_ROOT = join(process.cwd(), 'src');

const collectComponentFiles = (dir: string): string[] =>
  readdirSync(dir).flatMap((entry) => {
    const full = join(dir, entry);
    if (statSync(full).isDirectory()) {
      return entry === '__tests__' || entry === 'test' ? [] : collectComponentFiles(full);
    }
    if (!entry.endsWith('.tsx') || entry.includes('.test.')) return [];
    return [full];
  });

const BUTTON_PATTERN = /<button\b([^>]*?)>([\s\S]*?)<\/button>/g;

const hasAccessibleName = (attrs: string, body: string): boolean => {
  if (/aria-label[=\s]/.test(attrs) || /aria-labelledby[=\s]/.test(attrs)) return true;
  if (/\btitle=/.test(attrs)) return true;
  // Any content left after stripping nested JSX elements counts as a text label.
  const textOnly = body.replace(/<[^>]+>/g, '').replace(/\s+/g, '');
  return textOnly.length > 0;
};

describe('accessibility baseline', () => {
  const files = collectComponentFiles(SRC_ROOT);

  it('scans a meaningful number of components', () => {
    // Guards the guard: a broken walker must not report a vacuous pass.
    expect(files.length).toBeGreaterThan(15);
  });

  it('gives every button an accessible name', () => {
    const offenders: string[] = [];

    for (const file of files) {
      const source = readFileSync(file, 'utf8');
      for (const match of source.matchAll(BUTTON_PATTERN)) {
        if (!hasAccessibleName(match[1], match[2])) {
          const line = source.slice(0, match.index).split('\n').length;
          offenders.push(`${file.replace(`${process.cwd()}/`, '')}:${line}`);
        }
      }
    }

    // Icon-only buttons need aria-label; anything with visible text is fine.
    expect(offenders).toEqual([]);
  });

  it('gives every modal surface dialog semantics', () => {
    const modals = files.filter((file) => readFileSync(file, 'utf8').includes('aria-modal'));
    expect(modals.length).toBeGreaterThan(0);

    for (const file of modals) {
      const source = readFileSync(file, 'utf8');
      const relative = file.replace(`${process.cwd()}/`, '');
      expect(source, `${relative} declares aria-modal without role="dialog"`).toMatch(
        /role="dialog"/
      );
      expect(
        /aria-labelledby=/.test(source) || /aria-label=/.test(source),
        `${relative} is a modal with no accessible name`
      ).toBe(true);
      expect(source, `${relative} is a modal that cannot be dismissed with Escape`).toMatch(
        /'Escape'/
      );
      // Focus must be trapped, or Tab walks into the inert page behind. A
      // dialog may either implement the trap inline or delegate to the shared
      // `useDialogFocusTrap` hook, whose behaviour is covered by
      // src/components/ui/hooks/__tests__/useDialogFocusTrap.test.tsx.
      const trapsFocus = /'Tab'/.test(source) || /useDialogFocusTrap/.test(source);
      expect(trapsFocus, `${relative} is a modal with no Tab focus trap`).toBe(true);
    }
  });
});
