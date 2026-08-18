import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

import en from '../en/translation.json';
import zh from './translation.json';

type Bundle = Record<string, unknown>;

const flatten = (bundle: Bundle, prefix = ''): string[] =>
  Object.entries(bundle).flatMap(([key, value]) =>
    value !== null && typeof value === 'object'
      ? flatten(value as Bundle, `${prefix}${key}.`)
      : [`${prefix}${key}`]
  );

// Vitest runs with `frontend/` as the working directory. `import.meta.url` is
// not a file: URL under the jsdom environment, so it cannot be used here.
const SRC_ROOT = join(process.cwd(), 'src');

const collectSourceFiles = (dir: string): string[] =>
  readdirSync(dir).flatMap((entry) => {
    const full = join(dir, entry);
    if (statSync(full).isDirectory()) {
      return entry === '__tests__' || entry === 'test' ? [] : collectSourceFiles(full);
    }
    if (!/\.tsx?$/.test(entry) || /\.test\.tsx?$/.test(entry)) return [];
    return [full];
  });

// Matches t('some.key') and t('some.key', ...) but not t(variable).
const KEY_PATTERN = /\bt\(\s*['"]([a-zA-Z0-9_.]+)['"]/g;

const collectUsedKeys = (): Set<string> => {
  const keys = new Set<string>();
  for (const file of collectSourceFiles(SRC_ROOT)) {
    const source = readFileSync(file, 'utf8');
    for (const match of source.matchAll(KEY_PATTERN)) {
      keys.add(match[1]);
    }
  }
  return keys;
};

describe('zh translation bundle', () => {
  it('keeps key interface labels readable', () => {
    expect(zh.sidebar.dashboard).toBe('仪表盘');
    expect(zh.settings.title).toBe('设置');
    expect(zh.app.login).toBe('登录');
  });
});

describe('translation bundle parity', () => {
  it('has identical key sets in en and zh', () => {
    const enKeys = new Set(flatten(en as Bundle));
    const zhKeys = new Set(flatten(zh as Bundle));

    const missingFromZh = [...enKeys].filter((key) => !zhKeys.has(key)).sort();
    const missingFromEn = [...zhKeys].filter((key) => !enKeys.has(key)).sort();

    // Named explicitly so a failure does not cost a manual diff of two bundles.
    expect({ missingFromZh, missingFromEn }).toEqual({
      missingFromZh: [],
      missingFromEn: [],
    });
  });

  it('defines every key the application actually calls', () => {
    // Inline `t('key', 'Fallback')` defaults hide missing keys in English while
    // silently rendering English text in the Chinese UI. This is the guard.
    const used = collectUsedKeys();
    const enKeys = new Set(flatten(en as Bundle));
    const zhKeys = new Set(flatten(zh as Bundle));

    const missingFromEn = [...used].filter((key) => !enKeys.has(key)).sort();
    const missingFromZh = [...used].filter((key) => !zhKeys.has(key)).sort();

    expect({ missingFromEn, missingFromZh }).toEqual({
      missingFromEn: [],
      missingFromZh: [],
    });
  });

  it('finds a meaningful number of keys, so the scanner cannot silently pass', () => {
    expect(collectUsedKeys().size).toBeGreaterThan(150);
  });
});
