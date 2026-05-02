import { describe, expect, it } from 'vitest';

import { normalizeApiBaseOrigin } from './client';

describe('normalizeApiBaseOrigin', () => {
  it('strips trailing slashes and an existing /api suffix', () => {
    expect(normalizeApiBaseOrigin('https://example.com/api/')).toBe('https://example.com');
    expect(normalizeApiBaseOrigin('https://example.com///')).toBe('https://example.com');
    expect(normalizeApiBaseOrigin(' https://example.com/base/api ')).toBe('https://example.com/base');
  });
});
