import { describe, expect, it } from 'vitest';

import {
  buildFilterParams,
  formatDataDate,
  nextSortConfig,
  parseClusterParams,
  toggleClusterSelection,
} from '../dataPageFilters';

describe('parseClusterParams', () => {
  it('reads a comma-separated clusters list', () => {
    expect(parseClusterParams(new URLSearchParams('clusters=3,1,2'))).toEqual([3, 1, 2]);
  });

  it('accepts a single legacy cluster param', () => {
    expect(parseClusterParams(new URLSearchParams('cluster=7'))).toEqual([7]);
  });

  it('prefers clusters over the legacy cluster param', () => {
    expect(parseClusterParams(new URLSearchParams('clusters=1,2&cluster=9'))).toEqual([1, 2]);
  });

  it('drops non-numeric entries rather than yielding NaN filters', () => {
    expect(parseClusterParams(new URLSearchParams('clusters=1,abc,3'))).toEqual([1, 3]);
  });

  it('returns nothing when neither param is present', () => {
    expect(parseClusterParams(new URLSearchParams('q=nebula'))).toEqual([]);
  });

  it('returns nothing for a non-numeric legacy param', () => {
    expect(parseClusterParams(new URLSearchParams('cluster=abc'))).toEqual([]);
  });
});

describe('buildFilterParams', () => {
  it('writes a single cluster as `cluster`', () => {
    const result = buildFilterParams(new URLSearchParams(), '', [5]);
    expect(result.get('cluster')).toBe('5');
    expect(result.has('clusters')).toBe(false);
  });

  it('writes several clusters as a sorted `clusters` list', () => {
    const result = buildFilterParams(new URLSearchParams(), '', new Set([9, 2, 5]));
    expect(result.get('clusters')).toBe('2,5,9');
    expect(result.has('cluster')).toBe(false);
  });

  it('round-trips through parseClusterParams', () => {
    const written = buildFilterParams(new URLSearchParams(), '', new Set([4, 1]));
    expect(parseClusterParams(written)).toEqual([1, 4]);
  });

  it('trims the query and omits it when blank', () => {
    expect(buildFilterParams(new URLSearchParams(), '  nebula  ', []).get('q')).toBe('nebula');
    expect(buildFilterParams(new URLSearchParams(), '   ', []).has('q')).toBe(false);
  });

  it('preserves unrelated params such as month and topic', () => {
    const result = buildFilterParams(
      new URLSearchParams('month=2026-01&topic=ai&q=old&cluster=1'),
      'new',
      []
    );
    expect(result.get('month')).toBe('2026-01');
    expect(result.get('topic')).toBe('ai');
    expect(result.get('q')).toBe('new');
    // Stale cluster selection must be cleared, not merged.
    expect(result.has('cluster')).toBe(false);
  });
});

describe('nextSortConfig', () => {
  it('starts a new field ascending', () => {
    expect(nextSortConfig({ field: 'starred_at', direction: 'desc' }, 'name')).toEqual({
      field: 'name',
      direction: 'asc',
    });
  });

  it('toggles asc to desc on the same field', () => {
    expect(nextSortConfig({ field: 'name', direction: 'asc' }, 'name')).toEqual({
      field: 'name',
      direction: 'desc',
    });
  });

  it('toggles desc back to asc on the same field', () => {
    expect(nextSortConfig({ field: 'name', direction: 'desc' }, 'name')).toEqual({
      field: 'name',
      direction: 'asc',
    });
  });
});

describe('toggleClusterSelection', () => {
  it('adds a cluster that is not selected', () => {
    expect(toggleClusterSelection(new Set([1]), 2)).toEqual(new Set([1, 2]));
  });

  it('removes a cluster that is selected', () => {
    expect(toggleClusterSelection(new Set([1, 2]), 2)).toEqual(new Set([1]));
  });

  it('does not mutate the input set', () => {
    const original = new Set([1]);
    toggleClusterSelection(original, 2);
    expect(original).toEqual(new Set([1]));
  });
});

describe('formatDataDate', () => {
  it('renders a dash for a missing date', () => {
    expect(formatDataDate(null)).toBe('-');
    expect(formatDataDate(undefined)).toBe('-');
    expect(formatDataDate('')).toBe('-');
  });

  it('formats an ISO date', () => {
    expect(formatDataDate('2026-08-18T00:00:00Z')).not.toBe('-');
  });
});
