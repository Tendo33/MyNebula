import { describe, expect, it } from 'vitest';

import type { GraphNode } from '../../../types';
import {
  LANGUAGE_FACET_LIMIT,
  TAG_FACET_LIMIT,
  buildLanguageFacets,
  buildTagFacets,
} from '../commandPaletteFacets';

const node = (overrides: Partial<GraphNode> = {}): GraphNode =>
  ({
    id: 1,
    full_name: 'octo/nebula',
    name: 'nebula',
    language: 'TypeScript',
    ai_tags: [],
    topics: [],
    ...overrides,
  }) as GraphNode;

describe('buildLanguageFacets', () => {
  it('returns nothing without nodes', () => {
    expect(buildLanguageFacets(undefined)).toEqual([]);
    expect(buildLanguageFacets([])).toEqual([]);
  });

  it('ranks languages by repo count', () => {
    const nodes = [
      node({ id: 1, language: 'Python' }),
      node({ id: 2, language: 'TypeScript' }),
      node({ id: 3, language: 'Python' }),
      node({ id: 4, language: 'Python' }),
      node({ id: 5, language: 'TypeScript' }),
      node({ id: 6, language: 'Rust' }),
    ];

    expect(buildLanguageFacets(nodes)).toEqual([
      ['Python', 3],
      ['TypeScript', 2],
      ['Rust', 1],
    ]);
  });

  it('ignores nodes with no language', () => {
    expect(buildLanguageFacets([node({ language: undefined })])).toEqual([]);
  });

  it('caps the list at the facet limit', () => {
    const nodes = Array.from({ length: 30 }, (_, i) => node({ id: i, language: `Lang${i}` }));
    expect(buildLanguageFacets(nodes)).toHaveLength(LANGUAGE_FACET_LIMIT);
  });
});

describe('buildTagFacets', () => {
  it('returns nothing without nodes', () => {
    expect(buildTagFacets(undefined)).toEqual([]);
  });

  it('pools AI tags and GitHub topics into one namespace', () => {
    // The palette exposes a single "tag" facet, so the same label from either
    // source has to accumulate rather than appear twice.
    const nodes = [
      node({ id: 1, ai_tags: ['ai'], topics: ['ai'] }),
      node({ id: 2, ai_tags: [], topics: ['ai'] }),
    ];

    expect(buildTagFacets(nodes)).toEqual([['ai', 3]]);
  });

  it('ranks tags by count', () => {
    const nodes = [
      node({ id: 1, ai_tags: ['ml', 'ai'] }),
      node({ id: 2, ai_tags: ['ai'] }),
      node({ id: 3, ai_tags: ['ai'] }),
    ];

    expect(buildTagFacets(nodes)).toEqual([
      ['ai', 3],
      ['ml', 1],
    ]);
  });

  it('tolerates nodes with neither tags nor topics', () => {
    expect(buildTagFacets([node({ ai_tags: undefined, topics: undefined })])).toEqual([]);
  });

  it('caps the list at the facet limit', () => {
    const nodes = Array.from({ length: 40 }, (_, i) => node({ id: i, ai_tags: [`tag${i}`] }));
    expect(buildTagFacets(nodes)).toHaveLength(TAG_FACET_LIMIT);
  });
});
