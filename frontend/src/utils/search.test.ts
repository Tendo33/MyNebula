import { describe, expect, it } from 'vitest';

import {
  buildRepoSearchText,
  matchesClusterSearch,
  matchesFacetSearch,
  matchesRepoSearch,
  normalizeSearchQuery,
  parseStarsThreshold,
} from './search';

describe('search utils', () => {
  it('normalizes and parses stars threshold queries', () => {
    expect(normalizeSearchQuery('  Stars:>42  ')).toBe('stars:>42');
    expect(parseStarsThreshold('stars:>42')).toBe(42);
    expect(parseStarsThreshold('repo')).toBeNull();
  });

  it('matches repo search across shared text fields', () => {
    const repo = {
      name: 'nebula',
      full_name: 'openai/nebula',
      description: 'Semantic graph explorer',
      ai_summary: 'Visualize starred repositories',
      language: 'TypeScript',
      ai_tags: ['graph'],
      topics: ['visualization'],
      stargazers_count: 88,
    };

    const searchableText = buildRepoSearchText(repo);

    expect(matchesRepoSearch(repo, 'semantic', searchableText)).toBe(true);
    expect(matchesRepoSearch(repo, 'visualization', searchableText)).toBe(true);
    expect(matchesRepoSearch(repo, 'stars:>50', searchableText)).toBe(true);
    expect(matchesRepoSearch(repo, 'stars:>100', searchableText)).toBe(false);
  });

  it('matches clusters and facets with the same normalized query', () => {
    expect(
      matchesClusterSearch(
        { name: 'AI Tooling', description: 'Agent workflows', keywords: ['automation'] },
        'automation'
      )
    ).toBe(true);
    expect(matchesFacetSearch('TypeScript', 'script')).toBe(true);
    expect(matchesFacetSearch('TypeScript', 'python')).toBe(false);
  });
});
