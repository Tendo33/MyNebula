import { describe, expect, it } from 'vitest';

import { buildAdjacency, emptyFilters, filterNodes, hasActiveFilters } from './filters';
import { buildOverviewLinks, classifyLink, convexHull, nodeColor, nodeRadius, COLORS, type FocusContext } from './graph/style';
import { filterRepos, nextSort, sortRepos } from './pages/dataQuery';
import { buildTagFacets, matchesRepoSearch, normalizeTag, parseStarsThreshold } from './search';
import type { GraphNode } from './types';

const node = (id: number, patch: Partial<GraphNode> = {}): GraphNode => ({
  id,
  full_name: `o/repo-${id}`,
  name: `repo-${id}`,
  owner: 'o',
  html_url: '',
  stargazers_count: id * 10,
  cluster_id: id % 2,
  star_list_id: null,
  ...patch,
});

describe('search', () => {
  it('matches across text fields and supports stars:>N', () => {
    const repo = node(1, { ai_tags: ['Vector DB'], stargazers_count: 500 });
    expect(matchesRepoSearch(repo, 'vector')).toBe(true);
    expect(matchesRepoSearch(repo, 'stars:>499')).toBe(true);
    expect(matchesRepoSearch(repo, 'stars:>500')).toBe(false);
    expect(parseStarsThreshold(' STARS: > 10 ')).toBe(10);
  });

  it('pools ai tags and topics into one facet', () => {
    const facets = buildTagFacets([node(1, { ai_tags: ['rag'], topics: ['rag'] }), node(2, { topics: ['cli'] })]);
    expect(facets[0]).toEqual(['rag', 2]);
  });

  it('normalizes tags like the pipeline', () => {
    expect(normalizeTag(' AI_Agents ')).toBe('ai-agents');
    expect(normalizeTag('检索 增强')).toBe('检索-增强');
  });
});

describe('graph filters', () => {
  const nodes = [
    node(1, { language: 'Go', starred_at: '2024-01-02' }),
    node(2, { language: 'Rust', starred_at: '2024-03-02' }),
  ];
  const timeline = [
    { date: '2024-01', count: 1, repos: [], top_languages: [], top_topics: [] },
    { date: '2024-03', count: 1, repos: [], top_languages: [], top_topics: [] },
  ];

  it('combines cluster, language, query, and time range', () => {
    const filters = { ...emptyFilters(), languages: new Set(['Rust']) };
    expect(filterNodes(nodes, timeline, filters, new Map()).map((n) => n.id)).toEqual([2]);
    expect(filterNodes(nodes, timeline, { ...emptyFilters(), timeRange: [0, 0] }, new Map()).map((n) => n.id)).toEqual([1]);
    expect(filterNodes(nodes, timeline, { ...emptyFilters(), clusters: new Set([1]) }, new Map()).map((n) => n.id)).toEqual([1]);
    expect(hasActiveFilters(emptyFilters())).toBe(false);
  });

  it('builds an undirected adjacency index', () => {
    const index = buildAdjacency([[1, 2, 0.5]]);
    expect([...index.get(2)!]).toEqual([1]);
  });
});

describe('graph style', () => {
  const ctx: FocusContext = {
    selectedId: 1,
    hover: null,
    neighbors: new Set([3]),
    visible: new Set([1, 2, 3, 4]),
  };

  it('colors by focus: selected, linked neighbour, dimmed cluster-mate, ghost', () => {
    expect(nodeColor({ id: 1, cluster_id: 7, color: '#f00' }, ctx)).toBe('#f00');
    expect(nodeColor({ id: 2, cluster_id: 7, color: '#f00' }, ctx)).toBe(COLORS.NODE_DIM);
    expect(nodeColor({ id: 3, cluster_id: 9, color: '#0f0' }, ctx)).toBe('#0f0');
    expect(nodeColor({ id: 4, cluster_id: 9, color: '#0f0' }, ctx)).toBe(COLORS.NODE_DIM);
    expect(nodeColor({ id: 5, cluster_id: 9, color: '#0f0' }, ctx)).toBe(COLORS.GHOST_NODE);
    const hovering = { ...ctx, selectedId: undefined, neighbors: new Set<number>(), hover: { id: 1, cluster_id: 7 } };
    expect(nodeColor({ id: 2, cluster_id: 7, color: '#f00' }, hovering)).toBe('#f00');
  });

  it('shows only the backbone when nothing is focused', () => {
    const links = [
      { sourceId: 1, targetId: 2, weight: 0.9 },
      { sourceId: 2, targetId: 3, weight: 0.8 },
      { sourceId: 1, targetId: 3, weight: 0.1 },
    ];
    const overview = buildOverviewLinks(links);
    expect(overview.has(links[2])).toBe(false);
    const idle = { ...ctx, selectedId: undefined, neighbors: new Set<number>(), showTrajectories: true, clusterOf: new Map() };
    expect(classifyLink(links[2], idle, false)).toBe('hidden');
    expect(classifyLink(links[0], idle, true)).toBe('default');
    expect(classifyLink(links[0], { ...idle, showTrajectories: false }, true)).toBe('hidden');
  });

  it('keeps radius within bounds and hulls convex', () => {
    expect(nodeRadius(0)).toBe(5);
    expect(nodeRadius(10 ** 9)).toBe(30);
    const hull = convexHull([
      { x: 0, y: 0 },
      { x: 2, y: 0 },
      { x: 1, y: 1 },
      { x: 2, y: 2 },
      { x: 0, y: 2 },
    ]);
    expect(hull).toHaveLength(4);
  });
});

describe('data page query', () => {
  const nodes = [
    node(1, { name: 'b', topics: ['CLI'], starred_at: '2024-05-01' }),
    node(2, { name: 'a', language: 'Go', starred_at: '2024-06-01' }),
    node(3, { name: 'c', starred_at: null }),
  ];

  it('filters by topic and month', () => {
    const base = { query: '', clusters: new Set<number>(), month: null, topic: null };
    expect(filterRepos(nodes, { ...base, topic: 'cli' }).map((n) => n.id)).toEqual([1]);
    expect(filterRepos(nodes, { ...base, month: '2024-06' }).map((n) => n.id)).toEqual([2]);
  });

  it('sorts with missing values last in both directions', () => {
    expect(sortRepos(nodes, { field: 'starred_at', direction: 'desc' }).map((n) => n.id)).toEqual([2, 1, 3]);
    expect(sortRepos(nodes, { field: 'starred_at', direction: 'asc' }).map((n) => n.id)).toEqual([1, 2, 3]);
    expect(nextSort({ field: 'name', direction: 'asc' }, 'name')).toEqual({ field: 'name', direction: 'desc' });
    expect(nextSort({ field: 'name', direction: 'desc' }, 'stargazers_count').direction).toBe('asc');
  });
});
