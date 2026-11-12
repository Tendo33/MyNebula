import { describe, expect, it } from 'vitest';

import { buildAdjacency, emptyFilters, facetNodes, filterNodes, hasActiveFilters, timelineFacets } from './filters';
import { monthsFromTimeRange, readGraphLink, sharedViewParams, timeRangeFromMonths, writeGraphLink } from './graph/route';
import { buildOverviewLinks, classifyLink, convexHull, nodeColor, nodeRadius, COLORS, type FocusContext } from './graph/style';
import { filterRepos, nextSort, pageContaining, readLanguages, readMonth, readPage, readPageSize, readSort, clusterFacetCounts, sortRepos, starMonth } from './pages/dataQuery';
import { loadErrorDetail } from './loadError';
import { buildTagFacets, detailTopicLabels, matchesRepoSearch, normalizeTag, paletteTarget, parseStarsThreshold } from './search';
import type { GraphNode } from './types';

const node = (id: number, patch: Partial<GraphNode> = {}): GraphNode => ({
  id,
  full_name: `o/repo-${id}`,
  name: `repo-${id}`,
  owner: 'o',
  html_url: '',
  stargazers_count: id * 10,
  cluster_id: id % 2,
  star_list_ids: [],
  ...patch,
});

describe('search', () => {
  it('matches across text fields and supports stars:>N', () => {
    const repo = node(1, { ai_tags: ['Vector DB'], stargazers_count: 500 });
    expect(matchesRepoSearch(repo, 'vector')).toBe(true);
    expect(matchesRepoSearch(repo, 'stars:>499')).toBe(true);
    expect(matchesRepoSearch(repo, 'stars:>500')).toBe(false);
    expect(parseStarsThreshold(' STARS: > 10 ')).toBe(10);
    expect(paletteTarget('/data')).toBe('/data');
    expect(paletteTarget('/settings')).toBe('/');
    expect(paletteTarget('/graph')).toBe('/');
  });

  it('keeps GitHub topics filterable and leaves AI-only labels as text', () => {
    expect(detailTopicLabels(['RAG', 'cli'], ['rag', 'agent'])).toEqual([
      { label: 'RAG', filterable: true },
      { label: 'cli', filterable: true },
      { label: 'agent', filterable: false },
    ]);
    expect(detailTopicLabels(['a', 'b', 'c', 'd', 'e'], []).map((item) => item.label)).toEqual(['a', 'b', 'c', 'd']);
  });

  it('pools ai tags and topics into one facet', () => {
    const facets = buildTagFacets([node(1, { ai_tags: ['rag'], topics: ['rag'] }), node(2, { topics: ['cli'] })]);
    expect(facets[0]).toEqual(['rag', 2]);
  });

  it('normalizes tags like the pipeline', () => {
    expect(normalizeTag(' AI_Agents ')).toBe('agent');
    expect(normalizeTag('检索增强生成')).toBe('rag');
    expect(normalizeTag('检索 增强')).toBe('检索-增强');
    expect(matchesRepoSearch(node(1, { ai_tags: ['智能体'] }), 'agent')).toBe(true);
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

  it('keeps a repo that shares any selected star list', () => {
    const listed = [
      node(1, { star_list_ids: ['alpha', 'beta'] }),
      node(2, { star_list_ids: ['beta'] }),
      node(3, { star_list_ids: ['uncategorized'] }),
    ];
    expect(filterNodes(listed, [], { ...emptyFilters(), lists: new Set(['alpha']) }, new Map()).map((n) => n.id)).toEqual([1]);
    expect(filterNodes(listed, [], { ...emptyFilters(), lists: new Set(['beta']) }, new Map()).map((n) => n.id)).toEqual([1, 2]);
  });

  it('counts a facet without its own selection or the search box', () => {
    const filters = { ...emptyFilters(), languages: new Set(['Rust']), clusters: new Set([1]), query: 'zzz' };
    expect(facetNodes(nodes, timeline, filters, 'clusters').map((n) => n.id)).toEqual([2]);
    expect(facetNodes(nodes, timeline, filters, 'languages').map((n) => n.id)).toEqual([1]);
  });

  it('counts timeline months without the selected window or the search box', () => {
    const filters = { ...emptyFilters(), languages: new Set(['Go']), timeRange: [0, 0] as [number, number], query: 'zzz' };
    const facets = timelineFacets(
      [
        node(1, { language: 'Go', starred_at: '2024-01-02' }),
        node(2, { language: 'Rust', starred_at: '2024-01-03' }),
        node(3, { language: 'Go', starred_at: '2024-03-02' }),
      ],
      timeline,
      filters,
    );
    expect(facets.map((facet) => facet.count)).toEqual([1, 1]);
    expect(facets[0]?.topLanguages).toEqual(['Go']);
  });

  it('round-trips list ids and months instead of timeline indexes', () => {
    const timeline = [
      { date: '2024-01' },
      { date: '2024-03' },
      { date: '2024-06' },
    ];
    const params = writeGraphLink({
      nodeId: 7,
      clusters: [2, 1],
      lists: ['beta', 'alpha'],
      languages: ['Go'],
      query: 'stars:>1000',
      from: '2024-03',
      to: '2024-03',
    });
    expect(params.toString()).toBe('node=7&clusters=1%2C2&lists=alpha%2Cbeta&language=Go&q=stars%3A%3E1000&from=2024-03&to=2024-03');
    const link = readGraphLink(params);
    expect(link.lists).toEqual(['alpha', 'beta']);
    expect(link.languages).toEqual(['Go']);
    expect(readGraphLink(new URLSearchParams('languages=Python,Go')).languages).toEqual(['Python', 'Go']);
    expect(timeRangeFromMonths(timeline, link.from, link.to)).toEqual([1, 1]);
    expect(timeRangeFromMonths(timeline, '2024-01', '2024-06')).toBeNull();
    expect(monthsFromTimeRange(timeline, [0, 1])).toEqual({ from: '2024-01', to: '2024-03' });
    expect(readGraphLink(new URLSearchParams('tag=old&list=alpha')).query).toBe('old');
    expect(readGraphLink(new URLSearchParams('list=alpha')).lists).toEqual(['alpha']);
    const carried = sharedViewParams(
      new URLSearchParams('q=rag&node=7&cluster=0&lists=alpha&from=2024-03&to=2024-03&month=2024-06&sort=stargazers_count&page=2'),
    );
    expect(carried.toString()).toBe('q=rag&node=7&cluster=0');
    expect(sharedViewParams(new URLSearchParams('tag=agent&languages=Go,Python')).toString()).toBe('q=agent&languages=Go%2CPython');
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

describe('load error detail', () => {
  it('keeps a short reason and caps a long one', () => {
    expect(loadErrorDetail(' HTTP  500 ')).toBe('HTTP 500');
    expect(loadErrorDetail(null)).toBe('');
    expect(loadErrorDetail('x'.repeat(200))).toBe(`${'x'.repeat(179)}…`);
  });
});

describe('data page query', () => {
  const nodes = [
    node(1, { name: 'b', topics: ['CLI'], starred_at: '2024-05-01' }),
    node(2, { name: 'a', language: 'Go', starred_at: '2024-06-01' }),
    node(3, { name: 'c', starred_at: null }),
  ];

  it('filters by topic and month', () => {
    const base = { query: '', clusters: new Set<number>(), languages: new Set<string>(), month: null, topic: null };
    expect(filterRepos(nodes, { ...base, topic: 'cli' }).map((n) => n.id)).toEqual([1]);
    expect(filterRepos(nodes, { ...base, month: '2024-06' }).map((n) => n.id)).toEqual([2]);
    expect(filterRepos(nodes, { ...base, languages: new Set(['Go']) }).map((n) => n.id)).toEqual([2]);
    expect([...clusterFacetCounts(nodes, { ...base, languages: new Set(['Go']), clusters: new Set([1]) })]).toEqual([[0, 1]]);
  });

  it('sorts with missing values last in both directions', () => {
    expect(sortRepos(nodes, { field: 'starred_at', direction: 'desc' }).map((n) => n.id)).toEqual([2, 1, 3]);
    expect(sortRepos(nodes, { field: 'starred_at', direction: 'asc' }).map((n) => n.id)).toEqual([1, 2, 3]);
    expect(nextSort({ field: 'name', direction: 'asc' }, 'name')).toEqual({ field: 'name', direction: 'desc' });
    expect(nextSort({ field: 'name', direction: 'desc' }, 'stargazers_count').direction).toBe('asc');
  });

  it('reads a shared sort, page, and page size', () => {
    const params = new URLSearchParams('sort=stargazers_count&dir=asc&page=2&size=50');
    expect(readSort(params)).toEqual({ field: 'stargazers_count', direction: 'asc' });
    expect(readPage(params)).toBe(2);
    expect(readPageSize(params)).toBe(50);
    expect(readSort(new URLSearchParams('sort=nope'))).toEqual({ field: 'starred_at', direction: 'desc' });
    expect(readPage(new URLSearchParams('page=0'))).toBe(1);
    expect(pageContaining([10, 20, 30], 30, 2)).toBe(2);
    expect(pageContaining([10, 20, 30], 99, 2)).toBeNull();
    expect(readPageSize(new URLSearchParams('size=15'))).toBe(25);
    expect(readLanguages(new URLSearchParams('language=Python'))).toEqual(['Python']);
    expect(readLanguages(new URLSearchParams('languages=Go,Python'))).toEqual(['Go', 'Python']);
    expect(starMonth('2024-06-01T00:00:00Z')).toBe('2024-06');
    expect(starMonth(null)).toBeNull();
    expect(readMonth(new URLSearchParams('month=2024-06'))).toBe('2024-06');
    expect(readMonth(new URLSearchParams('month=nope'))).toBeNull();
  });
});
