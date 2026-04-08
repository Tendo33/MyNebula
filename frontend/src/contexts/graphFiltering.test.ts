import { describe, expect, it } from 'vitest';

import type { GraphData, TimelineData } from '../types';
import {
  buildGraphFilterIndexes,
  createVisibleNodeIds,
  filterVisibleClusters,
  filterVisibleEdges,
  filterVisibleNodes,
} from './graphFiltering';

const rawData: GraphData = {
  nodes: [
    {
      id: 1,
      github_id: 11,
      full_name: 'octo/nebula',
      name: 'nebula',
      description: 'Semantic explorer',
      language: 'TypeScript',
      html_url: 'https://example.com/1',
      owner: 'octo',
      x: 0,
      y: 0,
      z: 0,
      cluster_id: 1,
      color: '#111111',
      size: 1,
      star_list_id: 5,
      stargazers_count: 120,
      ai_summary: 'Graph tooling',
      ai_tags: ['graph'],
      topics: ['visualization'],
      starred_at: '2026-02-01T00:00:00Z',
      last_commit_time: '2026-02-03T00:00:00Z',
    },
    {
      id: 2,
      github_id: 22,
      full_name: 'octo/agent',
      name: 'agent',
      description: 'Automation workflows',
      language: 'Python',
      html_url: 'https://example.com/2',
      owner: 'octo',
      x: 1,
      y: 1,
      z: 1,
      cluster_id: 2,
      color: '#222222',
      size: 1,
      star_list_id: 6,
      stargazers_count: 40,
      ai_summary: 'Agent runtime',
      ai_tags: ['automation'],
      topics: ['agents'],
      starred_at: '2026-03-01T00:00:00Z',
      last_commit_time: '2026-03-03T00:00:00Z',
    },
  ],
  edges: [{ source: 1, target: 2, weight: 0.8 }],
  clusters: [
    { id: 1, name: 'Visualization', description: '', keywords: ['graph'], color: '#111111', repo_count: 1 },
    { id: 2, name: 'Automation', description: '', keywords: ['agent'], color: '#222222', repo_count: 1 },
  ],
  star_lists: [],
  total_nodes: 2,
  total_edges: 1,
  total_clusters: 2,
  total_star_lists: 0,
  version: 'snapshot-a',
  generated_at: '2026-04-08T00:00:00Z',
  request_id: 'req-1',
};

const timelineData: TimelineData = {
  points: [
    { date: '2026-02', count: 1, repos: [], top_languages: [], top_topics: [] },
    { date: '2026-03', count: 1, repos: [], top_languages: [], top_topics: [] },
  ],
  total_stars: 2,
  date_range: ['2026-02', '2026-03'],
};

describe('graph filtering', () => {
  it('keeps filtering behavior stable for combined filters', () => {
    const indexes = buildGraphFilterIndexes(rawData);
    const visibleNodes = filterVisibleNodes({
      rawData,
      timelineData,
      indexes,
      filters: {
        selectedClusters: new Set([1]),
        selectedStarLists: new Set(),
        searchQuery: 'graph',
        timeRange: [0, 0],
        minStars: 100,
        languages: new Set(['TypeScript']),
      },
    });

    expect(visibleNodes.map((node) => node.id)).toEqual([1]);
    const visibleNodeIds = createVisibleNodeIds(visibleNodes);
    expect(filterVisibleEdges(rawData.edges, visibleNodeIds)).toEqual([]);
    expect(filterVisibleClusters(rawData.clusters, visibleNodes).map((cluster) => cluster.id)).toEqual([1]);
  });

  it('supports stars threshold queries through the shared matcher path', () => {
    const indexes = buildGraphFilterIndexes(rawData);
    const visibleNodes = filterVisibleNodes({
      rawData,
      timelineData,
      indexes,
      filters: {
        selectedClusters: new Set(),
        selectedStarLists: new Set(),
        searchQuery: 'stars:>50',
        timeRange: null,
        minStars: 0,
        languages: new Set(),
      },
    });

    expect(visibleNodes.map((node) => node.id)).toEqual([1]);
  });
});
