import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, waitFor } from '@testing-library/react';
import type { ReactNode } from 'react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const graphApiMocks = vi.hoisted(() => ({
  getGraphDataV2: vi.fn(),
  getTimelineDataV2: vi.fn(),
  getGraphEdgesPageV2: vi.fn(),
}));

vi.mock('../../api/v2/graph', () => ({
  getGraphDataV2: graphApiMocks.getGraphDataV2,
  getTimelineDataV2: graphApiMocks.getTimelineDataV2,
  getGraphEdgesPageV2: graphApiMocks.getGraphEdgesPageV2,
}));

import { GraphProvider, useGraph, useNodeNeighbors } from '../GraphContext';

const node = (id: number) => ({
  id,
  github_id: id + 100,
  full_name: `octo/repo-${id}`,
  name: `repo-${id}`,
  description: null,
  language: 'TypeScript',
  html_url: `https://github.com/octo/repo-${id}`,
  owner: 'octo',
  owner_avatar_url: null,
  x: id,
  y: id,
  z: 0,
  cluster_id: 1,
  color: '#fff',
  size: 1,
  star_list_id: -1,
  star_list_name: 'Uncategorized',
  stargazers_count: 10,
  ai_summary: null,
  ai_tags: [],
  topics: [],
  starred_at: null,
  last_commit_time: null,
});

const graphPayload = {
  nodes: [node(1), node(2), node(3)],
  edges: [],
  clusters: [],
  star_lists: [],
  total_nodes: 3,
  total_edges: 2,
  total_clusters: 0,
  total_star_lists: 0,
  version: 'v1',
  generated_at: '2026-08-18T00:00:00Z',
  request_id: 'req-1',
};

const edgesPage = {
  edges: [
    { source: 1, target: 2, weight: 0.9 },
    { source: 2, target: 3, weight: 0.8 },
  ],
  next_cursor: null,
  total_edges: 2,
  version: 'v1',
  generated_at: '2026-08-18T00:00:00Z',
  request_id: 'req-1',
};

const seenIndexes: Array<Map<number, Set<number>>> = [];
const seenNeighbors: Array<Set<number>> = [];

const ConsumerA = () => {
  const { adjacencyIndex } = useGraph();
  seenIndexes.push(adjacencyIndex);
  const neighbors = useNodeNeighbors(2);
  seenNeighbors.push(neighbors);
  return <div data-testid="a">{[...neighbors].sort().join(',')}</div>;
};

const ConsumerB = () => {
  const { adjacencyIndex } = useGraph();
  seenIndexes.push(adjacencyIndex);
  return <div data-testid="b">{adjacencyIndex.size}</div>;
};

const wrapper = ({ children }: { children: ReactNode }) => (
  <QueryClientProvider
    client={
      new QueryClient({
        defaultOptions: { queries: { retry: false, gcTime: 0 } },
      })
    }
  >
    <GraphProvider enabled>{children}</GraphProvider>
  </QueryClientProvider>
);

describe('GraphProvider adjacency index', () => {
  beforeEach(() => {
    seenIndexes.length = 0;
    seenNeighbors.length = 0;
    graphApiMocks.getGraphDataV2.mockResolvedValue(graphPayload);
    graphApiMocks.getTimelineDataV2.mockResolvedValue({
      points: [],
      total_stars: 0,
      date_range: ['', ''],
      version: 'v1',
      generated_at: '2026-08-18T00:00:00Z',
      request_id: 'req-1',
    });
    graphApiMocks.getGraphEdgesPageV2.mockResolvedValue(edgesPage);
  });

  it('shares one index instance across every consumer', async () => {
    render(
      <>
        <ConsumerA />
        <ConsumerB />
      </>,
      { wrapper }
    );

    await waitFor(() => {
      expect(seenIndexes.at(-1)?.size).toBeGreaterThan(0);
    });

    // Both consumers in the final render pass must observe the same object.
    // Before this change each one built its own O(E) copy on every edge page.
    const finalPass = seenIndexes.slice(-2);
    expect(finalPass[0]).toBe(finalPass[1]);
  });

  it('resolves neighbours in both directions', async () => {
    const { getByTestId } = render(
      <>
        <ConsumerA />
        <ConsumerB />
      </>,
      { wrapper }
    );

    await waitFor(() => {
      expect(getByTestId('a').textContent).toBe('1,3');
    });
    // Three nodes appear in the adjacency map: 1, 2, and 3.
    expect(getByTestId('b').textContent).toBe('3');
  });
});
